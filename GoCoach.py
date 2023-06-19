from collections import deque
from Arena import Arena
from GoMCTS import MCTS
import numpy as np

from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.classic import go_v5 as go


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, log=False, logPath=''):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, t=self.nnet.netType)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.display = args['display']
        self.keepLog = log
        self.logPath = logPath
        self.p_loss_per_iteration = []
        self.v_loss_per_iteration = []
        self.winRate = []
        self.currentEpisode = 0
        self.sgf_output = ""

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        episode_step = 0

        # create the go environment for each episode
        # episode_env = go.env(board_size=self.args['board_size'])
        episode_env = go.env(board_size=self.args['board_size'])
        episode_env.reset()

        for agent in episode_env.agent_iter():
            episode_step += 1

            # get information about previous state
            obs, reward, termination, truncation, info = episode_env.last()
            canonical_form = self.game.get_pz_canonical_form(self.args['board_size'], obs)
            # temp reward instead of termination // truncation
            # in case of infinitely long game
            if reward != 0:
                if agent == 'black_0' and reward == 1:
                    print("Black Won!")
                elif agent == 'white_0' and reward == 1:
                    print("White Won!")
                elif agent == 'black_0' and reward == -1:
                    print("White Won!")
                elif agent == 'white_0' and reward == -1:
                    print("Black Won!")
                else:
                    print("Ended Early")

                print("Episode Complete\n")
                episode_env.close()

                return [(x[0], x[2], reward * ((-1) ** (x[1] != episode_env.agent_selection))) for x in train_examples]

            # End game if a player is winning by certain threshold
            score = episode_env.unwrapped._go.score()
            if ((score > self.args['by_score']) or (score < -self.args['by_score'])) and episode_step > 14:
                reward = 0
                if score > 0:
                    print("Black Won! By Score: ", score)
                    reward = 1
                else:
                    print("White Won! By Score: ", score)
                    reward = -1
                return [(x[0], x[2], reward * ((-1) ** (x[1] != episode_env.agent_selection))) for x in train_examples]

            temp = int(episode_step < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonical_form, episode_env, temp=temp)

            sym = self.game.getSymmetries(canonical_form, pi)
            for b, p in sym:
                train_examples.append([b, episode_env.agent_selection, p, None])

            action = np.random.choice(len(pi), p=pi)
            # print("Player: ", episode_env.agent_selection, "  Chose action: ", action)
            # print board state and useful information
            # current player is the player who is about to play next
            episode_env.step(action)
            score = episode_env.unwrapped._go.score()
            print("Current Score = ", score)
            if self.display == 2:
                print(
                    f"================Episode {self.currentEpisode} Step:{episode_step}=====Next Player:{agent}==========")

                self.game.display_pz_board(board_size=self.args['board_size'], observation=obs, agent=agent)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        iterHistory = {'ITER': [], 'ITER_DETAIL': [], 'PITT_RESULT': []}

        for i in range(1, self.args.numIters + 1):
            iterHistory['ITER'].append(i)

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                eps_time = AverageMeter()
                if self.display == 1:
                    bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    # print("{}th Episode:".format(eps+1))
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    self.currentEpisode = eps + 1
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()

                    if self.display == 1:
                        bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                            total=bar.elapsed_td, eta=bar.eta_td)
                        bar.next()

                if self.display == 1:
                    bar.finish()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                # print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            trainLog = self.nnet.train(trainExamples)
            self.p_loss_per_iteration.append(np.average(trainLog['P_LOSS'].to_numpy()))
            self.v_loss_per_iteration.append(np.average(trainLog['V_LOSS'].to_numpy()))
            if self.keepLog:
                trainLog.to_csv(self.logPath + 'ITER_{}_TRAIN_LOG.csv'.format(i))

            iterHistory['ITER_DETAIL'].append(self.logPath + 'ITER_{}_TRAIN_LOG.csv'.format(i))
            nmcts = MCTS(self.game, self.nnet, self.args)

            # clear sgf_output for next game and create new file
            self.sgf_output = ""
            self.create_sgf_file_for_game(iteration=i)

            print('\nPITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x, y: (pmcts.getActionProb(x, y, temp=1)),
                          lambda x, y: (nmcts.getActionProb(x, y, temp=1)), self.game, self.args.datetime,
                          display_value=self.display.value)
            pwins, nwins, draws, all_arena_games_history = arena.playGames(self.args.arenaCompare)
            self.winRate.append(nwins / self.args.arenaCompare)
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                iterHistory['PITT_RESULT'].append('R')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                iterHistory['PITT_RESULT'].append('A')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            pd.DataFrame(data=iterHistory).to_csv(self.logPath + 'ITER_LOG.csv')

            # combine all games into 1 tree and save to sgf output file
            self.combine_games_into_sgf_tree(iteration=i, games=all_arena_games_history)
            self.end_sgf_file_for_game(iteration=i, games=all_arena_games_history)

            self.saveTrainingPlots()

        pd.DataFrame(data=iterHistory).to_csv(self.logPath + 'ITER_LOG.csv')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile  # + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    # plot/save v/p loss after training
    # plot/save Arena Play Win Rates after arena
    def saveTrainingPlots(self):
        # close previous graph
        plt.close()

        plt.rcParams["figure.figsize"] = (17, 10)

        plt.subplot(2, 2, 1)
        plt.title("V Loss During Training")
        plt.ylabel('V Loss')
        plt.xlabel('Iteration')
        plt.locator_params(axis='x', integer=True, tight=True)
        plt.plot(self.v_loss_per_iteration, label="V Loss")

        plt.subplot(2, 2, 2)
        plt.title("P Loss During Training")
        plt.ylabel('P Loss')
        plt.xlabel('Iteration')
        plt.locator_params(axis='x', integer=True, tight=True)
        plt.plot(self.p_loss_per_iteration, label="P Loss")

        plt.subplot(2, 2, 3)
        plt.title('Arena Play Win Rates (New Model vs. Old Model)')
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate (%)')
        plt.locator_params(axis='x', integer=True, tight=True)
        plt.axhline(y=self.args.updateThreshold, color='b', linestyle='-')
        plt.plot(self.winRate, 'r', label='Win Rate')

        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        dpi = 100
        plt.savefig(f"logs/go/Training_Results/Training_Result_{self.args.datetime}.png", dpi=dpi)

    # Smart Game Format Helper Functions

    # create the file and a header for use later using sgf
    # https://en.wikipedia.org/wiki/Smart_Game_Format
    def create_sgf_file_for_game(self, iteration):
        sgf_file = open(f'logs/go/Game_History/Iteration {iteration}_Games_{self.args.datetime}.sgf', 'w')
        sgf_file.write(
            f"(;\nEV[AlphaGo Self-Play]\nGN[Iteration {iteration}]\nDT[{self.args.datetime}]\nPB[TCU_AlphaGo]\nPW[TCU_AlphaGo]"
            f"\nSZ[{self.game.getBoardSize()[0]}]\nKM[7.5]\nRU["
            f"Chinese]\n\n")
        sgf_file.close()

    def combine_games_into_sgf_tree(self, games, iteration):
        self.sgf_output += "("

        for game in range(0, 1):
            for move in range(len(games[game])):
                self.sgf_output += games[game][move]

        self.sgf_output += ")"

        for game_idx in range(1, len(games)):
            pointer = 0
            should_search = True
            for move_idx in range(len(games[game_idx])):
                current_move = games[game_idx][move_idx]
                if should_search:
                    # return pointer to fill in rest of game
                    pointer, should_search = self.search_sgf_string(pointer, current_move)
                else:
                    # split the output string and add in move where needed
                    p1 = self.sgf_output[:pointer]
                    p2 = self.sgf_output[pointer:]
                    self.sgf_output = p1 + current_move + p2
                    pointer += 6

        # write string into file
        sgf_file = open(f'logs/go/Game_History/Iteration {iteration}_Games_{self.args.datetime}.sgf', 'a')
        sgf_file.write(self.sgf_output)
        sgf_file.close()

    def search_sgf_string(self, ptr, move_to_add):
        # each move is 6 characters unless it is a passing move, then it is 4 characters

        move_char_count = 6

        if move_to_add.find("[]") != -1:
            # move_to_add is a passing move
            move_char_count = 4

        move_at_ptr = self.sgf_output[ptr:ptr + move_char_count]

        if move_at_ptr[0] == '(':
            # move_at_ptr is part of a branch declaration
            # search again without branch declaration
            return self.search_sgf_string(ptr + 1, move_to_add)

        # do the moves match?
        if move_to_add == move_at_ptr:
            # yes, moves match
            # search again with a new move_to_add
            return ptr + move_char_count, True

        else:
            # no, moves do not match

            # if at beginning of branch
            # jump to next branch at same level
            if self.sgf_output[ptr - 1] == '(':

                idx_left_paren = self.sgf_output.find('(', ptr)
                idx_right_paren = self.sgf_output.find(')', ptr)

                if idx_left_paren == -1:
                    # left paren '(' does not exist in substring
                    # make branch at end of output string
                    ptr = len(self.sgf_output) - 1
                    p1 = self.sgf_output

                    # add the move itself
                    branch = f'({move_to_add})'
                    self.sgf_output = p1 + branch

                    return ptr + move_char_count + 2, False

                # looking for ')(' pattern
                while idx_right_paren > idx_left_paren:
                    ptr = idx_right_paren
                    idx_left_paren = self.sgf_output.find('(', ptr)
                    idx_right_paren = self.sgf_output.find(')', ptr)

                return self.search_sgf_string(idx_left_paren, move_to_add)

            else:
                # move_at_ptr is not the first move of a branch
                # make a new branch

                # add a ")" before adding the move
                idx_of_next_paren = self.sgf_output.find(')', ptr)
                p1 = self.sgf_output[:idx_of_next_paren]
                p2 = self.sgf_output[idx_of_next_paren:]
                self.sgf_output = p1 + ")" + p2

                # add the move itself
                p1 = self.sgf_output[:ptr]
                p2 = self.sgf_output[ptr:]
                branch = f'({move_to_add})('
                self.sgf_output = p1 + branch + p2

                return ptr + move_char_count + 1, False

    def end_sgf_file_for_game(self, iteration, games):
        sgf_file = open(f'logs/go/Game_History/Iteration {iteration}_Games_{self.args.datetime}.sgf', 'a')
        sgf_file.write(f'\n\nC[Unformatted Log of Games: \n{games}\n]\n\n)')
        sgf_file.close()
