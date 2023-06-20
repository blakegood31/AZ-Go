import math
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

            # end game if too many moves have been played
            # equation roughly translates max moves to:
            # 5x5 = 120 moves
            # 7x7 = 173 moves
            # 9x9 = 227 moves
            # 19x19 = 498 moves
            if episode_step > 17651 * math.e**(0.0015 * self.game.getBoardSize()[0]) - 17664:
                reward = -0.0001
                print("maximum number of moves reached, game terminating...")
                return [(x[0], x[2], reward * ((-1) ** (x[1] != episode_env.agent_selection))) for x in train_examples]

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
            # print("Current Score = ", score)
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
                print('###########################ITER:{}###########################'.format(str(i)))

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

            # logging functions
            self.create_sgf_files_for_games(games=all_arena_games_history, iteration=i)
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

    def create_sgf_files_for_games(self, games, iteration):

        for game_idx in range(len(games)):
            file_name = f'logs/go/Game_History/Iteration {iteration}, Game {game_idx + 1} {self.args.datetime}.sgf'
            sgf_file = open(file_name, 'w')
            sgf_file.close()

            sgf_file = open(file_name, 'a')
            sgf_file.write(
                f"(;\nEV[AlphaGo Self-Play]\nGN[Iteration {iteration}]\nDT[{self.args.datetime}]\nPB[TCU_AlphaGo]\nPW[TCU_AlphaGo]"
                f"\nSZ[{self.game.getBoardSize()[0]}]\nRU["
                f"Chinese]\n\n")

            for move_idx in range(len(games[game_idx])):
                sgf_file.write(games[game_idx][move_idx])

            sgf_file.write("\n)")
            sgf_file.close()
