from collections import deque
from Arena import Arena
from GoMCTS import MCTS
from go.GoGame import display
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.classic import go_v5 as go


class Coach():
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

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        episode_step = 0

        # create the go environment for each episode
        env = go.env(board_size=self.args['board_size'])
        env = env
        env.reset()

        action_history = []

        for agent in env.agent_iter():
            episode_step += 1

            # get information about previous state
            obs, reward, termination, truncation, info = env.last()
            canonical_form = self.game.get_pz_canonical_form(self.args['board_size'], obs)

            if termination or truncation:
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
                env.close()

                return [(x[0], x[2], reward * ((-1) ** (x[1] != env.agent_selection))) for x in trainExamples]

            temp = int(episode_step < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonical_form, env, action_history, temp=temp)

            sym = self.game.getSymmetries(canonical_form, pi)
            for b, p in sym:
                trainExamples.append([b, env.agent_selection, p, None])

            action = np.random.choice(len(pi), p=pi)
            action_history.append(action)

            # print board state and useful information
            # current player is the player who is about to play next
            if self.display == 2:
                print(
                    f"================Episode {self.currentEpisode} Step:{episode_step}=====CURPLAYER:{agent}==========")

                if agent == "white_0":
                    is_white_player = 1
                    is_black_player = 0
                else:
                    is_white_player = 0
                    is_black_player = 1

                # 1 is always current player
                for i in range(self.args['board_size']):
                    for j in range(self.args['board_size']):
                        if obs['observation'][i, j, is_white_player] == 1:
                            print('W', end=' ')  # White stone
                        elif obs['observation'][i, j, is_black_player] == 1:
                            print('b', end=' ')  # Black stone
                        else:
                            print('.', end=' ')  # Empty intersection
                    print()  # New line for each row

            env.step(action)

    """
        while True:

            episodeStep += 1
            if self.display == 2:
                print("================Episode {} Step:{}=====CURPLAYER:{}==========".format(self.currentEpisode, episodeStep,
                                                                                          "White" if self.curPlayer == -1 else "Black"))
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            # get different symmetries/rotations of the board
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            action = np.random.choice(len(pi), p=pi)

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            if self.display == 2:
                print("BOARD updated:")
                # display(board)
                print(display(board))
            r, score = self.game.getGameEnded(board.copy(), self.curPlayer, returnScore=True)
            if r != 0:
                if self.display == 2:
                    print("Current episode ends, {} wins with score b {}, W {}.".format('Black' if r == -1 else 'White',
                                                                                        score[0], score[1]))

                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
            elif r == 0 and self.display == 2:
                print(f"Current score: b {score[0]}, W {score[1]}")
    """

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
            # bookkeeping
            print('###########################ITER:{}###########################'.format(str(i)))
            arena_log = open(f'logs/go/Game_Histories/Game_History_{self.args.datetime}.txt', 'a')
            arena_log.write("##########################################\n")
            arena_log.write("ITERATION: " + str(i) + "\n")
            arena_log.write("##########################################\n\n")
            arena_log.close()
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

            print('\nPITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x, y, z: (pmcts.getActionProb(x, y, z, temp=0)),
                          lambda x, y, z: (nmcts.getActionProb(x, y, z, temp=0)), self.game, self.args.datetime,
                          display=display,
                          displayValue=self.display.value)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, iter=i)
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
