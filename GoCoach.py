import multiprocessing
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
        game = self.game
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        mcts = MCTS(game, self.nnet, self.args)

        while True:
            episodeStep += 1
            if self.display == 1:
                print("\n================Episode {} Step:{}=====CURPLAYER:{}==========".format(self.currentEpisode + 1,
                                                                                             episodeStep,
                                                                                             "White" if curPlayer == -1 else "Black"))
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            # get different symmetries/rotations of the board
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])
            action = np.random.choice(len(pi), p=pi)

            board, curPlayer = game.getNextState(board, curPlayer, action)
            if self.display == 1:
                print("BOARD updated:")
                # display(board)
                print(display(board))
            r, score = game.getGameEnded(board.copy(), curPlayer, returnScore=True)
            if r != 0:
                if self.display == 1:
                    print("Current episode ends, {} wins with score b {}, W {}.".format('Black' if r == -1 else 'White',
                                                                                        score[0], score[1]))

                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
            elif r == 0 and self.display == 1:
                print(f"Current score: b {score[0]}, W {score[1]}")

    def append_eps_result(self, result):
        self.trainExamplesHistory.append(result)
        # print("Result Added.")

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        iterHistory = {'ITER': [], 'ITER_DETAIL': [], 'PITT_RESULT': []}

        for i in range(1, self.args.numIters + 1):
            iterHistory['ITER'].append(i)
            # bookkeeping
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(int(self.args.numEps / self.args.num_processes)):
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()

                    with multiprocessing.Pool(self.args.num_processes) as pool:

                        for _ in range(self.args.num_processes):
                            pool.apply_async(self.executeEpisode, callback=self.append_eps_result)

                        pool.close()
                        pool.join()

                    # increment bar
                    for k in range(self.args.num_processes):
                        bar.suffix = f'{eps * self.args.num_processes + k + 1}/{self.args.numEps} ' \
                                     f'Eps Time: {bar.elapsed_td / (eps + 1)}s ' \
                                     f'| Total: {bar.elapsed_td}' \
                                     f'| ETA: {(bar.elapsed_td / (eps + 1)) / self.args.num_processes * (self.args.numEps - eps)}'
                        bar.next()

                bar.finish()

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
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, self.args.datetime,
                          display=display,
                          displayValue=self.display.value)
            pwins, nwins, draws, outcomes = arena.playGames(self.args.arenaCompare)
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

            self.create_sgf_files_for_games(games=outcomes, iteration=i)
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

        plt.rcParams["figure.figsize"] = (18, 12)

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

        plt.savefig(f"logs/go/Training_Results/Training_Result_{self.args.datetime}.png")

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
