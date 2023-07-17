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
from DriveAPI import DriveAPI
import psutil


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
        self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

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
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        while True:

            episodeStep += 1
            if self.display == 1:
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
            if self.display == 1:
                print("BOARD updated:")
                # display(board)
                print(display(board))
            r, score = self.game.getGameEnded(board.copy(), self.curPlayer, returnScore=True)
            if r != 0:
                if self.display == 1:
                    print("Current episode ends, {} wins with score b {}, W {}.".format('Black' if r == -1 else 'White',
                                                                                        score[0], score[1]))

                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
            elif r == 0 and self.display == 1:
                print(f"Current score: b {score[0]}, W {score[1]}")


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        iterHistory = {'ITER': [], 'ITER_DETAIL': [], 'PITT_RESULT': []}
        upload_number = 1

        if self.args.load_model:
            self.loadLosses()

        for i in range(self.args.start_iter, self.args.numIters + 1):
            iterHistory['ITER'].append(i)
            print(f"######## Iteration {i} ########")
            # bookkeeping
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > self.args.start_iter:
                self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    # print("{}th Episode:".format(eps+1))
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    self.currentEpisode = eps + 1
                    self.iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()

                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                            total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()

                bar.finish()


            if self.args.distributed_training:
                #Create drive object
                drive = DriveAPI()
                downloads_count = 0

                #Get list of all files in Google Drive
                files = []
                for item in drive.items:
                    name = item['name']
                    files.append(name)

                #Get list of files already downloaded from drive
                checkpoint_dir = f'logs/go/{self.args.nettype}_MCTS_SimModified_checkpoint/{self.args.boardsize}/'
                downloaded_files = [str(file) for file in os.listdir(checkpoint_dir) if file.startswith('drive_')]

                #Find most recent batches of training examples
                print("Checking for new files from drive")
                best_found = False
                append_downloads = False
                for j in range(len(files)):
                    curr_file = files[j]
                    #Check what upload_num should be (used for model storage on drive)
                    if curr_file.startswith('best'):
                        best_found = True
                        best_num = int(curr_file.split('.')[0][4:])
                        if best_num >= upload_number:
                            upload_number = best_num + 1

                    #Check if file is a checkpoint and if it's been downloaded (stop downloading once latest model has been reached)
                    if "drive_checkpoint" in curr_file and not best_found:
                        if not curr_file in downloaded_files:
                            downloads_count += 1
                            #Download and store new file
                            print("Downloading Train Examples: ", drive.items[j]['name'])
                            drive.FileDownload(drive.items[j]['id'], drive.items[j]['name'])
                            file_path = os.path.join(self.args.checkpoint, drive.items[j]['name'])
                            self.loadDownloadedExamples(file_path)
                            append_downloads = True

                downloads_count = downloads_count * 5
                downloads_count += self.args.numEps
            else:
                downloads_count = self.args.numEps

            #Log how many games were added during each iteration
            file_name = f'logs/go/{self.args.nettype}_MCTS_SimModified_checkpoint/{self.args.boardsize}/Game_Counts.txt'
            if not os.path.isfile(file_name):
                counts_file = open(file_name, 'w')
                counts_file.close()
            counts_file = open(file_name, 'a')
            counts_file.write(f"\n Number of games added to train examples during iteration #{i}: {downloads_count} games\n")
            counts_file.close()

            # save the iteration examples to the history
            if not self.skipFirstSelfPlay or append_downloads:
                self.trainExamplesHistory.append(self.iterationTrainExamples)

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
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, self.args.datetime, display=display,
                          displayValue=self.display.value)
            pwins, nwins, draws, outcomes = arena.playGames(self.args.arenaCompare)
            self.winRate.append(nwins / self.args.arenaCompare)
            self.saveLosses()
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                iterHistory['PITT_RESULT'].append('R')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                if i == 1 and self.args.distributed_training and not self.args.load_model:
                    upload_path = os.path.join(self.args.checkpoint, 'temp.pth.tar')
                    drive.FileUpload(upload_path, upload_number)

            else:
                print('ACCEPTING NEW MODEL')
                iterHistory['PITT_RESULT'].append('A')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                if self.args.distributed_training:
                    upload_path = os.path.join(self.args.checkpoint, 'best.pth.tar')
                    drive.FileUpload(upload_path, upload_number)
                    upload_number += 1

            pd.DataFrame(data=iterHistory).to_csv(self.logPath + 'ITER_LOG.csv')

            self.create_sgf_files_for_games(games=outcomes, iteration=i)
            self.saveTrainingPlots()
            self.skipFirstSelfPlay = False

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
        examplesFile = modelFile #+ ".examples"
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

    def loadDownloadedExamples(self, file_path):
        examplesFile = file_path
        with open(examplesFile, "rb") as f:
            if len(self.iterationTrainExamples) == 0:
                examples = Unpickler(f).load()
                for i in range(len(examples)):
                    self.iterationTrainExamples += examples[i]
            else:
                examples = Unpickler(f).load()
                for i in range(len(examples)):
                    self.iterationTrainExamples += examples[i]
        self.skipFirstSelfPlay = False
        f.closed

    def saveLosses(self):
        #Save ploss, vloss, and winRate so graphs are consistent across training sessions
        folder = self.args.checkpoint
        if not os.path.exists(folder):
                os.makedirs(folder)
        vloss_filename = os.path.join(folder, "vlosses")
        ploss_filename = os.path.join(folder, "plosses")
        winRate_filename = os.path.join(folder, "winrates")
        with open(vloss_filename, "wb+") as f:
            Pickler(f).dump(self.v_loss_per_iteration)
        f.closed
        with open(ploss_filename, "wb+") as f:
            Pickler(f).dump(self.p_loss_per_iteration)
        f.closed
        with open(winRate_filename, "wb+") as f:
            Pickler(f).dump(self.winRate)
        f.closed

    def loadLosses(self):
        #Load in ploss, vloss, and winRates from previous iterations so graphs are consistent
        vlossFile = os.path.join(self.args.checkpoint, "vlosses")
        plossFile = os.path.join(self.args.checkpoint, "plosses")
        winrateFile = os.path.join(self.args.checkpoint, "winrates")
        if not os.path.isfile(vlossFile) or not os.path.isfile(plossFile):
            r = input("File with vloss or ploss not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(vlossFile, "rb") as f:
                self.v_loss_per_iteration = Unpickler(f).load()
            f.closed
            with open(plossFile, "rb") as f:
                self.p_loss_per_iteration = Unpickler(f).load()
            f.closed
            with open(winrateFile, "rb") as f:
                self.winRate = Unpickler(f).load()
            f.closed

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
            file_name = f'logs/go/Game_History/Iteration {iteration}, Game {game_idx + 1} {self.args.sgf_datetime}.sgf'
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