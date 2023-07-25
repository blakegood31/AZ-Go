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
import gc
import os


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, log=False, logPath='', NetType='RES'):
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
        self.NetType = NetType
        # TODO: Do we need this line? Doesn't seem to be used - HAL
        self.canonicalHistory = []

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
        x_boards = []
        y_boards = []
        c_boards = [np.ones((7, 7)), np.zeros((7, 7))]
        for i in range(4):
            x_boards.append(np.zeros((self.args.boardsize, self.args.boardsize)))
            y_boards.append(np.zeros((self.args.boardsize, self.args.boardsize)))
        while True:
            episodeStep += 1
            if self.display == 1:
                print("================Episode {} Step:{}=====CURPLAYER:{}==========".format(self.currentEpisode,
                                                                                             episodeStep,
                                                                                             "White" if self.curPlayer == -1 else "Black"))
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            player_board = c_boards[0] if self.curPlayer == 1 else c_boards[1]
            canonicalHistory, x_boards, y_boards = self.game.getCanonicalHistory(x_boards, y_boards, canonicalBoard.pieces, player_board)

            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonicalBoard, canonicalHistory, x_boards, y_boards, player_board, temp=temp)
            # get different symmetries/rotations of the board
            sym = self.game.getSymmetries(canonicalHistory, pi)
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
            x_boards, y_boards = y_boards, x_boards

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        # print('RAM Used start of learn (GB):', psutil.virtual_memory()[3] / 1000000000)
        iterHistory = {'ITER': [], 'ITER_DETAIL': [], 'PITT_RESULT': []}

        # helper distributed variables
        upload_number = 1
        new_model_accepted_in_previous_iteration = False

        # setup
        if self.args.distributed_training:
            drive = DriveAPI(self.args.nettype, self.args.boardsize)

        if self.args.load_model:
            self.loadLosses()

        # training loop
        for i in range(self.args.start_iter, self.args.numIters + 1):
            iterHistory['ITER'].append(i)
            games_played_during_iteration = 0

            if self.args.distributed_training:
                print(f"##### Iteration {i} Distributed Training #####")

                if i == 1:
                    # on first iteration, play X games, so a model can be updated to the drive before using the drive
                    print(f"First iteration. Play {int(self.args.numEps / 200)} self play games, so there is a model to upload to Google Drive.")

                    self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                    eps_time = AverageMeter()
                    bar = Bar('Self Play', max=self.args.numEps)
                    end = time.time()

                    for eps in range(int(self.args.numEps / 200)):
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
                    games_played_during_iteration = self.args.numEps
                else:
                    if not new_model_accepted_in_previous_iteration:
                        # download most recent training examples from the drive (until numEps is hit or files run out)
                        # previous examples are still valid training data
                        print("New model not accepted in previous iteration. Downloading from drive.")
                        games_played_during_iteration, upload_number = self.download_recent_games_from_google_drive(
                            downloads_threshold=self.args.numEps, upload_number=upload_number)
                    else:
                        print("New model accepted in previous iteration. Start polling games.")

                    if games_played_during_iteration >= self.args.numEps:
                        percent_complete(games_played_during_iteration, self.args.numEps,
                                         title="Self Play + Distributed Training", label="Games")

                    polling_tracker = 1
                    while games_played_during_iteration < self.args.numEps:
                        # play games and download from drive until limit is reached
                        print(f"There were not enough games on the drive. Starting polling session #{polling_tracker}.")
                        total_time = 0
                        for eps in range(self.args.polling_games):
                            start = time.time()
                            self.mcts = MCTS(self.game, self.nnet, self.args)
                            self.iterationTrainExamples += self.executeEpisode()
                            games_played_during_iteration += 1

                            end = time.time()
                            total_time += round(end - start, 2)
                            percent_complete(eps + 1, self.args.polling_games,
                                             title="Polling Games", label="Games",
                                             suffix=f"| Eps Time: {round(end - start, 2)} | Total Time: {round(total_time, 2)}")

                        # after polling games are played, check drive and download as many "new" files as possible
                        new_games, upload_number = self.download_recent_games_from_google_drive(
                            downloads_threshold=self.args.numEps - games_played_during_iteration,
                            upload_number=upload_number)
                        games_played_during_iteration += new_games

                        percent_complete(games_played_during_iteration, self.args.numEps,
                                         title="Self Play + Distributed Training", label="Games")

                        polling_tracker += 1

                        # spacers to ensure bar printouts are correct
                        print()
                        print()

            else:
                # normal (non-distributed) training loop
                print(f"######## Iteration {i} Episode Play ########")
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
                    games_played_during_iteration = self.args.numEps

            # Log how many games were added during each iteration
            file_name = f'logs/go/{self.args.nettype}_MCTS_SimModified_checkpoint/{self.args.boardsize}/Game_Counts.txt'
            if not os.path.isfile(file_name):
                counts_file = open(file_name, 'w')
                counts_file.close()
            counts_file = open(file_name, 'a')
            counts_file.write(
                f"\n Number of games added to train examples during iteration #{i}: {games_played_during_iteration} games\n")
            counts_file.close()

            # read trainExamples from local disk and use them for NN training
            if i != 1 or self.args.load_model:
                new_train_examples = self.iterationTrainExamples

                # update pathing
                checkpoint_dir = f'logs/go/{self.NetType}_MCTS_SimModified_checkpoint/{self.game.getBoardSize()[0]}/'
                checkpoint_files = [file for file in os.listdir(checkpoint_dir) if
                                    file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                self.args.load_folder_file = [
                    f'logs/go/{self.NetType}_MCTS_SimModified_checkpoint/{self.game.getBoardSize()[0]}/',
                    latest_checkpoint]

                self.loadTrainExamples()

                self.trainExamplesHistory.append(new_train_examples)
            else:
                # save the iteration examples to the history
                if not self.skipFirstSelfPlay:
                    self.trainExamplesHistory.append(self.iterationTrainExamples)

            # prune trainExamples to meet args recommendation
            while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print(f"Truncated trainExamplesHistory to {len(self.trainExamplesHistory)}. Length exceeded args limit.")
                self.trainExamplesHistory.pop(0)

            # prune trainExamples to meet ram requirement
            ramCap = self.args.ram_cap
            while int(psutil.virtual_memory()[3] / 1000000000) > ramCap and len(self.trainExamplesHistory) > 13:
                print(f"Truncated trainExamplesHistory to {len(self.trainExamplesHistory)}. Length exceeded ram limit.")
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

            # clear trainExamples after they are used
            self.trainExamplesHistory = []
            self.iterationTrainExamples.clear()
            trainExamples.clear()

            self.p_loss_per_iteration.append(np.average(trainLog['P_LOSS'].to_numpy()))
            self.v_loss_per_iteration.append(np.average(trainLog['V_LOSS'].to_numpy()))
            if self.keepLog:
                trainLog.to_csv(self.logPath + 'ITER_{}_TRAIN_LOG.csv'.format(i))

            iterHistory['ITER_DETAIL'].append(self.logPath + 'ITER_{}_TRAIN_LOG.csv'.format(i))

            nmcts = MCTS(self.game, self.nnet, self.args)

            print('\nPITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x, y, z, a, b: np.argmax(pmcts.getActionProb(x, y, z, a, b, temp=0)),
                          lambda x, y, z, a, b: np.argmax(nmcts.getActionProb(x, y, z, a, b, temp=0)), self.game, self.args.datetime, self.args,
                          display=display,
                          displayValue=self.display.value)
            pwins, nwins, draws, outcomes = arena.playGames(self.args.arenaCompare)
            self.winRate.append(nwins / self.args.arenaCompare)
            self.saveLosses()
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                new_model_accepted_in_previous_iteration = False
                iterHistory['PITT_RESULT'].append('R')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                if i == 1 and self.args.distributed_training and not self.args.load_model:
                    upload_path = os.path.join(self.args.checkpoint, 'temp.pth.tar')
                    drive.FileUpload(upload_path, upload_number)

            else:
                print('ACCEPTING NEW MODEL')
                new_model_accepted_in_previous_iteration = True
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
            # print('RAM Used before dump (GB):', psutil.virtual_memory()[3] / 1000000000)
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

    def loadDownloadedExamples(self, file_path):
        examplesFile = file_path
        with open(examplesFile, "rb") as f:
            try:
                examples = Unpickler(f).load()
                for i in range(len(examples)):
                    self.iterationTrainExamples += examples[i]
            except:
                print(f"Error loading file: {file_path}\nFile not found on local device. Maybe there was an issue downloading it?")
                pass
        self.skipFirstSelfPlay = False
        f.closed

    def saveLosses(self):
        # Save ploss, vloss, and winRate so graphs are consistent across training sessions
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
        # Load in ploss, vloss, and winRates from previous iterations so graphs are consistent
        vlossFile = os.path.join(self.args.checkpoint, "vlosses")
        plossFile = os.path.join(self.args.checkpoint, "plosses")
        winrateFile = os.path.join(self.args.checkpoint, "winrates")
        if not os.path.isfile(vlossFile) or not os.path.isfile(plossFile):
            r = input("File with vloss or ploss not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with graph information found. Read it.")
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

    # downloads games from Google Drive with a limit and returns number of games downloaded
    # and load them into self.iterationTrainExamples
    def download_recent_games_from_google_drive(self, downloads_threshold, upload_number):
        downloads_count = 0  # also used by counts_file to log number of games per iteration

        # Create drive object
        drive = DriveAPI(self.args.nettype, self.args.boardsize)
        # downloads_threshold = self.args.numEps

        # Get list of all files in Google Drive
        files = []
        for item in drive.items:
            name = item['name']
            files.append(name)

        # Find most recent batches of training examples
        best_found = False

        for j in range(len(files)):
            curr_file = files[j]

            # Check what upload_num should be (used for model storage on drive)
            if curr_file.startswith('best'):
                best_found = True
                best_num = int(curr_file.split('.')[0][4:])
                if best_num >= upload_number:
                    upload_number = best_num + 1

            # there is lag between when a download is queued and finished
            # add 5 to downloads_count here to trigger break statement on time
            if best_found and downloads_count + 5 > downloads_threshold:
                # print("Downloads threshold reached; downloads complete.")
                break

            # Check if file is a checkpoint and if it's been downloaded (stop downloading once latest model has been reached)
            if "drive_checkpoint" in curr_file:
                # if not best_found:
                try:
                    drive.FileDownload(drive.items[j]['id'], drive.items[j]['name'])
                    file_path = os.path.join(self.args.checkpoint, drive.items[j]['name'])
                    self.loadDownloadedExamples(file_path)
                    downloads_count += 5
                    percent_complete(downloads_count, downloads_threshold, title="Google Drive Game Download", label="Games")
                except:
                    pass

        del files
        gc.collect()
        print()
        # print("Downloading from Drive Complete")
        return downloads_count, upload_number


# progress bar print out for updated bar progress
def percent_complete(step, total_steps, bar_width=45, title="", label="", suffix="", print_perc=True):
    import sys

    # UTF-8 left blocks: 1, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
    utf_8s = ["█", "▏", "▎", "▍", "▌", "▋", "▊", "█"]
    perc = 100 * float(step) / float(total_steps)
    max_ticks = bar_width * 8
    num_ticks = int(round(perc / 100 * max_ticks))
    full_ticks = num_ticks / 8  # Number of full blocks
    part_ticks = num_ticks % 8  # Size of partial block (array index)

    disp = bar = ""  # Blank out variables
    bar += utf_8s[0] * int(full_ticks)  # Add full blocks into Progress Bar

    # If part_ticks is zero, then no partial block, else append part char
    if part_ticks > 0:
        bar += utf_8s[part_ticks]

    # Pad Progress Bar with fill character
    bar += "▒" * int((max_ticks / 8 - float(num_ticks) / 8.0))

    if len(title) > 0:
        disp = title + ": "  # Optional title to progress display

    # Print progress bar in green: https://stackoverflow.com/a/21786287/6929343
    disp += "\x1b[0;32m"  # Color Green
    disp += bar  # Progress bar to progress display
    disp += "\x1b[0m"  # Color Reset
    if print_perc:
        # If requested, append percentage complete to progress display
        if perc > 100.0:
            perc = 100.0  # Fix "100.04 %" rounding error
        disp += " {:6.2f}".format(perc) + " %"
    disp += f"   {step}/{total_steps} {label} {suffix}"

    # Output to terminal repetitively over the same line using '\r'.
    sys.stdout.write("\r" + disp)
    sys.stdout.flush()

    # print newline when finished
    if step >= total_steps:
        print()
