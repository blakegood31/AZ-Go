import glob
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paramiko
import psutil
from scp import SCPClient

from Arena import Arena
from GoMCTS import MCTS
from go.GoGame import display
from utils import status_bar


class Coach:
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

        if self.args.load_model:
            self.loadLosses()

        # training loop
        for i in range(self.args.start_iter, self.args.numIters + 1):
            iterHistory['ITER'].append(i)
            games_played_during_iteration = 0

            if self.args.distributed_training:
                print(f"##### Iteration {i} Distributed Training #####")

                if i == 1 and not self.args.load_model:
                    first_iteration_num_games = int(self.args.numEps / 20)

                    # on first iteration, play X games, so a model can be updated to lambda
                    print(f"First iteration. Play {first_iteration_num_games} self play games, so there is a model to upload to lambda.")

                    self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                    total_time = 0
                    for eps in range(first_iteration_num_games):
                        start_time = time.time()
                        self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                        self.iterationTrainExamples += self.executeEpisode()

                        # update bar print out
                        end_time = time.time()
                        total_time += round(end_time - start_time, 2)
                        status_bar(self.currentEpisode, first_iteration_num_games,
                                   title="Polling Games", label="Games",
                                   suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / self.currentEpisode} | Total: {round(total_time, 2)}")

                    games_played_during_iteration = first_iteration_num_games

                else:
                    self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                    if not new_model_accepted_in_previous_iteration:
                        # download most recent training examples from the drive (until numEps is hit or files run out)
                        # previous examples are still valid training data
                        print("New model not accepted in previous iteration. Downloading from lambda.")
                        games_played_during_iteration += self.scan_examples_folder_and_load(game_limit=self.args.numEps)
                        status_bar(games_played_during_iteration, self.args.numEps,
                                     title="Lambda Downloaded Games", label="Games")

                    else:
                        print("New model accepted in previous iteration. Start polling games.")

                    if games_played_during_iteration >= self.args.numEps:
                        status_bar(games_played_during_iteration, self.args.numEps,
                                         title="Self Play + Distributed Training", label="Games")

                    polling_tracker = 1
                    while games_played_during_iteration < self.args.numEps:
                        # play games and download from drive until limit is reached
                        print(f"Starting polling session #{polling_tracker}.")
                        total_time = 0
                        for eps in range(self.args.polling_games):
                            start_time = time.time()
                            self.mcts = MCTS(self.game, self.nnet, self.args)
                            self.iterationTrainExamples += self.executeEpisode()
                            games_played_during_iteration += 1

                            end_time = time.time()
                            total_time += round(end_time - start_time, 2)
                            status_bar(eps + 1, self.args.polling_games,
                                             title="Polling Games", label="Games",
                                             suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / (eps + 1)} | Total: {round(total_time, 2)}")

                        # after polling games are played, check drive and download as many "new" files as possible
                        num_downloads = self.scan_examples_folder_and_load(game_limit=self.args.numEps - games_played_during_iteration)
                        status_bar(num_downloads, self.args.numEps - games_played_during_iteration,
                                         title="Lambda Downloaded Games", label="Games")

                        print()

                        games_played_during_iteration += num_downloads

                        status_bar(games_played_during_iteration, self.args.numEps,
                                         title="Self Play + Distributed Training", label="Games")

                        polling_tracker += 1

                        # spacers to ensure bar printouts are correct
                        print()
                        print()

            else:
                if not self.skipFirstSelfPlay or i > self.args.start_iter:
                    # normal (non-distributed) training loop
                    print(f"######## Iteration {i} Episode Play ########")

                    self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                    total_time = 0
                    for eps in range(self.args.numEps):
                        start_time = time.time()

                        self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                        self.currentEpisode = eps + 1
                        self.iterationTrainExamples += self.executeEpisode()

                        end_time = time.time()
                        total_time += round(end_time - start_time, 2)
                        status_bar(self.currentEpisode, self.args.numEps,
                                   title="Self Play", label="Games",
                                   suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / self.currentEpisode} | Total: {round(total_time, 2)}")

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

            # # read trainExamples from local disk and use them for NN training
            # if i != 1 or self.args.load_model:
            #     new_train_examples = self.iterationTrainExamples
            #
            #     # update pathing
            #     checkpoint_dir = f'logs/go/{self.NetType}_MCTS_SimModified_checkpoint/{self.game.getBoardSize()[0]}/'
            #     checkpoint_files = [file for file in os.listdir(checkpoint_dir) if
            #                         file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
            #     latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            #     self.args.load_folder_file = [
            #         f'logs/go/{self.NetType}_MCTS_SimModified_checkpoint/{self.game.getBoardSize()[0]}/',
            #         latest_checkpoint]
            #
            #     self.loadTrainExamples()
            #
            #     self.trainExamplesHistory.append(new_train_examples)
            # else:
            #     # save the iteration examples to the history
            #     if not self.skipFirstSelfPlay:
            #         self.trainExamplesHistory.append(self.iterationTrainExamples)

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
            # self.trainExamplesHistory = []
            # self.iterationTrainExamples.clear()
            # trainExamples.clear()

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
                    new_model_accepted_in_previous_iteration = True
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                    self.send_model_to_server(username="username", server_address="server_address", remote_path="path")

            else:
                print('ACCEPTING NEW MODEL')
                new_model_accepted_in_previous_iteration = True
                iterHistory['PITT_RESULT'].append('A')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                if self.args.distributed_training:
                    self.send_model_to_server(username="username", server_address="server_address", remote_path="path")
                    upload_number += 1
                    self.wipe_examples_folder()

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

    def load_examples_from_path(self, file_path):
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

    ##### local distributed training helper functions ######
    def createSSHClient(self, server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    # doesn't currently rename the file... which I think is fine
    def send_model_to_server(self, username, server_address, remote_path):
        local_path = os.path.join(self.args.checkpoint, 'best.pth.tar')
        ssh = self.createSSHClient(server_address, 22, username, "password")
        scp = SCPClient(ssh.get_transport())
        scp.put(local_path, remote_path)
        print("New model uploaded.")

    def scan_examples_folder_and_load(self, game_limit):
        files = glob.glob("/")

        game_count = 0

        for f in files:
            # game_count >= game_limit, STOP
            if game_count >= game_limit:
                print("game limit reached")
                break

            # load in the file
            self.load_examples_from_path(f)

            # delete file from storage
            os.remove(f)

            # iterate game_count
            game_count += 1

        return game_count

    def wipe_examples_folder(self):
        files = glob.glob("/")

        for f in files:
            os.remove(f)
