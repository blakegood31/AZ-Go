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
import yaml
from scp import SCPClient

from Arena import Arena
from GoMCTS import MCTS
from go.GoGame import display
from utils import status_bar
from datetime import datetime


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. config is specified in config.yaml.
    """

    def __init__(self, game, nnet, config):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, config)  # the competitor network
        self.config = config
        self.mcts = MCTS(self.game, self.nnet, self.config)
        self.trainExamplesHistory = []  # history of examples from config["max_num_iterations_in_train_example_history"] latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.p_loss_per_iteration = []
        self.v_loss_per_iteration = []
        self.winRate = []
        self.currentEpisode = 0
        self.iterationTrainExamples = deque([], maxlen=self.config["max_length_of_queue"])

        self.date_time = datetime.now().strftime("%d-%m-%Y %H")
        self.latest_checkpoint = 0

        # if needed, import sensitive_config
        if config["enable_distributed_training"]:
            with open("sensitive.yaml", "r") as stream:
                try:
                    self.sensitive_config = yaml.safe_load(stream)
                    print(self.sensitive_config)

                except yaml.YAMLError as exc:
                    raise ValueError(exc)

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
        for i in range(8):
            x_boards.append(np.zeros((self.config["board_size"], self.config["board_size"])))
            y_boards.append(np.zeros((self.config["board_size"], self.config["board_size"])))
        while True:
            episodeStep += 1
            if self.config["display"] == 1:
                print("================Episode {} Step:{}=====CURPLAYER:{}==========".format(self.currentEpisode,
                                                                                             episodeStep,
                                                                                             "White" if self.curPlayer == -1 else "Black"))
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            player_board = (c_boards[0], c_boards[1]) if self.curPlayer == 1 else (c_boards[1], c_boards[0])
            canonicalHistory, x_boards, y_boards = self.game.getCanonicalHistory(x_boards, y_boards,
                                                                                 canonicalBoard.pieces, player_board)
            # print(canonicalHistory)
            temp = int(episodeStep < self.config["temperature_threshold"])
            pi = self.mcts.getActionProb(canonicalBoard, canonicalHistory, x_boards, y_boards, player_board, temp=temp)
            # get different symmetries/rotations of the board
            sym = self.game.getSymmetries(canonicalHistory, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            action = np.random.choice(len(pi), p=pi)

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            if self.config["display"] == 1:
                print("BOARD updated:")
                # display(board)
                print(display(board))
            r, score = self.game.getGameEnded(board.copy(), self.curPlayer, returnScore=True)
            if r != 0:
                if self.config["display"] == 1:
                    print("Current episode ends, {} wins with score b {}, W {}.".format('Black' if r == -1 else 'White',
                                                                                        score[0], score[1]))

                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
            elif r == 0 and self.config["display"] == 1:
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
        iterHistory = {'ITER': [], 'ITER_DETAIL': [], 'PITT_RESULT': []}

        if self.config["load_model"]:
            checkpoint_files = [file for file in os.listdir(self.config["checkpoint_directory"]) if
                                file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
            self.latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            start_iter = int(self.latest_checkpoint.split('_')[1].split('.')[0]) + 1
        else:
            start_iter = 1

        # helper distributed variables
        upload_number = 1
        new_model_accepted_in_previous_iteration = False

        if self.config["load_model"]:
            self.loadLosses()

        # training loop
        for i in range(start_iter, self.config["num_iterations"] + 1):
            iterHistory['ITER'].append(i)
            games_played_during_iteration = 0

            if self.config["enable_distributed_training"]:
                print(f"##### Iteration {i} Distributed Training #####")

                if i == 1 and not self.config["load_model"]:
                    first_iteration_num_games = int(self.config["num_self_play_episodes"] / 20)

                    # on first iteration, play X games, so a model can be updated to lambda
                    print(
                        f"First iteration. Play {first_iteration_num_games} self play games, so there is a model to upload to lambda.")

                    self.iterationTrainExamples = deque([], maxlen=self.config["max_length_of_queue"])

                    total_time = 0
                    for eps in range(first_iteration_num_games):
                        start_time = time.time()
                        self.mcts = MCTS(self.game, self.nnet, self.config)  # reset search tree
                        self.iterationTrainExamples += self.executeEpisode()

                        # update bar print out
                        end_time = time.time()
                        total_time += round(end_time - start_time, 2)
                        if eps == 0:
                            status_bar(eps + 1, first_iteration_num_games,
                                   title="1st Iter Games", label="Games",
                                   suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2)} | Total: {round(total_time, 2)}")
                        else:
                            status_bar(eps + 1, first_iteration_num_games,
                                   title="1st Iter Games", label="Games",
                                   suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / eps + 1} | Total: {round(total_time, 2)}")


                    games_played_during_iteration = first_iteration_num_games

                else:
                    self.iterationTrainExamples = deque([], maxlen=self.config["max_length_of_queue"])

                    if not new_model_accepted_in_previous_iteration:
                        # download most recent training examples from the drive (until numEps is hit or files run out)
                        # previous examples are still valid training data
                        print("New model not accepted in previous iteration. Downloading from lambda.")
                        games_played_during_iteration += self.scan_examples_folder_and_load(
                            game_limit=self.config["num_self_play_episodes"])
                        status_bar(games_played_during_iteration, self.config["num_self_play_episodes"],
                                   title="Lambda Downloaded Games", label="Games")

                    else:
                        print("New model accepted in previous iteration. Start polling games.")

                    if games_played_during_iteration >= self.config["num_self_play_episodes"]:
                        status_bar(games_played_during_iteration, self.config["num_self_play_episodes"],
                                   title="Self Play + Distributed Training", label="Games")

                    polling_tracker = 1
                    while games_played_during_iteration < self.config["num_self_play_episodes"]:
                        # play games and download from drive until limit is reached
                        print(f"Starting polling session #{polling_tracker}.")
                        total_time = 0
                        for eps in range(self.config["num_polling_games"]):
                            start_time = time.time()
                            self.mcts = MCTS(self.game, self.nnet, self.config)
                            self.iterationTrainExamples += self.executeEpisode()
                            games_played_during_iteration += 1

                            end_time = time.time()
                            total_time += round(end_time - start_time, 2)
                            status_bar(eps + 1, self.config["num_polling_games"],
                                       title="Polling Games", label="Games",
                                       suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / (eps + 1)} | Total: {round(total_time, 2)}")

                        # after polling games are played, check drive and download as many "new" files as possible
                        num_downloads = self.scan_examples_folder_and_load(
                            game_limit=self.config["num_self_play_episodes"] - games_played_during_iteration)
                        if self.config["num_self_play_episodes"] - games_played_during_iteration != 0:
                            status_bar(num_downloads, self.config["num_self_play_episodes"] - games_played_during_iteration,
                                             title="Lambda Downloaded Games", label="Games")

                        print()

                        games_played_during_iteration += num_downloads

                        status_bar(games_played_during_iteration, self.config["num_self_play_episodes"],
                                   title="Self Play + Distributed Training", label="Games")

                        polling_tracker += 1

                        # spacers to ensure bar printouts are correct
                        print()
                        print()

            else:
                if not self.skipFirstSelfPlay or i > start_iter:
                    # normal (non-distributed) training loop
                    print(f"######## Iteration {i} Episode Play ########")

                    self.iterationTrainExamples = deque([], maxlen=self.config["max_length_of_queue"])

                    total_time = 0
                    for eps in range(self.config["num_self_play_episodes"]):
                        start_time = time.time()

                        self.mcts = MCTS(self.game, self.nnet, self.config)  # reset search tree
                        self.currentEpisode = eps + 1
                        self.iterationTrainExamples += self.executeEpisode()

                        end_time = time.time()
                        total_time += round(end_time - start_time, 2)
                        status_bar(self.currentEpisode, self.config["num_self_play_episodes"],
                                   title="Self Play", label="Games",
                                   suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / self.currentEpisode} | Total: {round(total_time, 2)}")

                    games_played_during_iteration = self.config["num_self_play_episodes"]

            # Log how many games were added during each iteration
            file_name = self.config["train_logs_directory"] + "/Game_Counts.txt"
            if not os.path.isfile(file_name):
                counts_file = open(file_name, 'w')
                counts_file.close()
            counts_file = open(file_name, 'a')
            counts_file.write(
                f"\n Number of games added to train examples during iteration #{i}: {games_played_during_iteration} games\n")
            counts_file.close()

            # # read trainExamples from local disk and use them for NN training
            # if i != 1 or self.config["load_model"]:
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

            # prune trainExamples to meet config recommendation
            while len(self.trainExamplesHistory) > self.config["max_num_iterations_in_train_example_history"]:
                print(
                    f"Truncated trainExamplesHistory to {len(self.trainExamplesHistory)}. Length exceeded config limit.")
                self.trainExamplesHistory.pop(0)

            # prune trainExamples to meet ram requirement
            ramCap = self.config["ram_cap"]
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
            self.nnet.save_checkpoint(folder=self.config["checkpoint_directory"], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.config["checkpoint_directory"], filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.config)

            trainLog = self.nnet.train(trainExamples)

            # clear trainExamples after they are used
            # self.trainExamplesHistory = []
            # self.iterationTrainExamples.clear()
            # trainExamples.clear()

            self.p_loss_per_iteration.append(np.average(trainLog['P_LOSS'].to_numpy()))
            self.v_loss_per_iteration.append(np.average(trainLog['V_LOSS'].to_numpy()))
            trainLog.to_csv(self.config["train_logs_directory"] + '/ITER_{}_TRAIN_LOG.csv'.format(i))

            iterHistory['ITER_DETAIL'].append(self.config["train_logs_directory"] + '/ITER_{}_TRAIN_LOG.csv'.format(i))

            nmcts = MCTS(self.game, self.nnet, self.config)

            print('\nPITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x, y, z, a, b: np.argmax(pmcts.getActionProb(x, y, z, a, b, temp=0)),
                          lambda x, y, z, a, b: np.argmax(nmcts.getActionProb(x, y, z, a, b, temp=0)), self.game,
                          self.config)
            pwins, nwins, draws, outcomes = arena.playGames(self.config["num_arena_episodes"])
            self.winRate.append(nwins / self.config["num_arena_episodes"])
            self.saveLosses()
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.config["acceptance_threshold"]:
                print('REJECTING NEW MODEL')
                new_model_accepted_in_previous_iteration = False
                iterHistory['PITT_RESULT'].append('R')
                self.nnet.load_checkpoint(folder=self.config["checkpoint_directory"], filename='temp.pth.tar')
                if i == 1 and self.config["enable_distributed_training"] and not self.config["load_model"]:
                    new_model_accepted_in_previous_iteration = True
                    self.nnet.save_checkpoint(folder=self.config["checkpoint_directory"], filename='best.pth.tar')
                    self.send_model_to_server()

            else:
                print('ACCEPTING NEW MODEL')
                new_model_accepted_in_previous_iteration = True
                iterHistory['PITT_RESULT'].append('A')
                self.nnet.save_checkpoint(folder=self.config["checkpoint_directory"],
                                          filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.config["checkpoint_directory"], filename='best.pth.tar')
                if self.config["enable_distributed_training"]:
                    self.send_model_to_server()
                    upload_number += 1
                    self.wipe_examples_folder()

            pd.DataFrame(data=iterHistory).to_csv(self.config["train_logs_directory"] + '/ITER_LOG.csv')

            self.create_sgf_files_for_games(games=outcomes, iteration=i)
            self.saveTrainingPlots()
            self.skipFirstSelfPlay = False

        pd.DataFrame(data=iterHistory).to_csv(self.config["train_logs_directory"] + '/ITER_LOG.csv')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.config["checkpoint_directory"]
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            # print('RAM Used before dump (GB):', psutil.virtual_memory()[3] / 1000000000)
            is_error = True
            while is_error:
                try:
                    Pickler(f).dump(self.trainExamplesHistory)
                    is_error = False
                except:
                    is_error = True
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.config["checkpoint_directory"], self.latest_checkpoint)
        examplesFile = modelFile  # + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print(f"File with trainExamples found. Read it: {examplesFile}")
            with open(examplesFile, "rb") as f:
                is_error = True
                while is_error:
                    try:
                        self.trainExamplesHistory = Unpickler(f).load()
                        is_error = False
                    except:
                        is_error = True
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
                print(
                    f"Error loading file: {file_path}\nFile not found on local device. Maybe there was an issue downloading it?")
                pass
        self.skipFirstSelfPlay = False
        f.closed

    def saveLosses(self):
        # Save ploss, vloss, and winRate so graphs are consistent across training sessions
        folder = self.config["graph_directory"]
        if not os.path.exists(folder):
            os.makedirs(folder)
        vloss_filename = os.path.join(folder, "vlosses")
        ploss_filename = os.path.join(folder, "plosses")
        winRate_filename = os.path.join(folder, "winrates")
        with open(vloss_filename, "wb+") as f:
            is_error = True
            while is_error:
                try:
                    Pickler(f).dump(self.v_loss_per_iteration)
                    is_error = False
                except:
                    is_error = True
        f.closed
        with open(ploss_filename, "wb+") as f:
            is_error = True
            while is_error:
                try:
                    Pickler(f).dump(self.p_loss_per_iteration)
                    is_error = False
                except:
                    is_error = True
        f.closed
        with open(winRate_filename, "wb+") as f:
            is_error = True
            while is_error:
                try:
                    Pickler(f).dump(self.winRate)
                    is_error = False
                except:
                    is_error = True
        f.closed

    def loadLosses(self):
        # Load in ploss, vloss, and winRates from previous iterations so graphs are consistent
        vlossFile = os.path.join(self.config["graph_directory"], "vlosses")
        plossFile = os.path.join(self.config["graph_directory"], "plosses")
        winrateFile = os.path.join(self.config["graph_directory"], "winrates")
        if not os.path.isfile(vlossFile) or not os.path.isfile(plossFile):
            r = input("File with vloss or ploss not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with graph information found. Read it.")
            with open(vlossFile, "rb") as f:
                is_error = True
                while is_error:
                    try:
                        self.v_loss_per_iteration = Unpickler(f).load()
                        is_error = False
                    except:
                        is_error = True
            f.closed
            with open(plossFile, "rb") as f:
                is_error = True
                while is_error:
                    try:
                        self.p_loss_per_iteration = Unpickler(f).load()
                        is_error = False
                    except:
                        is_error = True
            f.closed
            with open(winrateFile, "rb") as f:
                is_error = True
                while is_error:
                    try:
                        self.winRate = Unpickler(f).load()
                        is_error = False
                    except:
                        is_error = True
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
        plt.axhline(y=self.config["acceptance_threshold"], color='b', linestyle='-')
        plt.plot(self.winRate, 'r', label='Win Rate')

        plt.savefig(self.config["graph_directory"] + f"/Training_Result{self.date_time}.png")

    def create_sgf_files_for_games(self, games, iteration):

        for game_idx in range(len(games)):
            # file_name = f'logs/go/Game_History/Iteration {iteration}, Game {game_idx + 1} {self.date_time}.sgf'
            file_name = self.config["game_history_directory"] + f"/Iteration {iteration}, Game {game_idx + 1} {self.date_time}.sgf"

            sgf_file = open(file_name, 'w')
            sgf_file.close()

            sgf_file = open(file_name, 'a')
            sgf_file.write(
                f"(;\nEV[AlphaGo Self-Play]\nGN[Iteration {iteration}]\nDT[{self.date_time}]\nPB[TCU_AlphaGo]\nPW[TCU_AlphaGo]"
                f"\nSZ[{self.game.getBoardSize()[0]}]\nRU["
                f"Chinese]\n\n")

            for move_idx in range(len(games[game_idx])):
                sgf_file.write(games[game_idx][move_idx])

            sgf_file.write("\n)")
            sgf_file.close()

    # local distributed training helper functions
    def createSSHClient(self, server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    # doesn't currently rename the file... which I think is fine
    def send_model_to_server(self):
        local_path = os.path.join(self.config["checkpoint_directory"], 'best.pth.tar')
        ssh = self.createSSHClient(self.sensitive_config["worker_server_address"], 22, self.sensitive_config["worker_username"], self.sensitive_config["worker_password"])
        scp = SCPClient(ssh.get_transport())
        scp.put(local_path, self.sensitive_config["distributed_models_directory"])
        print("New model uploaded.")

    def scan_examples_folder_and_load(self, game_limit):
        files = glob.glob(self.sensitive_config["distributed_examples_directory"] + "*")

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
        files = glob.glob(self.sensitive_config["distributed_examples_directory"] + "*")

        for f in files:
            os.remove(f)
