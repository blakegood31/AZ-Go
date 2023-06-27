import multiprocessing

import numpy as np

from go.GoGame import GoGame as Game
from pytorch_classification.utils import Bar, AverageMeter
import time


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, game, datetime, display=1, displayValue=0, num_processes=1, board_size=7):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.game = game
        self.display = display
        self.displayValue = displayValue
        self.datetime = datetime
        self.num_processes = num_processes
        self.board_size = board_size

        self.outcomes = []
        self.oneWon = 0
        self.twoWon = 0
        self.draws = 0

    def playGame(self, pmcts, nmcts):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        player1 = lambda x: np.argmax(pmcts.getActionProb(x, temp=0))
        player2 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
        players = [player2, None, player1]
        curPlayer = 1
        game = Game(self.board_size)
        board = game.getInitBoard()
        it = 0
        action_history = []
        verbose = True
        while game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                score = game.getScore(board)

                if self.displayValue == 1:
                    print("\nTurn ", str(it), "Player ", str(curPlayer))
                    print(self.display(board))
                    print(f"Current score: b {score[0]}, W {score[1]}")

            action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))
            player_name = "B" if curPlayer == 1 else "W"
            action_history.append(f";{player_name}[{game.action_space_to_GTP(action)}]")

            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                # assert valids[action] >0
            board, curPlayer = game.getNextState(board, curPlayer, action)

        if verbose:
            # assert(self.display)
            r, score = game.getGameEnded(board, 1, returnScore=True)

            if self.displayValue == 1:
                print("\nGame over: Turn ", str(it), "Result ", str(r))
                print(self.display(board))
                print(f"Final score: b {score[0]}, W {score[1]}\n")

        return [game.getGameEnded(board, 1), action_history]

    def handle_completed_game(self, array):
        game_result = array[0]
        action_history = array[1]
        self.outcomes += [action_history]

        if game_result == -1:
            self.oneWon += 1
        elif game_result == 1:
            self.twoWon += 1
        else:
            self.draws += 1
        print("Game Completed.")

    def playGames(self, num, pmcts, nmcts):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        originalNum = num
        num = int(num / 2)

        for _ in range(int(num / self.num_processes)):
            with multiprocessing.Pool(self.num_processes) as pool:
                for _ in range(self.num_processes):
                    pool.apply_async(self.playGame, args=(pmcts, nmcts,), callback=self.handle_completed_game)

                pool.close()
                pool.join()

        if originalNum % 2 == 1:
            num += 1

        for _ in range(int(num / self.num_processes)):
            with multiprocessing.Pool(self.num_processes) as pool:
                for _ in range(self.num_processes):
                    pool.apply_async(self.playGame, args=(nmcts, pmcts,), callback=self.handle_completed_game)

                pool.close()
                pool.join()

        return self.oneWon, self.twoWon, self.draws, self.outcomes
