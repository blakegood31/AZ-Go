import numpy as np
import time
from utils import status_bar


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, datetime, args, display=None, displayValue=0):
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
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.displayValue = displayValue
        self.datetime = datetime
        self.args = args

    def playGame(self, verbose=True):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        action_history = []
        x_boards = []
        y_boards = []
        c_boards = []
        c_boards.append(np.ones((7, 7)))
        c_boards.append(np.zeros((7, 7)))
        for i in range(4):
            x_boards.append(np.zeros((self.args.boardsize, self.args.boardsize)))
            y_boards.append(np.zeros((self.args.boardsize, self.args.boardsize)))
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                score = self.game.getScore(board)

                if self.displayValue == 1:
                    print("\nTurn ", str(it), "Player ", str(curPlayer))
                    print(self.display(board))
                    print(f"Current score: b {score[0]}, W {score[1]}")
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            player_board = c_boards[0] if curPlayer == 1 else c_boards[1]
            canonicalHistory, x_boards, y_boards = self.game.getCanonicalHistory(x_boards, y_boards,
                                                                                 canonicalBoard.pieces, player_board)
            # print("History used to make move: ", canonicalHistory)
            action = players[curPlayer + 1](canonicalBoard, canonicalHistory, x_boards, y_boards, player_board, )
            player_name = "B" if curPlayer == 1 else "W"
            action_history.append(f";{player_name}[{self.game.action_space_to_GTP(action)}]")

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                # assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            x_boards, y_boards = y_boards, x_boards

        if verbose:
            # assert(self.display)
            r, score = self.game.getGameEnded(board, 1, returnScore=True)

            if self.displayValue == 1:
                print("\nGame over: Turn ", str(it), "Result ", str(r))
                print(self.display(board))
                print(f"Final score: b {score[0]}, W {score[1]}\n")
        return self.game.getGameEnded(board, 1), action_history

    def playGames(self, num, verbose=True):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        total_time = 0

        eps = 0
        maxeps = int(num)
        originalNum = num

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        outcomes = []

        for _ in range(num):
            start_time = time.time()

            gameResult, action_history = self.playGame(verbose=verbose)
            outcomes.append(action_history)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

            eps += 1

            end_time = time.time()
            total_time += round(end_time - start_time, 2)
            status_bar(eps, maxeps,
                       title="Arena", label="Games",
                       suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / eps} | Total: {round(total_time, 2)}")


        self.player1, self.player2 = self.player2, self.player1

        if (originalNum % 2 == 1):
            num += 1

        for _ in range(num):
            start_time = time.time()

            gameResult, action_history = self.playGame(verbose=verbose)
            outcomes.append(action_history)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            eps += 1

            end_time = time.time()
            total_time += round(end_time - start_time, 2)
            status_bar(eps, maxeps,
                       title="Arena", label="Games",
                       suffix=f"| Eps: {round(end_time - start_time, 2)} | Avg Eps: {round(total_time, 2) / eps} | Total: {round(total_time, 2)}")

        return oneWon, twoWon, draws, outcomes
