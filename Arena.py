import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from pettingzoo.classic import go_v5 as go


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, datetime, display_value=0):
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
        self.player1 = player1  # Previous Model, black_0
        self.player2 = player2  # New Model, white_0
        self.game = game
        self.displayValue = display_value
        self.datetime = datetime
        self.game_num = 1
        self.komi = 1 if self.game.getBoardSize()[0]<=7 else 7.5
        self.by_score = 0.5 * (self.game.getBoardSize()[0] * self.game.getBoardSize()[0] + self.komi)

    def playGame(self):
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
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                score = self.game.getScore(board)

                if self.displayValue == 1:
                    print("\nTurn ", str(it), "Player ", str(curPlayer))
                    print(self.display(board))
                    print(f"Current score: b {score[0]}, W {score[1]}")

            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
            player_name = "B" if curPlayer == 1 else "W"
            action_history.append(f";{player_name}[{self.game.action_space_to_GTP(action)}]")

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                # assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        if verbose:
            # assert(self.display)
            r, score = self.game.getGameEnded(board, 1, returnScore=True)

            if self.displayValue == 1:
                print("\nGame over: Turn ", str(it), "Result ", str(r))
                print(self.display(board))
                print(f"Final score: b {score[0]}, W {score[1]}\n")
        return self.game.getGameEnded(board, 1), action_history

    def playGames(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            player_one_wins: games won by player1
            player_two_wins: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena Play', max=num)
        end = time.time()

        player_one_wins = 0
        player_two_wins = 0
        draws = 0

        outcomes = []

        for _ in range(num):
            gameResult, action_history = self.playGame(verbose=verbose)
            outcomes.append(action_history)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}\n'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        all_arena_games_history = []

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            outcomes.append(action_history)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            # bookkeeping + plot progress

            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}\n'.format(
                eps=self.game_num,
                maxeps=num,
                et=eps_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td)
            bar.next()

            self.game_num += 1

        bar.finish()
        
        return oneWon, twoWon, draws, outcomes
