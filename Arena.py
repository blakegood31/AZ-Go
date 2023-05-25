import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, datetime, display=None, displayValue=0):
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
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                score = self.game.getScore(board)
                arena_log = open(f'logs/go/Game_Histories/Game_History_{self.datetime}.txt', 'a')
                arena_log.write("Turn: " + str(it) + "   Player: " + str(curPlayer) + "\n")
                arena_log.write(self.display(board))
                arena_log.write(f"\nCurrent score: b {score[0]}, W {score[1]}\n")
                arena_log.write("\n\n")
                arena_log.close()

                # assert(self.display)
                if self.displayValue == 2:
                    print("\nTurn ", str(it), "Player ", str(curPlayer))
                    print(self.display(board))
                    print(f"Current score: b {score[0]}, W {score[1]}")

            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                # assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        if verbose:
            # assert(self.display)
            r, score = self.game.getGameEnded(board, 1, returnScore=True)

            arena_log = open(f'logs/go/Game_Histories/Game_History_{self.datetime}.txt', 'a')
            arena_log.write(
                "## Game over: Turn " + str(it) + " Result " + str(self.game.getGameEnded(board, 1)) + " ##\n")
            arena_log.write("Final Board Configuration: \n")
            arena_log.write(self.display(board))
            arena_log.write(f"\nFinal score: b (previous model) {score[0]}, W (current model) {score[1]}\n\n\n")
            arena_log.close()

            if self.displayValue == 2:
                print("\nGame over: Turn ", str(it), "Result ", str(r))
                print(self.display(board))
                print(f"Final score: b {score[0]}, W {score[1]}\n")

        return self.game.getGameEnded(board, 1)

    def playGames(self, num, iter, verbose=True):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)
        originalNum = num

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            arena_log = open(f'logs/go/Game_Histories/Game_History_{self.datetime}.txt', 'a')
            arena_log.write("#############################\n")
            arena_log.write("Playing Game #" + str(eps + 1) + "  (g" + str(eps + 1) + "i" + str(iter) + ")\n")
            arena_log.write("#############################\n\n")
            arena_log.close()

            gameResult = self.playGame(verbose=verbose)
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

        if(originalNum%2 == 1):
            num += 1

        for _ in range(num):
            arena_log = open(f'logs/go/Game_Histories/Game_History_{self.datetime}.txt', 'a')
            arena_log.write("#############################\n")
            arena_log.write("Playing Game #" + str(eps + 1) + "  (g" + str(eps + 1) + "i" + str(iter) + ")\n")
            arena_log.write("#############################\n\n")
            arena_log.close()
           
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}\n'.format(eps=eps,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        bar.finish()

        return oneWon, twoWon, draws
