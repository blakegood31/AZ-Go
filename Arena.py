import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
#from pettingzoo.classic import go_v5 as go
from go import PZGo as go


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
        action_history = []
        arena_env = go.env(board_size=self.game.getBoardSize()[0])
        arena_env.reset()
        step = 1
        outcome = 0

        for agent in arena_env.agent_iter():
            obs, reward, termination, truncation, info = arena_env.last()
            canonical_form = self.game.get_pz_canonical_form(self.game.getBoardSize()[0], obs)

            if termination or truncation:
                if self.game_num % 2 == 0:
                    # on even number games, new model = black = player 2
                    if agent == 'black_0' and reward == 1:
                        outcome = -1
                    elif agent == 'white_0' and reward == 1:
                        outcome = 1
                    elif agent == 'black_0' and reward == -1:
                        outcome = 1
                    elif agent == 'white_0' and reward == -1:
                        outcome = -1
                    arena_env.close()
                    return outcome
                else:
                    # on odd number games, old model = black = player 1
                    if agent == 'black_0' and reward == 1:
                        outcome = 1
                    elif agent == 'white_0' and reward == 1:
                        outcome = -1
                    elif agent == 'black_0' and reward == -1:
                        outcome = -1
                    elif agent == 'white_0' and reward == -1:
                        outcome = 1
                    arena_env.close()
                    return outcome
                
            #End game if a player is winning by certain threshold
            score = arena_env.unwrapped.getScore()
            if ((score > self.by_score) or (score < -self.by_score)) and step > 14: 
                outcome = 0
                print("Won by score")
                if self.game_num % 2 == 0:
                    if score > 0:
                        outcome = -1
                    else:
                        outcome = 1
                else:
                    if score > 0:
                        outcome = 1
                    else:
                        outcome = -1
                return outcome
            
            # determine player and moves
            # new model is player2
            # old model is player1
            # on even number games, new model starts
            # on odd number games, old model starts
            if self.game_num % 2 == 0:
                # new model gets to start (new model is black agent)
                if agent == "black_0":
                    # new model = black
                    moves = self.player2(canonical_form, arena_env, action_history)
                    current_player_num = 2
                else:
                    # old model = white
                    moves = self.player1(canonical_form, arena_env, action_history)
                    current_player_num = 1
            else:
                # old model gets to start (old model is black agent)
                if agent == "black_0":
                    # old model = black
                    moves = self.player1(canonical_form, arena_env, action_history)
                    current_player_num = 1
                else:
                    # new model = white
                    moves = self.player2(canonical_form, arena_env, action_history)
                    current_player_num = 2
            
            action = np.argmax(moves)
            action_history.append(action)

            if self.displayValue == 2:
                print(
                    f"================Game {self.game_num} Step:{step}=====Next Player:{current_player_num}==========")
                self.game.display_pz_board(board_size=self.game.getBoardSize()[0], observation=obs, agent=agent)
                if moves[25] == 1:
                    print(f"Player {current_player_num} chose to PASS")

            step += 1
            arena_env.step(action)

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
        bar = Bar('Arena.playGames', max=num)
        end = time.time()

        player_one_wins = 0
        player_two_wins = 0
        draws = 0

        for _ in range(num):
            game_result = self.playGame()
            if game_result == 1:
                player_one_wins += 1
            elif game_result == -1:
                player_two_wins += 1
            else:
                draws += 1

            # bookkeeping + plot progress
            self.game_num += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}\n'.format(
                eps=self.game_num,
                maxeps=num,
                et=eps_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td)
            bar.next()

        bar.finish()

        return player_one_wins, player_two_wins, draws
