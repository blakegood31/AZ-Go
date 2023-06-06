from __future__ import print_function
try:
    from Game import Game
    from GoLogic import Board
except:
    try:
        from alphabrain.Game import Game
        from alphabrain.go.GoLogic import Board
    except:
        from Game import Game
        from go.GoLogic import Board
import numpy as np


class GoGame(Game):
    def __init__(self, n=19):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return b

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # print("getting next state from perspect of player {} with action {}".format(player,action))

        b = board.copy()
        if action == self.n * self.n:
            return (b, -player)

        move = (int(action / self.n), action % self.n)

        b.execute_move(move,player)

        return (b, -player)

    # modified
    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0 for i in range(self.getActionSize())]
        b = board.copy()
        legalMoves = b.get_legal_moves(player)

        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1

        return np.array(valids)

    # modified
    def getGameEnded(self, board, player,returnScore=False):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        winner = 0
        (score_black, score_white) = self.getScore(board)
        by_score = 0.5 * (board.n * board.n + board.komi)

        if len(board.history) > 500:
            # stack overflow occurs when MCTS infinitely runs
            # to avoid this, end game with tie over a set number of moves
            print("#### MCTS Recursive Base Case Triggered ####")
            winner = 1e-4
        elif len(board.history) > 1:
            if (board.history[-1] is None and board.history[-2] is None\
                    and player == -1):
                if score_black > score_white:
                    winner = -1
                elif score_white > score_black:
                    winner = 1
                else:
                    # Tie
                    winner = 1e-4
            elif score_black > by_score or score_white > by_score:
                if score_black > score_white:
                    winner = -1
                elif score_white > score_black:
                    winner = 1
                else:
                    # Tie
                    winner = 1e-4
        if returnScore:
            return winner, (score_black, score_white)
        return winner

    def getScore(self, board):
        score_white = np.sum(board.pieces == -1)
        score_black = np.sum(board.pieces == 1)
        empties = zip(*np.where(board.pieces == 0))
        for empty in empties:
            # Check that all surrounding points are of one color
            if board.is_eyeish(empty, 1):
                score_black += 1
            elif board.is_eyeish(empty, -1):
                score_white += 1
        score_white += board.komi
        score_white -= board.passes_white
        score_black -= board.passes_black
        return (score_black, score_white)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        canonicalBoard=board.copy()

        canonicalBoard.pieces= board.pieces* player
        return canonicalBoard

    # modified
    def getSymmetries(self, canonicalForm, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []
        b_pieces = canonicalForm
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(b_pieces, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, canonicalForm):
        # 8x8 numpy array (canonical board)
        return np.array(canonicalForm).tostring()

    def getBoard(self, obs, agent):
        if agent == 'black_0':
            zero = 0
        else:
            zero = np.copysign(0, -1
            )
        canonicalForm = np.zeros((self.getBoardSize()[0], self.getBoardSize()[0]))
        for i in range(self.getBoardSize()[0]):
            for j in range(self.getBoardSize()[0]):
                if obs['observation'][i, j, 0] == 1:
                    canonicalForm[i, j] = 1
                elif obs['observation'][i, j, 1] == 1:
                    canonicalForm[i, j] = -1
                else:
                    canonicalForm[i, j] = zero  # Empty intersection
        return canonicalForm



def display(board):
    state = ""
    b_pieces = np.array(board.pieces)

    n = b_pieces.shape[0]

    for y in range(n):
        state = state + str(y) + " |"
    state += "\n"
    state += " -----------------------\n"
    for y in range(n):
        state += str(y) + "|"
        for x in range(n):
            piece = b_pieces[y][x]    # get the piece to print
            if piece == 1:
                state += "b "
            elif piece == -1:
                state += "W "
            else:
                if x == n:
                    state += "-"
                else:
                    state += "- "
        state += "|\n"

    state += " -----------------------"
    return state
