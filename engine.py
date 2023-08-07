import os
import sys

import numpy as np
import yaml

from go.GoGame import GoGame as Game
from go.GoGame import display
from go.pytorch.NNet import NNetWrapper as nn
from GoCoach import Coach
from GoMCTS import MCTS

with open(os.path.join(os.path.dirname(sys.argv[0]), 'config.yaml'), "r") as stream:
    try:
        config = yaml.safe_load(stream)
        # print(config)
    except yaml.YAMLError as exc:
        raise ValueError(exc)

VERSION = '1.0'

game = Game(config["board_size"])
neural_network = nn(game, config)

neural_network.load_checkpoint("", os.path.join(os.path.dirname(sys.argv[0]), 'model.tar'), cpu_only=True)

coach = Coach(game, neural_network, config)

# stones
BLACK = 1
WHITE = 2

# current board used
board = game.getInitBoard()
mcts = MCTS(game, neural_network, config)
coords = None
curPlayer = 1
x_boards = []
y_boards = []
c_boards = [np.ones((7, 7)), np.zeros((7, 7))]
for i in range(8):
    x_boards.append(np.zeros((config["board_size"], config["board_size"])))
    y_boards.append(np.zeros((config["board_size"], config["board_size"])))
canonicalBoard = game.getCanonicalForm(board, curPlayer)
player_board = c_boards[0]
canonicalHistory, x_boards, y_boards = game.getCanonicalHistory(x_boards, y_boards, canonicalBoard.pieces, player_board)


def print_board():
    global board
    print(display(board))


def set_board_size(command):
    # hook global variables
    global board

    # parse the board size
    size = int(command.split()[-1])

    # throw error if board size is not supported
    if size not in [7]:
        print('? current board size not supported\n')
        return

    # board = BOARDS[str(size)]
    g = Game(size)
    board = g.getInitBoard()


def clear_board():
    global board
    board = game.getInitBoard()


def generate_move(color):
    global board, mcts, curPlayer, x_boards, y_boards, c_boards, canonicalHistory, player_board

    if color == BLACK:
        curPlayer = 1
    else:
        curPlayer = -1

    canonicalBoard = game.getCanonicalForm(board, curPlayer)
    # canonicalBoard = g.getCanonicalForm(board, curPlayer)
    # find empty random square
    # action = np.argmax(mcts.getActionProb(canonicalBoard, temp=0))

    # make move on board
    # board, curPlayer = g.getNextState(board, curPlayer, action)
    player_board = c_boards[0] if curPlayer == 1 else c_boards[1]

    # Generate a move based on most recent board state
    action = np.argmax(mcts.getActionProb(canonicalBoard, canonicalHistory, x_boards, y_boards, player_board, temp=0))
    # Perform the move
    board, curPlayer = game.getNextState(board, curPlayer, action)

    row = config["board_size"] - int(action / config["board_size"])
    col_coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    col = col_coords[action % 7]
    coordinate = col + str(row)

    # Update histories to prepare for next move
    canonicalBoard = game.getCanonicalForm(board, curPlayer)
    player_board = c_boards[0] if curPlayer == 1 else c_boards[1]
    canonicalHistory, x_boards, y_boards = game.getCanonicalHistory(x_boards, y_boards, canonicalBoard.pieces,
                                                                 player_board)

    x_boards, y_boards = y_boards, x_boards

    # return the move
    return coordinate


# play command
def play(command):
    global board, curPlayer, x_boards, y_boards, c_boards, canonicalHistory
    # parse color
    curPlayer = 1 if command.split()[1] == 'B' else -1

    canonicalBoard = game.getCanonicalForm(board, curPlayer)

    player_idx = command.find(" ") + 1
    move_idx = command.find(" ", player_idx) + 1

    # player = gtp[player_idx]
    move = command[move_idx:]

    if move == "pass" or move == "PASS":
        action = config["board_size"] * config["board_size"]
    else:
        # parse square
        square_str = command.split()[-1]
        col = ord(square_str[0]) - ord('A') + 1 - (1 if ord(square_str[0]) > ord('I') else 0)
        row_count = int(square_str[1:]) if len(square_str[1:]) > 1 else ord(square_str[1:]) - ord('0')
        # row = (BOARD_RANGE - 1) - row_count
        action = ((config["board_size"] - row_count) * config["board_size"]) + (col - 1)
        # square = row * BOARD_RANGE + col

    # make move on board
    board, curPlayer = game.getNextState(board, curPlayer, action)

    # Update histories to prepare for next move
    canonicalBoard = game.getCanonicalForm(board, curPlayer)
    player_board = c_boards[0] if curPlayer == 1 else c_boards[1]
    canonicalHistory, x_boards, y_boards = game.getCanonicalHistory(x_boards, y_boards, canonicalBoard.pieces,
                                                                 player_board)

    # Player will switch, so switch x and y boards (current/opposing player histories)
    x_boards, y_boards = y_boards, x_boards


# GTP communication protocol
def gtp():
    # main GTP loop
    global curPlayer

    while True:
        # accept GUI command
        command = input()

        # handle commands
        if 'name' in command:
            print('= Go2AI\n')
        elif 'protocol_version' in command:
            print('= 1\n')
        elif 'version' in command:
            print('=', VERSION, '\n')
        elif 'list_commands' in command:
            print('= protocol_version\n')
        elif 'boardsize' in command:
            set_board_size(command)
            print('=\n')
        elif 'clear_board' in command:
            clear_board()
            print('=\n')
        elif 'showboard' in command:
            print('=')
            print_board()
        elif 'play' in command:
            play(command)
            print('=\n')
        elif 'genmove' in command:
            print('=', generate_move(BLACK if command.split()[-1] == 'B' else WHITE) + '\n')
        elif 'quit' in command:
            sys.exit()
        else:
            print('=\n')  # skip currently unsupported commands

        r = game.getGameEnded(board.copy(), curPlayer)
        if r != 0:
            print('= pass\n')
            print('= pass\n')


gtp()
