###################################
#
#   WALLY by Jonathan K. Millen
#     (reconstruction by CMK)
#
###################################

import sys
import os
import importlib.util

"""current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(current_dir)
print(sys.path[-1])"""

import random
from GoGame import GoGame as Game
from GoCoach import Coach
from NNet import NNetWrapper as nn
from utils import *
import os, sys
from enum import IntEnum
from datetime import datetime
from GoMCTS import MCTS
import psutil
import time
import numpy as np
from GoGame import display

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Display(IntEnum):
    DISPLAY_BAR = 0
    DISPLAY_BOARD = 1

VERSION = '1.0'

###################################
#
#          Piece encoding
#
###################################
#
# 0000 => 0    empty sqare
# 0001 => 1    black stone
# 0010 => 2    white stone
# 0100 => 4    stone marker
# 0111 => 7    offboard square
# 1000 => 8    liberty marker
#
# 0101 => 5    black stone marked
# 0110 => 6    white stone marked
#
###################################
NetType = 'RES'  # or 'RES'
tag = 'MCTS_SimModified'
BoardSize = 7

args = dotdict({
    'numIters': 2,
    'numEps': 2,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.54,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 200,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 3,

    'checkpoint': './logs/go/{}_checkpoint/{}/'.format(NetType + '_' + tag, BoardSize),
    'load_model': True,
    'numItersForTrainExamplesHistory': 25,
    'display': Display.DISPLAY_BOARD,
    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M"),
})

# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
# if args.load_model:
#     checkpoint_dir = f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/'
#     checkpoint_files = [file for file in os.listdir(checkpoint_dir) if
#                         file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
#     latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
#     args['load_folder_file'] = [f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/', latest_checkpoint]

g = Game(BoardSize)
nnet = nn(g, t=NetType)
logPath = './logs/go/{}_log/{}/'.format(NetType + '_' + tag, BoardSize)

if args.load_model:
    # nnet.load_checkpoint_from_plain_to_parallel("", resource_path('model.tar'), cpu_only=True)
    nnet.load_checkpoint("", resource_path('model.tar'), cpu_only=True)
c = Coach(g, nnet, args, log=True, logPath=logPath)

# if args.load_model:
    # c.loadTrainExamples()

# stones
BLACK = 1
WHITE = 2

# current board used
global board
board = g.getInitBoard()
coords = None

global mcts
mcts = MCTS(g, nnet, args)

global curPlayer
curPlayer = 0

# file markers
files = '     a b c d e f g h j k l m n o p q r s t'

# ASCII representation of stones
pieces = '.#o  bw +'


def print_board():
    global board
    # print('=\n')
    print(display(board))


# set Go ban size
def set_board_size(command):
    # hook global variables
    global board

    # parse the board size
    size = int(command.split()[-1])

    # throw error if board size is not supported
    if size not in [7, 9, 13, 19]:
        print('? current board size not supported\n')
        return

    # calculate current board size

    # board = BOARDS[str(size)]
    g = Game(size)
    board = g.getInitBoard()


# clear board
def clear_board():
    # clear groupd
    global board
    board = g.getInitBoard()


# generate random move
def make_random_move(color):
    global board, mcts, curPlayer

    if color == BLACK:
        curPlayer = 1
    else:
        curPlayer = -1

    canonicalBoard = g.getCanonicalForm(board, curPlayer)
    # find empty random square
    action = np.argmax(mcts.getActionProb(canonicalBoard, temp=0))

    # make move on board
    board, curPlayer = g.getNextState(board, curPlayer, action)

    row = BoardSize - int(action / (BoardSize))
    col_coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    col = col_coords[action % 7]
    coordinate = col + str(row)
    # return the move
    return coordinate


# play command
def play(command):
    global board, curPlayer
    # parse color
    curPlayer = 1 if command.split()[1] == 'B' else -1

    player_idx = command.find(" ") + 1
    move_idx = command.find(" ", player_idx) + 1

    # player = gtp[player_idx]
    move = command[move_idx:]

    if move == "pass" or move == "PASS":
        action = BoardSize * BoardSize
    else:
        # parse square
        square_str = command.split()[-1]
        col = ord(square_str[0]) - ord('A') + 1 - (1 if ord(square_str[0]) > ord('I') else 0)
        row_count = int(square_str[1:]) if len(square_str[1:]) > 1 else ord(square_str[1:]) - ord('0')
        # row = (BOARD_RANGE - 1) - row_count
        action = ((BoardSize - row_count) * BoardSize) + (col - 1)
        # square = row * BOARD_RANGE + col

    # make move on board
    board, curPlayer = g.getNextState(board, curPlayer, action)


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
            print('= 1\n');
        elif 'version' in command:
            print('=', VERSION, '\n')
        elif 'list_commands' in command:
            print('= protocol_version\n')
        elif 'boardsize' in command:
            set_board_size(command); print('=\n')
        elif 'clear_board' in command:
            clear_board(); print('=\n')
        elif 'showboard' in command:
            print('='); print_board()
        elif 'play' in command:
            play(command); print('=\n')
        elif 'genmove' in command:
            print('=', make_random_move(BLACK if command.split()[-1] == 'B' else WHITE) + '\n')
        elif 'quit' in command:
            sys.exit()
        else:
            print('=\n')  # skip currently unsupported commands

        r = g.getGameEnded(board.copy(), curPlayer)
        if r != 0:
            print('= pass\n')
            print('= pass\n')


gtp()