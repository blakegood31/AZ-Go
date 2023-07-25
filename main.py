from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import os, sys
from enum import IntEnum
from datetime import datetime
import torch.multiprocessing as mp
import time

sys.setrecursionlimit(5000)


class Display(IntEnum):
    DISPLAY_BAR = 0
    DISPLAY_BOARD = 1


BoardSize = 7
NetType = 'RES'  # or 'RES'
tag = 'MCTS_SimModified'

args = dotdict({
    # training parameters
    'numIters': 200,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.54,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 150,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 50,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.0,
    'numItersForTrainExamplesHistory': 30,

    # customization
    'load_model': True,
    'distributed_training': False,  # use Google Drive for computing on multiple machines
    'display': Display.DISPLAY_BAR,
    'ram_cap': 120,

    # utility
    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M"),
    'sgf_datetime': datetime.now().strftime("%d-%m-%Y %H"),
    'nettype': NetType,
    'boardsize': BoardSize,
    'start_time': time.time(),
    'checkpoint': './logs/go/{}_checkpoint/{}/'.format(NetType + '_' + tag, BoardSize),
})

if args.load_model:
    checkpoint_dir = f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/'
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if
                        file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    args['load_folder_file'] = [f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/', latest_checkpoint]
    args['start_iter'] = int(latest_checkpoint.split('_')[1].split('.')[0]) + 1

else:
    args['start_iter'] = 1

if __name__ == "__main__":
    mp.set_start_method('spawn')

    g = Game(BoardSize)
    nnet = nn(g, t=NetType)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    logPath = './logs/go/{}_log/{}/'.format(NetType + '_' + tag, BoardSize)
    try:
        os.makedirs(logPath)
    except:
        pass

    game_path = './logs/go/Game_History'
    try:
        os.makedirs(game_path)
    except:
        pass

    training_path = './logs/go/Training_Results'
    try:
        os.makedirs(training_path)
    except:
        pass

    if args.load_model:
        # if you are loading a checkpoint created from a model without DataParallel
        # use the load_checkpoint_from_plain_to_parallel() function
        # instead of the load_checkpoint() function
        nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')

    c = Coach(g, nnet, args, log=True, logPath=logPath, NetType=NetType)

    if args.load_model:
        c.skipFirstSelfPlay = True

    c.learn()
