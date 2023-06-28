from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import os, sys
from enum import IntEnum
from datetime import datetime
import torch.multiprocessing as mp

sys.setrecursionlimit(5000)


class Display(IntEnum):
    DISPLAY_BAR = 0
    DISPLAY_BOARD = 1


BoardSize = 7
NetType = 'RES'  # or 'RES'
tag = 'MCTS_SimModified'


args = dotdict({
    'numIters': 1000,
    'numEps': 400,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.54,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 150,         # Number of games moves for MCTS to simulate.
    'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    # 10 process maximum currently, do not set any higher
    'num_processes': 4,

    'checkpoint': './logs/go/{}_checkpoint/{}/'.format(NetType + '_' + tag, BoardSize),
    'load_model': True,
    'numItersForTrainExamplesHistory': 25,
    'display': Display.DISPLAY_BAR,
    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M"),
})
if args.load_model:
    checkpoint_dir = f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/'
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    args['load_folder_file'] = [f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/', latest_checkpoint]

if __name__ == "__main__":
    mp.set_start_method('spawn')

    g = Game(BoardSize)
    nnet = nn(g, t=NetType)

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
        # nnet.load_checkpoint_from_plain_to_parallel(args.checkpoint, 'best.pth.tar')

        nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')

    c = Coach(g, nnet, args, log=True, logPath=logPath)

    if args.load_model:
        print("Loading trainExamples from file")
        c.loadTrainExamples()

    c.learn()