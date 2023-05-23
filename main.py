from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import os,sys
from enum import Enum

sys.setrecursionlimit(5000)

class Display(Enum):
    NO_DISPLAY = 0
    DISPLAY_BAR = 1
    DISPLAY_BOARD = 2

BoardSize=7
NetType='CNN' # or 'RES'
tag='MCTS_SimModified'

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.54,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200,
    'arenaCompare': 50,
    'cpuct': 3,

    'checkpoint': './logs/go/{}_checkpoint/{}/'.format(NetType + '_' + tag, BoardSize),
    'load_model': False,
    'numItersForTrainExamplesHistory': 25,
    # temporary for debugging since enum is not working properly
    'display': 2
})

if __name__=="__main__":

    g = Game(BoardSize)
    nnet = nn(g, t=NetType)
    logPath='./logs/go/{}_log/{}/'.format(NetType + '_' + tag, BoardSize)
    try:
        os.makedirs(logPath)
    except:
        pass

    if args.load_model:
        nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')

    c = Coach(g, nnet, args, log=True, logPath=logPath)
    if args.load_model:
        print("Loading trainExamples from file")
        c.loadTrainExamples()
    c.learn()