from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import datetime
import time,os
BoardSize=7
args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 35,
    'cpuct': 1,

    'checkpoint': './HistoryLog/Go/checkpoint/{}/'.format(BoardSize),
    'load_model': False,
    'load_folder_file': ('./HistoryLog/Go/checkpoint/{}/'.format(BoardSize),'best.pth.tar'),
    'numItersForTrainExamplesHistory': 25,
    'display':False #True to display board, False to display progress bar
})

if __name__=="__main__":

    g = Game(BoardSize)
    nnet = nn(g)
    logPath='HistoryLog/Go/Log/{}'.format(BoardSize)
    try:
        os.mkdir(logPath)
    except:
        pass

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args,log=True,logPath=logPath)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
