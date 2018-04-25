from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import datetime
import time,os
args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.52,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 60,
    'cpuct': 1,

    'checkpoint': './HistoryLog/Go/checkpoint/',
    'load_model': False,
    'load_folder_file': ('./HistoryLog/Go/checkpoint/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 25,
    'display':False #True to display board, False to display progress bar
})

if __name__=="__main__":
    BoardSize=11
    g = Game(BoardSize)
    nnet = nn(g)
    logPath='HistoryLog/Go/{}'.format(BoardSize)
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
