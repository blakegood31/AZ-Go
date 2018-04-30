from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import datetime
import time,os
BoardSize=11
NetType='R'
tag='MCTS_SimModified'
#NetType='R'
args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200,
    'arenaCompare': 50,
    'cpuct': 3,

    'checkpoint': './HistoryLog/Go/{}_checkpoint/{}/'.format(NetType+'_'+tag,BoardSize),
    'load_model': False,
    'load_folder_file': ('./HistoryLog/Go/{}_checkpoint/{}/'.format(NetType+'_'+tag,BoardSize),'best.pth.tar'),
    'numItersForTrainExamplesHistory': 25,
    'display':False #True to display board, False to display progress bar
})

if __name__=="__main__":

    g = Game(BoardSize)
    nnet = nn(g,t='RES' if NetType=='R' else 'CNN')
    logPath='./HistoryLog/Go/{}_Log/{}/'.format(NetType+'_'+tag,BoardSize)
    os.makedirs(logPath)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args,log=True,logPath=logPath)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
