from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import os,sys
from enum import Enum
from datetime import datetime

sys.setrecursionlimit(5000)

class Display(Enum):
    NO_DISPLAY = 0
    DISPLAY_BAR = 1
    DISPLAY_BOARD = 2

BoardSize=5
NetType='CNN' # or 'RES'
tag='MCTS_SimModified'

args = dotdict({
    'numIters': 10,
    'numEps': 5,
    'tempThreshold': 15,
    'updateThreshold': 0.54,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200,
    'arenaCompare': 5,
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

    #########################################
    episode_log = open('logs/go/Game_History.txt', 'w')
    episode_log.write("############################################################\n")
    episode_log.write("This file contains visual representations of the game\n")
    episode_log.write("boards when pitting the current model against itself.\n\n")
    episode_log.write("Specific games and iterations can be found by searching\n")
    episode_log.write("the text file for g{game #}i{iteration #}.\n")
    episode_log.write("Example: Game 1 in Iteration 2 -- g1i2\n\n")
    episode_log.write("Data collected during training on " + str(datetime.now()) + "\n\n")
    episode_log.write("Player 1 is the current model; player -1 is the previous model.\n\n")
    episode_log.write("Total number of iterations: " + str(args['numIters']) + "\n")
    episode_log.write("Number of episodes per iteration: " + str(args['numEps']) + "\n")
    episode_log.write("Number of MCTS simulations: " + str(args['numMCTSSims']) + "\n")
    episode_log.write("Number of arena compares: " + str(args['arenaCompare']) + "\n")
    episode_log.write("############################################################\n\n")
    episode_log.close()

    if args.load_model:
        nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')

    c = Coach(g, nnet, args, log=True, logPath=logPath)
    if args.load_model:
        print("Loading trainExamples from file")
        c.loadTrainExamples()

    c.learn()