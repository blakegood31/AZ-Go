from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *
import os, sys
from enum import IntEnum
from datetime import datetime

sys.setrecursionlimit(5000)


class Display(IntEnum):
    NO_DISPLAY = 0
    DISPLAY_BAR = 1
    DISPLAY_BOARD = 2


BoardSize = 5
NetType = 'CNN'  # or 'RES'
tag = 'MCTS_SimModified'


args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.54,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 50,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 3,

    'checkpoint': './logs/go/{}_checkpoint/{}/'.format(NetType + '_' + tag, BoardSize),
    'load_model': False,
    'numItersForTrainExamplesHistory': 25,
    'display': Display.DISPLAY_BOARD,
    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M"),
})

if args.load_model:
    checkpoint_dir = f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/'
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.startswith('checkpoint_') and file.endswith('.pth.tar.examples')]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    args['load_folder_file'] = [f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/', latest_checkpoint]

if __name__ == "__main__":

    g = Game(BoardSize)
    nnet = nn(g, t=NetType)

    logPath = './logs/go/{}_log/{}/'.format(NetType + '_' + tag, BoardSize)
    try:
        os.makedirs(logPath)
    except:
        pass

    game_path = './logs/go/Game_Histories'
    try:
        os.makedirs(game_path)
    except:
        pass

    training_path = './logs/go/Training_Results'
    try:
        os.makedirs(training_path)
    except:
        pass

    # write header of Game_History.txt file
    arena_log = open(f'logs/go/Game_Histories/Game_History_{args.datetime}.txt', 'w')
    arena_log.write("############################################################\n")
    arena_log.write("This file contains visual representations of the game\n")
    arena_log.write("boards when pitting the current model against itself.\n\n")
    arena_log.write("Specific games and iterations can be found by searching\n")
    arena_log.write("the text file for g{game #}i{iteration #}.\n")
    arena_log.write("Example: Game 1 in Iteration 2 -- g1i2\n\n")
    arena_log.write(f"Data collected during training on {args.datetime}" + "\n\n")
    arena_log.write("Player 1 is the current model; player -1 is the previous model.\n\n")
    arena_log.write("Total number of iterations: " + str(args['numIters']) + "\n")
    arena_log.write("Number of episodes per iteration: " + str(args['numEps']) + "\n")
    arena_log.write("Number of MCTS simulations: " + str(args['numMCTSSims']) + "\n")
    arena_log.write("Number of arena compares: " + str(args['arenaCompare']) + "\n")
    arena_log.write("############################################################\n\n")
    arena_log.close()

    if args.load_model:
        nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')

    c = Coach(g, nnet, args, log=True, logPath=logPath)

    if args.load_model:
        print("Loading trainExamples from file")
        c.loadTrainExamples()

    c.learn()

    # add time completed training here...
    arena_log = open(f'logs/go/Game_Histories/Game_History_{args.datetime}.txt', 'a')
    time_completed = datetime.now().strftime("%d-%m-%Y %H:%M")
    arena_log.write(f"Training completed at: {time_completed}")
    arena_log.close()