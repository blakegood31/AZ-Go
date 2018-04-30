import Arena
from GoMCTS import MCTS
from go.GoGame import display
from go.GoGame import GoGame as game
from go.GoPlayers import *
from go.pytorch.NNet import NNetWrapper as nn

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
BoardSize=7
g = game(BoardSize)

# all players
rp = RandomPlayer(g).play
gp = GreedyGoPlayer(g).play
hp = HumanGoPlayer(g).play

# nnet players
NetType='CNN'

ResNet=nn(g,t='RES')

ResNet.load_checkpoint('./HistoryLog/Go/R_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
<<<<<<< HEAD
ResArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
=======
ResArgs = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
>>>>>>> ef4f12adbc392ff51f75311f96f6c3726e529756
ResMCTS = MCTS(g, ResNet, ResArgs)
ResPlayer = lambda x: np.argmax(ResMCTS.getActionProb(x, temp=0))


CNN=nn(g,t='CNN')
CNN.load_checkpoint('./HistoryLog/Go/C_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
CNNArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
CNNMCTS = MCTS(g, CNN, CNNArgs)
CNNPlayer = lambda x: np.argmax(CNNMCTS.getActionProb(x, temp=0))

arena = Arena.Arena(ResPlayer, CNNPlayer, g, display=display)
print(arena.playGames(3, verbose=True))
