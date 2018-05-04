
from .alphabrain import Arena
from .GoMCTS import MCTS
from .go.GoGame import display
from .go.GoGame import GoGame as game
from .go.GoPlayers import *
from .go.pytorch.NNet import NNetWrapper as nn

import numpy as np
from .utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
BoardSize=7
g = game(BoardSize)

# all players
hp = HumanGoPlayer(g).play

# nnet players
NetType='CNN'

ResNet=nn(g,t='RES')
ResNet.load_checkpoint('./HistoryLog/Go/R_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
ResArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
ResMCTS = MCTS(g, ResNet, ResArgs)
ResPlayer = lambda x: np.argmax(ResMCTS.getActionProb(x, temp=0))


CNN=nn(g,t='CNN')
CNN.load_checkpoint('./HistoryLog/Go/C_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
CNNArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
CNNMCTS = MCTS(g, CNN, CNNArgs)
CNNPlayer = lambda x: np.argmax(CNNMCTS.getActionProb(x, temp=0))



arena = Arena.Arena(ResPlayer, CNNPlayer, g, display=display)

print(arena.playGames(3, verbose=True))

class InterGame(object):

    def __init__(self,order,NetType='ResNet'):
        self.game=game(BoardSize)
        self.board=self.game.getInitBoard()
        self.n=self.game.getBoardSize()[0]
        self.playOrder=order
        if NetType=='ResNet':
            self.AlphaNet=nn(self.game,t='RES')
            self.AlphaNet.load_checkpoint('./HistoryLog/Go/R_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
            self.AlphaArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
            self.AlphaMCTS = MCTS(self.game, self.AlphaNet,self.AlphaArgs)
            self.Alpha= lambda x: np.argmax(self.AlphaMCTS.getActionProb(x, temp=0))
        else:
            self.AlphaNet=nn(self.game,t='CNN')
            self.AlphaNet.load_checkpoint('./HistoryLog/Go/C_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
            self.AlphaArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
            self.AlphaMCTS = MCTS(self.game, self.AlphaNet,self.AlphaArgs)
            self.Alpha= lambda x: np.argmax(self.AlphaMCTS.getActionProb(x, temp=0))

    def initialize(self):
        self.board=self.game.getInitBoard()

    def AlphaPlay(self):

        action = self.Alpha(self.game.getCanonicalForm(self.game.board,self.playOrder ))
        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.game.board, self.playOrder),1)

        if valids[action]==0:
            print(action)
            assert valids[action] >0
        self.game.board, curPlayer = self.game.getNextState(self.game.board, self.playOrder, action)

        return (int(action / self.n), action % self.n)

    def HumanPlay(self,move):
        x,y = [int(x) for x in move]
        action= self.game.n * x + y if x!= -1 else self.game.n ** 2
        self.game.board, curPlayer = self.game.getNextState(self.game.board, self.playOrder, action)

