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
HUMAN_FIRST=-1
HUMAN_SECOND=1

class InterGame(object):

    def __init__(self,order,NetType='ResNet'):
        self.game=game(BoardSize)
        self.board=self.game.getInitBoard()
        self.n=self.game.getBoardSize()[0]
        self.players=[self.AlphaPlay,None,self.HumanPlay]
        self.curPlayer=1
        if NetType=='ResNet':
            self.AlphaNet=nn(self.game,t='RES')
            self.AlphaNet.load_checkpoint('./HistoryLog/Go/R_MCTS_SimModified_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
            self.AlphaArgs = dotdict({'numMCTSSims': 2000, 'cpuct':17.3})
            self.AlphaMCTS = MCTS(self.game, self.AlphaNet,self.AlphaArgs)
            self.Alpha= lambda x: np.argmax(self.AlphaMCTS.getActionProb(x, temp=0))
        else:
            self.AlphaNet=nn(self.game,t='CNN')
            self.AlphaNet.load_checkpoint('./HistoryLog/Go/C_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
            self.AlphaArgs = dotdict({'numMCTSSims': 2000, 'cpuct':17.3})
            self.AlphaMCTS = MCTS(self.game, self.AlphaNet,self.AlphaArgs)
            self.Alpha= lambda x: np.argmax(self.AlphaMCTS.getActionProb(x, temp=0))

    def initialize(self):
        self.board=self.game.getInitBoard()

    def AlphaPlay(self,*move):

        action = self.Alpha(self.game.getCanonicalForm(self.board,self.curPlayer ))
        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.curPlayer),1)

        if valids[action]==0:
            print(action)
            assert valids[action] >0
        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

        return (int(action / self.n), action % self.n)

    def HumanPlay(self,move):
        x,y = [int(x) for x in move]
        action= self.game.n * x + y if x!= -1 else self.game.n ** 2
        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)
        return 
    def showBoard(self):
        display(self.board)
        