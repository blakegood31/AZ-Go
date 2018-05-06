try:
    from GoMCTS import MCTS
    from go.GoGame import display
    from go.GoGame import GoGame as game
    from go.GoPlayers import *
    from go.pytorch.NNet import NNetWrapper as nn
    import numpy as np
    from utils import *
except:
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
HUMAN_FIRST=-1
HUMAN_SECOND=1

class InterGame(object):

    def __init__(self,NetType='ResNet'):
        self.game=game(BoardSize)
        self.board=self.game.getInitBoard()
        self.n=self.game.getBoardSize()[0]
        self.players=[self.AlphaPlay,None,self.HumanPlay]
        self.curPlayer=1
        self.gameStatus=0
        if NetType=='ResNet':
            self.AlphaNet=nn(self.game,t='RES')
            self.AlphaNet.load_checkpoint('/home/zc1213/course/alphabackend/alphabrain/HistoryLog/Go/R_Ver2_checkpoint/{}/'.format(BoardSize),'RVer2.best.pth.tar')
            self.AlphaArgs = dotdict({'numMCTSSims': 2000, 'cpuct':17.3})
            self.AlphaMCTS = MCTS(self.game, self.AlphaNet,self.AlphaArgs)
            self.Alpha= lambda x: np.argmax(self.AlphaMCTS.getActionProb(x, temp=0))
        else:
            self.AlphaNet=nn(self.game,t='CNN')
            self.AlphaNet.load_checkpoint('/home/zc1213/course/alphabackend/alphabrain/HistoryLog/Go/C_checkpoint/{}/'.format(BoardSize),'best.pth.tar')
            self.AlphaArgs = dotdict({'numMCTSSims': 2000, 'cpuct':17.3})
            self.AlphaMCTS = MCTS(self.game, self.AlphaNet,self.AlphaArgs)
            self.Alpha= lambda x: np.argmax(self.AlphaMCTS.getActionProb(x, temp=0))

    def initialize(self):
        self.board=self.game.getInitBoard()
    def judgeGame(self):
        self.gameStatus=self.game.getGameEnded(self.board,self.curPlayer)
        if self.gameStatus==-1:
            print("player 1 lost.")
            return -1
        elif self.gameStatus==1:
            print("player 1 won.")
            return 1
        else:
            print("game continues.")
            return 0
    def AlphaPlay(self,*move):
        assert(self.judgeGame()==0)
        
        action = self.Alpha(self.game.getCanonicalForm(self.board,self.curPlayer ))
        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.curPlayer),1)

        if valids[action]==0:
            print(action)
            assert valids[action] >0
        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

        return (int(action / self.n), action % self.n)

    def HumanPlay(self,move):
        assert(self.judgeGame()==0)
        x,y = [int(x) for x in move]
        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.curPlayer),1)
        action= self.game.n * x + y if x!= -1 else self.game.n ** 2
        if valids[action]==0:
            print("Invalid Move!")
            return
        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)
        return 
    
    def showBoard(self):
        display(self.board)


class InterGameTest(object):

    def __init__(self,NetType='ResNet'):
        self.game=game(BoardSize)
        self.board=self.game.getInitBoard()
        self.n=self.game.getBoardSize()[0]
        self.players=[self.AlphaPlay,None,self.HumanPlay]
        self.curPlayer=1
        self.gameStatus=0

    def initialize(self):
        self.board=self.game.getInitBoard()
    def judgeGame(self):
        self.gameStatus=self.game.getGameEnded(self.board,self.curPlayer)
        if self.gameStatus==-1:
            print("player 1 lost.")
            return -1
        elif self.gameStatus==1:
            print("player 1 won.")
            return 1
        else:
            print("game continues.")
            return 0
    def AlphaPlay(self,*move):

        return (3,3)

    def HumanPlay(self,move):
        assert(self.judgeGame()==0)
        x,y = [int(x) for x in move]
        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.curPlayer),1)
        action= self.game.n * x + y if x!= -1 else self.game.n ** 2
        if valids[action]==0:
            print("Invalid Move!")
            return
        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)
        return

    def showBoard(self):
        display(self.board)
