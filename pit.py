
try:
    import Arena
    from GoMCTS import MCTS
    from go.GoGame import display
    from go.GoGame import GoGame as game
    from go.GoPlayers import *
    from go.pytorch.NNet import NNetWrapper as nn
    from utils import *
except:
    import Arena
    from GoMCTS import MCTS
    from .go.GoGame import display
    from .go.GoGame import GoGame as game
    from .go.GoPlayers import *
    from .go.pytorch.NNet import NNetWrapper as nn
    from utils import *

import numpy as np
import pandas as pd


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
ResNet.load_checkpoint('./HistoryLog/Go/R_Ver2_checkpoint/{}/'.format(BoardSize),'RVer2.best.pth.tar')
ResArgs = dotdict({'numMCTSSims': 3000, 'cpuct':17.0})
ResMCTS = MCTS(g, ResNet, ResArgs)
ResPlayer = lambda x: np.argmax(ResMCTS.getActionProb(x, temp=0))


CNN=nn(g,t='CNN')
CNN.load_checkpoint('./HistoryLog/Go/C_checkpoint/{}/'.format(BoardSize),'checkpoint_4.pth.tar')
CNNArgs = dotdict({'numMCTSSims': 250, 'cpuct':3.0})
CNNMCTS = MCTS(g, CNN, CNNArgs)
CNNPlayer = lambda x: np.argmax(CNNMCTS.getActionProb(x, temp=0))

arena = Arena.Arena(ResPlayer, CNNPlayer, g, display=display)

def tour():
    pathAtt='./HistoryLog/Go/'
    Rcand={
        'R1_10':[pathAtt+'R_Ver1_checkpoint/7/','checkpoint_11.pth.tar'],
        'R1_40':[pathAtt+'R_Ver1_checkpoint/7/','checkpoint_47.pth.tar'],
        'R1_B':[pathAtt+'R_Ver1_checkpoint/7/','best.pth.tar'],
        'R2_B':[pathAtt+'R_Ver2_checkpoint/7/','best.pth.tar'],
        'R3_B':[pathAtt+'R_Ver3_checkpoint/7/','best.pth.tar']
    }
    Ccand={
        'C_10':[pathAtt+'C_Ver1_checkpoint/7/','checkpoint_6.pth.tar'],
        'C_40':[pathAtt+'C_Ver1_checkpoint/7/','checkpoint_40.pth.tar'],
        'C_B':[pathAtt+'C_Ver1_checkpoint/7/','best.pth.tar']
    }
    
    compares=[
        ('R1_10','C_10'),
        ('R1_40','C_40'),
        ('R1_B','C_B'),
        ('R2_B','C_B'),
        ('R3_B','C_B'),
        ('R1_B','R2_B'),
        ('R1_B','R3_B'),
        ('R2_B','R3_B')
    ]
    res=[]
    for c in [
        ('R1_10','C_10')]:
        print(c)
        p1type='RES' if c[0][0]=='R' else 'CNN'
        p2type='RES' if c[1][0]=='R' else 'CNN'
        p1checkpoint=Rcand[c[0]] if c[0][0]=='R' else Ccand[c[0]]
        p2checkpoint=Rcand[c[1]] if c[1][0]=='R' else Ccand[c[1]]

        print(p1type,p2type)
        print(p1checkpoint,p2checkpoint)
        
        Net1=nn(g,t=p1type)
        Net1.load_checkpoint(p1checkpoint[0],p1checkpoint[1])
        Args1 = dotdict({'numMCTSSims': 3000, 'cpuct':17.5})
        MCTS1 = MCTS(g, Net1, Args1)
        Player1 = lambda x: np.argmax(MCTS1.getActionProb(x, temp=0))


        Net2=nn(g,t=p2type)
        Net2.load_checkpoint(p2checkpoint[0],p2checkpoint[1])
        Args2 = dotdict({'numMCTSSims': 3000 if p2type=='RNN' else 250, 'cpuct':17.5 if p2type=='RNN' else 3.0})
        MCTS2 = MCTS(g, Net2,Args2)
        Player2 = lambda x: np.argmax(MCTS2.getActionProb(x, temp=0))

        arena = Arena.Arena(Player1, Player2, g, display=display)
        _res=arena.playGames(10, verbose=True)
        res.append(_res)
    result={'1win':[],
    '2win':[],
    'draw':[]
    }
    for r in res:
        result['1win'].append(r[0])
        result['2win'].append(r[1])
        result['draw'].append(r[2])
    pd.DataFrame(data=result).to_csv('reuslt.csv')