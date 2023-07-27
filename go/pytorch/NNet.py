import os
import time
from collections import OrderedDict

import numpy as np
import sys

from torch import nn

sys.path.append('../../')

import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable

from .GoAlphaNet import AlphaNetMaker as NetMaker
from .GoNNet import GoNNet

try:
    from utils import *
    from pytorch_classification.utils import Bar, AverageMeter
    from NeuralNet import NeuralNet
except:
    from ...utils import *
    from ...pytorch_classification.utils import Bar, AverageMeter
    from ...NeuralNet import NeuralNet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.0,
    'epochs': 10,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})

print(args)


class NNetWrapper(NeuralNet):
    def __init__(self, game, t='RES'):
        self.netType = t
        if t == 'RES':
            netMkr = NetMaker(game, args)
            self.nnet = netMkr.makeNet()
            # self.nnet = nn.DataParallel(self.nnet)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.nnet.to(device)
        else:
            self.nnet = GoNNet(game, args)
            self.nnet = nn.DataParallel(self.nnet)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.nnet.to(device)

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        trainLog = {
            'EPOCH': [],
            'P_LOSS': [],
            'V_LOSS': []
        }

        # dynamically change the batch size as the number of training examples increases
        batch_size = 64 * round((len(examples) / 100) / 64)
        if batch_size < 64:
            batch_size = 64

        for epoch in range(args.epochs):
            # print('EPOCH ::: ' + str(epoch + 1))
            trainLog['EPOCH'].append(epoch)
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Network', max=int(len(examples) / batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / batch_size):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                #Convert board histories to stacks as defined in paper
                temp_boards = list(boards)
                for i in range(len(temp_boards)):
                    temp_boards[i] = np.stack(temp_boards[i])
                boards = tuple(temp_boards)
                
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                boards, target_pis, target_vs = Variable(boards), Variable(target_pis), Variable(target_vs)

                # measure data loading time
                data_time.update(time.time() - end)
                # compute output
                out_pi, out_v = self.nnet(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.data.item(), boards.size(0))
                v_losses.update(l_v.data.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Epoch: {epoch:} | Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f} | Batch_size: {batch_size:}'.format(
                    epoch=epoch + 1,
                    batch=batch_idx,
                    size=int(len(examples) / batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                    batch_size=batch_size
                )

                bar.next()

            trainLog['P_LOSS'].append(pi_losses.avg)
            trainLog['V_LOSS'].append(v_losses.avg)
            bar.finish()

        #### plot avg pi loss and v loss for all epochs in iteration

        return pd.DataFrame(data=trainLog)

    def predict(self, board_list):
        """
        board: np array with board
        """
        # preparing input
        board = np.stack(board_list)
        #print("stack length: ", len(board))
        board = torch.FloatTensor(board.astype(np.float64))
        #print("stack length2: ", len(board))
        if args.cuda: board = board.contiguous().cuda()
        board = Variable(board, requires_grad=False)
        #print("stack length3: ", len(board))
        board = board.view(9, self.board_x, self.board_y)
        #print("stack length4: ", len(board))

        self.nnet.eval()

        pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='R_checkpoint', filename='R_checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='R_checkpoint', filename='R_checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise BaseException("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def load_checkpoint_from_plain_to_parallel(self, folder='R_checkpoint', filename='R_checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module." + k  # add 'module.' of dataparallel, so it works with examples from plain model
            new_state_dict[name] = v

        self.nnet.load_state_dict(new_state_dict)