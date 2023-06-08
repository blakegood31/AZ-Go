import math
import time
import sys
import numpy as np
from pettingzoo.classic import go_v5 as go
from go import PZGo


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def get_stack_size(self):
        size = 2  # current frame and caller's frame always exist
        while True:
            try:
                sys._getframe(size)
                size += 1
            except ValueError:
                return size - 1  # subtract current frame

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.smartSimNum = 10 * (self.game.getBoardSize()[0] ** 2)
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonical_board, env, action_history, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        # print('current sim numbers:{}'.format(max(self.args.numMCTSSims,self.smartSimNum)))
        start_time = time.time()

        for i in range(max(self.args.numMCTSSims, self.smartSimNum)):

            # make a copy of the environment
            search_env = PZGo.env(board_size=self.args['board_size'])
            search_env.reset()
            search_env.unwrapped.deep_copy(env.unwrapped)

            # search_env = go.env(board_size=self.args['board_size'])
            #
            # if len(action_history) == 0:
            #     search_env.reset()
            # else:
            #     search_env.reset()
            #     for i in range(len(action_history)):
            #         search_env.step(action_history[i])

            self.search(search_env, canonical_board)

            # release resources used by env
            search_env.close()

        obs, reward, termination, truncation, info = env.last()
        s = self.game.stringRepresentation(canonical_board)

        counts = np.array([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])
        valids = np.array(obs['action_mask'])

        self.smartSimNum = 10 * (np.count_nonzero(valids))

        if np.sum(counts) == 0:
            counts = valids
        else:
            counts *= valids

        if temp == 0:
            if np.sum(counts) == 0:
                bestA = obs['observation'].shape[0] * obs['observation'].shape[0]
            else:
                bestA = np.argmax(counts)

            try:
                assert (valids[bestA] != 0)
            except:
                print("temp=0, assert valids[bestA]!=0 !!!")
                print("current valids:", valids)
                flag_Qsa = False
                flag_Nsa = False
                if s in self.Ps:
                    print("s in p! Which measn it's been visited, has the probability of each action", self.Ps[s])
                for _ in range(self.game.getActionSize()):
                    if (s, _) in self.Nsa:
                        print(_, "in Nsa! which measn its value is calculated to ", self.Nsa[(s, _)])
                    else:
                        flag_Nsa = True
                        print(_, "no Nsa value, set 0 by default in counts=[...]!")

                    if (s, _) in self.Qsa:
                        print(_, "in! Qsa with value:", self.Qsa[(s, _)])
                    else:
                        flag_Qsa = True
                        print(_, "no Qsa value")

                    if flag_Nsa and flag_Qsa:
                        print("no nsa, no qsa")
                    if flag_Nsa and not flag_Qsa:
                        print("no nsa, has qsa")
                    if not flag_Nsa and flag_Qsa:
                        print("has nsa, no qsa")

                print(counts)

            probs = [0 for i in range(len(counts))]
            probs[bestA] = 1

            for _ in range(self.game.getActionSize()):
                if probs[_] > 0:
                    assert (valids[_] > 0)

            # debug
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time for get action prob: {:.2f} seconds".format(elapsed_time))

            return probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]

        for _ in range(self.game.getActionSize()):
            if probs[_] > 0:
                assert (valids[_] > 0)

        # debug
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time for get action prob: {:.2f} seconds".format(elapsed_time))

        return probs * valids

    def search(self, search_env, canonical_board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        obs, reward, termination, truncation, info = search_env.last()

        if reward != 0:
            if search_env.agent_selection == 'black_0' and reward == 1:
                return reward
            elif search_env.agent_selection == 'white_0' and reward == 1:
                return -reward
            elif search_env.agent_selection == 'black_0' and reward == -1:
                return reward
            else:
                return -reward

        s = self.game.stringRepresentation(canonical_board)

        # leaf node
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonical_board)

            valids = np.array(obs['action_mask'])

            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # re-normalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # not a leaf node
        valids = np.array(obs['action_mask'])
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a] != 0:
                if (s, a) in self.Qsa and self.Qsa[(s, a)] is not None:
                    # print("Values for s,a: Qsa: ", self.Qsa[(s,a)], "  Psa: ", self.Ps[s][a], "  Ns: ", self.Ns[s], "  Nsa: ", self.Nsa[(s,a)])
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    # print("Values for s,a: Psa: ", self.Ps[s][a], "  Ns: ", self.Ns[s])
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        assert (valids[a] != 0)

        try:
            search_env.step(a)
            obs, reward, termination, truncation, info = search_env.last()

            next_s = self.game.getBoard(obs, search_env.agent_selection)
        except:
            valids = np.array(obs['action_mask'])
            valids[-1] = 0
            if all(element == 0 for element in valids):
                valids[-1] = 1
            self.Vs[s] = valids
            cur_best = -float('inf')
            best_act = -1

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a] != 0:
                    if (s, a) in self.Qsa and self.Qsa[(s, a)] is not None:
                        u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                1 + self.Nsa[(s, a)])
                    else:
                        u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            try:
                search_env.step(a)
                obs, reward, termination, truncation, info = search_env.last()
                next_s = self.game.getBoard(obs, search_env.agent_selection)
            except:
                return -1e-4

        v = self.search(search_env, next_s)

        if (s, a) in self.Qsa:
            assert (valids[a] != 0)
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return -v
