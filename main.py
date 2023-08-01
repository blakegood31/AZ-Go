import os
import sys
from GoCoach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
import yaml

sys.setrecursionlimit(5000)

if __name__ == "__main__":

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    # create logs subdirectories
    if not os.path.exists(config["checkpoint_directory"]):
        os.makedirs(config["checkpoint_directory"])
    if not os.path.exists(config["game_history_directory"]):
        os.makedirs(config["game_history_directory"])
    if not os.path.exists(config["graph_directory"]):
        os.makedirs(config["graph_directory"])
    if not os.path.exists(config["train_logs_directory"]):
        os.makedirs(config["train_logs_directory"])

    game = Game(config["board_size"])
    neural_network = nn(game, config)

    if config["load_model"]:
        # if you are loading a checkpoint created from a model without DataParallel
        # use the load_checkpoint_from_plain_to_parallel() function
        # instead of the load_checkpoint() function
        neural_network.load_checkpoint(config["checkpoint_directory"], 'best.pth.tar')

    c = Coach(game, neural_network, config)

    if config["load_model"]:
        c.skipFirstSelfPlay = True

    c.learn()
