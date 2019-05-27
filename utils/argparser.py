# -*- coding: utf-8 -*-
"""Commad-line argument parser module"""
import argparse
import enum
import copy
import yaml
import os


class ArgsEnum(enum.Enum):
    REQUIRED = 1


class Args(object):
    def __init__(self, description="", **arguments):
        self.args = dict()
        self.description = description

        for key, details in _DEFAULT_ARGS.items():
            details = copy.deepcopy(details)

            if key in arguments:
                details["default"] = arguments[key]

            # if default is ArgsEnum.REQUIRED:
            #     del details['default']
            #     details['required'] = True

            self.args[key] = details

    def parse(self):
        parser = argparse.ArgumentParser(description=self.description)
        for var, details in self.args.items():
            # set option_string to --<arg_name>
            parser.add_argument("--" + var, **details)
        return parser.parse_args()


class ArgsFromFile(object):
    configs_fname = "configs.yaml"

    def __init__(self):
        # TODO: fix required
        # 1. get the command-line arguments
        cmargs = Args().parse()

        # 2. load the default configs file
        self.load_configs(os.path.join(cmargs.configs_path, self.configs_fname))

        # 3. re-apply the command-line arguments
        for k, v in vars(cmargs).items():
            if v is not None:
                self.__dict__[k] = v

    def load_configs(self, filename):
        with open(filename, "r") as cfile:
            data = yaml.load(cfile)
        if not isinstance(data, dict):
            raise ValueError("The config file should represent a dictionary")
        self.__dict__.update(data)

    def store_configs(self, path):
        if path is not None:
            with open(os.path.join(path, self.configs_fname), "w") as cfile:
                yaml.dump(self.__dict__, cfile)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# The option_string is always gonna be --<arg_name>
# And everything is always assumed to be required if the default is not present
_DEFAULT_ARGS = {
    "name": {"help": "Name of the experiment."},
    "desc": {"help": "Description of the experiment."},
    # env params
    "env": {"help": "Name of the environment to use."},
    "c2d": {"type": str2bool, "help": "Wheter to discretize the action space or not."},
    "add_timestep": {
        "type": str2bool,
        "help": "Whether to add the time-step to the observation space",
    },
    # paths
    "save_path": {"type": str},
    "save_interval": {"type": int},
    "log_dir": {"type": str},
    "load_interval": {"type": int},
    "load_path": {"type": str, "help": "Just runs a previously saved model."},
    "configs_path": {
        "type": str,
        "default": ".",
        "help": "Path to the directory containing the `configs.yaml` file",
    },
    # model parameters
    "rnn": {"type": str2bool},
    "data_parallel": {"type": str2bool},
    "model_type": {"type": str},
    "num_layers": {"type": int},
    "hidden_size": {"type": int},
    # training parameters
    "num_steps": {"type": int},
    "num_total_frames": {"type": int},
    "num_processes": {"type": int},
    "lr": {"type": float},
    "pol_lr": {"type": float},
    "vf_lr": {"type": float},
    "lr_decay_gamma": {"type": float, "default": 0.992},
    "weight_decay": {"type": float},
    "decay_gamma": {"type": float},
    "gae_lambda": {"type": float},
    "use_peb": {"type": str2bool},
    # mixed pg
    "epoch_per_iter": {
        "type": int,
        "help": "Number of gradient updates to perform each time",
    },
    "batch_size": {
        "type": int,
        "help": "Size of the mini-batches used for optimization",
    },
    "num_samples": {
        "type": int,
        "help": "Number of samples to use per gradient update",
    },
    "moving_average_tau": {
        "type": float,
        "help": "Target smoothing coefficient (τ) used for the value network",
    },
    "log_stdev": {
        "type": float,
        "help": "The amount of fixed noise used in the policy",
    },
    "clip_eps": {
        "type": float,
        "help": "The epsilon value used to clip the gradients in PPO",
    },
    "mixing_ratio": {
        "type": float,
        "help": "how much DDPG style and how much PPO style gradient should be used:\n  mixing_ratio * L_ddpg + (1 - mixing_ratio) * L_ppo",
    },
    "l2_reg_w": {"type": float, "help": "L2 regularization weight"},
    "max_grad_norm": {
        "type": float,
        "help": "Gradient clipping: maximum gradient l2 norm",
    },
    # evaluation/render
    "render": {"type": str2bool, "help": "Renders the environment"},
    "store": {"type": str2bool},
    "record": {"type": str2bool},
    "plot": {"type": str2bool, "default": True},
    "evaluate": {"type": str2bool, "default": False},
    "eval_epis": {"type": int, "default": 50},
    "ep_length": {"type": int},
    "new_action_prob": {"type": float},
    "max_action": {"type": float},
    "store_video": {"type": str2bool},
    # misc
    "cuda": {"type": int, "help": "Whether to use Cuda for training or not"},
    "seed": {"type": int, "help": "The random seed: helps replication"},
}
