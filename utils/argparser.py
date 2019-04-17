# -*- coding: utf-8 -*-
'''Commad-line argument parser module'''
import argparse
import enum
import copy
import yaml

class ArgsEnum(enum.Enum):
    REQUIRED = 1


class Args(object):
    def __init__(self, description='', **arguments):
        self.args = dict()
        self.description = description

        for key, default in arguments.items():
            details = dict()
            if key in _DEFAULT_ARGS:
                details = copy.deepcopy(_DEFAULT_ARGS[key])

            details['default'] = default

            if default is ArgsEnum.REQUIRED:
                del details['default']
                details['required'] = True
    
            self.args[key] = details

    def parse(self):
        parser = argparse.ArgumentParser(
            description=self.description
        )
        for var, details in self.args.items():
            # set option_string to --<arg_name>
            parser.add_argument('--' + var, **details)
        return parser.parse_args()


class ArgsFromFile(Args):
    def __init__(self, filename):
        with open(filename, 'r') as cfile:
            data = yaml.load(cfile)
        super().__init__(**data)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# The option_string is always gonna be --<arg_name>
# And everything is always assumed to be required if the default is not present
_DEFAULT_ARGS = {
    # env params
    'env': {
        'default': 'pendulum',
        'help': 'Name of the environment to use.',
    },
    'c2d': {
        'type': str2bool,
        'help': 'Wheter to discretize the action space or not.',
    },
    'add_timestep': {
        'type': str2bool,
        'help': 'Whether to add the time-step to the observation space'
    },

    # paths
    'save_path': {
        'type': str,
    },
    'save_interval': {
        'type': int,
    },
    'load_path': {
        'type': str,
    },
    'load_interval': {
        'type': int,
    },
    'load_path': {
        'type': str,
        'help': 'Just runs a previously saved model.',
    },

    # model parameters
    'rnn': {
        'type': str2bool,
    },
    'data_parallel': {
        'type': str2bool,
    },
    'model_type': {
        'type': str,
    },
    'num_layers': {
        'type': int,
    },
    'hidden_size': {
        'type': int,
    },

    # training parameters
    'num_steps': {
        'type': int,
    },
    'num_total_frames': {
        'type': int,
    },
    'num_processes': {
        'type': int,
    },
    'lr': {
        'type': float,
    },
    'pol_lr': {
        'type': float,
    },
    'vf_lr': {
        'type': float,
    },
    'weight_decay': {
        'type': float,
    },
    'decay_gamma': {
        'type': float,
    },
    'gae_lambda': {
        'type': float,
    },
    'use_peb': {
        'type': str2bool,
    },

    # mixed pg
    'epoch_per_iter': {
        'type': int,
        'help': 'Number of gradient updates to perform each time'
    },
    'batch_size': {
        'type': int,
        'help': 'Size of the mini-batches used for optimization'
    },
    'num_samples': {
        'type': int,
        'help': 'Number of samples to use per gradient update',
    },
    'moving_average_tau': {
        'type': float,
        'default': 0.05,
        'help': 'Target smoothing coefficient (τ) used for the value network',
    },
    'log_stdev': {
        'type': float,
        'help': 'The amount of fixed noise used in the policy',
    },
    'clip_eps': {
        'type': float,
        'default': 0.2,
        'help': 'The epsilon value used to clip the gradients in PPO',
    },
    'mixing_ratio': {
        'type': float,
        'help': 'how much DDPG style and how much PPO style gradient should be used:\n  mixing_ratio * L_ddpg + (1 - mixing_ratio) * L_ppo',
    },
    'l2_reg_w': {
        'type': float,
        'help': 'L2 regularization weight',
    },
    'max_grad_norm': {
        'type': float,
        'help': 'Gradient clipping: maximum gradient l2 norm',
    },

    # evaluation/render
    'render': {
        'type': str2bool,
        'help': 'Renders the environment'
    },
    'store': {
        'type': str2bool,
    },
    'record': {
        'type': str2bool,
    },
    'eval_steps': {
        'type': int,
    },
    'ep_length': {
        'type': int,
    },
    'new_action_prob': {
        'type': float,
    },
    'max_action': {
        'type': float,
    },
    'store_video': {
        'type': str2bool,
    },

    # misc
    'cuda': {
        'type': str2bool,
        'help': 'Whether to use Cuda for training or not',
    },
    'seed': {
        'type': int,
        'help': 'The random seed: helps replication'
    },
}
