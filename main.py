#! usr/bin/env python

"""
Gabriel Fernandez
github.com/gabriel80808
Date: 31 July 2018

DPP: Dynamic Policy Programming
Model: Simplified Hexapod

Usage Example:
    python main.py --legs 1 --render -roll 20 -max_time_step 2000
"""

from misc.kernel import adaptive_isotropic_gaussian_kernel
from replay_buff.sql_replay_buffer import SimpleReplayBuffer
from q_v_funcs.value_functions import NNQFunction
from policies.policies import StochasticNNPolicy
from misc.sampler import SimpleSampler
from algos.sql import SQLAlgorithm
import tensorflow.contrib.layers as layers
import tensorflow as tf
import argparse
import envs  # Needed to init envs
import gym

SHARED_PARAMS = {
    'seed': [1, 2, 3],
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}
ENV_PARAMS = {
    1: {  # 3 DoF
        'prefix': '1L',
        'env_name': 'one-leg',
        'max_path_length': 1000,
        'n_epochs': 500,
        'reward_scale': 30,
    },
    2: {  # 6 DoF
        'prefix': '2L',
        'env_name': 'two-leg',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'reward_scale': 30,
    },
    3: {  # 9 DoF
        'prefix': '3L',
        'env_name': 'three-leg',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 30,
        'max_pool_size': 1E7,
    },
    4: {  # 12 DoF
        'prefix': '4L',
        'env_name': 'four-leg',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'reward_scale': 10,
    },
    5: {  # 15 DoF
        'prefix': '5L',
        'env_name': 'five-leg',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 300,
    },
    6: {  # 18 DoF
        'prefix': '6L',
        'env_name': 'six-leg',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': [1, 3, 10, 30, 100, 300]
    },
    11: {  # 6 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': '11L',
        'env_name': 'dual-leg',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    },
    22: {  # 12 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': '22L',
        'env_name': 'quad-leg',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    },
    33: {  # 9 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': '3L',
        'env_name': 'triple-leg',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    }
}
DEFAULT_ENV = 1
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse():
    """Pass in arguments form user for experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--legs', '-l', type=int, choices=AVAILABLE_ENVS,
        default=DEFAULT_ENV, help='No. of Legs')
    parser.add_argument('--render', '-r', action='store_true', help='Render')
    parser.add_argument('-roll', type=int, default=20, help='   Number of rollouts')
    parser.add_argument('--max_time_step', '-max', type=int, help='Max time step')
    args = parser.parse_args()
    return args


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    args = parse()
    env_params = ENV_PARAMS[args.legs]
    params = SHARED_PARAMS
    params.update(env_params)

    env = gym.make('Hex1-v0')
    policy = StochasticNNPolicy(env.spec, hidden_layer_sizes=(128, 128))
    qf = NNQFunction(env.spec, hidden_layer_sizes=(128, 128))
    pool = SimpleReplayBuffer(env.spec, max_replay_buffer_size=1E6)
    sampler = SimpleSampler(max_path_length = 1000,
                            min_pool_size = 1E6,
                            batch_size = 128)

    algo = SQLAlgorithm(
           env=env,
           policy=policy,
           qf=qf,
           pool=pool,
           sampler=sampler,
           n_epochs=10,  # 1000
           n_train_repeat=1,
           epoch_length=1000,  # 1000
           eval_n_episodes=10,
           eval_render=False,
           plotter=None,
           policy_lr=1E-3,
           qf_lr=1E-3,
           value_n_particles=16,
           td_target_update_interval=1,
           kernel_fn=adaptive_isotropic_gaussian_kernel,
           kernel_n_particles=16,
           kernel_update_ratio=0.5,
           discount=0.99,
           reward_scale=1,
           use_saved_qf=False,
           use_saved_policy=False,
           save_full_state=False,
           train_qf=True,
           train_policy=True
           )

    algo.train()
