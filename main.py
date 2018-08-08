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

from six_legged_env import SixLeggedEnv
from replay_buffer import ReplayBuffer
from schedules import LinearSchedule
from utils import ObservationInput
from graphs import build_train
import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
import argparse
import tf_util
import logger
import itertools


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

    with tf_util.make_session():
        env = SixLeggedEnv(args.legs)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        tf_util.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
