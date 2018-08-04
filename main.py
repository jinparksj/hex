#! usr/bin/env python
"""
Gabriel Fernandez
github.com/gabriel80808
Date: 31 July 2018

DPP: Dynamic Policy Programming
Model: Simplified Hexapod

Usage Example:
    python main.py Hex -r -run 5
"""
from six_legged_env import SixLeggedEnv
import tensorflow as tf
import numpy as np
import argparse


ALREADY_INITIALIZED = set()


def parse():
    """Pass in arguments form user for experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--legs', '-l', type=int, default=1, help='No. Legs: 1, 2, 3, 4, 5, 6, 11, 22, 33')
    parser.add_argument('--render', '-r', action='store_true', help='Render')
    parser.add_argument('-roll', type=int, default=20, help='   Number of rollouts')
    parser.add_argument('-max_time_step', '-max', type=int, help='Max time step')
    args = parser.parse_args()
    return args


def gen_data(args, m=None):
    with tf.Session():
        new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
        tf.get_default_session().run(tf.variables_initializer(new_variables))
        ALREADY_INITIALIZED.update(new_variables)

        env = SixLeggedEnv(args.legs)
        max_steps = args.max_time_step or 200

        returns = []
        observations = []
        actions = []
        for i in range(args.roll):
            print('gen iter', i+1)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = env.action_space.sample()
                actions.append(action)
                observations.append(obs)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps: break
            returns.append(totalr)
    return np.array(np.squeeze(observations)), np.array(np.squeeze(actions))


if __name__ == '__main__':
    args = parse()
    obs_data, act_data = gen_data(args)
