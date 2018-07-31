#! usr/bin/env python
"""
Gabriel Fernandez
github.com/gabriel80808

Behavioral Cloning with simple Feedforward NN applied to hexapod model

Usage Example:

python behavioralCloning.py Hex -r -run 5
"""
import tensorflow as tf
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import random
import sys

# Initialization for running
sys.path.append("envs")
from six_legged_env import SixLeggedEnv
ALREADY_INITIALIZED = set()


# Define cyclical motion for hexapod motor positions
class Cyc(object):
    def __init__(self, time_step=random.randint(1, 101), substep=1):
        expert = [[-0.12909389, 0.3230989, -0.3151643, 0.05632893, 0.31895516,
                   -0.28769827, 0.04774928, 0.31189033, -0.26120125, -0.12909389,
                   0.3230989, -0.3151643, 0.05632893, 0.31895516, -0.28769827,
                   0.04774928, 0.31189033, -0.26120125],

                  [-0.10478499, 0.38209646, -0.3385682, 0.0283867, 0.31950387,
                   -0.29026163, 0.06442661, 0.37072305, -0.30841235, -0.10478499,
                   0.32191046, -0.30392185, 0.0283867, 0.37657554, -0.3232424,
                   0.06442661, 0.31634586, -0.2768472],

                  [-0.08333333, 0.44055556, -0.35722222, -0.05632893, 0.31895516,
                   -0.28769827, 0.08333333, 0.44055556, -0.35722222, -0.04774928,
                   0.31189033, -0.26120125, 0., 0.44055556, -0.35722222,
                   0.12909389, 0.3230989, -0.3151643],

                  [-0.06442661, 0.37072305, -0.30841235, -0.08342262, 0.31798558,
                   -0.28344571, 0.10478499, 0.38209646, -0.3385682, -0.033006,
                   0.30633063, -0.24426278, -0.0283867, 0.37657554, -0.3232424,
                   0.15652619, 0.3233781, -0.32475787],

                  [-0.04774928, 0.31189033, -0.26120125, -0.10933571, 0.31652358,
                   -0.27753331, 0.12909389, 0.3230989, -0.3151643, -0.0199322,
                   0.29969919, -0.22609982, -0.05632893, 0.31895516, -0.28769827,
                   0.18723439, 0.32295912, -0.33263113],

                  [-0.06442661, 0.31634586, -0.2768472, -0.05632893, 0.37548604,
                   -0.32039295, 0.10478499, 0.32191046, -0.30392185, -0.04774928,
                   0.36350472, -0.29134229, -0.0283867, 0.31950387, -0.29026163,
                   0.12909389, 0.38618119, -0.35138642],

                  [-0.08333333, 0.31968091, -0.29111806, 0., 0.44055556,
                   -0.35722222, 0.08333333, 0.31968091, -0.29111806, -0.08333333,
                   0.44055556, -0.35722222, 0., 0.31968091, -0.29111806,
                   0.08333333, 0.44055556, -0.35722222],

                  [-0.10478499, 0.32191046, -0.30392185, 0.0283867, 0.37657554,
                   -0.3232424, 0.06442661, 0.31634586, -0.2768472, -0.10478499,
                   0.38209646, -0.3385682, 0.0283867, 0.31950387, -0.29026163,
                   0.06442661, 0.37072305, -0.30841235]]
        self.expert = expert
        self.time_step = time_step
        self.pos = time_step % 8
        self.current = expert[self.pos]
        self.next = expert[(self.pos+1) % 8]
        self.last = expert[self.pos - 1 if self.pos - 1 != -1 else 7]
        self.sub = substep

    # Update to next position
    def update_next(self):
        self.last = self.expert[self.pos]
        self.current = self.expert[(self.pos+1) % 8]
        self.next = self.expert[(self.pos+2) % 8]
        self.pos = (self.pos+1) % 8
        return None

    # Get move based on time_step
    def get_move_t(self, time_step):
        self.pos = time_step % 8
        self.time_step = time_step
        self.update_next()
        return self.expert[self.pos]

    # Get move based on current move if get next will update all positions
    def get_move_f(self, frame='next', step=1):
        if (step+1) % self.sub == 0:
            if frame == 'next':
                move = self.next
                self.update_next()
            elif frame == 'current':
                move = self.current
            elif frame == 'last':
                move = self.last
        else:
            move = self.current
        return move


    # Get current frame
    def get_frame(self):
        return self.current


# Loads args
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Enter a valid model: Hex')
    parser.add_argument('--render', '-r', action='store_true', help='Render')
    parser.add_argument('-run', type=int, default=20, help='Number of rollouts')
    parser.add_argument('-max_time_step', '-max', type=int, help='Max time step')
    parser.add_argument('-sub', type=int, default=1, help='Delay for hexapod to reach position')

    args = parser.parse_args()
    args.envname = args.model

    return args


# Collect observations from sim
def gen_data(args, m=None):
    with tf.Session():
        new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
        tf.get_default_session().run(tf.variables_initializer(new_variables))
        ALREADY_INITIALIZED.update(new_variables)

        env = SixLeggedEnv()
        max_steps = args.max_time_step or 200

        returns = []
        observations = []
        actions = []
        if m is not None:
            # Load model here, don't load outside session
            model_h5 = load_model('models/BC/' + args.envname + '_Master_' +
                                  str(args.run) + '_' + str(args.sub) + '.h5py')
        for i in range(args.run):
            print('gen iter', i+1)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            policy = Cyc(substep=args.sub)
            while not done:
                action = policy.get_move_f(frame='next', step=steps)
                actions.append(action)
                observations.append(obs)
                if m is not None:
                    model_action = (model_h5.predict(obs.reshape(1, len(obs)), batch_size=64, verbose=0))
                else:
                    model_action = action
                obs, r, done, _ = env.step(model_action)
                totalr += r
                steps += 1
                env.render()
                if args.render and m == 'final':
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps: break
            returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

    return np.array(np.squeeze(observations)), np.array(np.squeeze(actions))


# Simple Feedforward Neural Network
def simple_nn(obs_size, act_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(obs_size, )))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(act_size, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    args = parse()
    obs_data, act_data = gen_data(args)
    obs_train, obs_test, act_train, act_test = train_test_split(
        obs_data, act_data, test_size=0.01, random_state=245323)
    model = simple_nn(obs_data.shape[1], act_data.shape[1])
    model.fit(obs_train, act_train, batch_size=64, epochs=10, verbose=1)
    model.evaluate(obs_test, act_test, verbose=0)
    model.save('models/BC/'+args.envname+'_Master_'+str(args.run)+'_'+str(args.sub)+'.h5py')
    gen_data(args, 'final')
