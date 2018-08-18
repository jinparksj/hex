"""Example script for training from an existing Q-function and Policy"""
from schema.algos.sql.sql_kernel import adaptive_isotropic_gaussian_kernel
from schema.replay_buff.replay_buffer import SimpleReplayBuffer
from schema.algos.sql.sql_instrument import run_sql_experiment
from schema.qv_funcs.value_functions import NNQFunction
from schema.policies.policies import StochasticNNPolicy
from schema.utils.utils import timestamp, PROJECT_PATH
from schema.launch_exp.variant import VariantGenerator
from schema.envs.base.normalized_env import normalize
from schema.sampler.sampler import SimpleSampler
from schema.algos.sql.sql import SQLAlgorithm
from schema.envs.base.gym_env import GymEnv
import tensorflow as tf
import argparse
import joblib

SHARED_PARAMS = {
    'seed': [1, 2, 3],
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 2,  # 1000
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}
ENV_PARAMS = { # Envs for Hex see __init__ in envs dir
    1: {  # 3 DoF
        'prefix': '1l',
        'env_name': 'Hex1-v0',
        'max_path_length': 2,  # 1000
        'n_epochs': 2,  # 500
        'reward_scale': 30,
    },
    2: {  # 6 DoF
        'prefix': '2l',
        'env_name': 'Hex2-v0',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'reward_scale': 30,
    },
    3: {  # 9 DoF
        'prefix': '3l',
        'env_name': 'Hex3-v0',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 30,
        'max_pool_size': 1E7,
    },
    4: {  # 12 DoF
        'prefix': '4l',
        'env_name': 'Hex4-v0',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'reward_scale': 10,
    },
    5: {  # 15 DoF
        'prefix': '5l',
        'env_name': 'Hex5-v0',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 300,
    },
    6: {  # 18 DoF
        'prefix': '6l',
        'env_name': 'Hex6-v0',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': [1, 3, 10, 30, 100, 300]
    },
    11: {  # 6 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': '11l',
        'env_name': 'Hex11-v0',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    },
    22: {  # 12 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': '22l',
        'env_name': 'Hex22-v0',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    },
    33: {  # 9 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': '33l',
        'env_name': 'Hex33-v0',
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
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--env', type=int, choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV, help='No. of Legs')
    parser.add_argument('--exp_name', '-n', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params.update(env_params)
    params['file'] = args.file

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def run_experiment(variant):
    # TODO: shouldn't need to provide log_dir, bug
    env = normalize(GymEnv(variant['env_name'],
                           log_dir=PROJECT_PATH + "/data"))

    pool = SimpleReplayBuffer(env_spec=env.spec,
                              max_replay_buffer_size=variant['max_pool_size'])

    sampler = SimpleSampler(max_path_length=variant['max_path_length'],
                            min_pool_size=variant['max_path_length'],
                            batch_size=variant['batch_size'])

    with tf.Session().as_default():
        data = joblib.load(variant['file'])
        if 'algo' in data.keys():
            saved_qf = data['algo'].qf
            saved_policy = data['algo'].policy
        else:
            saved_qf = data['qf']
            saved_policy = data['policy']

        algorithm = SQLAlgorithm(epoch_length=variant['epoch_length'],
                                 n_epochs=variant['n_epochs'],
                                 n_train_repeat=variant['n_train_repeat'],
                                 eval_render=False,
                                 eval_n_episodes=1,
                                 sampler=sampler,
                                 env=env,
                                 pool=pool,
                                 qf=saved_qf,
                                 policy=saved_policy,
                                 kernel_fn=adaptive_isotropic_gaussian_kernel,
                                 kernel_n_particles=variant['kernel_particles'],
                                 kernel_update_ratio=variant['kernel_update_ratio'],
                                 value_n_particles=variant['value_n_particles'],
                                 td_target_update_interval=variant['td_target_update_interval'],
                                 qf_lr=variant['qf_lr'],
                                 policy_lr=variant['policy_lr'],
                                 discount=variant['discount'],
                                 reward_scale=variant['reward_scale'],
                                 use_saved_qf=True,
                                 use_saved_policy=True,
                                 save_full_state=False)

        algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        full_experiment_name = variant['prefix']
        full_experiment_name += '-' + args.exp_name + '-' + str(i).zfill(2)

        run_sql_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=full_experiment_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=True)


if __name__ == '__main__':
    args = parse()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator, args)
