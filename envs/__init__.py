from gym.envs.registration import register

observation_dim = 137
action_dim = 3

register(
    id='Hex1-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim,
            'legs': 1},
)

register(
    id='Hex2-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*2,
            'legs': 2},
)

register(
    id='Hex3-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*3,
            'legs': 3},
)

register(
    id='Hex4-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*4,
            'legs': 4},
)

register(
    id='Hex5-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*5,
            'legs': 5},
)

register(
    id='Hex6-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*6,
            'legs': 6},
)

# Two legs that move but are on opposite sides
register(
    id='Hex11-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*2,
            'legs': 11},
)

# Four legs that move but pairs on opposite sides
register(
    id='Hex22-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*4,
            'legs': 22},
)

# Three legs that move separated by legs that don't
register(
    id='Hex33-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': observation_dim,
            'action_dim': action_dim*3,
            'legs': 33},
)
