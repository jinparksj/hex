from gym.envs.registration import register

register(
    id='Hex1-v0',
    entry_point='envs.hex_env:SixLeggedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    nondeterministic=True,
    kwargs={'observation_dim': 137,
            'action_dim': 3},
)
