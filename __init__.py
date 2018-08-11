from gym.envs.registration import register

register(
    id='Hex-v0',
    entry_point='env:SixLeggedEnv',
)
