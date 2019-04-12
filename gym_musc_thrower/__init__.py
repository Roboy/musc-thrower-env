from gym.envs.registration import register

register(
    id='musc-thrower-v0',
    entry_point='gym_musc_thrower.envs:MuscThrowerEnv',
)