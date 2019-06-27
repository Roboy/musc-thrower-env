import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1, A2C, DDPG

env = gym.make('gym_musc_thrower:musc-thrower-v0')
# env = gym.make('gym_musc_thrower:biker-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# from stable_baselines.ddpg.policies import MlpPolicy
# model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log="./biking_tensorboard/")#, normalize_returns=True, reward_scale=10.0)

from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy, MlpPolicy, CnnPolicy
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./throwing_tensorboard_ppo/")#, normalize_returns=True, reward_scale=10.0)
model.learn(total_timesteps=1000000)
model.save("test")
print("learn done")
print()
obs = env.reset()
for i in range(1000):
    print("step %i"%i)
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    # env.render()
