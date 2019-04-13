import gym

from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1, A2C

env = gym.make('gym_musc_thrower:musc-thrower-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=2)
model.learn(total_timesteps=10000)

print()
obs = env.reset()
for i in range(1000):
    print("step %i"%i)
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    # env.render()
