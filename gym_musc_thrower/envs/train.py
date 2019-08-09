import gym

from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines import PPO2, PPO1, A2C, DDPG, ACER, ACKTR, DQN, GAIL, TRPO, SAC

env = gym.make('gym_musc_thrower:musc-thrower-v0')
# env = gym.make('gym_musc_thrower:biker-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
# env = VecCheckNan(env, raise_exception=True)
# from stable_baselines.ddpg.policies import MlpPolicy
# model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log="./biking_tensorboard/")#, normalize_returns=True, reward_scale=10.0)

from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy, MlpPolicy, CnnPolicy
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./throwing_tensorboard_ppo2/")#  vf_coef=1.0, ent_coef=0.01, n_steps=32, nminibatches=4, noptepochs=3, cliprange=0.3, gamma=0.8, lam=0.9) # normalize_returns=Truelearning_rate=0.00003,)
# try:
model.learn(total_timesteps=10000)
# except Exception as e:
#     print(str(e))
#     pass
model.save("test10000")
# print("learn done")
# # print()

# print("testing")
# model.load(load_path='test.pkl')
obs = env.reset()
for i in range(100):
    print("step %i"%i)
    # import pdb; pdb.set_trace()
    action = env.action_space.sample()
    # action, _states = model.predict(obs)
    print("train")
    print(action)
    obs, rewards, done, info = env.step(action)
    if done:
        env.reset()
    # env.render()
