# import rclpy
# from std_srvs.srv import Empty, Trigger
#
# rclpy.init()
# node = rclpy.create_node('minimal_client')
# print(node.get_service_names_and_types())
#
# cli = node.create_client(Trigger, '/roboy/simulation/step')
# cli = node.create_client(Trigger, '/roboy/simulation/joint/detach')
# while not cli.wait_for_service(timeout_sec=1.0):
#     node.get_logger().info('service not available, waiting again...')
# future = cli.call_async(Trigger.Request())
# rclpy.spin_until_future_complete(node, future)
#
# node.destroy_node()
# rclpy.shutdown()

import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1, A2C, DDPG, ACER, ACKTR, DQN, GAIL, TRPO, SAC

env = gym.make('gym_musc_thrower:musc-thrower-v0')
# env = gym.make('gym_musc_thrower:biker-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# from stable_baselines.ddpg.policies import MlpPolicy
# model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log="./biking_tensorboard/")#, normalize_returns=True, reward_scale=10.0)

from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy, MlpPolicy, CnnPolicy
model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log="./throwing_tensorboard_ppo2/")#, normalize_returns=True, reward_scale=10.0)
# model.learn(total_timesteps=1000)
# model.save("test")
# print("learn done")
# print()

print("testing")
model.load(load_path='test.pkl')
obs = env.reset()
for i in range(100):
    print("step %i"%i)
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    # env.render()
