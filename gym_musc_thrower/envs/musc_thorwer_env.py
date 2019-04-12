import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rclpy
from std_srvs.srv import Empty, Trigger
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import GetLinkState
from roboy_middleware_msgs.msg import MotorCommand
import numpy as np


class MuscThrowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.node = rclpy.create_node("musc-thrower-env")
        self.reset_srv = self.node.create_client(Empty, "/gazebo/reset_simulation")
        self.com_srv = self.node.create_client(GetLinkState, "/gazebo/get_link_state")
        self.link_states_sub = self.node.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_cb)
        self.step_srv = self.node.create_client(Trigger, "/roboy/simulation/step")
        self.command_pub = self.node.create_publisher(MotorCommand, "/roboy/middleware/MotorCommand")

        self.motor_command = MotorCommand()
        self.motor_command.id = 3
        self.motor_command.motors = [0, 1, 2, 3]
        self.motor_command.set_points = [0, 0, 0, 0]

        self.curr_episode = -1
        self.action_space = spaces.Discrete(500)
        self.observation_space = spaces.Box(0, np.inf)

        self.ball_xyz = None
        self.upper_link_xyz = None

        self.ball_hit_ground = False
        self.ball_hit_location = None

    def step(self, action):
        ball_xy = self.ball_xyz[:2]
        ee_xy = self.upper_link_xyz[:2]

        if not self.ball_hit_ground and self.ball_xyz[2] <= 0.055:
            self.ball_hit_ground = True
            self.ball_hit_location = self.ball_xyz

        if self.ball_hit_ground:
            ball_hit_xy = self.ball_hit_location[:2]
            reward = np.linalg.norm(ball_hit_xy - ee_xy)
        else:
            reward = np.linalg.norm(ball_xy - ee_xy)

        self.do_simulation()

        observations = self.get_observations()
        done = False
        return observations, reward, done

    def do_simulation(self):
        self.command_pub.publish(self.motor_command)
        self.step_srv.call()

    def link_states_cb(self, msg):
        ball_idx = msg.name.index("musc-le-ball::ball")
        upper_link_idx = msg.name.index("musc-le-ball::upper_link")
        ball_com = msg.pose[ball_idx].position
        upper_link_com = msg.pose[upper_link_idx].position
        self.ball_xyz = np.array([ball_com.x, ball_com.y, ball_com.z])
        self.upper_link_xyz = np.array([upper_link_com.x, upper_link_com.y, upper_link_com.z])

    def get_body_com(self, link):
        self.com_srv.call(link_name=link, reference_frame="world")

    def reset(self):
        self.reset_srv.call()

    def render(self, mode='human'):
        pass

    def close(self):
        self.node.destroy_node()
