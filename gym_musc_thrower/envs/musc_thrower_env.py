import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import rclpy
from std_msgs.msg import Int32, Bool, Float64
from std_srvs.srv import Empty, Trigger
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetLinkState, SpawnModel, DeleteModel
from roboy_middleware_msgs.msg import MotorCommand
import numpy as np
import logging
import time

class MuscThrowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("musc_thrower_env")
        # self.spawn_srv = self.node.create_client(SpawnModel, '/gazebo/spawn_sdf_model')
        # self.delete_srv = self.node.create_client(DeleteModel, '/gazebo/delete_model')
        # self.reset_srv = self.node.create_client(Empty, "/gazebo/reset_simulation")
        self.com_srv = self.node.create_client(GetLinkState, "/gazebo/get_link_state")
        self.unpause_srv = self.node.create_client(Empty, "/gazebo/unpause_physics")
        # self.link_states_sub = self.node.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_cb)
        self.joint_states_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_states_cb)
        self.ball_sub = self.node.create_subscription(Float64, "/roboy/simulation/ball/speed", self.ball_speed_cb)
        self.step_srv = self.node.create_client(Trigger, "/roboy/simulation/step")
        self.detach_srv = self.node.create_client(Trigger, "/roboy/simulation/joint/detach")
        self.atach_srv = self.node.create_client(Trigger, "/roboy/simulation/joint/atach")
        self.command_pub = self.node.create_publisher(MotorCommand, "/roboy/middleware/MotorCommand")
        # self.step_pub = self.node.create_publisher(Int32, "/roboy/simulation/step")
        # self.detach_pub = self.node.create_publisher(Bool, "/roboy/simulation/joint/detach")

        # self.spawn_request = SpawnModel.Request()
        # self.spawn_request.model_name = "musc-le-ball"
        # f = open("/home/missxa/.gazebo/models/musc-le-ball/model.sdf")
        # self.spawn_request.model_xml = f.read()

        # self.delete_request = DeleteModel.Request()
        # self.delete_request.model_name = self.spawn_request.model_name

        self.motor_command = MotorCommand()
        self.motor_command.id = 3
        self.motor_command.motors = [0, 1, 2, 3]
        self.motor_command.set_points = [0.0, 0.0, 0.0, 0.0]

        self.curr_episode = -1
        # self.model_exists = False
        # actions = [displacement1, displacement2, detach_ball]
        low = np.array([-2.5, -2.50, -2.50, -2.50, -1])
        high = np.array([2.50, 2.50, 2.50, 2.50, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32) # spaces.MultiDiscrete([500, 500, 500, 500, 2])
        # self.action_space =spaces.MultiDiscrete([500, 500, 500, 500, 5])
        # self.action_space = spaces.Box(np.array([0]*5), np.array([500]*5))
        low = np.array([-1.0472, -1.39626, (-1)*np.inf, (-1)*np.inf])#, (-1)*np.inf, (-1)*np.inf, (-1)*np.inf])
        high = np.array([1.0472, 1.39626, np.inf, np.inf])#, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # self.joint_pos = None
        # self.joint_vel = None

        self.ball_xyz = self.prev_ball_xyz = None
        self.upper_link_xyz = None

        self.ball_detached = False
        self.ball_hit_ground = False
        self.ball_hit_location = None
        for srv in [self.unpause_srv, self.step_srv, self.atach_srv, self.detach_srv, self.com_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('%s service not available, waiting again...'%srv.srv_name)

        self.reset()
        self.base_xyz = self.get_body_com("musc-le-ball::base")

        print("init done")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_episode += 1
        # print("episode %i"%(self.curr_episode))

        self.do_simulation(action)
        done = False
        self.ball_hit_ground = False

        self.prev_ball_xyz = self.ball_xyz
        self.ball_xyz = self.get_body_com("musc-le-ball::ball")

        if (self.curr_episode == 0):
            self.prev_ball_xyz = self.ball_xyz


        ball_y = self.ball_xyz[1]


        if self.ball_xyz[2] <= 0.055:
            self.ball_hit_ground = True
            self.ball_hit_location = self.ball_xyz

        if self.ball_hit_ground:
            print("hit")
            done = True
        #     ball_hit_xy = self.ball_hit_location[:2]
        #     reward = np.linalg.norm(ball_hit_xy - ee_xy)
        # elif self.ball_detached:  # flying
            # print("flying")
            # print(ball_xy)
            # print(ee_xy)
            # print(np.linalg.norm(ball_xy - ee_xy))
        # else:
        # if self.ball_detached and
            # reward = 100*np.linalg.norm(ball_y - self.base_xyz[1])
        reward = self.ball_speed
        if (abs(self.ball_speed) < 0.001):
            reward = -1

        # reward += np.linalg.norm(self.prev_ball_xyz[:2] - self.ball_xyz[:2])
        # else:  # still attached
        #     print("still attached")
        #     reward = 0

        print("reward: %f"%reward)

        observations = self.get_observations()

        return observations, reward, done, dict(reward=reward, episode=self.curr_episode)

    def do_simulation(self, action):

        # print("Action: ")
        # print(action)
        self.motor_command.set_points = [(x+2.5)*100.0 for x in action[:4]]
        self.command_pub.publish(self.motor_command)
        if (not self.ball_detached and action[-1] > 0.9):
            # import pdb; pdb.set_trace()
            print("RELEASED")
            # msg = Bool()
            # msg.data = True
            # self.detach_pub.publish(msg)
            future = self.detach_srv.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self.node, future)
            self.ball_detached = True
        # msg = Int32()
        # msg.data = 100
        # self.step_pub.publish(msg)
        future = self.step_srv.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, future)

    def get_observations(self):
        # msg = Int32()
        # msg.data = 1
        #
        # while (self.joint_pos is None):
        #     rclpy.spin_once(self.node)
        #     # self.step_pub.publish(msg)
        ball_xyz = self.get_body_com("musc-le-ball::ball")
        obs = np.concatenate((self.joint_pos, self.joint_vel)) #np.append(np.concatenate((self.joint_pos, self.joint_vel)), ball_xyz)
        # print("Observation: ")
        # print(obs)
        return obs

    def ball_speed_cb(self, msg):
        self.ball_speed = msg.data

    def link_states_cb(self, msg):
        # print("updated link states")
        ball_idx = msg.name.index("musc-le-ball::ball")
        upper_link_idx = msg.name.index("musc-le-ball::upper_arm")
        ball_com = msg.pose[ball_idx].position
        upper_link_com = msg.pose[upper_link_idx].position
        self.ball_xyz = np.array([ball_com.x, ball_com.y, ball_com.z])
        self.upper_link_xyz = np.array([upper_link_com.x, upper_link_com.y, upper_link_com.z])

    def joint_states_cb(self, msg):
        # print("updated joint states")
        upper_joint_idx = msg.name.index("upper_joint")
        lower_joint_idx = msg.name.index("lower_joint")

        self.joint_pos = np.array([msg.position[lower_joint_idx], msg.position[upper_joint_idx]])
        self.joint_vel = np.array([msg.velocity[lower_joint_idx], msg.velocity[upper_joint_idx]])

    def get_body_com(self, link):
        req = GetLinkState.Request(link_name=link, reference_frame="world")
        future = self.com_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            pos = future.result().link_state.pose.position
            return np.array([pos.x, pos.y, pos.z])

    # def spawn_model(self):
    #     future = self.spawn_srv.call_async(self.spawn_request)
    #     rclpy.spin_until_future_complete(self.node, future)
    #     if future.result() is None:
    #         raise RuntimeError('exception while calling service: %r' % future.exception())
    #     self.model_exists = True
    #
    # def delete_model(self):
    #     future = self.delete_srv.call_async(self.delete_request)
    #     rclpy.spin_until_future_complete(self.node, future)
    #     if future.result() is None:
    #         raise RuntimeError('exception while calling service: %r' % future.exception())
    #     self.model_exists = False

    def reset(self):
        print("reset simulation")
        # future = self.reset_srv.call_async(Empty.Request())
        self.ball_hit_ground = False
        self.ball_hit_location = None
        self.ball_xyz = self.prev_ball_xyz = None
        self.upper_link_xyz = None
        req = Trigger.Request()
        future = self.atach_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        # import pdb; pdb.set_trace()
        self.ball_detached = False

        req = Empty.Request()
        future = self.unpause_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        #     rclpy.spin_until_future_complete(self.node, future)

        # self.spawn_srv = self.node.create_client(SpawnModel, '/gazebo/spawn_sdf_model')
        # self.delete_srv = self.node.create_client(DeleteModel, '/gazebo/delete_model')
        # self.reset_srv = self.node.create_client(Empty, "/gazebo/reset_simulation")
        # self.com_srv = self.node.create_client(GetLinkState, "/gazebo/get_link_state")
        # self.link_states_sub = self.node.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_cb)
        # self.joint_states_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_states_cb)
        # self.step_srv = self.node.create_client(Trigger, "/roboy/simulation/step")
        # self.detach_srv = self.node.create_client(Trigger, "/roboy/simulation/joint/detach")
        # self.command_pub = self.node.create_publisher(MotorCommand, "/roboy/middleware/MotorCommand")
        # self.step_pub = self.node.create_publisher(Int32, "/roboy/simulation/step")
        #
        # self.detach_pub = self.node.create_publisher(Bool, "/roboy/simulation/joint/detach")
        # rclpy.spin_until_future_complete(self.node, future)


        print("done")

    def render(self, mode='human'):
        pass

    def close(self):
        self.node.destroy_node()
