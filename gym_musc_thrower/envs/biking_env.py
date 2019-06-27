import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import rclpy
from std_msgs.msg import Int32, Bool, Float64
from std_srvs.srv import Empty, Trigger
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetLinkState, SpawnModel, DeleteModel, GetJointProperties
from roboy_middleware_msgs.msg import MotorCommand
from roboy_simulation_msgs.srv import GetJointVelocity
import numpy as np
import logging
import time
import pickle


class BikingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("biking_env")
        # self.spawn_srv = self.node.create_client(SpawnModel, '/gazebo/spawn_sdf_model')
        # self.delete_srv = self.node.create_client(DeleteModel, '/gazebo/delete_model')
        self.reset_srv = self.node.create_client(Empty, "/gazebo/reset_simulation")
        self.com_srv = self.node.create_client(GetLinkState, "/gazebo/get_link_state")
        self.joint_state_srv = self.node.create_client(GetJointProperties, "/gazebo/get_joint_properties")
        self.unpause_srv = self.node.create_client(Empty, "/gazebo/unpause_physics")
        self.pause_srv = self.node.create_client(Empty, "/gazebo/pause_physics")
        # self.link_states_sub = self.node.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_cb)
        # self.joint_states_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_states_cb)
        # self.ball_sub = self.node.create_subscription(Float64, "/roboy/simulation/ball/speed", self.ball_speed_cb)
        self.step_srv = self.node.create_client(Trigger, "/roboy/simulation/step")
        self.joint_vel_srv = self.node.create_client(GetJointVelocity, "/roboy/simulation/joint/velocity")
        # self.detach_srv = self.node.create_client(Trigger, "/roboy/simulation/joint/detach")
        # self.atach_srv = self.node.create_client(Trigger, "/roboy/simulation/joint/atach")
        self.command_pub = self.node.create_publisher(MotorCommand, "/roboy/middleware/MotorCommand")
        # self.step_pub = self.node.create_publisher(Int32, "/roboy/simulation/step")
        # self.detach_pub = self.node.create_publisher(Bool, "/roboy/simulation/joint/detach")

        # self.spawn_request = SpawnModel.Request()
        # self.spawn_request.model_name = "musc-le-ball"
        # f = open("/home/missxa/.gazebo/models/musc-le-ball/model.sdf")
        # self.spawn_request.model_xml = f.read()

        # self.delete_request = DeleteModel.Request()
        # self.delete_request.model_name = self.spawn_request.model_name
        # self.trajectory = pickle.load(open("/home/missxa/workspace/cardsflow_ws/src/CARDSflow/robots/biking_legs3/traj.p", "rb"))
        # self.trajectory = self.trajectory[5:100]
        self.motor_command = MotorCommand()
        self.motor_command.id = 3
        self.motor_command.motors = range(12)#[6,7,8,9,10,11]
        self.motor_command.set_points = [-2.50, -2.50, -2.50, -2.5, -2.50, -2.50, -2.50, -2.50, -2.50, -2.5, -2.50, -2.50]

        self.curr_episode = -1
        self.step_gazebo = True
        self.joint_pedal_vel = 0
        # self.model_exists = False
        # actions = [displacement1, displacement2, detach_ball]
        self.coef =2
        self.max = 1.0
        self.min = (-1)*self.max
        low = np.array([self.min]*len(self.motor_command.set_points))
        high = np.array([self.max]*len(self.motor_command.set_points))
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32) # spaces.MultiDiscrete([500, 500, 500, 500, 2])
        self.action_space = spaces.MultiDiscrete([self.coef]*len(self.motor_command.set_points))
        # self.action_space = spaces.Box(np.array([0]*5), np.array([500]*5))
        # observation_space: position, velocity
        obs_len = 6
        self.obs_coef = 5
        low = np.array([self.min*self.obs_coef]*obs_len)#, (-1)*np.inf, (-1)*np.inf, (-1)*np.inf])
        high = np.array([self.max*self.obs_coef]*obs_len)#, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # self.joint_pos = None
        # self.joint_vel = None

        # self.ball_xyz = self.prev_ball_xyz = None
        # self.upper_link_xyz = None
        #
        # self.ball_detached = False
        # self.ball_hit_ground = False
        # self.ball_hit_location = None
        for srv in [self.reset_srv, self.com_srv, self.unpause_srv, self.step_srv, self.joint_vel_srv, self.joint_state_srv]:
        # for srv in [self.unpause_srv, self.step_srv, self.atach_srv, self.detach_srv, self.com_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('%s service not available, waiting again...'%srv.srv_name)

        self.reset()
        # self.base_xyz = self.get_body_com("musc-le-ball::base")

        print("init done")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_episode += 1
        print("episode %i"%(self.curr_episode))

        self.do_simulation(action)

        # if self.curr_episode == len(self.trajectory) - 10:
        if abs(self.distance_driven) > 1.5 :
            done = True
        else:
            done = False

        if self.curr_episode == 0 or done:
            self.bike_xyz_init = self.get_body_com("biking_legs3::bike")
        self.bike_xyz = self.get_body_com("biking_legs3::bike")

        self.distance_driven = self.bike_xyz_init[0] - self.bike_xyz[0] #np.linalg.norm(self.bike_xyz_init - self.bike_xyz)

        self.prev_joint_pedal_vel = self.joint_pedal_vel
        self.joint_pedal_vel = self.get_joint_vel("joint_pedal")


        observations = self.get_observations()
        eps = 0.01
        punishment = 0
        if self.curr_episode != 0:
            for i in range(len(self.joints_pos)):
                if abs(self.joints_pos[i] - self.prev_joints_pos[i]) < eps:
                    punishment += 5
                else:
                    punishment -= 5

        # if self.joints_pos[2] < -1.8:
        #     punishment += 100

        # reward = 0
        # for i in range(len(self.joints_pos)):
        #     reward += 1/abs(self.joints_pos[i]-self.trajectory[self.curr_episode%len(self.trajectory)][i])

            # if abs(self.joints_pos[i]-self.trajectory[self.curr_episode%len(self.trajectory)][i]) < eps:
            #     reward += 1
            # else:
            #     reward -= 1

        # if self.joint_pedal_vel >= 0:
        #     punishment *= 10

        # reward = 1/reward
        reward =  (-1)*(self.joint_pedal_vel)*10  #- punishment*10 #- self.distance_driven*10 -
        # if (reward > 0 and self.joint_pedal_vel < 0 and self.prev_joint_pedal_vel < 0):
        #     reward *= 10

        # x = shin[0]
        # z = shin[2]
        # set_x = self.trajectory[self.curr_episode][0]
        # set_z = self.trajectory[self.curr_episode][1]
        # l = np.array([self.shin_left[0], 0, self.shin_left[2]])
        # l_s = np.array([set_x, 0, set_z])
        #
        # set_x = self.trajectory[self.curr_episode][2]
        # set_z = self.trajectory[self.curr_episode][3]
        # r = np.array([self.shin_right[0], 0, self.shin_right[2]])
        # r_s = np.array([set_x, 0, set_z])
        #
        # reward = np.linalg.norm(l-l_s)*10 + 10*np.linalg.norm(r - r_s)
        # reward = (x - set_x)**2 + (z - set_z)**2 + (x_r - set_x_r)**2 + (z_r - set_z_r)**2
        print("\nreward %f \n"%reward)
        return observations, reward, done, dict(reward=reward, episode=self.curr_episode)

    def do_simulation(self, action):

        print("Action: ")
        print(action)
        # import pdb; pdb.set_trace()
        print("motor command")
        self.motor_command.set_points = [x*400.0 for x in action]
        # self.motor_command.set_points = [(x-self.min)*self.coef for x in action]
        self.command_pub.publish(self.motor_command)
        print(self.motor_command.set_points)
        # if (not self.ball_detached and action[-1] > 0.9):
        #     # import pdb; pdb.set_trace()
        #     print("RELEASED")
        #     # msg = Bool()
        #     # msg.data = True
        #     # self.detach_pub.publish(msg)
        #     future = self.detach_srv.call_async(Trigger.Request())
        #     rclpy.spin_until_future_complete(self.node, future)
        #     self.ball_detached = True
        # msg = Int32()
        # msg.data = 100
        # self.step_pub.publish(msg)

        if self.step_gazebo:
            future = self.step_srv.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self.node, future)
        else:
            time.sleep(0.5)

    def get_observations(self):
        # bike_com = np.array(self.get_body_com("biking_legs3::bike"))
        # bike_vel = self.get_joint_vel("joint_pedal")
        # self.pedal_right = self.get_body_com("pedal_right", "kurbel")
        # self.pedal_left = self.get_body_com("pedal_left", "kurbel")
        self.joints_pos = []
        self.prev_joints_pos = self.joints_pos
        self.shin_left = self.get_body_com("biking_legs3::shin_left")
        self.shin_right = self.get_body_com("biking_legs3::shin_right")


        for j in ["joint_hip_right", "joint_knee_right", "joint_foot_right",
                    "joint_foot_left",  "joint_hip_left", "joint_knee_left"]:
            self.joints_pos.append(self.get_joint_pos(j))

        # obs = np.array(self.shin_left)
        # obs = np.append(obs, self.shin_right)
        # obs = np.append(obs, pedal_left)
        # obs = np.append(obs, np.array(self.joints_pos))
        obs = np.array(self.joints_pos)
        print("Observation: ")
        print(obs)
        return obs

    def get_joint_vel(self, name):
        req = GetJointVelocity.Request()
        req.name = name
        future = self.joint_vel_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return future.result().velocity


    def ball_speed_cb(self, msg):
        self.ball_speed = msg.data

    def link_states_cb(self, msg):
        print("updated link states")
        bike_idx = msg.name.index("biking_legs3::bike")
        if self.curr_episode == -1 or self.done == True:
            bike_com_0 = msg.pose[bike_idx].position
            self.bike_xyz_init = np.array([bike_com_0.x, bike_com_0.y, bike_com_0.z])
        bike_com = msg.pose[bike_idx].position
        self.bike_xyz = np.array([bike_com.x, bike_com.y, bike_com.z])

        print("bike pos: ")
        print(self.bike_xyz_init)
        print(self.bike_xyz)
        # ball_idx = msg.name.index("musc-le-ball::ball")
        # upper_link_idx = msg.name.index("musc-le-ball::upper_arm")
        # ball_com = msg.pose[ball_idx].position
        # upper_link_com = msg.pose[upper_link_idx].position
        # self.ball_xyz = np.array([ball_com.x, ball_com.y, ball_com.z])
        # self.upper_link_xyz = np.array([upper_link_com.x, upper_link_com.y, upper_link_com.z])

    def joint_states_cb(self, msg):
        self.joint_states = []
        for j in ["joint_hip_right", "joint_knee_right", "joint_foot_right",
                    "joint_foot_left",  "joint_hip_left", "joint_knee_left"]:
            idx = msg.name.index(j)
            self.joint_states.append(msg.position[idx])
        # print("updated joint states")
        # joint_pedal_idx = msg.name.index("joint_pedal")
        # self.joint_pedal_vel = msg.velocity[joint_pedal_idx]

        # upper_joint_idx = msg.name.index("upper_joint")
        # lower_joint_idx = msg.name.index("lower_joint")
        #
        # self.joint_pos = np.array([msg.position[lower_joint_idx], msg.position[upper_joint_idx]])
        # self.joint_vel = np.array([msg.velocity[lower_joint_idx], msg.velocity[upper_joint_idx]])

    def get_joint_pos(self, name):
        req = GetJointProperties.Request()
        req.joint_name = name
        future = self.joint_state_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result is not None:
            return future.result().position[0]

    def get_body_com(self, link, reference_frame="world"):
        req = GetLinkState.Request(link_name=link, reference_frame=reference_frame)
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
        req = Empty.Request()
        future = self.reset_srv.call_async(req)
        # self.ball_hit_ground = False
        # self.ball_hit_location = None
        # self.ball_xyz = self.prev_ball_xyz = None
        # self.upper_link_xyz = None
        # req = Trigger.Request()
        # future = self.atach_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        # import pdb; pdb.set_trace()
        # self.ball_detached = False
        #
        if not self.step_gazebo:
            req = Empty.Request()
            future = self.unpause_srv.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)

        # time.sleep(2)
        # future = self.pause_srv.call_async(req)
        # rclpy.spin_until_future_complete(self.node, future)
        self.distance_driven = 0.0
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
