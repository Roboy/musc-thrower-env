from gym_musc_thrower.envs.musc_env import MuscEnv
import numpy as np
from gym import error, spaces, utils, logger
from gym.utils import seeding

import rospy
from std_srvs.srv import Empty, Trigger, EmptyRequest, TriggerRequest
from std_msgs.msg import Float64
import time

class MuscThrowerEnv(MuscEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.ball_sub = rospy.Subscriber("/roboy/simulation/ball/speed", Float64, self.ball_speed_cb)
        self.detach_srv = rospy.ServiceProxy("/roboy/simulation/joint/detach", Trigger)
        self.atach_srv = rospy.ServiceProxy("/roboy/simulation/joint/atach", Trigger)
        # self.step_pub = self.node.create_publisher(Int32, "/roboy/simulation/step")
        # self.detach_pub = self.node.create_publisher(Bool, "/roboy/simulation/joint/detach")

        # self.spawn_request = SpawnModel.Request()
        # self.spawn_request.model_name = "musc-le-ball"
        # f = open("/home/missxa/.gazebo/models/musc-le-ball/model.sdf")
        # self.spawn_request.model_xml = f.read()

        # self.delete_request = DeleteModel.Request()
        # self.delete_request.model_name = self.spawn_request.model_name

        low = np.array([-2.5, -2.50, -2.50, -2.50, -1])
        high = np.array([2.50, 2.50, 2.50, 2.50, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32) # spaces.MultiDiscrete([500, 500, 500, 500, 2])
        # self.action_space =spaces.MultiDiscrete([500, 500, 500, 500, 5])
        # self.action_space = spaces.Box(np.array([0]*5), np.array([500]*5))
        low = np.array([ -1.0472, -1.39626,-5.0, -5.0])# , (-1)*np.inf, (-1)*np.inf, (-1)*np.inf])
        high = np.array([1.0472, 1.39626,5.0, 5.0])#, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # self.joint_pos = None
        # self.joint_vel = None

        self.ball_xyz = self.prev_ball_xyz = None
        self.upper_link_xyz = None

        self.ball_detached = False
        self.ball_hit_ground = False
        self.ball_hit_location = None
        self.ball_speed = 0
        super().__init__(4, False)
        self.step_gazebo = False
        self.base_xyz = self.get_body_com("musc-le-ball::base")
        print("init done")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_episode += 1
        print("episode %i"%(self.curr_episode))

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
            reward = abs(ball_y)
            # reward = self.flying_speed
            done = True
        else:
            reward = 0.0
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


        # reward /= 2.0
        # if (abs(reward) < 0.001):
        #     reward = 0
        # else:
        #     reward += abs(ball_y)

        # reward += np.linalg.norm(self.prev_ball_xyz[:2] - self.ball_xyz[:2])
        # else:  # still attached
        #     print("still attached")
        #     reward = 0

        print("\n====== \n reward: %f \n======= \n"%reward)

        observations = self.get_observations()


        return observations, reward, done, dict(reward=reward, episode=self.curr_episode)

    def do_simulation(self, action):

        # time.sleep(1)
        # print("Action: ")
        # print(action)
        self.motor_command.set_points = [(x+2.5)*100.0 for x in action[:4]]
        self.command_pub.publish(self.motor_command)
        time.sleep(0.5)
        if (not self.ball_detached and action[-1] > 0.9):
            # import pdb; pdb.set_trace()
            print("RELEASED")
            # msg = Bool()
            # msg.data = True
            # self.detach_pub.publish(msg)
            future = self.detach_srv(TriggerRequest())
            #rclpy.spin_until_future_complete(self.node, future)
            self.ball_detached = True
            self.ball_xyz = self.get_body_com("musc-le-ball::ball")
            # import pdb; pdb.set_trace()
            self.flying_speed = 0
            i = 0.0
            while self.ball_xyz[2] >= 0.055:
                i += 1
                # import pdb; pdb.set_trace()
                # rospy.loginfo("waiting for the ball to land")
                time.sleep(0.01)
                self.step_srv()
                self.flying_speed += self.ball_speed
                self.ball_xyz = self.get_body_com("musc-le-ball::ball")
            self.flying_speed /= i
        # msg = Int32()
        # msg.data = 100
        # self.step_pub.publish(msg)
        if self.step_gazebo:
            future = self.step_srv(TriggerRequest())
        #rclpy.spin_until_future_complete(self.node, future)

    def get_observations(self):
        # msg = Int32()
        # msg.data = 1
        #
        # while (self.joint_pos is None):
        #     rclpy.spin_once(self.node)
        #     # self.step_pub.publish(msg)
        ball_xyz = self.get_body_com("musc-le-ball::ball")
        self.joint_pos = [self.get_joint_pos("lower_joint"), self.get_joint_pos("lower_joint")]
        self.joint_vel = [self.get_joint_vel("lower_joint"), self.get_joint_vel("lower_joint")]
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

    # def joint_states_cb(self, msg):
    #     # print("updated joint states")
    #     upper_joint_idx = msg.name.index("upper_joint")
    #     lower_joint_idx = msg.name.index("lower_joint")
    #
    #     self.joint_pos = np.array([msg.position[lower_joint_idx], msg.position[upper_joint_idx]])
    #     self.joint_vel = np.array([msg.velocity[lower_joint_idx], msg.velocity[upper_joint_idx]])
    #
    # def get_body_com(self, link):
    #     req = GetLinkState.Request(link_name=link, reference_frame="world")
    #     future = self.com_srv(req)
    #     #rclpy.spin_until_future_complete(self.node, future)
    #
    #     if future.result() is not None:
    #         pos = future.result().link_state.pose.position
    #         return np.array([pos.x, pos.y, pos.z])

    # def spawn_model(self):
    #     future = self.spawn_srv(self.spawn_request)
    #     #rclpy.spin_until_future_complete(self.node, future)
    #     if future.result() is None:
    #         raise RuntimeError('exception while calling service: %r' % future.exception())
    #     self.model_exists = True
    #
    # def delete_model(self):
    #     future = self.delete_srv(self.delete_request)
    #     #rclpy.spin_until_future_complete(self.node, future)
    #     if future.result() is None:
    #         raise RuntimeError('exception while calling service: %r' % future.exception())
    #     self.model_exists = False

    def reset(self):
        print("reset simulation")
        # future = self.reset_srv(Empty.Request())
        self.ball_hit_ground = False
        self.ball_hit_location = None
        self.ball_xyz = self.prev_ball_xyz = None
        self.upper_link_xyz = None

        #rclpy.spin_until_future_complete(self.node, future)
        # import pdb; pdb.set_trace()
        self.ball_detached = False
        rospy.loginfo("Resetting simulation...")
        req = EmptyRequest()
        future = self.reset_srv(req)

        req = TriggerRequest()
        future = self.atach_srv(req)
        #rclpy.spin_until_future_complete(self.node, future)

        req = EmptyRequest()
        if not self.step_gazebo:
            future = self.pause_srv(req)
        else:
            self.unpause_srv(req)

            #rclpy.spin_until_future_complete(self.node, future)

        #rclpy.spin_until_future_complete(self.node, future)
        #     #rclpy.spin_until_future_complete(self.node, future)

        # self.spawn_srv = rospy.ServiceProxy(SpawnModel, '/gazebo/spawn_sdf_model')
        # self.delete_srv = rospy.ServiceProxy(DeleteModel, '/gazebo/delete_model')
        # self.reset_srv = rospy.ServiceProxy(Empty, "/gazebo/reset_simulation")
        # self.com_srv = rospy.ServiceProxy(GetLinkState, "/gazebo/get_link_state")
        # self.link_states_sub = rospy.Subscriber(LinkStates, "/gazebo/link_states", self.link_states_cb)
        # self.joint_states_sub = rospy.Subscriber(JointState, "/joint_states", self.joint_states_cb)
        # self.step_srv = rospy.ServiceProxy(Trigger, "/roboy/simulation/step")
        # self.detach_srv = rospy.ServiceProxy(Trigger, "/roboy/simulation/joint/detach")
        # self.command_pub = self.node.create_publisher(MotorCommand, "/roboy/middleware/MotorCommand")
        # self.step_pub = self.node.create_publisher(Int32, "/roboy/simulation/step")
        #
        # self.detach_pub = self.node.create_publisher(Bool, "/roboy/simulation/joint/detach")
        # #rclpy.spin_until_future_complete(self.node, future)


        print("done")

    def render(self, mode='human'):
        pass

    def close(self):
        self.node.destroy_node()
