import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import rospy
# from std_msgs.msg import Int32, Bool, Float64
from std_srvs.srv import Empty, Trigger, EmptyRequest, TriggerRequest
# from gazebo_msgs.msg import LinkStates
# from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest, SpawnModel, SpawnModelRequest, DeleteModelRequest, DeleteModel, GetJointPropertiesRequest, GetJointProperties, GetJointPropertiesRequest
from roboy_middleware_msgs.msg import MotorCommand
from roboy_simulation_msgs.srv import GetJointVelocity, GetJointVelocityRequest
import numpy as np
import inspect, re


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)

class MuscEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, motor_count, step_gazebo):
        rospy.init_node("musc_gym_env", disable_signals=True)

        # self.spawn_srv = rospy.ServiceProxy(SpawnModel, '/gazebo/spawn_sdf_model')
        # self.delete_srv = rospy.ServiceProxy(DeleteModel, '/gazebo/delete_model')
        self.reset_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.com_srv = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.joint_state_srv = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)
        self.unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        # self.link_states_sub = self.node.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_cb)
        # self.joint_states_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_states_cb)
        self.step_srv = rospy.ServiceProxy("/roboy/simulation/step", Trigger)
        self.joint_vel_srv = rospy.ServiceProxy("/roboy/simulation/joint/velocity", GetJointVelocity)
        self.command_pub = rospy.Publisher("/roboy/middleware/MotorCommand", MotorCommand)

        self.motor_command = MotorCommand()
        self.motor_command.id = 3
        self.motor_command.motors = range(motor_count)
        self.motor_command.set_points = [0.0]*motor_count

        self.curr_episode = -1
        self.step_gazebo = step_gazebo

        map = {self.reset_srv: "/gazebo/reset_simulation",
            self.com_srv: "/gazebo/get_link_state",
            self.joint_state_srv: "/gazebo/get_joint_properties",
            self.unpause_srv: "/gazebo/unpause_physic",
            self.pause_srv: "/gazebo/pause_physics",
            self.step_srv: "/roboy/simulation/step",
            self.joint_vel_srv: "/roboy/simulation/joint/velocity"}
        for srv in [self.reset_srv, self.com_srv, self.unpause_srv, self.step_srv, self.joint_vel_srv, self.joint_state_srv]:
            rospy.loginfo('connecting to %s...'%map[srv])
            srv.wait_for_service()

        self.reset()

        print("init done")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def do_simulation(self, action):

        self.motor_command.set_points = [x*1.0 for x in action]
        self.command_pub.publish(self.motor_command)
        rospy.loginfo("Motor command: " + self.motor_command.set_points)

        if self.step_gazebo:
            future = self.step_srv(TriggerRequest())
            # #rclpy.spin_until_future_complete(self.node, future)

    def get_joint_vel(self, name):
        req = GetJointVelocityRequest()
        req.name = name
        future = self.joint_vel_srv(req)
        # #rclpy.spin_until_future_complete(self.node, future)
        return future.velocity

    def get_joint_pos(self, name):
        req = GetJointPropertiesRequest()
        req.joint_name = name
        future = self.joint_state_srv(req)
        # #rclpy.spin_until_future_complete(self.node, future)
        if future is not None:
            return future.position[0]

    def get_body_com(self, link, reference_frame="world"):
        req = GetLinkStateRequest(link_name=link, reference_frame=reference_frame)
        future = self.com_srv(req)
        #rclpy.spin_until_future_complete(self.node, future)

        if future is not None:
            pos = future.link_state.pose.position
            return np.array([pos.x, pos.y, pos.z])

    # def spawn_model(self):
    #     future = self.spawn_srv(self.spawn_request)
    #     #rclpy.spin_until_future_complete(self.node, future)
    #     if future is None:
    #         raise RuntimeError('exception while calling service: %r' % future.exception())
    #     self.model_exists = True
    #
    # def delete_model(self):
    #     future = self.delete_srv(self.delete_request)
    #     #rclpy.spin_until_future_complete(self.node, future)
    #     if future is None:
    #         raise RuntimeError('exception while calling service: %r' % future.exception())
    #     self.model_exists = False

    def reset(self):
        rospy.loginfo("Resetting simulation...")
        req = EmptyRequest()
        future = self.reset_srv(req)
        #rclpy.spin_until_future_complete(self.node, future)

        if not self.step_gazebo:
            req = EmptyRequest()
            future = self.unpause_srv(req)
            #rclpy.spin_until_future_complete(self.node, future)

        rospy.loginfo("Resetting simulation done")

    def render(self, mode='human'):
        pass

    def close(self):
        rospy.signal_shutdown("sutdown requested by user")
