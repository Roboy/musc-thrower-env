import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import rclpy
# from std_msgs.msg import Int32, Bool, Float64
from std_srvs.srv import Empty, Trigger
# from gazebo_msgs.msg import LinkStates
# from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetLinkState, SpawnModel, DeleteModel, GetJointProperties
from roboy_middleware_msgs.msg import MotorCommand
from roboy_simulation_msgs.srv import GetJointVelocity
import numpy as np


class MuscEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, motor_count, step_gazebo):
        rclpy.init()
        self.node = rclpy.create_node("musc_gym_env")
        # self.spawn_srv = self.node.create_client(SpawnModel, '/gazebo/spawn_sdf_model')
        # self.delete_srv = self.node.create_client(DeleteModel, '/gazebo/delete_model')
        self.reset_srv = self.node.create_client(Empty, "/gazebo/reset_simulation")
        self.com_srv = self.node.create_client(GetLinkState, "/gazebo/get_link_state")
        self.joint_state_srv = self.node.create_client(GetJointProperties, "/gazebo/get_joint_properties")
        self.unpause_srv = self.node.create_client(Empty, "/gazebo/unpause_physics")
        self.pause_srv = self.node.create_client(Empty, "/gazebo/pause_physics")
        # self.link_states_sub = self.node.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_cb)
        # self.joint_states_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_states_cb)
        self.step_srv = self.node.create_client(Trigger, "/roboy/simulation/step")
        self.joint_vel_srv = self.node.create_client(GetJointVelocity, "/roboy/simulation/joint/velocity")
        self.command_pub = self.node.create_publisher(MotorCommand, "/roboy/middleware/MotorCommand")

        self.motor_command = MotorCommand()
        self.motor_command.id = 3
        self.motor_command.motors = range(motor_count)
        self.motor_command.set_points = [0.0]*motor_count

        self.curr_episode = -1
        self.step_gazebo = step_gazebo

        for srv in [self.reset_srv, self.com_srv, self.unpause_srv, self.step_srv, self.joint_vel_srv, self.joint_state_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('%s service not available, waiting again...'%srv.srv_name)

        self.reset()

        print("init done")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def do_simulation(self, action):

        self.motor_command.set_points = [x*1.0 for x in action]
        self.command_pub.publish(self.motor_command)
        self.node.get_logger().info("Motor command: " + self.motor_command.set_points)

        if self.step_gazebo:
            future = self.step_srv.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self.node, future)

    def get_joint_vel(self, name):
        req = GetJointVelocity.Request()
        req.name = name
        future = self.joint_vel_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return future.result().velocity

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
        self.node.get_logger().info("Resetting simulation...")
        req = Empty.Request()
        future = self.reset_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        if not self.step_gazebo:
            req = Empty.Request()
            future = self.unpause_srv.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)

        self.node.get_logger().info("Resetting simulation done")

    def render(self, mode='human'):
        pass

    def close(self):
        self.node.destroy_node()
