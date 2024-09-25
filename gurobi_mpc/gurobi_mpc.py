
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
import numpy as np
import gurobipy as gp

class GurobiMPCNode(Node):
    def __init__(self):
        super().__init__('gurobi_mpc_node') 

        # MPC parameters
        self.horizon = 20
        self.dt = 0.1

        # State: [x, y, theta]
        # Input: [v, omega]

        # Weights for the cost function 
        self.Q = np.diag([1, 1, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.Qf = np.diag([1, 1, 0.1])

        # Box constraints
        self.x_min = 0
        self.x_max = 10
        self.y_min = 0
        self.y_max = 10
        self.theta_min = -np.pi
        self.theta_max = np.pi
        self.v_min = -1
        self.v_max = 1
        self.omega_min = -1
        self.omega_max = 1

        # current state and goal state
        self.current_state = np.zeros(3)
        self.goal_state = np.zeros(3)

        # Setup optimization problem
        self.setup_mpc()

        # ROS2 publishers and subscribers
        self.state_sub = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.current_state_callback,
            10)
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_state_callback,
            10)
        
        self.path_sub = self.create_subscription(
            Path,
            '/path',
            self.path_callback,
            10)

        self.control_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # Timer for continuous control in seconds
        self.control_timer = self.create_timer(0.1, self.control_callback)

        self.get_logger().info('GurobiMPCNode has been initialized')


    def setup_mpc(self):
        self.model = gp.Model("mpc")
        self.model.Params.OutputFlag = 0

        # Decision variables
        self.x = self.model.addVars(self.horizon + 1, lb=self.x_min, ub=self.x_max, name="x")
        self.y = self.model.addVars(self.horizon + 1, lb=self.y_min, ub=self.y_max, name="y")
        self.theta = self.model.addVars(self.horizon + 1, lb=self.theta_min, ub=self.theta_max, name="theta")
        self.v = self.model.addVars(self.horizon, lb=self.v_min, ub=self.v_max, name="v")
        self.omega = self.model.addVars(self.horizon, lb=self.omega_min, ub=self.omega_max, name="omega")

        # Objective function
        obj = 0
        for k in range(self.horizon):
            state_error = np.array([self.x[0, k] - self.goal_state[0],
                                    self.y[1, k] - self.goal_state[1],
                                    self.theta[2, k] - self.goal_state[2]])
            obj += state_error @ self.Q @ state_error + \
                   np.array([self.v[k], self.omega[k]]) @ self.R @ np.array([self.v[k], self.omega[k]])

        # Terminal cost
        state_error = np.array([self.x[0, self.horizon] - self.goal_state[0],
                                self.y[1, self.horizon] - self.goal_state[1],
                                self.theta[2, self.horizon] - self.goal_state[2]])
        obj += state_error @ self.Qf @ state_error

        self.model.setObjective(obj)

        # Constraints: Kinematic model
        for k in range(self.horizon):
            self.model.addConstr(
                self.x[0, k + 1] == self.x[0, k] + self.dt * self.v[k] * gp.cos(self.theta[2, k])
            )
            self.model.addConstr(
                self.y[1, k + 1] == self.y[1, k] + self.dt * self.v[k] * gp.sin(self.theta[2, k])
            )
            self.model.addConstr(
                self.theta[2, k + 1] == self.theta[2, k] + self.dt * self.omega[k]
            )

        # Initial state constraint
        self.model.addConstr(self.x[0, 0] == self.current_state[0])
        self.model.addConstr(self.y[1, 0] == self.current_state[1])
        self.model.addConstr(self.theta[2, 0] == self.current_state[2])

        self.model.update()

        # CasADi symbols
        # self.x = ca.SX.sym('x', 3)
        # self.u = ca.SX.sym('u', 2)

        # # Differential drive kinematics
        # x_dot = self.u[0] * ca.cos(self.x[2])
        # y_dot = self.u[0] * ca.sin(self.x[2])
        # theta_dot = self.u[1]

        # x_next = self.x + self.dt * ca.vertcat(x_dot, y_dot, theta_dot)
        # self.f = ca.Function('f', [self.x, self.u], [x_next])

        # # Optimization variables
        # self.opt_x = ca.SX.sym('opt_x', 3, self.horizon + 1)
        # self.opt_u = ca.SX.sym('opt_u', 2, self.horizon)

        # # Parameters
        # self.p = ca.SX.sym('p', 3)  # Initial state
        # self.ref = ca.SX.sym('ref', 3)  # Goal state

        # # Cost function
        # obj = 0
        # for k in range(self.horizon):
        #     state_error = self.opt_x[:, k] - self.ref
        #     obj += ca.mtimes([state_error.T, self.Q, state_error]) + \
        #            ca.mtimes([self.opt_u[:, k].T, self.R, self.opt_u[:, k]])
        
        # state_error = self.opt_x[:, self.horizon] - self.ref
        # obj += ca.mtimes([state_error.T, self.Qf, state_error])
        
        # # Define the constraints
        # g = []
        # lbg = []
        # ubg = []
        # lbx = []
        # ubx = []

        # # Constraints
        # g = []
        # for k in range(self.horizon):
        #     g.append(self.opt_x[:, k+1] - self.f(self.opt_x[:, k], self.opt_u[:, k]))
        
        # # Initial condition constraint
        # g.append(self.opt_x[:, 0] - self.p)

        # # NLP problem
        # nlp = {'x': ca.vertcat(ca.reshape(self.opt_x, -1, 1), ca.reshape(self.opt_u, -1, 1)),
        #        'f': obj,
        #        'g': ca.vertcat(*g),
        #        'p': ca.vertcat(self.p, self.ref)}

        # # Create solver
        # opts = {'ipopt.print_level': 0, 'print_time': 0}
        # self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def current_state_callback(self, msg):
        current_x = msg.poses[0].position.x 
        current_y = msg.poses[0].position.y
        _, _, current_yaw = self.euler_from_quaternion(msg.poses[0].orientation)

        self.current_state = np.array([current_x, current_y, current_yaw]) 
        self.get_logger().info(f'Received state update: x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f}')

    def goal_state_callback(self, msg):
        self.goal_pose = msg.pose
        self.is_goal_received = True

        goal_x = msg.poses[0].position.x
        goal_y = msg.poses[0].position.y
        _, _, goal_yaw = self.euler_from_quaternion(msg.poses[0].orientation)

        self.goal_state = np.array([goal_x, goal_y, goal_yaw])
        self.get_logger().info(f'Received goal update: x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f}')

    def path_callback(self, msg):
        self.path = msg
        self.is_path_received = True

        self.ref_path = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses])

    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr

        return q
    
    def control_callback(self):
        # Update initial state constraint
        self.model.getVarByName("x[0]").lb = self.current_state[0]
        self.model.getVarByName("x[0]").ub = self.current_state[0]
        self.model.getVarByName("y[0]").lb = self.current_state[1]
        self.model.getVarByName("y[0]").ub = self.current_state[1]
        self.model.getVarByName("theta[2,0]").lb = self.current_state[2]
        self.model.getVarByName("theta[2,0]").ub = self.current_state[2]

        # Solve the optimization problem
        self.model.optimize()

        # Get the optimal control inputs
        v_opt = self.v[0].X
        omega_opt = self.omega[0].X

        # Publish control input
        control_msg = Twist()
        control_msg.linear.x = float(v_opt)
        control_msg.angular.z = float(omega_opt)
        self.control_pub.publish(control_msg)

        self.get_logger().info(f'Published velocities: linear={v_opt:.2f}, angular={omega_opt:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = GurobiMPCNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# 
# ros2 run gurobi_mpc gurobi_mpc
# 