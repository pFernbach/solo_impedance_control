import numpy as np

from pybullet_utils import bullet_client

from example_robot_data.robots_loader import getModelPath
from loop import Loop

SOLO8=True

class pybullet_simulator:
    def __init__(self, dt=0.001, q_init=None, root_init = None, env_name="plane.urdf", env_package = None, urdf_name="solo12.urdf", force_control=False, use_gui = True):
        
        import pybullet as opyb
        import pybullet_data        

        self.ENV_NAME = env_name
        self.ROBOT_URDF_NAME = urdf_name
        self.URDF_SUBPATH = "/solo_description/robots"
        self.use_gui = use_gui

        self.q_init = q_init
        if self.q_init is None:
            self.q_init = np.array([[0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()

        # Start the client for PyBullet
        self.pyb = bullet_client.BulletClient(opyb.GUI if use_gui else opyb.DIRECT)
        # p.GUI for graphical version
        # p.DIRECT for non-graphical version

        # Load horizontal plane
        if env_package is None:
            self.pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            from rospkg import RosPack
            rp = RosPack()
            package_path = rp.get_path(env_package) + "/urdf/"
            self.pyb.setAdditionalSearchPath(package_path)
            print("Environment path : ", package_path)
        print("Load environment urdf : ", self.ENV_NAME)
        self.planeId = self.pyb.loadURDF(self.ENV_NAME)

        # Set the gravity
        self.pyb.setGravity(0, 0, -9.81)

        # Load Quadruped robot
        robotStartPos = [0, 0, 0.235 + 0.0045]
        robotStartOrientation = self.pyb.getQuaternionFromEuler([0.0, 0.0, 0.0])  # -np.pi/2
        self.pyb.setAdditionalSearchPath(getModelPath(self.URDF_SUBPATH) + self.URDF_SUBPATH)
        self.robotId = self.pyb.loadURDF(self.ROBOT_URDF_NAME, robotStartPos, robotStartOrientation)

        # Disable default motor control for revolute joints
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.pyb.setJointMotorControlArray(self.robotId,
                                      jointIndices=self.revoluteJointIndices,
                                      controlMode=self.pyb.VELOCITY_CONTROL,
                                      targetVelocities=[0.0 for m in self.revoluteJointIndices],
                                      forces=[0.0 for m in self.revoluteJointIndices])

        # Initialize the robot in a specific configuration
        self.pyb.resetJointStatesMultiDof(self.robotId, self.revoluteJointIndices, self.q_init)  # q0[7:])

        if force_control:
            # Enable torque control for revolute joints
            jointTorques = [0.0 for m in self.revoluteJointIndices]
            self.pyb.setJointMotorControlArray(self.robotId,
                                          self.revoluteJointIndices,
                                          controlMode=self.pyb.TORQUE_CONTROL,
                                          forces=jointTorques)

        # Set time step for the simulation
        self.pyb.setTimeStep(dt)

        # init members that will store state:
        self.jointStates = None
        self.baseState = None
        self.baseVel = None
        self.qmes12 = None
        self.vmes12 = None

        # Change camera position
        if root_init is not None:
            self.pyb.resetDebugVisualizerCamera(cameraDistance=0.8,
                                         cameraYaw=-30,
                                         cameraPitch=-35,
                                         cameraTargetPosition=[root_init[0], root_init[1], 0.1])

    def retrieve_pyb_data(self):
        """Retrieve the position and orientation of the base in world frame as well as its linear and angular velocities
        """
        # Retrieve data from the simulation
        self.jointStates = self.pyb.getJointStates(self.robotId, self.revoluteJointIndices)  # State of all joints
        self.baseState = self.pyb.getBasePositionAndOrientation(self.robotId)  # Position and orientation of the trunk
        self.baseVel = self.pyb.getBaseVelocity(self.robotId)  # Velocity of the trunk


        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.qmes12 = np.vstack((np.array([self.baseState[0]]).T, np.array([self.baseState[1]]).T,
                                 np.array([[state[0] for state in self.jointStates]]).T))
        self.vmes12 = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[state[1] for state in self.jointStates]]).T))

        return 0


    def camera_follow(self, root):
        state = self.pyb.getDebugVisualizerCamera()
        current_root = state[-1]
        x = max(current_root[0], root[0])
        y = current_root[1]
        z = current_root[2]
        self.pyb.resetDebugVisualizerCamera(cameraDistance=state[-2],
                                       cameraYaw=state[-4],
                                       cameraPitch=state[-3],
                                       cameraTargetPosition=[x, y, z])




class VelocityControlLoop(Loop):
    """
    Class used to call pybullet at a given frequency
    """
    def __init__(self, pyb_sim, period, q_t, dq_t, display_func = None):
        """
        Constructor
        :param pyb_sim: instance of pybullet_simulator class
        :param period: the time step between each new frame
        :param q_t: the joint position trajectory, stored in a Curves object
        :param dq_t: the joint velocity trajectory, stored in a Curves object
        :param display_func: function pointer, if provided this function is called
        with the wholebody configuration (including free-flyer) after each simulation step
        """
        self.pyb_sim = pyb_sim
        self.display_func = display_func
        self.q_t = q_t
        self.dq_t = dq_t
        self.t = q_t.min()
        self.t_max = q_t.max()
        super().__init__(period)

    def loop(self, signum, frame):
        self.t += self.period
        if self.t > self.t_max:
            self.stop()
        # Set control for all joints
        self.pyb_sim.pyb.setJointMotorControlArray(self.pyb_sim.robotId,
                                      self.pyb_sim.revoluteJointIndices,
                                      controlMode=self.pyb_sim.pyb.POSITION_CONTROL,
                                      targetPositions=self.q_t(self.t)[7:],
                                      targetVelocities=self.dq_t(self.t)[6:])
        if SOLO8:
            for i in range(4):
                self.pyb_sim.pyb.setJointMotorControl2(self.pyb_sim.robotId,
                                          self.pyb_sim.revoluteJointIndices[i*3],
                                          controlMode=self.pyb_sim.pyb.VELOCITY_CONTROL,
                                          targetVelocity=0.)  
        # Compute one step of simulation
        self.pyb_sim.pyb.stepSimulation()
        
        # Get position/orientation of the base and angular position of actuators
        self.pyb_sim.retrieve_pyb_data()
        # update camera position to follow the root position
        if self.pyb_sim.use_gui:
            self.pyb_sim.camera_follow(self.pyb_sim.baseState[0])

        # display the motion with display_func if required:
        if self.display_func is not None and self.pyb_sim.qmes12 is not None:
            self.display_func(self.pyb_sim.qmes12.reshape(-1,1))




class TorqueControlLoop(Loop):
    """
    Class used to call pybullet at a given frequency
    """
    def __init__(self, pyb_sim, period, q_t, dq_t, tau_t, display_func = None):
        """
        Constructor
        :param pyb_sim: instance of pybullet_simulator class
        :param period: the time step between each new frame
        :param q_t: the joint position trajectory, stored in a Curves object
        :param dq_t: the joint velocity trajectory, stored in a Curves object
        :param display_func: function pointer, if provided this function is called
        with the wholebody configuration (including free-flyer) after each simulation step
        """
        self.pyb_sim = pyb_sim
        self.display_func = display_func
        self.q_t = q_t
        self.dq_t = dq_t
        self.tau_t = tau_t
        self.t = q_t.min()
        self.t_max = q_t.max()
        self.kd = np.ones([12,1]) * 0.3
        for i in range(4):
            self.kd[i*3] = 1.
        self.kp = np.ones([12,1]) * 3.
        super().__init__(period)

    def loop(self, signum, frame):
        self.t += self.period
        if self.t > self.t_max:
            self.stop()
        # get current joint positions and velocities:
        self.pyb_sim.retrieve_pyb_data()
        q = self.pyb_sim.qmes12[7:]
        v = self.pyb_sim.vmes12[6:]
        #print("q = ", q)
        #print("size q : ", q.shape)
        #print("size v : ", v.shape)
        # compute error from with the reference:
        desired_q = self.q_t(self.t)[7:].reshape(-1,1)
        desired_v = self.dq_t(self.t)[6:].reshape(-1,1)
        desired_tau = self.tau_t(self.t).reshape(-1,1)
        #print("desired size q : ", desired_q.shape)
        #print("desired size v : ", desired_v.shape)
        #print("desired size tau : ", desired_tau.shape)
        pos_error = desired_q - q
        vel_error = desired_v - v
        #print("pos error shape : ", pos_error.shape)
        #print("pos error : ", pos_error)
        #print("vel error : ", vel_error)
        tau = desired_tau + self.kp * pos_error + self.kd * vel_error
        if  np.linalg.norm(tau) > 10:
            print("large torque !!! :", tau)
            print("pos error : ", pos_error)
            print("vel error : ", vel_error)
            print("desired torque : ", desired_tau)
            self.stop()
        #tau = desired_tau
        #print("command size tau : ", tau.shape)
        # Set control for all joints
        self.pyb_sim.pyb.setJointMotorControlArray(self.pyb_sim.robotId,
                                      self.pyb_sim.revoluteJointIndices,
                                      controlMode=self.pyb_sim.pyb.TORQUE_CONTROL,
                                      forces=tau)
        if SOLO8:
            for i in range(4):
                self.pyb_sim.pyb.setJointMotorControl2(self.pyb_sim.robotId,
                                          self.pyb_sim.revoluteJointIndices[i*3],
                                          controlMode=self.pyb_sim.pyb.VELOCITY_CONTROL,
                                          targetVelocity=0.)  
        # Compute one step of simulation
        self.pyb_sim.pyb.stepSimulation()
        
        # update camera position to follow the root position
        if self.pyb_sim.use_gui:
            self.pyb_sim.camera_follow(self.pyb_sim.baseState[0])

        # display the motion with display_func if required:
        if self.display_func is not None and self.pyb_sim.qmes12 is not None:
            self.display_func(self.pyb_sim.qmes12.reshape(-1,1))



