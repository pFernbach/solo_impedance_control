import numpy as np

import pybullet as pyb  # Pybullet server
import pybullet_data

from example_robot_data.robots_loader import getModelPath
from loop import Loop

class pybullet_simulator:
    def __init__(self, dt=0.001, q_init=None, root_init = None, env_name="plane.urdf", urdf_name="solo12.urdf", force_control=False):

        self.ENV_NAME = env_name
        self.ROBOT_URDF_NAME = urdf_name
        self.URDF_SUBPATH = "/solo_description/robots"

        self.q_init = q_init
        if self.q_init is None:
            self.q_init = np.array([[0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()

        # Start the client for PyBullet
        physicsClient = pyb.connect(pyb.GUI)
        # p.GUI for graphical version
        # p.DIRECT for non-graphical version

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pyb.loadURDF(self.ENV_NAME)

        # Set the gravity
        pyb.setGravity(0, 0, -9.81)

        # Load Quadruped robot
        robotStartPos = [0, 0, 0.235 + 0.0045]
        robotStartOrientation = pyb.getQuaternionFromEuler([0.0, 0.0, 0.0])  # -np.pi/2
        pyb.setAdditionalSearchPath(getModelPath(self.URDF_SUBPATH) + self.URDF_SUBPATH)
        self.robotId = pyb.loadURDF(self.ROBOT_URDF_NAME, robotStartPos, robotStartOrientation)

        # Disable default motor control for revolute joints
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        pyb.setJointMotorControlArray(self.robotId,
                                      jointIndices=self.revoluteJointIndices,
                                      controlMode=pyb.VELOCITY_CONTROL,
                                      targetVelocities=[0.0 for m in self.revoluteJointIndices],
                                      forces=[0.0 for m in self.revoluteJointIndices])

        # Initialize the robot in a specific configuration
        pyb.resetJointStatesMultiDof(self.robotId, self.revoluteJointIndices, self.q_init)  # q0[7:])

        if force_control:
            # Enable torque control for revolute joints
            jointTorques = [0.0 for m in self.revoluteJointIndices]
            pyb.setJointMotorControlArray(self.robotId,
                                          self.revoluteJointIndices,
                                          controlMode=pyb.TORQUE_CONTROL,
                                          forces=jointTorques)

        # Set time step for the simulation
        pyb.setTimeStep(dt)

        # init members that will store state:
        self.jointStates = None
        self.baseState = None
        self.baseVel = None
        self.qmes12 = None
        self.vmes12 = None

        # Change camera position
        if root_init is not None:
            pyb.resetDebugVisualizerCamera(cameraDistance=0.8,
                                         cameraYaw=-30,
                                         cameraPitch=-35,
                                         cameraTargetPosition=[root_init[0], root_init[1], 0.1])

    def retrieve_pyb_data(self):
        """Retrieve the position and orientation of the base in world frame as well as its linear and angular velocities
        """

        # Retrieve data from the simulation
        self.jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)  # State of all joints
        self.baseState = pyb.getBasePositionAndOrientation(self.robotId)  # Position and orientation of the trunk
        self.baseVel = pyb.getBaseVelocity(self.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.qmes12 = np.vstack((np.array([self.baseState[0]]).T, np.array([self.baseState[1]]).T,
                                 np.array([[state[0] for state in self.jointStates]]).T))
        self.vmes12 = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[state[1] for state in self.jointStates]]).T))

        return 0


class SimulatorLoop(Loop):
    """
    Class used to call pybullet at a given frequency
    """
    def __init__(self, pyb_sim, period, q_t, dq_t):
        """
        Constructor
        :param pyb_sim: instance of pybullet_simulator class
        :param period: the time step between each new frame
        :param q_t: the joint position trajectory, stored in a Curves object
        :param dq_t: the joint velocity trajectory, stored in a Curves object
        """
        self.pyb_sim = pyb_sim
        self.q_t = q_t
        self.dq_t = dq_t
        self.t = q_t.min()
        self.t_max = q_t.max()
        super().__init__(period)

    def loop(self, signum, frame):
        self.t += self.period
        if self.t > self.t_max:
            self.stop()

        # Get position/orientation of the base and angular position of actuators
        self.pyb_sim.retrieve_pyb_data()

        # update camera position to follow the root position
        camera_follow(self.pyb_sim.baseState[0])

        # Set control for all joints
        pyb.setJointMotorControlArray(self.pyb_sim.robotId,
                                      self.pyb_sim.revoluteJointIndices,
                                      controlMode=pyb.POSITION_CONTROL,
                                      targetPositions=self.q_t(self.t)[7:],
                                      targetVelocities=self.dq_t(self.t)[6:])

        # Compute one step of simulation
        pyb.stepSimulation()




def camera_follow(root):
    state = pyb.getDebugVisualizerCamera()
    current_root = state[-1]
    x = max(current_root[0], root[0])
    y = current_root[1]
    z = current_root[2]
    pyb.resetDebugVisualizerCamera(cameraDistance=state[-2],
                                   cameraYaw=state[-4],
                                   cameraPitch=state[-3],
                                   cameraTargetPosition=[x, y, z])


