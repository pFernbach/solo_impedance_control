import numpy as np
import pybullet as pyb  # Pybullet server
import time
import argparse
import curves
from multicontact_api import ContactSequence
from pyb_solo_simulator import pybullet_simulator
from loop import Loop

DT = 0.002


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


class SimulatorLoop(Loop):
    """
    Class used to call pybullet at a given frequency
    """
    def __init__(self, period, q_t, dq_t):
        """
        Constructor
        :param period: the time step between each new frame
        :param q_t: the joint position trajectory, stored in a Curves object
        :param dq_t: the joint velocity trajectory, stored in a Curves object
        """
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
        pyb_sim.retrieve_pyb_data()

        # update camera position to follow the root position
        camera_follow(pyb_sim.baseState[0])

        # Set control for all joints
        pyb.setJointMotorControlArray(pyb_sim.robotId,
                                      pyb_sim.revoluteJointIndices,
                                      controlMode=pyb.POSITION_CONTROL,
                                      targetPositions=self.q_t(self.t)[7:],
                                      targetVelocities=self.dq_t(self.t)[6:])

        # Compute one step of simulation
        pyb.stepSimulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start simulation for solo from a planned motion")
    parser.add_argument('filename', type=str,
                        help="the absolute path of a multicontact_api.ContactSequence serialized file")
    args = parser.parse_args()
    filename = args.filename

    #  Load contact sequence
    cs = ContactSequence(0)
    cs.loadFromBinary(filename)

    # extract trajectories from the file:
    q_t = cs.concatenateQtrajectories()  # with the freeflyer configuration
    dq_t = cs.concatenateDQtrajectories()  # with the freeflyer configuration
    ddq_t = cs.concatenateDDQtrajectories()  # with the freeflyer configuration
    tau_t = cs.concatenateTauTrajectories()  # joints torques
    # Get time interval from planning:
    t_min = q_t.min()
    t_max = q_t.max()
    print("## Complete duration of the motion loaded: ", t_max - t_min)

    # Sanity checks:
    assert t_min < t_max
    assert dq_t.min() == t_min
    assert ddq_t.min() == t_min
    assert tau_t.min() == t_min
    assert dq_t.max() == t_max
    assert ddq_t.max() == t_max
    assert tau_t.max() == t_max
    assert q_t.dim() == 19
    assert dq_t.dim() == 18
    assert ddq_t.dim() == 18
    assert tau_t.dim() == 12

    # Build the simulator wrapper class:
    q_init = q_t(t_min)[7:].reshape(-1, 1)  # without the freeflyer
    root_init = q_t(t_min)[:7]
    pyb_sim = pybullet_simulator(dt=DT, q_init=q_init)

    # Change camera position
    pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-30, cameraPitch=-35,
                                   cameraTargetPosition=[root_init[0], root_init[1], 0.1])

    print("Press enter to start the motion")
    input()
    # Start the control loop:
    SimulatorLoop(DT, q_t, dq_t)

    # compute final position error of the base
    root_final_desired = cs.contactPhases[-1].q_t(cs.contactPhases[-1].timeFinal)[:2]
    pyb_sim.retrieve_pyb_data()
    root_final = pyb_sim.baseState[0][:2]
    print("## Final position error of the base (x, y) :", root_final_desired - root_final)
