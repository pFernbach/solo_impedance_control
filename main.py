import numpy as np
import pybullet as pyb  # Pybullet server
import time
import argparse
import curves
from multicontact_api import ContactSequence
from pyb_solo_simulator import pybullet_simulator

DT = 0.002

def camera_follow(root):
    state = pyb.getDebugVisualizerCamera()
    current_root = state[-1]
    x = max(current_root[0], root[0])
    y = current_root[1]
    z = current_root[2]
    pyb.resetDebugVisualizerCamera(cameraDistance=state[-2], cameraYaw=state[-4], cameraPitch=state[-3],
                                   cameraTargetPosition=[x, y, z])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start simulation for solo from a planned motion")
    parser.add_argument('filename', type=str, help="the absolute path of a multicontact_api.ContactSequence serialized file")
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
    q_init = q_t(t_min)[7:].reshape(-1,1)  # without the freeflyer
    root_init = q_t(t_min)[:7]
    pyb_sim = pybullet_simulator(dt=DT, q_init=q_init)

    # Change camera position
    pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-30, cameraPitch=-35,
                                   cameraTargetPosition=[root_init[0], root_init[1], 0.1])

    t = t_min
    while t < t_max:
        # Get position/orientation of the base and angular position of actuators
        pyb_sim.retrieve_pyb_data()

        camera_follow(pyb_sim.baseState[0])

        # Vector that contains torques
        q = q_t(t)[7:]
        dq = dq_t(t)[6:]
        ddq = ddq_t(t)[6:]
        tau = tau_t(t)

        # Set control for all joints
        pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                      controlMode=pyb.POSITION_CONTROL, targetPositions = q, targetVelocities = dq)

        # Compute one step of simulation
        pyb.stepSimulation()
        #input()
        time.sleep(DT) # TODO: use LOOP helper class from mlp instead
        t += DT
