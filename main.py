import numpy as np
import pybullet as pyb  # Pybullet server
import time
import argparse
import curves
from multicontact_api import ContactSequence
from pyb_solo_simulator import pybullet_simulator, camera_follow, SimulatorLoop


DT = 0.002


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
    pyb_sim = pybullet_simulator(dt=DT, q_init=q_init, root_init = root_init)

    print("Press enter to start the motion")
    input()
    # Start the control loop:
    SimulatorLoop(pyb_sim, DT, q_t, dq_t)

    # compute final position error of the base
    root_final_desired = cs.contactPhases[-1].q_t(cs.contactPhases[-1].timeFinal)[:2]
    pyb_sim.retrieve_pyb_data()
    root_final = pyb_sim.baseState[0][:2]
    print("## Final position error of the base (x, y) :", root_final_desired - root_final)
