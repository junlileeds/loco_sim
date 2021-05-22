"""Example of whole body controller on A1 robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
from absl import logging

import time
import scipy.interpolate

import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from robots import a1
from robots import robot_config

from whole_body_controller import gait_generator as gait_generator_lib
from whole_body_controller import com_velocity_estimator
from whole_body_controller import openloop_gait_generator
from whole_body_controller import raibert_swing_leg_controller
from whole_body_controller import torque_stance_leg_controller
from whole_body_controller import locomotion_controller

flags.DEFINE_float("max_time_secs", 3., "maximum time to run the robot.")
FLAGS = flags.FLAGS


_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 m/s reduce this to 0.13s)


# # Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)

def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 1.6
    vy = 1.2
    wz = 0.8

    time_points = (0, 5, 10, 15, 20, 25, 30)
    # time_points = (0, 0.5, 1, 1.5, 2.0, 2.5, 3.0)
    speed_points = ((0,0,0,0), (0,0,0,wz), (vx,0,0,0), (0,0,0,-wz),
                    (0,-vy,0,0), (0,0,0,0), (0,0,0,wz))

    speed = scipy.interpolate.interp1d(time_points, 
                                       speed_points,
                                       kind="linear",
                                       fill_value="extrapolate",
                                       axis=0)(t)
    return speed[0:3], speed[3], False


def _setup_controller(robot):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0,0)
    desired_twisting_speed = 0

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE
    )

    window_size = 20  # don't use real robot
    state_estimator = com_velocity_estimator.COMVelocityEstimator(
        robot=robot, 
        window_size=window_size)
    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=robot.MPC_BODY_HEIGHT,   
        foot_clearance=0.01
    )
    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot.MPC_BODY_HEIGHT
    )
    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset
    )
    return controller

def _update_controller_params(controller, lin_speed, ang_speed):
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed

def main(argv):
    """Runs the locomotion controller example."""
    del argv

    # Step 1: Construct simulator
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Step 2: Construct robot class
    robot = a1.A1(p, 
                  motor_control_mode=robot_config.MotorControlMode.HYBRID, 
                  enable_action_interpolation=False, 
                  reset_time=2, 
                  time_step=0.002,
                  action_repeat=1)

    # Step 3: Consruct controller
    controller = _setup_controller(robot)
    controller.reset()

    # Step 4: Receive speed commands 
    command_function = _generate_example_linear_angular_speed

    start_time = robot.GetTimeSinceReset()
    current_time = start_time
    states, actions = [], []

    while current_time - start_time < FLAGS.max_time_secs:
        start_time_robot = current_time
        start_time_wall = time.time()
        # Step 5: Updates the controller behavior parameters
        lin_speed, ang_speed, e_stop = command_function(current_time)

        if e_stop:
            logging.info("E-stop kicked, exiting...")
            break

        _update_controller_params(controller, lin_speed, ang_speed)

        # Step 6: Update the controlller and get next action
        controller.update()
        hybrid_action, info = controller.get_action()

        # Step 7: Record states and actions
        states.append(
            dict(timestamp=robot.GetTimeSinceReset(), 
                 base_rpy=robot.GetBaseRollPitchYaw(), 
                 motor_angles=robot.GetMotorAngles(),
                 base_vel=robot.GetBaseVelocity(), 
                 base_vels_body_frame=controller.state_estimator.com_velocity_body_frame, 
                 base_rpy_rate=robot.GetBaseRollPitchYawRate(), 
                 motor_vels=robot.GetMotorVelocities(), 
                 contacts=robot.GetFootContacts(), 
                 qp_sol=info['qp_sol'])
        )
        actions.append(hybrid_action)

        # Step 8: Apply new action to robot
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()

        # Step 9: Add sleep time
        expected_duration = current_time - start_time_robot
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)


if __name__ == "__main__":
    app.run(main)