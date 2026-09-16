"""Posture costs sharing one terrain estimate per policy step."""


def terrain_hip_pose_l1(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    error = (robot.data.joint_pos[:, asset_cfg.joint_ids]
             - robot.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(-1)
    return error * env.v7_posture.update().hip_scale


def terrain_leg_pose_l1(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    error = (robot.data.joint_pos[:, asset_cfg.joint_ids]
             - robot.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(-1)
    return error * env.v7_posture.update().leg_scale


def terrain_roll_pitch_l2(env):
    return env.v7_posture.update().tilt_cost


def flat_wheel_position_l2(env):
    return env.v7_wheel_support.update().position_cost


def flat_wheelbase_l2(env):
    return env.v7_wheel_support.update().wheelbase_cost


def flat_track_width_l2(env):
    return env.v7_wheel_support.update().track_width_cost


def _stop_command_gate(env):
    commands = env.command_manager.get_command("base_velocity")
    cfg = env.scene["robot"].actuators["wheels"].cfg.stop_pi
    # This is a command condition, independent of measured drift and PI's
    # enabled flag. Pure lateral/yaw commands do not become stop commands.
    return ((commands[:, :2].norm(dim=-1) < cfg.enter_linear_mps)
            & (commands[:, 2].abs() < cfg.enter_yaw_rps)).float()


def stop_wheel_speed_l2(env, rim_speed_scale):
    from robot_lab.assets.x5 import X5_WHEEL_RADIUS
    actuator = env.scene["robot"].actuators["wheels"]
    speed = env.scene["robot"].data.joint_vel[:, actuator.joint_indices] * X5_WHEEL_RADIUS
    return _stop_command_gate(env) * (speed / rim_speed_scale).square().mean(-1)


def stop_wheel_target_l2(env, rim_speed_scale):
    from robot_lab.assets.x5 import X5_WHEEL_RADIUS
    # Penalize the policy request BEFORE the actuator overrides it to zero.
    # Raw actions/history retain their original 16D observation contract.
    requested = env.action_manager.get_term("wheel_vel").processed_actions * X5_WHEEL_RADIUS
    return _stop_command_gate(env) * (requested / rim_speed_scale).square().mean(-1)
