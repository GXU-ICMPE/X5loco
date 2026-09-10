"""CPU value tests for the independent X5 reward experiment; no simulation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/x5_rewards_v1/rewards.py"
spec = importlib.util.spec_from_file_location("x5_rewards_v1_cpu", REWARDS_PATH)
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


@pytest.fixture
def wheel_case():
    """Use different joint/body index orders, as actual scene resolution does."""
    joint_ids = [1, 3, 5, 7]
    body_ids = [6, 0, 4, 2]
    robot = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.full((1, 9), 99.0)))
    robot.data.joint_vel[:, joint_ids] = 0.0
    sensor = SimpleNamespace(data=SimpleNamespace(
        current_air_time=torch.zeros(1, 8),
        net_forces_w_history=torch.full((1, 3, 8, 3), 100.0),
    ))
    sensor.data.current_air_time[:, body_ids] = 0.2
    sensor.data.net_forces_w_history[:, :, body_ids, :] = 0.0
    env = SimpleNamespace(
        scene={"robot": robot, "contacts": sensor},
        num_envs=1, device="cpu", step_dt=0.02,
        episode_length_buf=torch.tensor([100]), common_step_counter=100,
    )
    params = dict(
        asset_cfg=SimpleNamespace(name="robot", joint_ids=joint_ids),
        sensor_cfg=SimpleNamespace(name="contacts", body_ids=body_ids),
        wheel_radius=0.1, speed_deadband=0.5,
        air_time_grace=0.1, contact_threshold=5.0, reset_grace=0.3,
    )
    return env, params


def test_pose_deadband_allows_small_terrain_adjustments_and_ignores_unselected_joints():
    default = torch.tensor([[10.0, 0.75, 10.0, -1.5, 10.0, 0.0]])
    offsets = torch.tensor([[100.0, 0.05, 100.0, -0.1, 100.0, 0.0]])
    asset = SimpleNamespace(data=SimpleNamespace(
        joint_pos=default + offsets, default_joint_pos=default,
    ))
    env = SimpleNamespace(scene={"robot": asset})
    cfg = SimpleNamespace(name="robot", joint_ids=[1, 3, 5])
    actual = rewards.joint_pose_deadband_l1(env, cfg, deadband=0.1)
    torch.testing.assert_close(actual, torch.zeros(1), atol=1e-6, rtol=0)


def test_pose_deadband_counts_only_excess_and_is_sign_symmetric():
    default = torch.zeros(2, 3)
    asset = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.tensor([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]]),
        default_joint_pos=default,
    ))
    env = SimpleNamespace(scene={"robot": asset})
    cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2])
    actual = rewards.joint_pose_deadband_l1(env, cfg, deadband=0.1)
    torch.testing.assert_close(actual, torch.tensor([0.3, 0.3]))


def test_air_wheel_penalty_is_mean_over_all_four_wheels(wheel_case):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = torch.tensor([[10., 0., 0., 0.]])
    # One airborne wheel at 1 m/s has 0.5 m/s excess: 0.5^2 / 4.
    actual = rewards.wheel_air_spin_l2(env, **params)
    torch.testing.assert_close(actual, torch.tensor([0.0625]))


def test_air_wheel_penalty_respects_physical_units_and_rotation_sign(wheel_case):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = torch.tensor([[10., -10., 20., -20.]])
    actual = rewards.wheel_air_spin_l2(env, **params)
    # Rim speeds [1, 1, 2, 2], squared excesses [0.25, 0.25, 2.25, 2.25].
    torch.testing.assert_close(actual, torch.tensor([1.25]))
    env.scene["robot"].data.joint_vel *= 0.5
    params["wheel_radius"] *= 2
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), actual)


@pytest.mark.parametrize("rim_speed", [0.0, 0.2, 0.5])
def test_air_wheel_deadband_allows_slow_prespins(wheel_case, rim_speed):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = rim_speed / params["wheel_radius"]
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), torch.zeros(1))


@pytest.mark.parametrize("air_time", [0.0, 0.05, 0.1])
def test_air_wheel_grace_does_not_penalize_short_liftoffs(wheel_case, air_time):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = 20.0
    env.scene["contacts"].data.current_air_time[:, params["sensor_cfg"].body_ids] = air_time
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), torch.zeros(1))


def test_air_wheel_contact_history_uses_vector_norm_and_correct_body_order(wheel_case):
    env, params = wheel_case
    joint_ids, body_ids = params["asset_cfg"].joint_ids, params["sensor_cfg"].body_ids
    env.scene["robot"].data.joint_vel[:, joint_ids] = torch.tensor([[20., 10., 10., 10.]])
    # Contact in an older sample must exempt wheel zero even with stale air time;
    # the 6 N force is horizontal, as a wheel can touch the side of a step.
    env.scene["contacts"].data.net_forces_w_history[0, 2, body_ids[0], 0] = 6.0
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), torch.tensor([0.1875]))


def test_air_wheel_rolling_contacts_have_no_penalty(wheel_case):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = 24.0
    env.scene["contacts"].data.net_forces_w_history[:, 0, params["sensor_cfg"].body_ids, 2] = 300.0
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), torch.zeros(1))


def test_air_wheel_result_is_invariant_to_consistent_wheel_permutation(wheel_case):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = torch.tensor([[10., 12., 15., 20.]])
    env.scene["contacts"].data.current_air_time[0, params["sensor_cfg"].body_ids[2]] = 0.02
    before = rewards.wheel_air_spin_l2(env, **params)
    order = [2, 0, 3, 1]
    params["asset_cfg"].joint_ids = [params["asset_cfg"].joint_ids[i] for i in order]
    params["sensor_cfg"].body_ids = [params["sensor_cfg"].body_ids[i] for i in order]
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), before)


def test_air_wheel_reset_grace_is_evaluated_per_environment(wheel_case):
    env, params = wheel_case
    env.num_envs = 2
    env.scene["robot"].data.joint_vel = env.scene["robot"].data.joint_vel.repeat(2, 1)
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = 10.0
    data = env.scene["contacts"].data
    data.current_air_time = data.current_air_time.repeat(2, 1)
    data.net_forces_w_history = data.net_forces_w_history.repeat(2, 1, 1, 1)
    env.episode_length_buf = torch.tensor([0, 100])
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), torch.tensor([0.0, 0.25]))


@pytest.mark.parametrize("global_step,expected", [(0, 0.0), (24, 0.0), (25, 0.25)])
def test_air_wheel_startup_grace_cannot_be_bypassed_by_randomized_episode_lengths(wheel_case, global_step, expected):
    env, params = wheel_case
    env.scene["robot"].data.joint_vel[:, params["asset_cfg"].joint_ids] = 10.0
    # RSL may randomize the episode counter before the first rollout. A large
    # counter must not make a just-started simulator look settled already.
    env.episode_length_buf.fill_(10000)
    env.common_step_counter = global_step
    params["reset_grace"] = 0.5
    torch.testing.assert_close(rewards.wheel_air_spin_l2(env, **params), torch.tensor([expected]))


def gate_update(gate, commands, step):
    # Binary-exact times make the intended boundary independent of rounding.
    return gate.update(
        commands, step=step, dt=0.125,
        linear_threshold=0.1, yaw_threshold=0.2, settle_time=0.375,
    ).clone()


def test_quiet_gate_requires_both_translation_and_yaw_to_be_quiet():
    commands = torch.tensor([
        [0., 0., 0.], [0., 0., 0.3], [0., 0.2, 0.],
        [-0.2, 0., 0.], [0.09, 0.09, 0.], [0.02, 0., -0.05],
    ])
    gate = rewards.QuietCommandGate(len(commands), "cpu")
    assert not gate_update(gate, commands, 0).any()
    assert not gate_update(gate, commands, 1).any()
    actual = gate_update(gate, commands, 2)
    assert actual.dtype == torch.bool
    assert actual.tolist() == [True, False, False, False, False, True]


def test_quiet_gate_duplicate_reads_do_not_advance_settling_time():
    commands = torch.zeros(1, 3)
    gate = rewards.QuietCommandGate(1, "cpu")
    for _ in range(6):
        assert not gate_update(gate, commands, 0).item()
    assert not gate_update(gate, commands, 1).item()
    assert gate_update(gate, commands, 2).item()


def test_quiet_gate_same_step_motion_switch_immediately_cancels_readiness():
    quiet, moving = torch.zeros(1, 3), torch.tensor([[0., 0., 0.3]])
    gate = rewards.QuietCommandGate(1, "cpu")
    for step in range(3):
        gate_update(gate, quiet, step)
    assert gate_update(gate, quiet, 2).item()
    assert not gate_update(gate, moving, 2).item()
    assert not gate_update(gate, quiet, 2).item()
    assert not gate_update(gate, quiet, 3).item()
    assert not gate_update(gate, quiet, 4).item()
    assert gate_update(gate, quiet, 5).item()


def test_quiet_gate_partial_reset_preserves_other_environments():
    commands = torch.zeros(2, 3)
    gate = rewards.QuietCommandGate(2, "cpu")
    for step in range(3):
        gate_update(gate, commands, step)
    gate.reset(torch.tensor([0]))
    assert gate_update(gate, commands, 3).tolist() == [False, True]
    assert gate_update(gate, commands, 4).tolist() == [False, True]
    assert gate_update(gate, commands, 5).tolist() == [True, True]
    gate.reset()
    assert gate_update(gate, commands, 6).tolist() == [False, False]


def test_quiet_gate_resumes_full_grace_after_a_moving_command():
    commands = torch.zeros(1, 3)
    gate = rewards.QuietCommandGate(1, "cpu")
    for step in range(3):
        gate_update(gate, commands, step)
    assert not gate_update(gate, torch.tensor([[0., -0.3, 0.]]), 3).item()
    assert not gate_update(gate, commands, 4).item()
    assert not gate_update(gate, commands, 5).item()
    assert gate_update(gate, commands, 6).item()
