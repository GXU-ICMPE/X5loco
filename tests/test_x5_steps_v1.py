"""Synthetic trajectory tests for X5 completed steps; no simulator or policy."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/robot_lab/robot_lab/tasks/x5_rewards_v1/stepping.py"
spec = importlib.util.spec_from_file_location("x5_completed_steps_cpu", SOURCE)
stepping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stepping)


def trajectory(
    *, swing_feet=(0,), command=(0.0, 0.3, 0.0), world_direction=(0.0, 1.0),
    terrain=False, base_progress=0.04, foot_progress=0.08,
    peak_lift=0.06, peak_body_lift=0.0, landing_height=0.0,
    air_samples=2, num_envs=1, translation=(0.0, 0.0, 0.0),
):
    """A grounded stance, stationary liftoff, lifted swing, then touchdown.

    The liftoff sample precedes leg movement so taking the first airborne sample
    as the origin does not silently inflate the measured clearance or progress.
    """
    feet = torch.tensor([[
        [0.3, -0.2, 0.1], [0.3, 0.2, 0.1],
        [-0.3, -0.2, 0.1], [-0.3, 0.2, 0.1],
    ]]).repeat(num_envs, 1, 1)
    base = torch.tensor([[0.0, 0.0, 0.55]]).repeat(num_envs, 1)
    offset = torch.tensor(translation)
    feet += offset
    base += offset
    direction = torch.tensor(world_direction)
    cmd = torch.tensor([command]).repeat(num_envs, 1)
    world_command = direction.unsqueeze(0).repeat(num_envs, 1) * cmd[:, :2].norm(dim=-1, keepdim=True)

    def frame(step):
        return dict(
            step=step, dt=0.05,
            wheel_pos_w=feet.clone(), base_pos_w=base.clone(),
            commands_b=cmd.clone(), command_xy_w=world_command.clone(),
            contacts=torch.ones(num_envs, 4, dtype=torch.bool),
            terrain_eligible=torch.full((num_envs,), terrain, dtype=torch.bool),
            episode_time=torch.full((num_envs,), 2.0 + 0.05 * step),
        )

    frames = [frame(0)]
    for sample in range(air_samples):
        current = frame(sample + 1)
        current["contacts"][:, list(swing_feet)] = False
        if sample > 0:
            current["wheel_pos_w"][:, list(swing_feet), :2] += direction * foot_progress * 0.5
            current["wheel_pos_w"][:, list(swing_feet), 2] += peak_lift
            current["base_pos_w"][:, :2] += direction * base_progress * 0.5
            current["base_pos_w"][:, 2] += peak_body_lift
        frames.append(current)
    landed = frame(air_samples + 1)
    landed["wheel_pos_w"][:, list(swing_feet), :2] += direction * foot_progress
    landed["wheel_pos_w"][:, list(swing_feet), 2] += landing_height
    landed["base_pos_w"][:, :2] += direction * base_progress
    landed["base_pos_w"][:, 2] += landing_height
    frames.append(landed)
    return frames


def run(frames, tracker=None):
    if tracker is None:
        tracker = stepping.CompletedStepTracker(frames[0]["wheel_pos_w"].shape[0], 4, "cpu")
    values = [tracker.update(**frame).clone() for frame in frames]
    return tracker, values


def test_lateral_step_pays_only_at_touchdown_and_averages_all_wheels():
    tracker, values = run(trajectory())
    for value in values[:-1]:
        torch.testing.assert_close(value, torch.zeros(1))
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))
    assert tracker.accepted_events.tolist() == [[True, False, False, False]]


def test_two_supported_swinging_feet_receive_two_wheel_shares():
    tracker, values = run(trajectory(swing_feet=(0, 3)))
    torch.testing.assert_close(values[-1], torch.tensor([0.5]))
    assert tracker.accepted_events.tolist() == [[True, False, False, True]]


def test_step_quality_scales_with_relative_lift_and_actual_base_progress():
    # 3 cm leg lift / 6 cm target, and 2 cm base progress / 4 cm target.
    _, values = run(trajectory(peak_lift=0.05, peak_body_lift=0.02, base_progress=0.02))
    torch.testing.assert_close(values[-1], torch.tensor([0.0625]))


def test_step_quality_is_capped_instead_of_rewarding_extreme_hops():
    _, values = run(trajectory(peak_lift=2.0, base_progress=2.0, foot_progress=2.0))
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))


@pytest.mark.parametrize("landing_height,peak_lift,peak_body_lift", [(0.08, 0.12, 0.06), (-0.08, 0.06, 0.0)])
def test_steps_up_and_down_use_local_swing_geometry(landing_height, peak_lift, peak_body_lift):
    _, values = run(trajectory(
        command=(0.3, 0., 0.), world_direction=(1., 0.), terrain=True,
        landing_height=landing_height, peak_lift=peak_lift, peak_body_lift=peak_body_lift,
    ))
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))


def test_world_height_and_xy_origin_do_not_change_reward():
    _, original = run(trajectory())
    _, translated = run(trajectory(translation=(13., -7., 8.)))
    torch.testing.assert_close(translated[-1], original[-1], atol=2e-5, rtol=1e-5)


def test_forward_body_command_at_90_degree_yaw_projects_progress_along_world_y():
    _, values = run(trajectory(command=(0.3, 0., 0.), world_direction=(0., 1.), terrain=True))
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))


def test_normal_yaw_rotation_of_world_command_does_not_change_body_command_intent():
    frames = trajectory(command=(0., 0.3, 0.5))
    # The base turns during the swing; the body-frame translation stays the same.
    frames[2]["command_xy_w"] = torch.tensor([[-0.1, 0.28]])
    frames[3]["command_xy_w"] = torch.tensor([[-0.15, 0.26]])
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))


def test_forward_flat_rolling_context_does_not_request_steps():
    _, values = run(trajectory(command=(0.3, 0., 0.), world_direction=(1., 0.), terrain=False))
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_obstacle_context_is_latched_at_takeoff():
    frames = trajectory(command=(0.3, 0., 0.), world_direction=(1., 0.), terrain=True)
    frames[2]["terrain_eligible"].fill_(False)
    frames[-1]["terrain_eligible"].fill_(False)
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))


def test_entering_eligible_terrain_midair_does_not_reclassify_a_flat_takeoff():
    frames = trajectory(command=(0.3, 0., 0.), world_direction=(1., 0.), terrain=False)
    frames[2]["terrain_eligible"].fill_(True)
    frames[-1]["terrain_eligible"].fill_(True)
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_continuous_rolling_is_neither_rewarded_nor_penalized_as_stepping():
    frames = trajectory(terrain=True)
    for frame in frames:
        frame["contacts"].fill_(True)
    _, values = run(frames)
    for value in values:
        torch.testing.assert_close(value, torch.zeros(1))


@pytest.mark.parametrize("command", [(0., 0., 0.), (0., 0., 0.5), (0.01, 0.01, 0.)])
def test_stationary_or_pure_yaw_commands_do_not_reward_lifting(command):
    _, values = run(trajectory(command=command, terrain=True))
    torch.testing.assert_close(values[-1], torch.zeros(1))


@pytest.mark.parametrize("base_progress", [0.0, -0.04, 0.005])
def test_lifting_without_sufficient_forward_base_progress_is_not_rewarded(base_progress):
    _, values = run(trajectory(base_progress=base_progress))
    torch.testing.assert_close(values[-1], torch.zeros(1))


@pytest.mark.parametrize("foot_progress", [0.0, -0.08, 0.02])
def test_base_motion_does_not_replace_progress_of_the_swinging_foot(foot_progress):
    _, values = run(trajectory(foot_progress=foot_progress))
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_body_only_bobbing_does_not_count_as_leg_lift():
    _, values = run(trajectory(peak_lift=0.06, peak_body_lift=0.06))
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_lowering_body_under_stationary_wheel_does_not_count_as_world_lift():
    _, values = run(trajectory(peak_lift=0., peak_body_lift=-0.08))
    torch.testing.assert_close(values[-1], torch.zeros(1))


@pytest.mark.parametrize("air_samples", [1, 14])
def test_contact_chatter_and_prolonged_flight_are_rejected(air_samples):
    _, values = run(trajectory(air_samples=air_samples))
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_airtime_upper_boundary_allows_a_touchdown_after_exactly_point_six_seconds():
    # Twelve policy intervals are exactly the configured maximum in physical
    # time; float32 accumulation must not invalidate the swing one sample early.
    _, values = run(trajectory(air_samples=12))
    torch.testing.assert_close(values[-1], torch.tensor([0.25]))


@pytest.mark.parametrize("swing_feet", [(0, 1, 2), (0, 1, 2, 3)])
def test_one_support_or_whole_body_flight_does_not_earn_stepping_bonus(swing_feet):
    _, values = run(trajectory(swing_feet=swing_feet))
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_touching_a_vertical_wall_is_not_a_supported_touchdown():
    frames = trajectory()
    landed = frames[-1]
    landed["support_contacts"] = landed["contacts"].clone()
    landed["support_contacts"][:, 0] = False
    # The swinging wheel reports contact, but its force does not provide upward
    # support. Three other supporting wheels must not qualify this wall strike.
    tracker, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))
    assert not tracker.accepted_events.any()


def test_wall_contacts_do_not_supply_the_two_supports_required_for_takeoff():
    frames = trajectory()
    for frame in frames[:-1]:
        frame["support_contacts"] = torch.zeros_like(frame["contacts"])
        frame["support_contacts"][:, 1] = True
    # All wheel bodies initially report contacts, but only one is supported.
    # Real support appearing at landing cannot retroactively validate takeoff.
    frames[-1]["support_contacts"] = frames[-1]["contacts"].clone()
    tracker, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))
    assert not tracker.accepted_events.any()


def test_losing_support_mid_swing_invalidates_the_event_even_after_support_returns():
    frames = trajectory(air_samples=3)
    frames[2]["contacts"].fill_(False)
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_command_reversal_during_swing_invalidates_the_event_even_if_it_reverses_back():
    frames = trajectory()
    frames[2]["commands_b"] *= -1
    frames[2]["command_xy_w"] *= -1
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_zero_command_mid_swing_invalidates_the_event():
    frames = trajectory()
    frames[2]["commands_b"].zero_()
    frames[2]["command_xy_w"].zero_()
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_reset_grace_blocks_spawn_landing_rewards():
    frames = trajectory()
    for frame in frames:
        frame["episode_time"].fill_(frame["step"] * frame["dt"])
    _, values = run(frames)
    torch.testing.assert_close(values[-1], torch.zeros(1))


def test_touchdown_reads_are_idempotent_and_next_grounded_step_has_no_event():
    frames = trajectory()
    tracker, values = run(frames)
    torch.testing.assert_close(tracker.update(**frames[-1]), values[-1])
    assert tracker.accepted_events.tolist() == [[True, False, False, False]]
    grounded = deepcopy(frames[-1])
    grounded["step"] += 1
    grounded["episode_time"] += grounded["dt"]
    torch.testing.assert_close(tracker.update(**grounded), torch.zeros(1))
    assert not tracker.accepted_events.any()


def test_repeated_airborne_reads_do_not_accumulate_air_duration():
    frames = trajectory(air_samples=1)
    tracker = stepping.CompletedStepTracker(1, 4, "cpu")
    tracker.update(**frames[0])
    for _ in range(10):
        tracker.update(**frames[1])
    torch.testing.assert_close(tracker.update(**frames[-1]), torch.zeros(1))


def test_partial_reset_midair_discards_only_the_reset_environments_swing():
    frames = trajectory(num_envs=2)
    tracker, _ = run(frames[:-1])
    tracker.reset(torch.tensor([0]))
    actual = tracker.update(**frames[-1])
    torch.testing.assert_close(actual, torch.tensor([0., 0.25]))
    assert tracker.accepted_events.tolist() == [[False, False, False, False], [True, False, False, False]]


def test_full_reset_midair_does_not_create_a_phantom_touchdown_event():
    frames = trajectory()
    tracker, _ = run(frames[:-1])
    tracker.reset()
    torch.testing.assert_close(tracker.update(**frames[-1]), torch.zeros(1))


def test_skipped_steps_discard_incomplete_swing_history():
    frames = trajectory()
    tracker, _ = run(frames[:-1])
    frames[-1]["step"] += 5
    torch.testing.assert_close(tracker.update(**frames[-1]), torch.zeros(1))


def test_starting_observation_in_air_does_not_count_as_a_recorded_liftoff():
    frames = trajectory()
    tracker, values = run(frames[1:])
    torch.testing.assert_close(values[-1], torch.zeros(1))
    assert not tracker.accepted_events.any()
