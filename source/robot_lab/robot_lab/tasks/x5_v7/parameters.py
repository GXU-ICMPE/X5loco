"""Terrain/posture thresholds; reward-only geometry does not enter actor input."""

from dataclasses import dataclass
from math import isfinite


@dataclass
class X5V7PostureParameters:
    scanner_name: str = "height_scanner"
    slope_transition_deg: tuple[float, float] = (3.0, 8.0)
    residual_transition_m: tuple[float, float] = (0.008, 0.030)
    jump_transition_m: tuple[float, float] = (0.020, 0.050)
    lateral_transition: tuple[float, float] = (0.05, 0.20)
    yaw_transition: tuple[float, float] = (0.10, 0.40)
    stop_linear_threshold: float = 0.03
    stop_yaw_threshold: float = 0.05
    quiet_linear_threshold: float = 0.05
    quiet_yaw_threshold: float = 0.10
    stand_dwell_s: float = 0.30
    stand_ramp_s: float = 0.50
    # Multipliers of the inherited -0.05 hip / -0.01 thigh+calf L1 weights.
    hip_stand_scale: float = 2.0
    hip_maneuver_scale: float = 0.5
    hip_slope_scale: float = 0.5
    hip_rough_scale: float = 0.3
    leg_stand_scale: float = 3.0
    leg_maneuver_scale: float = 0.3
    leg_slope_scale: float = 0.3
    leg_rough_scale: float = 0.1
    flat_tilt_allowance_deg: float = 3.0
    slope_tilt_margin_deg: float = 5.0
    slope_tilt_cap_deg: float = 30.0
    rough_tilt_allowance_deg: float = 15.0
    nonflat_tilt_scale: float = 0.5
    invalid_tilt_allowance_deg: float = 10.0

    def validate(self):
        for name, value in vars(self).items():
            if isinstance(value, (float, int)) and (not isfinite(value) or value <= 0):
                raise ValueError(f"V7 posture {name} must be finite and positive")
            if isinstance(value, (tuple, list)) and (
                len(value) != 2 or not all(isfinite(v) for v in value) or not 0 <= value[0] < value[1]
            ):
                raise ValueError(f"V7 posture {name} must have ordered, nonnegative thresholds")
        if not (self.flat_tilt_allowance_deg <= self.slope_tilt_margin_deg < self.slope_tilt_cap_deg < 90
                and self.rough_tilt_allowance_deg < 90 and self.invalid_tilt_allowance_deg < 90):
            raise ValueError("V7 tilt allowances must preserve a finite penalty for excessive body tilt")


@dataclass
class X5V7WheelSupportParameters:
    # URDF link-frame FK at HAA=0, HFE=0.75, KFE=-1.5 rad, in base coordinates.
    # Order is RF, LF, RH, LH, matching X5_FOOT_BODY_NAMES. These are wheel
    # axle/link origins, not wheel COMs or points rotating around the rim.
    # Recalibrate this reference if the asset or nominal joint pose changes.
    reference_wheel_xy_m: tuple[tuple[float, float], ...] = (
        (0.378088, -0.326500),
        (0.378512, 0.326500),
        (-0.378512, -0.326500),
        (-0.378088, 0.326500),
    )
    longitudinal_tolerance_m: float = 0.04
    lateral_tolerance_m: float = 0.03
    position_error_scale_m: float = 0.05
    wheelbase_shortening_tolerance_m: float = 0.06
    wheelbase_error_scale_m: float = 0.05
    track_width_shortening_tolerance_m: float = 0.06
    track_width_error_scale_m: float = 0.04

    def validate(self):
        reference = self.reference_wheel_xy_m
        if (len(reference) != 4 or any(len(xy) != 2 for xy in reference)
                or not all(isfinite(v) for xy in reference for v in xy)):
            raise ValueError("V7 wheel reference must contain four finite XY pairs in RF/LF/RH/LH order")
        for name, value in vars(self).items():
            if name != "reference_wheel_xy_m" and (not isfinite(value) or value <= 0):
                raise ValueError(f"V7 wheel support {name} must be finite and positive")
        for front, rear in ((0, 2), (1, 3)):
            if not reference[front][0] > 0 > reference[rear][0]:
                raise ValueError("V7 wheel reference must place front/rear axles on their nominal sides of base")
            if reference[front][0] - reference[rear][0] <= self.wheelbase_shortening_tolerance_m:
                raise ValueError("V7 wheelbase tolerance must leave a positive minimum wheelbase")
        for right, left in ((0, 1), (2, 3)):
            if not reference[left][1] > 0 > reference[right][1]:
                raise ValueError("V7 wheel reference must place left/right axles on their nominal sides of base")
            if reference[left][1] - reference[right][1] <= self.track_width_shortening_tolerance_m:
                raise ValueError("V7 track-width tolerance must leave a positive minimum width")
