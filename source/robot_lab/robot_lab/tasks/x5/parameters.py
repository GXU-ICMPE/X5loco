"""X5 physical ratings and observation/height bounds; no reward rescaling.

Ratings are ordered HAA/HFE/KFE for RF, LF, RH, LH, then the four wheels.
The three physical penalties use the original Legbot functions and weights.
"""

X5_EFFORT = (300.0, 250.0, 250.0) * 4 + (30.0,) * 4
X5_SPEED = (6.0, 6.0, 8.0) * 4 + (30.0,) * 4

BASE_HEIGHT_TARGET = 0.55  # Loaded flat-ground stance measured at approximately 0.5502 m.
HEIGHT_SCAN_OFFSET = 0.63  # 0.5 + (0.55 - 0.42): keep nominal terrain-feature centering.
CONTACT_FORCE_CLIP = (0.0, 3000.0)  # About 10x nominal per-wheel support, not a safety limit.
JOINT_TORQUE_CLIP = (-300.0, 300.0)  # Includes the largest X5 effort limit.
# Max nominal discrete speed reversal: 2 * 30 / 0.0025 = 24000 rad/s^2.
# Add headroom; this is an observation bound, not an acceleration constraint.
JOINT_ACCELERATION_CLIP = (-30000.0, 30000.0)
