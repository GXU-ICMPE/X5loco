#!/usr/bin/env bash
# V8 from scratch, with the existing 512-env PhysX memory limits.
set -euo pipefail

X5_V8_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${X5_TRAIN_PYTHON:-}" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" == "x3w_isaaclab" && -n "${CONDA_PREFIX:-}" ]]; then
    X5_TRAIN_PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x /home/xgy/miniforge3/envs/x3w_isaaclab/bin/python ]]; then
    X5_TRAIN_PYTHON=/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python
  else
    X5_TRAIN_PYTHON="$(command -v python || true)"
  fi
fi
if [[ -z "$X5_TRAIN_PYTHON" ]] || ! command -v "$X5_TRAIN_PYTHON" >/dev/null 2>&1; then
  echo '请先 conda activate x3w_isaaclab，或设置 X5_TRAIN_PYTHON=/path/to/python。' >&2
  exit 1
fi
cd "$X5_V8_ROOT"
export PYTHONPATH="$X5_V8_ROOT/source/rsl_rl:$X5_V8_ROOT/source/robot_lab${PYTHONPATH:+:$PYTHONPATH}"
echo "[X5 V8] Python: $X5_TRAIN_PYTHON"

exec "$X5_TRAIN_PYTHON" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-v8 \
  --num_envs 512 --max_iterations 30000 --seed 42 \
  --run_name v8_pose_axis_curriculum_512_s42 --device cuda:0 --headless \
  env.sim.physx.gpu_max_rigid_contact_count=1048576 \
  env.sim.physx.gpu_max_rigid_patch_count=262144 \
  env.sim.physx.gpu_found_lost_pairs_capacity=262144 \
  env.sim.physx.gpu_found_lost_aggregate_pairs_capacity=1048576 \
  env.sim.physx.gpu_total_aggregate_pairs_capacity=262144 \
  env.sim.physx.gpu_collision_stack_size=67108864 \
  agent.resume=false \
  "$@"
