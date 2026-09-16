"""Independent V8 state lifecycle; same observation/action layout and stop PI."""

from robot_lab.tasks.x5.env import X5Env
from .state import X5V8TerrainState, X5V8MotionState


class X5V8Env(X5Env):
    def load_managers(self):
        super().load_managers()
        # Existing tilt/terrain helpers consume this attribute.
        self.v7_posture = X5V8TerrainState(self)
        self.v8_motion = X5V8MotionState(self)

    def _reset_idx(self, env_ids):
        log = {}
        if len(env_ids):
            if hasattr(self, "v7_posture"):
                log.update(self.v7_posture.reset(env_ids))
            if hasattr(self, "v8_motion"):
                log.update(self.v8_motion.reset(env_ids))
            if hasattr(self, "scene"):
                actuator = self.scene["robot"].actuators["wheels"]
                if hasattr(actuator, "episode_metrics"):
                    log.update({key.replace("X5V7/", "X5V8/"): value
                                for key, value in actuator.episode_metrics(env_ids).items()})
        super()._reset_idx(env_ids)
        self.extras["log"].update(log)
