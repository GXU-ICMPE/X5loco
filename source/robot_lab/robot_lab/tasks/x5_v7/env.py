"""V6 control/observations with shared, resettable V7 posture reward state."""

from robot_lab.tasks.x5.env import X5Env

from .state import X5V7PostureState
from .wheel_support import X5V7WheelSupportState


class X5V7Env(X5Env):
    def load_managers(self):
        super().load_managers()
        self.v7_posture = X5V7PostureState(self)
        self.v7_wheel_support = X5V7WheelSupportState(self)

    def _reset_idx(self, env_ids):
        log = self.v7_posture.reset(env_ids) if hasattr(self, "v7_posture") else {}
        if hasattr(self, "v7_wheel_support"):
            log.update(self.v7_wheel_support.reset(env_ids))
        if len(env_ids) and hasattr(self, "scene"):
            actuator = self.scene["robot"].actuators["wheels"]
            if hasattr(actuator, "episode_metrics"):
                log.update(actuator.episode_metrics(env_ids))
        super()._reset_idx(env_ids)
        self.extras["log"].update(log)
