"""X5 environment using the existing action-history manager without delay."""

from isaaclab.envs import ManagerBasedRLEnv

from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2


class X5Env(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        # This shared manager is robot-agnostic: it only tracks action history.
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print("[X5Env] Legbot task with X5 asset/control and physical-unit calibration.")
