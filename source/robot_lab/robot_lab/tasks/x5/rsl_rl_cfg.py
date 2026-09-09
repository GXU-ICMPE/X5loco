"""Unmodified MoE-CTS learner in an isolated X5 experiment namespace."""

from isaaclab.utils import configclass

from robot_lab.tasks.legbot.rsl_rl_cfg import LegbotMoECTSRunnerCfg


@configclass
class X5MoECTSRunnerCfg(LegbotMoECTSRunnerCfg):
    experiment_name = "x5_moe_cts"
