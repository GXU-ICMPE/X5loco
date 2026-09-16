"""Moderately tighten the existing exponential tracking kernels after warmup."""


def tracking_precision(env, env_ids, start_it=1000, end_it=5000,
                       initial_std=0.5, final_std=0.4, steps_per_iteration=24):
    if not (0 <= start_it < end_it and 0 < final_std <= initial_std and steps_per_iteration > 0):
        raise ValueError("Invalid X5 tracking precision curriculum")
    iteration = env.common_step_counter // steps_per_iteration
    progress = min(1.0, max(0.0, (iteration - start_it) / (end_it - start_it)))
    std = initial_std + progress * (final_std - initial_std)
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
        term = env.reward_manager.get_term_cfg(name)
        term.params["std"] = std
        env.reward_manager.set_term_cfg(name, term)
    return std
