"""Config compatibility only: no simulation, backward pass, or optimizer step."""

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MoECTS
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS


def construct(**compat_kwargs):
    torch.manual_seed(42)
    obs = TensorDict(
        {name: torch.randn(4, size) for name, size in (("policy", 530), ("single_obs", 53), ("critic", 291))},
        batch_size=[4],
    )
    policy = ActorCriticMoECTS(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 16,
        actor_hidden_dims=[16], critic_hidden_dims=[16],
        teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16, 16],
        expert_num=8, latent_dim=32,
    )
    storage = RolloutStorageCTS("rl", 4, 3, 24, obs, [16])
    return MoECTS(policy, storage, 4, **compat_kwargs), torch.get_rng_state()


def optimizer_parameter_names(algorithm, optimizer):
    names = {id(param): name for name, param in algorithm.policy.named_parameters()}
    return [[names[id(param)] for param in group["params"]] for group in optimizer.param_groups]


def test_new_config_defaults_are_a_noop():
    legacy, legacy_rng = construct()
    current, current_rng = construct(optimizer="adam", share_cnn_encoders=False)
    assert torch.equal(legacy_rng, current_rng)
    assert legacy.policy.state_dict().keys() == current.policy.state_dict().keys()
    for name, tensor in legacy.policy.state_dict().items():
        assert torch.equal(tensor, current.policy.state_dict()[name]), name
    for name in ("optimizer", "optimizer_stu_enc"):
        old_optimizer = getattr(legacy, name)
        new_optimizer = getattr(current, name)
        assert type(old_optimizer) is type(new_optimizer) is torch.optim.Adam
        assert old_optimizer.defaults == new_optimizer.defaults
        assert old_optimizer.state_dict() == new_optimizer.state_dict()
        assert optimizer_parameter_names(legacy, old_optimizer) == optimizer_parameter_names(current, new_optimizer)
        assert len(old_optimizer.state) == len(new_optimizer.state) == 0
    assert current.teacher_env_idxs.tolist() == legacy.teacher_env_idxs.tolist() == [1, 2, 3]
    assert current.student_env_idxs.tolist() == legacy.student_env_idxs.tolist() == [0]
    assert current.storage.step == legacy.storage.step == 0
    assert current.storage.actions.shape == legacy.storage.actions.shape == (24, 4, 16)
    assert not any(param.grad is not None for param in current.policy.parameters())


@pytest.mark.parametrize("optimizer", ["adamw", "sgd", "rmsprop"])
def test_other_optimizers_fail_explicitly(optimizer):
    with pytest.raises(ValueError, match="original Adam optimizers"):
        MoECTS(None, None, 4, optimizer=optimizer)


def test_cnn_sharing_is_not_silently_enabled():
    with pytest.raises(ValueError, match="CNN encoder sharing is not implemented"):
        MoECTS(None, None, 4, share_cnn_encoders=True)


def test_unknown_config_fields_still_fail():
    with pytest.raises(TypeError, match="unexpected keyword argument 'optimzer'"):
        MoECTS(None, None, 4, optimzer="adam")
