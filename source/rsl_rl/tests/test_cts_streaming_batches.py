"""Check sample identity and optimizer parity against the materialized CTS batches."""

import copy
import types
import weakref

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MoECTS
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS


def materialized_batches(storage, count, epochs, indices):
    """Reference: original env-major transpose/flatten and teacher-first batches."""
    teacher, student = indices
    nt, ns = len(teacher) // count, len(student) // count
    fields = [getattr(storage, name).transpose(0, 1).flatten(0, 1) for name in (
        "observations", "actions", "values", "advantages", "returns",
        "actions_log_prob", "mu", "sigma",
    )]
    result = []
    for _ in range(epochs):
        for i in range(count):
            batch = tuple(torch.cat((
                field[teacher[i * nt:(i + 1) * nt]],
                field[student[i * ns:(i + 1) * ns]],
            ), dim=0).detach() for field in fields)
            result.append((*batch, (None, None), None))
    return result


@pytest.mark.parametrize("num_envs,teachers,steps,batches", [(8, 6, 5, 4), (512, 384, 24, 4)])
def test_batches_preserve_samples_order_epochs_and_rng(num_envs, teachers, steps, batches, monkeypatch):
    obs = TensorDict({"policy": torch.zeros(num_envs, 3), "critic": torch.zeros(num_envs, 2)}, [num_envs])
    storage = RolloutStorageCTS("rl", num_envs, teachers, steps, obs, [2])
    for data in (*storage.observations.values(), storage.actions, storage.values, storage.advantages,
                 storage.returns, storage.actions_log_prob, storage.mu, storage.sigma):
        data.copy_(torch.arange(data.numel()).reshape(data.shape))
    torch.manual_seed(42)
    indices = storage.mini_batch_indices()
    rng = torch.get_rng_state()
    expected = materialized_batches(storage, batches, 3, indices)

    def forbidden_flatten(*args, **kwargs):
        raise AssertionError("Streaming batches must not flatten the full TensorDict")

    monkeypatch.setattr(TensorDict, "flatten", forbidden_flatten)
    seen = 0
    for actual, reference in zip(storage.mini_batch_generator(batches, 3, indices=indices), expected, strict=True):
        for a, b in zip(actual[:8], reference[:8], strict=True):
            if isinstance(a, TensorDict):
                for key in a.keys():
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert actual[8:] == reference[8:]
        seen += 1
    assert seen == batches * 3
    assert torch.equal(rng, torch.get_rng_state())


def test_generator_releases_previous_observation_batch():
    obs = TensorDict({"policy": torch.zeros(8, 3)}, [8])
    storage = RolloutStorageCTS("rl", 8, 6, 24, obs, [2])
    generator = storage.mini_batch_generator(4, 5)
    first = next(generator)
    ref = weakref.ref(first[0])
    del first
    second = next(generator)
    assert ref() is None
    assert second[0].batch_size == torch.Size([48])


def make_algorithm():
    torch.manual_seed(19)
    obs = TensorDict({key: torch.randn(8, size) for key, size in (
        ("policy", 530), ("single_obs", 53), ("critic", 291),
    )}, [8])
    policy = ActorCriticMoECTS(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 16,
        actor_hidden_dims=[16], critic_hidden_dims=[16],
        teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16, 16],
        expert_num=8, latent_dim=32,
    )
    storage = RolloutStorageCTS("rl", 8, 6, 24, obs, [16])
    algorithm = MoECTS(policy, storage, 8, num_learning_epochs=5, num_mini_batches=4)
    with torch.inference_mode():
        for step in range(24):
            algorithm.act(obs)
            obs = obs.apply(torch.randn_like)
            dones = torch.zeros(8, dtype=torch.bool)
            dones[0] = step == 12
            algorithm.process_env_step(obs, torch.randn(8), dones, {"time_outs": dones})
        algorithm.compute_returns(obs)
    return algorithm


def test_streamed_update_matches_materialized_update_and_phase_order():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        streamed, reference = make_algorithm(), make_algorithm()
        cache = []

        def cached_generator(storage, count, epochs, *, indices=None):
            if not cache:
                cache.extend(materialized_batches(storage, count, epochs, indices))
            yield from cache

        reference.storage.mini_batch_generator = types.MethodType(cached_generator, reference.storage)
        updates = []
        original_ppo_step = streamed.optimizer.step
        original_student_step = streamed.optimizer_stu_enc.step

        def ppo_step(*args, **kwargs):
            updates.append("ppo")
            return original_ppo_step(*args, **kwargs)

        def student_step(*args, **kwargs):
            updates.append("student")
            return original_student_step(*args, **kwargs)

        streamed.optimizer.step = ppo_step
        streamed.optimizer_stu_enc.step = student_step
        before = copy.deepcopy(streamed.policy.state_dict())
        torch.manual_seed(123)
        expected = reference.update()
        reference_rng = torch.get_rng_state()
        torch.manual_seed(123)
        actual = streamed.update()
        assert updates == ["ppo"] * 20 + ["student"] * 20
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
        assert torch.equal(reference_rng, torch.get_rng_state())
        assert streamed.learning_rate == reference.learning_rate
        assert streamed.storage.step == reference.storage.step == 0
        for name, tensor in streamed.policy.state_dict().items():
            torch.testing.assert_close(tensor, reference.policy.state_dict()[name], rtol=1e-6, atol=1e-7)
        for optimizer_name in ("optimizer", "optimizer_stu_enc"):
            a = getattr(streamed, optimizer_name).state_dict()
            b = getattr(reference, optimizer_name).state_dict()
            assert a["param_groups"] == b["param_groups"]
            for param, state in a["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b["state"][param][name], rtol=1e-6, atol=1e-7)
        for prefix in ("actor.", "teacher_encoder.", "student_moe_encoder."):
            assert any(not torch.equal(value, before[name]) for name, value in streamed.policy.state_dict().items()
                       if name.startswith(prefix))
    finally:
        torch.set_num_threads(previous_threads)
