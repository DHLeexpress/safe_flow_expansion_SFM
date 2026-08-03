from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

import sfm_hp100_early_acquisition as E
from sfm_hp100_ball_core.expansion import Verification


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)
        self.d = 20
        self.u_max = 2.0

    def sample(self, n, context, nfe, temp, initial_noise):
        del context, nfe, temp
        return initial_noise.reshape(n, 10, 2).clamp(-2.0, 2.0)

    def phi_s_from_x0(self, plans, context, base, s):
        del context, s
        return torch.stack((plans.reshape(len(plans), -1).mean(1),
                            base.reshape(len(base), -1).mean(1)), dim=1)


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = _Policy()
        self.nfe = 1

    def _policy_context(self, context):
        return context.reshape(-1)[:1]

    def cfm_loss(self, *_args, **_kwargs):
        raise AssertionError("no-update diagnostic called cfm_loss")


@dataclass
class _State:
    scenario_id: int
    steps: int = 0
    status: str | None = None


class _Task:
    def __init__(self, positive_steps=2):
        self.positive_steps = int(positive_steps)
        self.reset_rows = []

    def reset(self, gamma, episode, seed):
        self.reset_rows.append((float(gamma), int(episode), int(seed)))
        return _State(scenario_id=300_000 + int(seed) % 1_000_000_000)

    def context(self, state, gamma):
        return torch.tensor([float(state.steps), float(gamma)])

    def verify(self, context, candidates, gamma):
        del gamma
        positive = int(round(float(context[0]))) < self.positive_steps
        return tuple(
            Verification(
                valid=positive, hp_eligible=True, margin=1.0,
                execution_cost=float(index), progress_eligible=True,
                error=False, step_margin=float(index) / 10.0,
            )
            for index in range(len(candidates))
        )

    def advance(self, state, candidate):
        del candidate
        state.steps += 1
        return state

    def terminal(self, state):
        return state.status


def _run(task, *, K=2, B=1, std=1.0):
    return E.run_diagnostic(
        _Adapter(), task, seeds=(2,), max_steps=30, K=K, B=B,
        flow_base_std=std, beta=1e-3, rbf_lengthscale=.5,
        rbf_noise=1e-2, execution_rule="max_step_margin",
        step_margin_weight=700_000.0, parallel_episodes=16,
        verifier_workers=1, audit_unselected_at_nvp=True,
    )


def test_no_update_and_paired_16_by_7_lineages_with_nvp_accounting():
    payload, queries, contexts = _run(_Task(positive_steps=2))
    assert payload["policy_unchanged"] is True
    assert payload["policy_state_sha256_before"] == payload["policy_state_sha256_after"]
    assert len(payload["lineages"]) == 16 * 7
    assert {row["status"] for row in payload["lineages"]} == {"nvp"}
    assert {row["nvp_step"] for row in payload["lineages"]} == {2}
    assert {row["executed_steps"] for row in payload["lineages"]} == {2}
    assert len(queries) == 16 * 7 * 3
    assert len(contexts) == 16 * 7 * 3
    assert payload["summary"]["pooled"]["selected_B"]["Dplus"] == 16 * 7 * 2
    assert payload["summary"]["pooled"]["selected_B"]["Dminus"] == 16 * 7
    for replica in range(16):
        ids = {
            row["scenario_id"] for row in payload["lineages"]
            if row["replica"] == replica
        }
        assert len(ids) == 1


def test_max_step_margin_uses_native_cost_only_as_tie_break():
    results = [
        Verification(True, True, 1.0, 0.0, step_margin=.1),
        Verification(True, True, 1.0, 100.0, step_margin=.2),
        Verification(True, True, 1.0, -100.0, step_margin=.2),
    ]
    chosen, scores = E._select(
        results, rule="max_step_margin", step_margin_weight=0.0,
    )
    assert chosen == 2
    assert scores == [-.1, -.2, -.2]


def test_batched_sampling_preserves_base_std_scaling():
    adapter = _Adapter()
    contexts = [torch.tensor([0.0, .1]), torch.tensor([0.0, 1.0])]
    _, base1, _, _ = E._sample_blocks(
        adapter, contexts, (11, 12), K=4, flow_base_std=1.0,
    )
    _, base2, _, _ = E._sample_blocks(
        adapter, contexts, (11, 12), K=4, flow_base_std=2.0,
    )
    assert torch.equal(base2[0], 2.0 * base1[0])
    assert torch.equal(base2[1], 2.0 * base1[1])


@pytest.mark.parametrize("K,B", [(0, 1), (2, 0), (2, 3)])
def test_invalid_candidate_budgets_fail(K, B):
    with pytest.raises(ValueError):
        E.run_diagnostic(
            _Adapter(), _Task(), seeds=(2,), max_steps=30, K=K, B=B,
            flow_base_std=1.0, beta=1e-3, rbf_lengthscale=.5,
            rbf_noise=1e-2, execution_rule="max_step_margin",
            step_margin_weight=0.0, parallel_episodes=16,
            verifier_workers=1, audit_unselected_at_nvp=False,
        )


def test_parallel_episode_contract_is_fixed_at_16():
    with pytest.raises(ValueError, match="exactly 16"):
        E.run_diagnostic(
            _Adapter(), _Task(), seeds=(2,), max_steps=30, K=2, B=1,
            flow_base_std=1.0, beta=1e-3, rbf_lengthscale=.5,
            rbf_noise=1e-2, execution_rule="max_step_margin",
            step_margin_weight=0.0, parallel_episodes=8,
            verifier_workers=1, audit_unselected_at_nvp=False,
        )
