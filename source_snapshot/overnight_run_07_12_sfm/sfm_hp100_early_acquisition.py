"""Early-horizon, no-update acquisition diagnostic for HP100 SFM.

This runner deliberately stops before any CFM replay.  It asks whether a larger
proposal/query budget can keep the frozen pretrained policy out of early NVP
states.  All seven gammas share the same scenario and flow-noise coordinates.
Only selected-B exact full-H queries may be executed; an optional K-B check at
NVP is audit-only and can never alter execution or enter a buffer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_approval_smoke as APPROVAL
from sfm_hp100_ball_core.expansion import RBFPosterior, _OrderedVerifier, _counter_seed
import sfm_scene as SS


VERSION = "sfm_hp100_early_acquisition_v1"
STATUS = "SFM_HP100_EARLY_ACQUISITION_COMPLETE"
ROUND = 1
RETRY_BATCH = 0
H = 10
EXPECTED_PARALLEL_EPISODES = 16


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def model_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _candidate_sha256(candidate: torch.Tensor) -> str:
    value = candidate.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _terminal_status(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {"SUCCESS": "success", "COLLISION": "collision", "OOB": "oob"}
    if value not in mapping:
        raise RuntimeError(f"unexpected terminal status {value!r}")
    return mapping[value]


def _select(
    results: Sequence,
    *,
    rule: str,
    step_margin_weight: float,
) -> tuple[int | None, list[float | None]]:
    eligible = [
        index for index, result in enumerate(results)
        if not result.error and result.valid and result.progress_eligible
    ]
    scores: list[float | None] = [None] * len(results)
    if not eligible:
        return None, scores
    if rule == "max_step_margin":
        if any(results[index].step_margin is None for index in eligible):
            raise ValueError("max_step_margin requires a margin for every exact positive")
        chosen = min(
            eligible,
            key=lambda index: (
                -float(results[index].step_margin),
                float(results[index].execution_cost),
                index,
            ),
        )
        for index in eligible:
            scores[index] = -float(results[index].step_margin)
        return chosen, scores
    if rule != "weighted_cost":
        raise ValueError(f"unknown execution rule {rule!r}")
    return APPROVAL.select_exact_positive(
        results, step_margin_weight=step_margin_weight,
    )


@torch.inference_mode()
def _sample_blocks(
    adapter: PORT.HP100ExpansionPolicy,
    contexts: Sequence[torch.Tensor],
    seeds: Sequence[int],
    *,
    K: int,
    flow_base_std: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Generator]]:
    """Sample all active contexts in one GPU flow solve while preserving CRN."""
    if len(contexts) != len(seeds):
        raise ValueError("contexts and seeds differ")
    device = next(adapter.parameters()).device
    dtype = adapter.policy.head.weight.dtype
    bases, tokens, generators = [], [], []
    for context, seed in zip(contexts, seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        base = torch.randn(
            K, adapter.policy.d, generator=generator, device=device, dtype=dtype,
        ) * float(flow_base_std)
        token = adapter._policy_context(context).to(device=device, dtype=dtype)
        bases.append(base)
        tokens.append(token.reshape(1, -1).expand(K, -1))
        generators.append(generator)
    flat_base = torch.cat(bases)
    flat_token = torch.cat(tokens)
    flat_plans = adapter.policy.sample(
        len(flat_base), flat_token, nfe=adapter.nfe, temp=1.0,
        initial_noise=flat_base,
    ).detach()
    flat_features = adapter.policy.phi_s_from_x0(
        flat_plans, flat_token, flat_base, s=0.9,
    ).detach()
    plans = list(flat_plans.split(K))
    features = list(flat_features.split(K))
    plan_bases = [base.reshape(K, H, 2) for base in bases]
    return plans, plan_bases, features, generators


def _plan_diagnostics(candidates: torch.Tensor, u_max: float) -> dict:
    values = candidates.detach().cpu().to(torch.float32).reshape(len(candidates), -1)
    saturation = float((values.abs() >= float(u_max) - 1.0e-6).float().mean())
    rounded = np.round(values.numpy(), decimals=5)
    unique_ratio = float(len(np.unique(rounded, axis=0)) / len(rounded))
    distance = (
        float(torch.pdist(values).mean() / math.sqrt(values.shape[1]))
        if len(values) > 1 else 0.0
    )
    return {
        "component_saturation_fraction": saturation,
        "unique_plan_fraction": unique_ratio,
        "mean_pairwise_control_distance": distance,
    }


def _summary(lineages: Sequence[dict], query_rows: Sequence[dict], max_steps: int) -> dict:
    def one(rows: Sequence[dict], queries: Sequence[dict]) -> dict:
        nvp_times = [row["nvp_step"] for row in rows if row["status"] == "nvp"]
        # Non-NVP lineages are right-censored at max_steps for the NVP clock.
        nvp_clock = [row["nvp_step"] if row["status"] == "nvp" else max_steps for row in rows]
        positive = sum(row["y"] == 1 for row in queries)
        negative = sum(row["y"] == 0 and not row["error"] for row in queries)
        errors = sum(bool(row["error"]) for row in queries)
        contexts = {(row["seed"], row["gamma"], row["replica"], row["step"]) for row in queries}
        positive_contexts = {
            (row["seed"], row["gamma"], row["replica"], row["step"])
            for row in queries if row["y"] == 1
        }
        survived_30 = sum(
            row["status"] == "success" or row["executed_steps"] >= 30
            for row in rows
        )
        return {
            "lineages": len(rows),
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "S30": float(survived_30 / len(rows)) if rows else None,
            "survived_30_count": int(survived_30),
            "RMST_to_max_steps": float(np.mean(nvp_clock)) if rows else None,
            "right_censored_nvp_clock_median": float(np.median(nvp_clock)) if rows else None,
            "nvp_step": {
                "count": len(nvp_times),
                "q25": (float(np.quantile(nvp_times, .25)) if nvp_times else None),
                "median": (float(np.median(nvp_times)) if nvp_times else None),
                "q75": (float(np.quantile(nvp_times, .75)) if nvp_times else None),
            },
            "selected_B": {
                "contexts": len(contexts), "Dplus": positive, "Dminus": negative,
                "unresolved": errors,
                "positive_fraction": (
                    float(positive / (positive + negative))
                    if positive + negative else None
                ),
                "positive_context_fraction": (
                    float(len(positive_contexts) / len(contexts)) if contexts else None
                ),
            },
        }

    pooled = one(lineages, query_rows)
    by_gamma = {}
    for gamma in map(float, SS.GAMMAS):
        rows = [row for row in lineages if float(row["gamma"]) == gamma]
        queries = [row for row in query_rows if float(row["gamma"]) == gamma]
        by_gamma[f"{gamma:g}"] = one(rows, queries)
    return {"pooled": pooled, "per_gamma": by_gamma}


@torch.inference_mode()
def run_diagnostic(
    adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    seeds: Sequence[int],
    max_steps: int,
    K: int,
    B: int,
    flow_base_std: float,
    beta: float,
    rbf_lengthscale: float,
    rbf_noise: float,
    execution_rule: str,
    step_margin_weight: float,
    parallel_episodes: int,
    verifier_workers: int,
    audit_unselected_at_nvp: bool,
) -> tuple[dict, list[dict], list[dict]]:
    if parallel_episodes != EXPECTED_PARALLEL_EPISODES:
        raise ValueError("canonical diagnostic requires exactly 16 lineages per gamma")
    if max_steps < 30:
        raise ValueError("max_steps must be at least 30")
    if K < 1 or B < 1 or B > K:
        raise ValueError("require 1 <= B <= K")
    if not math.isfinite(flow_base_std) or flow_base_std <= 0.0:
        raise ValueError("flow_base_std must be finite and positive")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if verifier_workers < 1 or not seeds:
        raise ValueError("verifier_workers and seeds must be nonempty/positive")
    if execution_rule == "weighted_cost" and (
        not math.isfinite(step_margin_weight) or step_margin_weight < 0.0
    ):
        raise ValueError("step-margin weight must be finite and nonnegative")

    before_hash = model_state_sha256(adapter)
    posterior = RBFPosterior(rbf_lengthscale, rbf_noise)
    posterior.set_buffer(None)
    device = next(adapter.parameters()).device
    episodes: list[dict[str, Any]] = []
    for seed in map(int, seeds):
        for gamma in map(float, SS.GAMMAS):
            for replica in range(parallel_episodes):
                reset_seed = _counter_seed(seed, "reset", ROUND, RETRY_BATCH, replica)
                state = task.reset(gamma, replica, reset_seed)
                episodes.append({
                    "seed": seed, "gamma": gamma, "replica": replica,
                    "scenario_id": int(state.scenario_id), "state": state,
                    "status": None, "nvp_step": None, "executed_steps": 0,
                })

    query_rows: list[dict] = []
    context_rows: list[dict] = []
    plan_stats: list[dict] = []
    timers = defaultdict(float)
    verifier_candidates = 0
    audit_candidates = 0
    with _OrderedVerifier(task, verifier_workers) as verifier:
        for step in range(max_steps):
            active = [row for row in episodes if row["status"] is None]
            if not active:
                break
            started = time.perf_counter()
            contexts = [
                task.context(row["state"], row["gamma"]).detach().to(device)
                for row in active
            ]
            gather_seeds = [
                _counter_seed(
                    row["seed"], "gather", ROUND, RETRY_BATCH,
                    row["replica"], step,
                )
                for row in active
            ]
            candidate_blocks, base_blocks, feature_blocks, generators = _sample_blocks(
                adapter, contexts, gather_seeds, K=K, flow_base_std=flow_base_std,
            )
            timers["flow_and_phi"] += time.perf_counter() - started

            prepared = []
            started = time.perf_counter()
            for row, context, candidates, bases, features, generator in zip(
                active, contexts, candidate_blocks, base_blocks, feature_blocks, generators,
            ):
                sigma = posterior.sigma(features).detach().cpu()
                selected, selected_sigma, conditional_ess = posterior.acquire(
                    features, B, beta, generator,
                )
                prepared.append({
                    "episode": row, "context": context, "candidates": candidates,
                    "bases": bases, "sigma": sigma, "selected": selected,
                    "selected_sigma": selected_sigma,
                    "conditional_ess": conditional_ess,
                    "queried": candidates[selected],
                })
                plan_stats.append(_plan_diagnostics(candidates, adapter.policy.u_max))
            timers["empty_rbf_acquisition"] += time.perf_counter() - started

            started = time.perf_counter()
            verified = verifier.verify_many([
                (row["context"], row["queried"], row["episode"]["gamma"])
                for row in prepared
            ])
            timers["selected_B_verifier"] += time.perf_counter() - started
            verifier_candidates += sum(len(row["queried"]) for row in prepared)

            nvp_audits = []
            for index, (prepared_row, results) in enumerate(zip(prepared, verified)):
                if len(results) != B:
                    raise RuntimeError("verifier returned the wrong selected-B size")
                if any(result.error for result in results):
                    raise RuntimeError("verifier error is unresolved, not Dminus")
                chosen, scores = _select(
                    results, rule=execution_rule,
                    step_margin_weight=step_margin_weight,
                )
                prepared_row["results"] = results
                prepared_row["chosen"] = chosen
                prepared_row["scores"] = scores
                if chosen is None and audit_unselected_at_nvp and B < K:
                    selected_set = set(prepared_row["selected"])
                    remaining = [candidate for candidate in range(K) if candidate not in selected_set]
                    prepared_row["audit_ids"] = remaining
                    nvp_audits.append((index, remaining))

            audit_results: dict[int, Sequence] = {}
            if nvp_audits:
                started = time.perf_counter()
                rows = verifier.verify_many([
                    (
                        prepared[index]["context"],
                        prepared[index]["candidates"][remaining],
                        prepared[index]["episode"]["gamma"],
                    )
                    for index, remaining in nvp_audits
                ])
                timers["nvp_KminusB_audit"] += time.perf_counter() - started
                for (index, remaining), results in zip(nvp_audits, rows):
                    if any(result.error for result in results):
                        raise RuntimeError("NVP K-B audit encountered verifier error")
                    audit_results[index] = results
                    audit_candidates += len(remaining)

            for index, prepared_row in enumerate(prepared):
                episode = prepared_row["episode"]
                results = prepared_row["results"]
                selected = prepared_row["selected"]
                chosen = prepared_row["chosen"]
                identity = {
                    "seed": episode["seed"], "gamma": episode["gamma"],
                    "replica": episode["replica"], "scenario_id": episode["scenario_id"],
                    "step": step,
                }
                positive_count = 0
                queried_cpu = prepared_row["queried"].detach().cpu()
                for local, (candidate_id, result) in enumerate(zip(selected, results)):
                    y = int(bool(result.valid and result.progress_eligible))
                    positive_count += y
                    query_rows.append(identity | {
                        "candidate_id": int(candidate_id), "selected_slot": local,
                        "y": y, "error": bool(result.error),
                        "chosen": bool(local == chosen),
                        "native_cost": float(result.execution_cost),
                        "step_margin": float(result.step_margin),
                        "selection_score": prepared_row["scores"][local],
                        "sigma": float(prepared_row["selected_sigma"][local]),
                        "candidate_sha256": _candidate_sha256(queried_cpu[local]),
                        "first_action": queried_cpu[local, 0].tolist(),
                    })

                oracle_positive = None
                nvp_cause = None
                if chosen is None:
                    if index in audit_results:
                        oracle_positive = any(
                            result.valid and result.progress_eligible
                            for result in audit_results[index]
                        )
                        nvp_cause = "acquisition_miss" if oracle_positive else "proposal_failure"
                    else:
                        nvp_cause = "selected_B_all_negative"
                    episode["status"] = "nvp"
                    episode["nvp_step"] = step
                else:
                    episode["state"] = task.advance(
                        episode["state"], prepared_row["queried"][chosen],
                    )
                    episode["executed_steps"] += 1
                    episode["status"] = _terminal_status(task.terminal(episode["state"]))
                context_rows.append(identity | {
                    "selected_B_positive": positive_count,
                    "selected_B_all_negative": bool(chosen is None),
                    "oracle_K_positive_at_nvp": oracle_positive,
                    "nvp_cause": nvp_cause,
                    "chosen_candidate_id": (
                        None if chosen is None else int(selected[chosen])
                    ),
                    "marginal_sigma_mean": float(prepared_row["sigma"].mean()),
                    "marginal_ESS_over_K": 1.0,
                    "conditional_ESS_over_remaining": list(map(float, prepared_row["conditional_ess"])),
                })
            if step + 1 == max_steps:
                for row in episodes:
                    if row["status"] is None:
                        row["status"] = "cutoff"

    after_hash = model_state_sha256(adapter)
    if before_hash != after_hash:
        raise RuntimeError("no-update diagnostic changed the policy state")
    if any(row["status"] is None for row in episodes):
        raise RuntimeError("diagnostic left an unterminated lineage")
    lineages = [
        {
            "seed": row["seed"], "gamma": row["gamma"], "replica": row["replica"],
            "scenario_id": row["scenario_id"], "status": row["status"],
            "nvp_step": row["nvp_step"], "executed_steps": row["executed_steps"],
        }
        for row in episodes
    ]
    causes = Counter(
        row["nvp_cause"] for row in context_rows if row["nvp_cause"] is not None
    )
    payload = {
        "status": STATUS, "version": VERSION,
        "scientific_role": "acquisition-only; no replay, optimizer, or checkpoint write",
        "config": {
            "round": ROUND, "max_steps": max_steps, "K": K, "B": B, "H": H,
            "flow_base_std": flow_base_std, "beta": beta,
            "rbf": {"buffer_rows": 0, "lengthscale": rbf_lengthscale, "noise": rbf_noise},
            "round1_note": (
                "empty GP: marginal sigma is constant, marginal ESS/K=1, uplift=0; "
                "only sequential within-B RBF conditioning diversifies later draws"
            ),
            "execution_rule": execution_rule,
            "execution_step_margin_weight": step_margin_weight,
            "parallel_episodes_per_gamma": parallel_episodes,
            "gammas": list(map(float, SS.GAMMAS)), "seeds": list(map(int, seeds)),
            "audit_unselected_KminusB_at_NVP": audit_unselected_at_nvp,
        },
        "policy_state_sha256_before": before_hash,
        "policy_state_sha256_after": after_hash,
        "policy_unchanged": True,
        "summary": _summary(lineages, query_rows, max_steps),
        "nvp_audit": dict(sorted(causes.items())),
        "counts": {
            "selected_B_verifier_queries": verifier_candidates,
            "audit_only_KminusB_queries": audit_candidates,
            "contexts": len(context_rows),
        },
        "proposal_diagnostics": {
            key: float(np.mean([row[key] for row in plan_stats]))
            for key in plan_stats[0]
        },
        "timers_seconds": dict(sorted(timers.items())),
        "lineages": lineages,
    }
    return payload, query_rows, context_rows


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
                       choices=SS.SCIENTIFIC_EVAL_PROFILES)
    value.add_argument("--scenario-start", type=int, default=300_000)
    value.add_argument("--seeds", default="2")
    value.add_argument("--max-steps", type=int, default=40)
    value.add_argument("--K", type=int, default=64)
    value.add_argument("--B", type=int, default=16)
    value.add_argument("--flow-base-std", type=float, default=2.0)
    value.add_argument("--beta", type=float, default=5.0e-4)
    value.add_argument("--rbf-lengthscale", type=float, default=0.8932)
    value.add_argument("--rbf-noise", type=float, default=1.0e-2)
    value.add_argument("--execution-rule", choices=("max_step_margin", "weighted_cost"),
                       default="max_step_margin")
    value.add_argument("--execution-step-margin-weight", type=float, default=700_000.0)
    value.add_argument("--parallel-episodes", type=int, default=16)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--no-audit-unselected-at-nvp", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    adapter, checkpoint_payload, checkpoint_sha, trainable = APPROVAL._load_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, args.device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile, scenario_start=args.scenario_start,
    ).attach_context_encoder(adapter.policy)
    started = time.time()
    payload, queries, contexts = run_diagnostic(
        adapter, task, seeds=tuple(int(item) for item in args.seeds.split(",")),
        max_steps=args.max_steps, K=args.K, B=args.B,
        flow_base_std=args.flow_base_std, beta=args.beta,
        rbf_lengthscale=args.rbf_lengthscale, rbf_noise=args.rbf_noise,
        execution_rule=args.execution_rule,
        step_margin_weight=args.execution_step_margin_weight,
        parallel_episodes=args.parallel_episodes,
        verifier_workers=args.verifier_workers,
        audit_unselected_at_nvp=not args.no_audit_unselected_at_nvp,
    )
    payload["wall_seconds"] = time.time() - started
    payload["checkpoint"] = {
        "path": str(Path(args.checkpoint).resolve()), "sha256": checkpoint_sha,
        "scientific_status": checkpoint_payload.get("scientific_status"),
        "trainable_parameter_names": trainable,
    }
    query_path = output / "selected_B.jsonl"
    context_path = output / "contexts.jsonl"
    _write_jsonl(query_path, queries)
    _write_jsonl(context_path, contexts)
    payload["artifacts"] = {
        "selected_B": {"path": str(query_path), "sha256": APPROVAL.sha256_file(query_path)},
        "contexts": {"path": str(context_path), "sha256": APPROVAL.sha256_file(context_path)},
    }
    marker = output / "EARLY_ACQUISITION_COMPLETE.json"
    _write_json(marker, payload)
    print(json.dumps({
        "status": STATUS, "output": str(output),
        "S30": payload["summary"]["pooled"]["S30"],
        "marker": str(marker),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
