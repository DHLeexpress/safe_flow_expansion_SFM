# Claude handoff: HP100 Safe Flow Expansion

## 0. Start here

Work from the latest pushed `origin/master` in a private worktree. Do not edit
the shared checkout or reuse another agent's output root.

```bash
cd /home/dohyun/projects
git -C safe_flow_expansion_SFM fetch origin
FROZEN_SHA="$(git -C safe_flow_expansion_SFM rev-parse origin/master)"
git -C safe_flow_expansion_SFM worktree add \
  /home/dohyun/projects/safe_flow_expansion_SFM-claude-hp100 \
  "$FROZEN_SHA"
cd /home/dohyun/projects/safe_flow_expansion_SFM-claude-hp100
git switch -c agent/claude-hp100-expansion-$(date +%Y%m%d)
python scripts/show_claude_hp100_handoff.py
```

Show the checker output before editing. New outputs belong under a new root:

```text
/data3/research1/claude_sfm_hp100_<FROZEN_SHA7>/
```

## 1. Immutable canonical inputs

| artifact | bundled path | SHA-256 |
|---|---|---|
| checkpoint | `checkpoints/hp100_pretrained_r0_258999ae.pt` | `258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44` |
| pretraining report | `provenance/hp100_pretrain_20260802/pretraining_report.json` | `19f83f865056db73aea656d617680c1078ed5cf69e0a5fdb68cc4f5c14b74cbd` |
| dataset manifest | `provenance/hp100_pretrain_20260802/dataset_manifest.json` | `44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94` |
| ID M50 | `provenance/hp100_pretrain_20260802/id_m50.json` | `58df71d0a25b801c47f9f0da8077704eddb0b52eda18c072de45cad2c3818961` |
| OOD M50 | `provenance/hp100_pretrain_20260802/ood_m50.json` | `1708348be707868d93e8e878f069cb66c0605fea33ab1f4319dd3d3cf1b0ce4e` |
| branch video | `assets/hp100_20260802/branch_viz/hp100_id_ood_branch.mp4` | `dd1269a392b3dd14161d2392ad4ee69d278c3bd76253639dac892206370c8547` |
| SafeMPPI mechanism | `assets/hp100_20260802/expert_mechanism/hp100_safemppi_mechanism_g0p2_ep109.mp4` | `416b77aeb60b36a06e365dcbb06a343b038bea3a4e671d4c4df00b0f74f69876` |

External Helios data:

```text
/data3/research1/sfm_hp100_certified_weighted_500x7_2671a94
```

Report source provenance precisely:

- dataset collection: `2671a9447b7b914053dce5fe9be2a0aae6c67a8d`;
- pretraining: `e9164e5a6e70b86cecae4660e7732f8ecc6a93f7`;
- exact branch renderer: `b659526`;
- integrated source snapshot: `ef35f1f2df89a2131d9e0a21e0d7095a2d1d7b1d`.

The promoted file is the selected epoch-119 state plus authenticated metadata.
Do not compare its container hash to the intermediate `ckpt_119.pt` hash and
claim a mismatch; compare the state dictionary and config.

## 2. Locked model and dataset contract

The dataset is ID-only: 20 pedestrians at 0.5-1.0 m/s, seven gammas, exactly
500 successful SafeMPPI lineages per gamma. It contains 204,297 contexts and
201,075 eligible CFM targets. OOD never entered model selection.

The target is the current SafeMPPI accepted-set weighted H10 plan. A row is
eligible only if at least one of 2,048 candidates survives and the weighted
plan independently passes the same frozen nominal-polytope H10 recheck.
All-rejected fallback rows and failed weighted-plan rechecks are provenance,
not positive CFM targets.

Architecture:

- `v3-sfm-hp100-residual`;
- 10x32x100 newest-to-oldest Hp history;
- Conv 10->16->32, angular pooling only, no radial pooling, visual token 128;
- GRU-16 over the previous 16 robot controls; low token 48;
- residual width 256, two blocks; output 10x2;
- action and velocity componentwise capped to [-2,2];
- current-position tangent geometry, `predict_gain=0`;
- 16 nominal support faces; 32 observation rays are a different object.

For the declared HP100 expansion control, call
`configure_head_only_expansion()`: CNN, GRU, low encoder, and trunk remain
frozen; only the final head is trainable.

## 3. Locked raw baselines

Canonical raw evaluation is temperature 1, NFE 8, one unguided H10 sample per
context, first clipped action, with no GP, verifier selector, Kazuki guidance,
MPPI refinement, fallback, or privileged lookahead.

| distribution | episode bank | SR | CR | timeout | Validity | clearance | time |
|---|---|---:|---:|---:|---:|---:|---:|
| ID | ep0=150000, M50/gamma | .9571 | .0429 | 0 | .7906 | .3010 m | 5.617 s |
| OOD | ep0=250000, M50/gamma | .5600 | .4371 | .0029 | .4769 | .1086 m | 7.147 s |

Both use noise seed 20260802 and the same declared noise-bank contract. The
promotion bank at ep0=14000 is a different experiment; do not mix its numbers
with these public rows.

There is no authenticated locked HP100 Kazuki M50 result yet. Establish it on
these exact scenario ranges with `sfm_hp100_kazuki_eval.py`, checkpoint above,
shared clipped dynamics, `safe_coef=.3`, `goal_coef=.5`, and no shield,
templates, privileged lookahead, or fallback.

## 4. Exact safety semantics

The pretraining target gate is one frozen current-tangent nominal polytope.
Expansion/evaluation label truth comes from the candidate-specific exact GREEN
moving-pedestrian verifier: every sensed or predicted-entering pedestrian,
analytic H1-H10 faces, exactly 16 artificial outer faces, and shared clipped
dynamics.

Window-level Validity is the fraction of terminal-truncated executed windows
that are in bounds, collision-free, and GREEN-certified. A current proposal's
blue/red label is not trajectory Validity. Goal reach ends the executed episode
audit; it must not silently shorten a queried candidate's H10 certificate.

## 5. Required next task: exact HP100 port

Port the existing neutral B1 mechanism additively to HP100; do not rewrite the
legacy Hp10 path.

Control contract:

- OOD gather: 40 pedestrians at 1.0-2.0 m/s;
- qualification: two synchronous episodes per gamma, T=180, H=10;
- K=16, B=4, generation temperature 1, NFE 8;
- store each proposal's original Gaussian x0;
- representation
  \(z=\operatorname{normalize}\phi_{\theta_0}((1-s)x_0+sU,s,c)\), s=.9;
- calibrate RBF length scale from exactly 50 balanced HP100 embeddings; never
  reuse Hp10 ell=.242108...;
- GP uses only the previous round's executed full-H positives, gamma-balanced
  cap 512, lambda=1e-2, adaptive normalized ESS target .5, fixed within round;
- exact verifier resolves selected B=4;
- ordinary admissible execution uses max one-step nominal-Hp margin;
- an all-negative guided execution retains y=0 and enters only isolated D0;
  D0 never enters D+, GP, beta calibration, or certificate Validity;
- replay whole D+ and then isolated D0, every eligible sample exactly once per
  declared inner pass;
- head-only control: batch 128, lr 1e-5, one inner pass, alpha 0;
- no expert replay, prox, anchor, curriculum, rollback, recovery start, or
  privileged MPC in the control arm.

Before a run, prove strict checkpoint admission, cached r0 equivalence, exact
freeze behavior, x0/context/gamma/lineage serialization, D0 isolation, and one
exact branch visualization. Same-context probes are diagnostics only.

## 6. Evaluation funnel

Do not run M50 for every arm and round.

1. Every arm/round: fixed disjoint raw M10 screen.
2. Best eligible checkpoint per arm: fresh M50.
3. Global winner: untouched disjoint M100.

Temperature-one results are always mandatory. Any per-gamma temperature must
be selected on a separate calibration bank, frozen, then evaluated once on a
new confirmation bank.

Primary OOD objective: lower CR and raise Validity and successful clearance
relative to HP100 r0 while retaining liveness and the intended gamma-dependent
safety/time behavior. ID is a preservation audit. Do not claim a Kazuki win
until the locked comparator uses the matching confirmation scenarios.

## 7. Do not change

- pretrained state/config;
- ID/OOD definitions or banks after viewing results;
- capped-dynamics ordering;
- current-tangent `predict_gain=0` feature contract;
- exact GREEN solver, 16 outer faces, full-H semantics;
- raw temperature-one baseline;
- locked Kazuki coefficients;
- dataset labels or OOD-free promotion claim;
- shared checkout, shared output roots, or another agent's branch.

No artificial positive relabeling, hidden fallback, post-confirmation
temperature tuning, privileged MPC, or changed collision/timeout semantics.

## 8. Required delivery

Return frozen source SHA and clean-worktree proof; every checkpoint/dataset/
scene/source hash; locked HP100 Kazuki ID/OOD M50; HP100 RBF preflight; one
round with D/D+/D0, beta/ESS/uplift and exact replay accounting; r0/r1 raw M10
and selected disjoint M50; exact branch MP4/final PNG; per-gamma six metrics;
all deviations; and a byte/SHA manifest for every artifact.

Stop fail-closed on any provenance, label, solver, or equivalence mismatch. Do
not start a long sweep until the one-round delivery is complete.

## Copy-paste prompt for Claude

> Pull the latest `DHLeexpress/safe_flow_expansion_SFM` master into a new
> private worktree and read `CLAUDE_HP100_EXPANSION_HANDOFF.md` completely.
> Run `python scripts/show_claude_hp100_handoff.py` first and show its full
> authenticated output before editing. Preserve the checkpoint, raw baselines,
> clipped dynamics, current-tangent HP100 observation, exact GREEN verifier,
> scenes, and fixed banks. First measure the locked HP100 Kazuki comparator on
> the existing ID/OOD M50 scenario banks. Then additively port the canonical
> neutral B1 collector/update to HP100 with the declared head-only freeze and
> recalibrate RBF length scale from 50 balanced HP100 embeddings; never reuse
> Hp10 ell. Prove r0 equivalence and every storage/label/freeze invariant, then
> run only a two-episode-per-gamma, one-round OOD qualification. Use GPUs 1 and
> 3 in parallel without killing foreign processes, write only under a new
> `/data3/research1/claude_sfm_hp100_<sha7>/` root, and commit/push only your
> private branch. Stop fail-closed on any provenance, label, solver, or
> equivalence mismatch; do not launch a long sweep until the one-round delivery
> is complete.
