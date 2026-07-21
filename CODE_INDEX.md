# SFM B1 code index

All paths are relative to `source_snapshot/`. The snapshot preserves the
original directory layout because the authenticated source uses `_paths.py`.

## Active SFM pipeline

| file | role | blind spot / warning |
|---|---|---|
| `overnight_run_07_12_sfm/grid_policy_sfm.py` | Hp10 residual conditional flow policy and explicit-design CFM loss. | GRU sees robot controls, not pedestrian velocity or identity. |
| `overnight_run_07_12_sfm/sfm_hp_history.py` | Leak-free newest-to-oldest Hp10 construction. | Ten frames are a finite temporal summary. |
| `overnight_run_07_12_sfm/sfm_b1_expert.py` | Faithful SafeMPPI expert definition used for demonstrations. | Dataset tensors are external; this is not called during expansion. |
| `overnight_run_07_12_sfm/stage3_pretrain_sfm.py` | Trajectory-disjoint, gamma/trajectory-balanced CFM pretraining and ID gate. | The original dataset serialization launcher is absent from the authenticated source commit. |
| `overnight_run_07_12_sfm/sfm_scene.py` | Hashed training, ID, moderate OOD, severe OOD, and density-only scene profiles. | Constant spawn law may not span all crowd processes. |
| `overnight_run_07_12_sfm/sfm_metrics2.py` | H10 rollout, moving-pedestrian prediction, exact angular-interval fitted-face certificate, K16 outer faces. | Not a generic conic solver; assumes constant pedestrian velocity. |
| `overnight_run_07_12_sfm/sfm_b1_rbf.py` | RBF posterior, sequential without-replacement acquisition, adaptive ESS beta. | Kernel uncertainty can be blind in a collapsed representation. |
| `overnight_run_07_12_sfm/sfm_b1_cost.py` | Nominal-Hp admissibility and max-margin / frozen SafeMPPI-cost selectors. | The current sweep activates max margin only; local margin is not recursive feasibility. |
| `overnight_run_07_12_sfm/sfm_b1_store.py` | D/D+ shards, q75 gamma-balanced GP retention, hierarchy mass, positive/signed replay. | W2 and cap512 deliberately forget older acquisition support. |
| `overnight_run_07_12_sfm/sfm_b1_expand.py` | 56-replica frozen macro-round, K16/B4 gathering, NVP, update, checkpoint. | Controller-induced outcomes are not raw evaluation. |
| `overnight_run_07_12_sfm/sfm_b1_curve_eval.py` | Canonical per-round raw M10 and locked-temperature evaluation. | Finite M10 is development monitoring, not confirmation. |
| `overnight_run_07_12_sfm/sfm_b1_alpha_steps_sweep.py` | Nine-arm alpha×epoch scheduler, shortlist, M50 screen, M100 confirmation. | Temperature calibration must remain disjoint and separately labeled. |
| `overnight_run_07_12_sfm/run_sfm_b1_alpha_inner_sweep.sh` | Helios provenance/resource gate and eight-slot launcher. | Assumes physical GPUs 1/3 and external `/data3/research1`. |
| `overnight_run_07_12_sfm/sfm_b1_benchmark.py` | Fixed-bank raw and Kazuki benchmark. | Comparator semantics differ from raw flow. |
| `overnight_run_07_12_sfm/sfm_kazuki.py` | Goal/safety guidance plus MPPI refinement using the Hp10 prior. | Zero guidance still leaves MPPI refinement and warm start. |
| `overnight_run_07_12_sfm/sfm_b1_query_diagnostic.py` | Candidate-specific K/B certificate traces. | Explanatory evidence; never training or evaluation data. |
| `overnight_run_07_12_sfm/sfm_b1_density_viz.py` | Read-only mechanism renderer. | Outcome-conditioned density visualization must not support performance claims. |

## Transitive closure

| path | purpose |
|---|---|
| `overnight_run_2026-07-02/` | shared grid features and policy compatibility layer |
| `overnight_run_2026-07-01/` | nominal polar grid, local frame, and verifier geometry |
| `overnight_run_today/src/` | base flow-policy and dynamics compatibility imports |
| `cfm_mppi/` | SFM agents/render utilities and SafeMPPI adapter |
| `ieee_compact_polytope_verifier_package/` | fitted-polytope reference implementation |
| `overnight_run_2026-06-28/best_area_mode4.json` | frozen SafeMPPI configuration payload |

The transitive directories contain historical utilities not active in the
current method. Presence in the snapshot is dependency provenance, not method
endorsement.

## Tests

`source_snapshot/overnight_run_07_12_sfm/analysis/` contains the 20 focused
SFM tests. They cover Hp10 leakage, checkpoint freezing, verifier semantics,
RBF acquisition, GP retention, replay mass and visit counts, alpha=0 exactness,
cost matching, fixed-bank curves, artifact contracts, scheduler slots, and
runtime gates. The authenticated latest worktree passed 118 tests before the
current run.

`tests/test_workbook_contract.py` validates this standalone package without
rerunning the expensive study.

