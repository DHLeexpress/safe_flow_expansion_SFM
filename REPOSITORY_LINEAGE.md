# Repository lineage

## Answer to “when did the two repositories branch?”

They did not branch in Git. There is no common Git ancestor between
`DHLeexpress/safe_flow_expansion` and `DHLeexpress/safe_flow_expansion_SFM`.
The latter remote was empty before this handoff.

The **conceptual task fork** occurred in the shared `safeMPPI` repository at:

```text
e6fcebe278d076459df379c4fc0739e9cd18acff
2026-07-20 00:48:01 PDT
Implement SFM Hp10 B1 study
```

This was 4 h 38 min after the static standalone workbook's initial commit
`8ae3d99` (2026-07-19 20:09:39 PDT). The static and SFM lines were then updated
in parallel. Therefore neither repository should merge the other's Git history
as if one were a normal branch.

## Shared mechanism

Both repositories implement:

- conditional flow proposals over H=10 control windows;
- uncertainty-tilted finite-budget verifier acquisition;
- deterministic full-window labels and separate D/D+ bookkeeping;
- fail-closed execution from verified candidates;
- recent-window hierarchical replay;
- raw, untilted evaluation isolated from the gathering controller.

## Task-specific divergence

| component | static sister | SFM repository |
|---|---|---|
| dynamics / scene | double integrator, fixed circular obstacles | robot plus moving SFM pedestrians |
| observation | low7 + 32×32 nominal-polytope embedding | Hp10: 10×16×12 nominal signed-field history + low5 + control GRU |
| global modes | U/R routes around fixed obstacle | left/right/yield interaction modes across scenarios |
| verifier | fitted static trajectory polytope / SOCP-oriented checker | moving-face H10 certificate under constant-velocity pedestrians |
| uncertainty memory | static B1 RBF mechanism | RBF W2, cap512, per-gamma q75 retention |
| execution | static nominal-Hp max margin | moving-scene nominal-Hp max margin among full-H positives |
| OOD | enlarged/fused static obstacle | 40 pedestrians at 1–2 m/s vs 20 at .5–1 m/s |

## SFM commit milestones in `safeMPPI`

| commit | date (PDT) | change |
|---|---|---|
| `e6fcebe` | Jul 20 00:48 | Hp10+B1 implementation begins |
| `103476d` | Jul 20 02:15 | frozen selection and negative-replay semantics |
| `b2caf9a` | Jul 20 12:07 | explicit ID/OOD evaluation |
| `60be313` | Jul 20 23:05 | exact K16 verifier shared by expansion and visualization |
| `b27df76` | Jul 20 23:25 | matched-ID/double-shift deployment and visual provenance |
| `f0142ff` | Jul 21 02:13 | matched Kazuki cost; optimizer steps exposed |
| `d112ad2` | Jul 21 11:53 | full-H replay and gamma-balanced GP memory revision |
| `47800df` | Jul 21 12:28 | bounded runtime and `/data3/research1` artifact contract |
| `e5ab47b` | Jul 21 12:51 | eight scheduler slots for the current sweep |

The sister repository's `master` stayed focused on the static B1 workbook at
`94cc262`; its active paper branches later added static-scene figures and
controller studies. Those are not SFM results and are not copied here.

