from __future__ import annotations

import sfm_hp100_preterminal_sweep as S


def test_declared_grid_is_exactly_three_by_three():
    arms = S.declared_arms()
    assert len(arms) == 9
    assert {arm.lambda_weight for arm in arms} == {0.0, 70_000.0, 140_000.0}
    assert {arm.negative_alpha for arm in arms} == {0.0, 0.001, 0.01}
    assert len({arm.name for arm in arms}) == 9


def test_development_key_enforces_liveness_before_safety_gain():
    baseline = dict(
        SR=.56, CR=.44, timeout=0.0, Validity=.48,
        successful_clearance=.11, successful_time_to_goal=7.0,
    )
    safe_live = dict(
        SR=.55, CR=.30, timeout=.01, Validity=.60,
        successful_clearance=.13, successful_time_to_goal=8.0,
    )
    collapsed = dict(
        SR=.20, CR=.10, timeout=.70, Validity=.90,
        successful_clearance=.20, successful_time_to_goal=15.0,
    )
    assert S.development_key(safe_live, baseline, round_index=2) > S.development_key(
        collapsed, baseline, round_index=1,
    )


def test_strict_win_requires_all_declared_safety_improvements():
    baseline = dict(
        SR=.56, CR=.44, timeout=0.0, Validity=.48,
        successful_clearance=.11, successful_time_to_goal=7.0,
    )
    winner = dict(
        SR=.55, CR=.30, timeout=.01, Validity=.60,
        successful_clearance=.13, successful_time_to_goal=8.0,
    )
    assert S.strict_win(winner, baseline)
    assert not S.strict_win(winner | {"successful_clearance": .10}, baseline)
