from __future__ import annotations

import scripts.collect_vq2_lc095_late_phase_local_teacher_features as lc095


def test_lc095_dense_late_phase_contract() -> None:
    assert lc095.AGENTS == lc095.EPISODES == 512
    assert lc095.PHASE_MIN == 6
    assert lc095.PHASE_MAX_EXCLUSIVE == 24
    assert lc095.STEP_LIMIT == 2_048
    assert lc095.MINIMUM_RECORDS_PER_PHASE == 2_000


def test_lc095_parent_is_source_locked() -> None:
    lc095.verify_inputs()


def test_lc095_environment_is_teacher_local_only() -> None:
    original = lc095.base.dagger_config
    try:
        lc095.base.dagger_config = lambda _: ({"env": {}}, [])
        config, _ = lc095.local_teacher_config(object())
    finally:
        lc095.base.dagger_config = original
    environment = config["env"]
    assert environment["gate_local_start_curriculum"] == 1
    assert environment["gate_local_start_probability"] == 1.0
    assert environment["gate_local_start_gate_min"] == 6
    assert environment["gate_local_start_gate_max_exclusive"] == 24
    assert environment["teacher_action_blend"] == 0.0
