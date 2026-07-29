from __future__ import annotations

from pathlib import Path

import pytest

import scripts.eval_vq2_variable_gate_oracle as variable_oracle
from scripts.eval_vq2_variable_gate_oracle import (
    ACCEPTANCE_COUNTS,
    EPISODE_SECONDS,
    MINIMUM_SUCCESS_RATE,
    fixed_overrides,
    mixed_variable_environment,
    native_step_budget,
    run_admission,
    variable_environment,
    variable_oracle_passes,
)


def _passing_metrics(num_gates: int, success_rate: float = 1.0) -> dict[str, float]:
    metrics = {
        "env/n": 512.0,
        "env/success_rate": success_rate,
        "env/crash": 0.0,
        "env/out_of_order": 0.0,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        f"env/gate_count{num_gates}_episode": 1.0,
        f"env/gate_count{num_gates}_success": success_rate,
    }
    for gate in range(num_gates):
        metrics[f"env/ordered_gate{gate}_sampled"] = success_rate
        metrics[f"env/ordered_gate{gate}_radial"] = 0.05
    return metrics


def test_variable_oracle_contract_is_full_course_and_count_agnostic() -> None:
    assert ACCEPTANCE_COUNTS == (5, 8, 11, 12)
    assert MINIMUM_SUCCESS_RATE == 0.99
    assert EPISODE_SECONDS == 360.0
    for count in range(5, 13):
        environment = variable_environment(count)
        assert environment["num_gates"] == count
        assert environment["gate_radius"] == 0.75
        assert environment["teacher_alignment_governor"] == 1
        assert environment["teacher_action_blend"] == 1.0
        assert environment["teacher_roll_until_gate_index"] == count
        assert environment["observable_gate_index_denominator"] == 16.0
        assert environment["gate_position_require_valid_course"] == 1
        assert environment["gate_position_min_forward_gap_m"] == 8.0
        assert environment["gate_position_max_segment_distance_m"] == 40.0
        assert environment["sitl_plant_domain_randomize"] == 0


def test_variable_environment_rejects_out_of_range_count() -> None:
    with pytest.raises(ValueError, match=r"\[5, 12\]"):
        variable_environment(4)
    with pytest.raises(ValueError, match=r"\[5, 12\]"):
        variable_environment(13)


def test_mixed_profile_samples_per_instance_and_retains_six_anchor() -> None:
    environment = mixed_variable_environment(seed=429021)
    assert environment["num_gates"] == 6
    assert environment["num_gates_per_env_randomize"] == 1
    assert environment["num_gates_per_env_min"] == 5
    assert environment["num_gates_per_env_max"] == 12
    assert environment["num_gates_per_env_seed"] == 429021
    assert environment["teacher_roll_until_gate_index"] == 16


def test_variable_oracle_admits_99_percent_but_never_collision() -> None:
    success_rate = 507.0 / 512.0
    metrics = _passing_metrics(12, success_rate)
    assert variable_oracle_passes(metrics, episodes=512, num_gates=12)
    metrics["env/crash"] = 1.0 / 512.0
    assert not variable_oracle_passes(metrics, episodes=512, num_gates=12)


def test_variable_oracle_requires_every_counted_gate_and_envelope() -> None:
    metrics = _passing_metrics(11)
    assert variable_oracle_passes(metrics, episodes=512, num_gates=11)
    metrics["env/ordered_gate10_sampled"] = 0.0
    assert not variable_oracle_passes(metrics, episodes=512, num_gates=11)


def _count_report(
    *,
    count: int,
    agents: int,
    episodes: int,
    seed: int,
    sources: dict[str, str],
    extension_path: str,
    source_commit: str,
    runtime: dict[str, str],
) -> dict[str, object]:
    metrics = _passing_metrics(count)
    metrics["env/n"] = float(episodes)
    environment = variable_environment(count)
    return {
        "schema": "vq2_variable_gate_oracle_count_report_v1",
        "tag": f"vq2_vg002_variable_gate_oracle_{count}g_{episodes}",
        "num_gates": count,
        "agents": agents,
        "episodes": episodes,
        "seed": seed,
        "precision_bytes": 4,
        "compiled_extension_path": extension_path,
        "source_commit": source_commit,
        "runtime": runtime,
        "native_steps": native_step_budget(
            max_steps=int(environment["max_steps"]),
            episodes=episodes,
            agents=agents,
        ),
        "wall_time_seconds": 1.0,
        "minimum_success_rate": MINIMUM_SUCCESS_RATE,
        "admitted": True,
        "loader_overrides": fixed_overrides(agents=agents, seed=seed),
        "fixed_environment": environment,
        "metrics": metrics,
        "source_sha256": sources,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "student_checkpoints_written": 0,
            "submission_authorized": False,
        },
    }


def test_admission_resumes_only_missing_count_without_rewriting_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "vg002"
    sources = {"source": "locked"}
    extension_path = "/workspace/pufferlib/_C.so"
    source_commit = "a" * 40
    runtime = {"runtime": "locked"}
    monkeypatch.setattr(
        variable_oracle,
        "current_source_sha256",
        lambda: (sources, extension_path, source_commit),
    )
    monkeypatch.setattr(variable_oracle, "current_runtime_manifest", lambda: runtime)
    first_calls: list[int] = []

    def interrupted_count(**kwargs: int) -> dict[str, object]:
        count = kwargs["num_gates"]
        first_calls.append(count)
        if count == 8:
            raise InterruptedError("host interruption")
        return _count_report(
            count=count,
            agents=kwargs["agents"],
            episodes=kwargs["episodes"],
            seed=kwargs["seed"],
            sources=sources,
            extension_path=extension_path,
            source_commit=source_commit,
            runtime=runtime,
        )

    monkeypatch.setattr(variable_oracle, "run_count", interrupted_count)
    with pytest.raises(InterruptedError, match="host interruption"):
        run_admission(
            output_root=output,
            counts=(5, 8),
            agents=2,
            episodes=2,
            seed=900,
        )
    count5_before = (output / "count_5.json").read_bytes()
    assert first_calls == [5, 8]
    assert not (output / "report.json").exists()

    resumed_calls: list[int] = []

    def resumed_count(**kwargs: int) -> dict[str, object]:
        count = kwargs["num_gates"]
        resumed_calls.append(count)
        return _count_report(
            count=count,
            agents=kwargs["agents"],
            episodes=kwargs["episodes"],
            seed=kwargs["seed"],
            sources=sources,
            extension_path=extension_path,
            source_commit=source_commit,
            runtime=runtime,
        )

    monkeypatch.setattr(variable_oracle, "run_count", resumed_count)
    aggregate = run_admission(
        output_root=output,
        counts=(5, 8),
        agents=2,
        episodes=2,
        seed=900,
        resume=True,
    )
    assert aggregate["admitted"] is True
    assert aggregate["completed_counts"] == [5, 8]
    assert resumed_calls == [8]
    assert (output / "count_5.json").read_bytes() == count5_before

    monkeypatch.setattr(
        variable_oracle,
        "run_count",
        lambda **_: pytest.fail("completed resume reran a count"),
    )
    assert run_admission(
        output_root=output,
        counts=(5, 8),
        agents=2,
        episodes=2,
        seed=900,
        resume=True,
    ) == aggregate


def test_admission_resume_rejects_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "vg002"
    sources = {"source": "locked"}
    extension_path = "/workspace/pufferlib/_C.so"
    source_commit = "b" * 40
    runtime = {"runtime": "locked"}
    monkeypatch.setattr(
        variable_oracle,
        "current_source_sha256",
        lambda: (sources, extension_path, source_commit),
    )
    monkeypatch.setattr(variable_oracle, "current_runtime_manifest", lambda: runtime)
    monkeypatch.setattr(
        variable_oracle,
        "run_count",
        lambda **kwargs: _count_report(
            count=kwargs["num_gates"],
            agents=kwargs["agents"],
            episodes=kwargs["episodes"],
            seed=kwargs["seed"],
            sources=sources,
            extension_path=extension_path,
            source_commit=source_commit,
            runtime=runtime,
        ),
    )
    run_admission(
        output_root=output,
        counts=(5,),
        agents=2,
        episodes=2,
        seed=901,
    )
    monkeypatch.setattr(
        variable_oracle,
        "current_source_sha256",
        lambda: ({"source": "changed"}, extension_path, source_commit),
    )
    with pytest.raises(RuntimeError, match="source_sha256"):
        run_admission(
            output_root=output,
            counts=(5,),
            agents=2,
            episodes=2,
            seed=901,
            resume=True,
        )
    monkeypatch.setattr(
        variable_oracle,
        "current_source_sha256",
        lambda: (sources, extension_path, source_commit),
    )
    monkeypatch.setattr(
        variable_oracle,
        "current_runtime_manifest",
        lambda: {"runtime": "changed"},
    )
    with pytest.raises(RuntimeError, match="runtime"):
        run_admission(
            output_root=output,
            counts=(5,),
            agents=2,
            episodes=2,
            seed=901,
            resume=True,
        )
