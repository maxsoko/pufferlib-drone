import json

from scripts.analyze_gate2_gate3_transition import analyze, build_report


def _observation(*, forward, right, closing, visible=True, size=32):
    import math

    values = [0.0] * size
    values[0] = math.tanh(-closing / 5.0)
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    if size == 32:
        values[24 + 2] = 1.0
    return values


def _write_report(path, *, passed, gate3_rows):
    samples = [
        {
            "elapsed_s": 1.0,
            "official_active_gate_index": 1,
            "observation": _observation(forward=2.0, right=0.1, closing=4.0),
            "normalized_action": [0.0, 0.0, 0.0, 0.0],
        }
    ]
    samples.extend(gate3_rows)
    path.write_text(
        json.dumps(
            {
                "official_active_gate_index": 3 if passed else 2,
                "policy_trace": {"samples": samples},
                "sitl": {"latest_telemetry": {"collision_id": None}},
            }
        ),
        encoding="utf-8",
    )


def _gate3_row(elapsed, *, forward, right, closing, roll=0.0):
    return {
        "elapsed_s": elapsed,
        "official_active_gate_index": 2,
        "observation": _observation(
            forward=forward, right=right, closing=closing
        ),
        "normalized_action": [0.0, roll, 0.0, 0.0],
    }


def test_analyze_identifies_severe_transition_and_missing_admission(tmp_path):
    path = tmp_path / "failed.json"
    _write_report(
        path,
        passed=False,
        gate3_rows=[
            _gate3_row(2.0, forward=28.0, right=-16.0, closing=0.0, roll=1.0),
            _gate3_row(3.0, forward=20.0, right=3.0, closing=-2.0, roll=-1.0),
        ],
    )
    report = analyze(path)
    assert report["passed_gate3"] is False
    assert report["severe_far_off_axis_samples"] == 1
    assert report["first_close_range_admission"] is None
    assert report["receding_visible_samples"] == 1
    assert report["full_positive_roll_samples"] == 1
    assert report["full_negative_roll_samples"] == 1


def test_build_report_separates_pass_and_failure_counts(tmp_path):
    passed = tmp_path / "passed.json"
    failed = tmp_path / "failed.json"
    _write_report(
        passed,
        passed=True,
        gate3_rows=[
            _gate3_row(2.0, forward=11.0, right=-1.0, closing=4.0),
        ],
    )
    _write_report(
        failed,
        passed=False,
        gate3_rows=[
            _gate3_row(2.0, forward=25.0, right=10.0, closing=0.0),
        ],
    )
    report = build_report([passed, failed])
    assert report["source_count"] == 2
    assert report["passing_source_count"] == 1
    assert report["failing_source_count"] == 1
    assert report["summary"]["passing_with_close_range_admission"] == 1
    assert report["summary"]["failing_with_severe_far_off_axis"] == 1


def test_analyze_accepts_historical_23_value_prefix_observation(tmp_path):
    path = tmp_path / "legacy.json"
    row = _gate3_row(2.0, forward=10.0, right=0.2, closing=3.0)
    row["observation"] = _observation(
        forward=10.0, right=0.2, closing=3.0, size=23
    )
    _write_report(path, passed=True, gate3_rows=[row])
    report = analyze(path)
    assert report["gate3_samples"] == 1
    assert report["first_close_range_admission"] is not None
