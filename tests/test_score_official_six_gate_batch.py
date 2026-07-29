import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_official_six_gate_batch as scorer


def _fixture(tmp_path, name, *, index, crash=False, accepted=False, finish=-1, exit_code=1):
    summary = tmp_path / f"{name}_summary.json"
    poststop = tmp_path / f"{name}_poststop.json"
    summary.write_text(json.dumps({"attempts": [{
        "exit_code": exit_code,
        "official_active_gate_index": index,
        "official_race_finish_time_ns": finish,
        "smoke": {
            "acceptance_passed": accepted,
            "crash_detected": crash,
            "invalid_run": crash,
            "sitl": {"duration_s": 12.0},
        },
    }]}), encoding="utf-8")
    poststop.write_text(json.dumps({
        "latest_telemetry": {"race_status": {
            "active_gate_index": index,
            "race_finish_time_ns": finish,
        }}
    }), encoding="utf-8")
    return {"variant": name.split("_")[0], "tag": name,
            "summary_path": str(summary), "poststop_path": str(poststop)}


def test_valid_finish_dominates_faster_partial_result(tmp_path):
    manifest = {"target_gate_count": 6, "attempts": [
        _fixture(tmp_path, "finish_1", index=6, accepted=True,
                 finish=20_000_000_000, exit_code=0),
        _fixture(tmp_path, "fast_1", index=5, accepted=False, exit_code=1),
    ]}
    result = scorer.score_manifest(manifest)
    assert result["leader"] == "finish"
    assert result["variants"][0]["valid_finishes"] == 1


def test_poststop_delayed_gate_progress_is_authoritative(tmp_path):
    record = _fixture(tmp_path, "delayed_1", index=1, crash=True)
    post = json.loads(Path(record["poststop_path"]).read_text())
    post["latest_telemetry"]["race_status"]["active_gate_index"] = 2
    Path(record["poststop_path"]).write_text(json.dumps(post), encoding="utf-8")
    result = scorer.score_attempt(record, target_gate_count=6)
    assert result["in_run_official_gate_count"] == 1
    assert result["poststop_official_gate_count"] == 2
    assert result["official_gate_count"] == 2
    assert result["valid_finish"] is False


def test_worst_then_median_gate_progress_precedes_time(tmp_path):
    records = [
        _fixture(tmp_path, "robust_1", index=3),
        _fixture(tmp_path, "robust_2", index=3),
        _fixture(tmp_path, "spiky_1", index=5),
        _fixture(tmp_path, "spiky_2", index=1),
    ]
    result = scorer.score_manifest({"target_gate_count": 6, "attempts": records})
    assert result["leader"] == "robust"
    assert result["variants"][0]["minimum_official_gate_count"] == 3


def test_non_six_gate_manifest_is_rejected():
    try:
        scorer.score_manifest({"target_gate_count": 3, "attempts": []})
    except ValueError as exc:
        assert "must be 6" in str(exc)
    else:
        raise AssertionError("expected six-gate target rejection")
