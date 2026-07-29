import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_tangent_channel_source_contract import (
    N698_REPORT_SHA256,
    N699_REPORT_SHA256,
    _n698_legacy_rejection_exact,
    _n699_failure_is_only_legacy_field,
    audit,
)


ROOT = Path(__file__).resolve().parents[1]
N698 = ROOT / "logs/drone_race_vq2_informed_dreamer/n698_n587_low_sensitivity_progress_channel/report.json"
N699 = ROOT / "logs/drone_race_vq2_informed_dreamer/n699_n587_actor_tangent_progress_channel/report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_legacy_n698_contract_accepts_absent_direction_mode_only() -> None:
    source = _load(N698)
    assert "direction_mode" not in source
    assert _n698_legacy_rejection_exact(source)
    wrong = copy.deepcopy(source)
    wrong["direction_mode"] = "actor_tangent_null"
    assert not _n698_legacy_rejection_exact(wrong)


def test_n699_failure_is_isolated_to_legacy_contract_gate() -> None:
    source = _load(N699)
    assert _n699_failure_is_only_legacy_field(source)
    wrong = copy.deepcopy(source)
    wrong["process_gates"]["model_tensors_exact"] = False
    assert not _n699_failure_is_only_legacy_field(wrong)


def test_current_immutable_reports_authorize_only_implementation_smoke(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    report = audit(
        argparse.Namespace(n698_report=N698, n699_report=N699, report=report_path)
    )
    assert report["source_report_hashes"] == {
        "n698": N698_REPORT_SHA256,
        "n699": N699_REPORT_SHA256,
    }
    assert all(report["audit_gates"].values())
    assert report["corrected_process_passed"] is True
    assert report["corrected_channel_admitted"] is True
    assert report["implementation_smoke_authorized"] is True
    assert report["model_loaded"] == 0
    assert report["optimizer_steps"] == 0
    assert report["flightsim_packets"] == 0


def test_direct_cli_has_only_report_inputs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_tangent_channel_source_contract.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--n698-report" in result.stdout
    assert "--n699-report" in result.stdout
    assert "--report" in result.stdout
    assert "checkpoint" not in result.stdout
    assert "dataset" not in result.stdout
