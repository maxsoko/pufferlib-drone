import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_tangent_validation_readout import (
    _n702_rejection_exact,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


ROOT = Path(__file__).resolve().parents[1]
N702_REPORT = ROOT / "logs/drone_race_vq2_informed_dreamer/n702_n587_three_epoch_tangent_donor/report.json"
PARENT = ROOT / "logs/drone_race_vq2_informed_dreamer/n587_all1024_actor16_direction/model.pt"
CHILD = ROOT / "logs/drone_race_vq2_informed_dreamer/n702_n587_three_epoch_tangent_donor/model.pt"
TRAIN = ROOT / "logs/drone_race_vq2_informed_dreamer/n681_n680_materialized_partitions/train.npz"
VALIDATION = ROOT / "logs/drone_race_vq2_informed_dreamer/n681_n680_materialized_partitions/validation.npz"


def _hashes() -> dict[str, str]:
    return {
        "parent_checkpoint": _sha256(PARENT),
        "child_checkpoint": _sha256(CHILD),
        "train_dataset": _sha256(TRAIN),
        "validation_dataset": _sha256(VALIDATION),
    }


def test_n702_rejection_contract_is_exact_and_fail_closed() -> None:
    report = json.loads(N702_REPORT.read_text(encoding="utf-8"))
    assert _n702_rejection_exact(report, _hashes())
    wrong = copy.deepcopy(report)
    wrong["progress_gates"]["direction_at_least_minimum"] = True
    assert not _n702_rejection_exact(wrong, _hashes())


def test_direct_cli_has_no_sealed_test_or_optimizer_argument() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_tangent_validation_readout.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--validation-dataset" in result.stdout
    assert "test-dataset" not in result.stdout
    assert "optimizer" not in result.stdout
