from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_n219_predictor_adaptation.sh"


def test_n219_adapts_promoted_parent_under_predictor_without_intervention():
    source = TRAIN.read_text(encoding="utf-8")
    assert "0000000000065536.bin" in source
    assert "--env.sitl-gate-motion-predict-dropout 1" in source
    assert "--env.teacher-action-blend 0" in source
    assert "--env.w-action-teacher 200" in source
    assert "--env.start-gate-index 3" in source


def test_n219_is_native_only():
    source = TRAIN.read_text(encoding="utf-8")
    assert "FlightSim" not in source
    assert "31000" not in source
    assert "Mavlink" not in source
