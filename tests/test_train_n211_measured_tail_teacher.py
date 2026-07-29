from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_n211_measured_tail_teacher.sh"
SCREEN = ROOT / "scripts" / "screen_n211_measured_tail_teacher.sh"


def test_n211_is_policy_only_measured_entry_fine_tuning():
    text = TRAIN.read_text(encoding="utf-8")
    assert "n211_measured_tail_teacher" in text
    assert "0000000000589824.bin" in text
    assert "--env.start-gate-index 3" in text
    assert "--env.mixed-start-curriculum 0" in text
    assert "--env.gate-position-domain-randomize 0" in text
    assert "--env.observable-gate-phase-onehot 0" in text
    assert "policy_callable" not in text


def test_n211_teacher_is_reward_only_roll_thrust_from_gate4():
    text = TRAIN.read_text(encoding="utf-8")
    assert "--env.w-action-teacher 50" in text
    assert "--env.teacher-course-spline 1" in text
    assert "--env.teacher-pitch-from-gate-index 6" in text
    assert "--env.teacher-roll-from-gate-index 3" in text
    assert "--env.teacher-roll-until-gate-index 6" in text
    assert "--env.teacher-thrust-from-gate-index 3" in text
    assert "--env.teacher-pitch-weight 0" in text


def test_n211_screen_is_exact_common_256_cohort():
    text = SCREEN.read_text(encoding="utf-8")
    assert 'EPISODES="${EPISODES:-256}"' in text
    assert '--eval-episodes "$EPISODES" --require-exact-episodes' in text
    assert '--vec.total-agents "$EPISODES"' in text
    assert "--env.start-gate-index 3" in text
    assert "--env.gate-position-domain-randomize 0" in text
