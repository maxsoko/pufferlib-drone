from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_bounded.ps1"


def test_vq2_gate2_bounded_launcher_is_puffer_only_hash_pinned_and_bounded():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "policy_callable_vq2_gate2_composite.py" in text
    assert "a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6" in text
    assert "6389d30a6c03eb680d0cb205e91b52690c05cf74bf9871b6eaee1897779e838a" in text
    assert '"--official-reset-on-start"' in text
    assert '"--target-gate-count", "2"' in text
    assert '"--stop-after-official-gate-index", "2"' in text
    assert '"--min-gate-passes", "2"' in text
    assert '"--require-official-race-progress"' in text
    assert "ValidateRange(12, 14)" in text
    assert "--policy-shadow-only" not in text
    assert "Submission" not in text
