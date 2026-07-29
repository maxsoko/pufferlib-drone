from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_shadow.ps1"


def test_vq2_gate2_launcher_is_shadow_only_and_hash_pinned():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "policy_callable_vq2_gate2_composite.py" in text
    assert "a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6" in text
    assert "6389d30a6c03eb680d0cb205e91b52690c05cf74bf9871b6eaee1897779e838a" in text
    assert "1998a3370df919bf0eaf16e57d3928de41850fefac9bc0ea026d0ba783d653bc" in text
    assert '"--policy-shadow-only"' in text
    assert '"--no-arm-on-start"' in text
    assert "--official-reset-on-start" not in text
    assert "--target-gate-count" not in text
    assert "--stop-after-official-gate-index" not in text
    assert "--min-gate-passes" not in text
