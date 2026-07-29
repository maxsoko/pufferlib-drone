from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_windows_aigp_race.ps1"


def test_reuses_live_reset_ready_race_regardless_of_age() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "$raceResetReady = (" in source
    assert "[int]$raceReport.base_mode -eq 193" in source
    assert "[int]$raceReport.system_status -eq 4" in source
    assert "Existing race is MAVLink-31000 reset-ready" in source
    assert "$raceAgeS -le $MaxReusableRaceAgeS" not in source


def test_ui_recovery_is_reserved_for_non_reset_ready_state() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    reset_ready_block = source.split("$raceResetReady = (", 1)[1].split(
        "if ($null -ne $ExistingSim)", 1
    )[0]

    assert "return" in reset_ready_block
    assert "Invoke-InactiveRaceRecovery" not in reset_ready_block
    assert "Race is not MAVLink-31000 reset-ready" in source


def test_clicks_are_window_relative_and_recovery_requires_live_readiness() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "GetWindowRect(IntPtr hWnd, out RECT rect)" in source
    assert "$rect.Left + $X, $rect.Top + $Y" in source
    assert source.count("[int]$raceReport.base_mode -eq 193") >= 2
    assert source.count("[int]$raceReport.system_status -eq 4") >= 2
    assert "Race did not become MAVLink-31000 reset-ready" in source
    assert "Invoke-Click 480 805" in source
    assert "Invoke-Click 480 755" not in source
    assert "[int]$EventY = 277" in source
    assert "[int]$RaceY = 872" in source
