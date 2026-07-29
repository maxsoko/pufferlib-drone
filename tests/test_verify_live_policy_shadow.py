import hashlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "verify_live_policy_shadow", SCRIPTS / "verify_live_policy_shadow.py"
)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class FakeModel:
    def reset_state(self):
        pass

    def infer(self, observation):
        return [observation[0], observation[1], observation[2], observation[3]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_verify_shadow_report_checks_no_control_phase_abi_hashes_and_replay(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in verify.LOCAL_MANIFEST_PATHS.items()
    }
    manifest["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": digest(checkpoint),
    }
    observation = [0.0] * 32
    observation[:4] = [0.1, -0.2, 0.3, -0.4]
    observation[10] = 1.0
    observation[23] = 0.0
    observation[24] = 1.0
    report = {
        "acceptance_passed": True,
        "control_mode": "policy-attitude",
        "control_inputs": {
            "policy_shadow_only": True,
            "arm_on_start": False,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "policy_shadow_cadence_probe": {
                "commands_counted": 60,
                "measurement_duration_s": 1.0,
                "effective_command_hz": 60.0,
                "command_rate_violations": 0,
                "mavlink_setpoints_sent": 0,
            },
            "official_reset_start": {"reset_sent": False},
            "state_estimator": {"calibration": {"samples": 60}},
        },
        "sitl": {
            "commands_sent": 0,
            "command_kind": "shadow_policy_attitude_no_setpoint",
            "telemetry": {"messages_seen": 100, "race_statuses": 10},
        },
        "vision": {"frames_seen": 10, "detector_detections": 8},
        "policy_trace": {
            "inference_ticks": 1,
            "deployment_manifest": manifest,
            "samples": [
                {
                    "official_active_gate_index": 0,
                    "observation": observation,
                    "normalized_action": observation[:4],
                }
            ],
        },
    }

    result = verify.verify_shadow_report(report, checkpoint, model=FakeModel())

    assert result["passed"] is True
    assert result["blockers"] == []
    assert result["max_action_error"] == 0.0
    assert result["phase_counts"] == [1, 0, 0, 0, 0, 0]
    assert all(result["deployment_manifest_matches"].values())

    report["sitl"]["duration_s"] = 1.0
    slow = verify.verify_shadow_report(
        report, checkpoint, model=FakeModel(), min_inference_hz=2.0
    )
    assert "policy_inference_rate_too_low:1.000<2.000" in slow["blockers"]


def test_verify_shadow_report_accepts_composite_manifest(tmp_path):
    gate4 = tmp_path / "gate4.bin"
    gate5 = tmp_path / "gate5.bin"
    gate6 = tmp_path / "gate6.bin"
    for path in (gate4, gate5, gate6):
        path.write_bytes(path.name.encode())
    callable_path = ROOT / "scripts" / "policy_callable_six_gate_composite.py"
    manifest_paths = dict(verify.LOCAL_MANIFEST_PATHS)
    manifest_paths["policy_callable"] = callable_path
    manifest = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in manifest_paths.items()
    }
    manifest["checkpoint"] = {"path": str(gate4), "sha256": digest(gate4)}
    manifest["gate4_checkpoint"] = {
        "path": str(gate4),
        "sha256": digest(gate4),
    }
    manifest["gate5_checkpoint"] = {
        "path": str(gate5),
        "sha256": digest(gate5),
    }
    manifest["gate6_checkpoint"] = {
        "path": str(gate6),
        "sha256": digest(gate6),
    }
    observation = [0.0] * 32
    observation[:4] = [0.1, -0.2, 0.3, -0.4]
    observation[10] = 1.0
    observation[24] = 1.0
    report = {
        "acceptance_passed": True,
        "control_mode": "policy-attitude",
        "control_inputs": {
            "policy_shadow_only": True,
            "arm_on_start": False,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "policy_shadow_cadence_probe": {
                "commands_counted": 60,
                "measurement_duration_s": 1.0,
                "effective_command_hz": 60.0,
                "command_rate_violations": 0,
                "mavlink_setpoints_sent": 0,
            },
            "official_reset_start": {"reset_sent": False},
            "state_estimator": {"calibration": {"samples": 60}},
        },
        "sitl": {
            "commands_sent": 0,
            "command_kind": "shadow_policy_attitude_no_setpoint",
            "telemetry": {"messages_seen": 100, "race_statuses": 10},
        },
        "vision": {"frames_seen": 10, "detector_detections": 8},
        "policy_trace": {
            "inference_ticks": 1,
            "deployment_manifest": manifest,
            "samples": [
                {
                    "official_active_gate_index": 0,
                    "observation": observation,
                    "normalized_action": observation[:4],
                }
            ],
        },
    }

    result = verify.verify_shadow_report(
        report,
        gate4,
        model=FakeModel(),
        policy_callable_path=callable_path,
        additional_checkpoints={
            "gate4_checkpoint": gate4,
            "gate5_checkpoint": gate5,
            "gate6_checkpoint": gate6,
        },
    )

    assert result["passed"] is True
    assert result["blockers"] == []
    assert result["deployment_manifest_matches"]["gate6_checkpoint"] is True


def test_verify_shadow_report_accepts_vq2_gate2_composite_manifest(tmp_path):
    prefix = tmp_path / "prefix.bin"
    gate2 = tmp_path / "gate2.bin"
    prefix.write_bytes(b"prefix")
    gate2.write_bytes(b"gate2")
    callable_path = ROOT / "scripts" / "policy_callable_vq2_gate2_composite.py"
    manifest_paths = dict(verify.LOCAL_MANIFEST_PATHS)
    manifest_paths["policy_callable"] = callable_path
    manifest = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in manifest_paths.items()
    }
    manifest["checkpoint"] = {"path": str(prefix), "sha256": digest(prefix)}
    manifest["gate2_checkpoint"] = {
        "path": str(gate2), "sha256": digest(gate2)
    }
    observation = [0.0] * 32
    observation[:4] = [0.1, -0.2, 0.3, -0.4]
    observation[10] = 1.0
    observation[24] = 1.0
    report = {
        "acceptance_passed": True,
        "control_mode": "policy-attitude",
        "control_inputs": {
            "policy_shadow_only": True,
            "arm_on_start": False,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "policy_shadow_cadence_probe": {
                "commands_counted": 60,
                "measurement_duration_s": 1.0,
                "effective_command_hz": 60.0,
                "command_rate_violations": 0,
                "mavlink_setpoints_sent": 0,
            },
            "official_reset_start": {"reset_sent": False},
            "state_estimator": {"calibration": {"samples": 60}},
        },
        "sitl": {
            "commands_sent": 0,
            "command_kind": "shadow_policy_attitude_no_setpoint",
            "telemetry": {"messages_seen": 100, "race_statuses": 10},
            "duration_s": 1.0,
        },
        "vision": {"frames_seen": 10, "detector_detections": 8},
        "policy_trace": {
            "inference_ticks": 1,
            "deployment_manifest": manifest,
            "samples": [{
                "official_active_gate_index": 0,
                "observation": observation,
                "normalized_action": observation[:4],
            }],
        },
    }

    result = verify.verify_shadow_report(
        report,
        prefix,
        model=FakeModel(),
        policy_callable_path=callable_path,
        additional_checkpoints={"gate2_checkpoint": gate2},
    )

    assert result["passed"] is True
    assert result["deployment_manifest_matches"]["gate2_checkpoint"] is True


def test_verify_shadow_report_accepts_hybrid_prefix_confidence_only_before_gate4(
    tmp_path,
):
    prefix = tmp_path / "prefix.bin"
    gate4 = tmp_path / "gate4.bin"
    gate5 = tmp_path / "gate5.bin"
    gate6 = tmp_path / "gate6.bin"
    for path in (prefix, gate4, gate5, gate6):
        path.write_bytes(path.name.encode())
    callable_path = ROOT / "scripts" / "policy_callable_six_gate_hybrid.py"
    manifest_paths = dict(verify.LOCAL_MANIFEST_PATHS)
    manifest_paths["policy_callable"] = callable_path
    manifest = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in manifest_paths.items()
    }
    manifest["checkpoint"] = {"path": str(prefix), "sha256": digest(prefix)}
    for name, path in (
        ("gate4_checkpoint", gate4),
        ("gate5_checkpoint", gate5),
        ("gate6_checkpoint", gate6),
    ):
        manifest[name] = {"path": str(path), "sha256": digest(path)}

    def observation(gate_index, confidence):
        values = [0.0] * 32
        values[:4] = [0.1, -0.2, 0.3, -0.4]
        values[10] = 1.0
        values[23] = gate_index / 6.0
        values[24 + gate_index] = 1.0
        values[30] = confidence
        values[31] = 0.625 if gate_index < 3 else 0.0
        return values

    samples = [
        {
            "official_active_gate_index": 0,
            "observation": observation(0, 0.625),
            "normalized_action": [0.1, -0.2, 0.3, -0.4],
        },
        {
            "official_active_gate_index": 3,
            "observation": observation(3, 0.0),
            "normalized_action": [0.1, -0.2, 0.3, -0.4],
        },
    ]
    report = {
        "acceptance_passed": True,
        "control_mode": "policy-attitude",
        "control_inputs": {
            "policy_shadow_only": True,
            "arm_on_start": False,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "policy_shadow_cadence_probe": {
                "commands_counted": 60,
                "measurement_duration_s": 1.0,
                "effective_command_hz": 60.0,
                "command_rate_violations": 0,
                "mavlink_setpoints_sent": 0,
            },
            "official_reset_start": {"reset_sent": False},
            "state_estimator": {"calibration": {"samples": 60}},
        },
        "sitl": {
            "commands_sent": 0,
            "command_kind": "shadow_policy_attitude_no_setpoint",
            "telemetry": {"messages_seen": 100, "race_statuses": 10},
        },
        "vision": {"frames_seen": 10, "detector_detections": 8},
        "policy_trace": {
            "inference_ticks": 2,
            "deployment_manifest": manifest,
            "samples": samples,
        },
    }

    result = verify.verify_shadow_report(
        report,
        prefix,
        model=FakeModel(),
        policy_callable_path=callable_path,
        additional_checkpoints={
            "gate4_checkpoint": gate4,
            "gate5_checkpoint": gate5,
            "gate6_checkpoint": gate6,
        },
        hybrid_prefix_confidence=True,
    )

    assert result["passed"] is True
    assert result["blockers"] == []


def test_verify_shadow_report_rejects_hybrid_confidence_in_tail(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in verify.LOCAL_MANIFEST_PATHS.items()
    }
    manifest["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": digest(checkpoint),
    }
    observation = [0.0] * 32
    observation[:4] = [0.1, -0.2, 0.3, -0.4]
    observation[10] = 1.0
    observation[23] = 0.5
    observation[27] = 1.0
    observation[30] = 0.25
    report = {
        "acceptance_passed": True,
        "control_mode": "policy-attitude",
        "control_inputs": {
            "policy_shadow_only": True,
            "arm_on_start": False,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "policy_shadow_cadence_probe": {
                "commands_counted": 60,
                "effective_command_hz": 60.0,
                "command_rate_violations": 0,
                "mavlink_setpoints_sent": 0,
            },
            "official_reset_start": {"reset_sent": False},
            "state_estimator": {"calibration": {"samples": 60}},
        },
        "sitl": {
            "commands_sent": 0,
            "command_kind": "shadow_policy_attitude_no_setpoint",
            "telemetry": {"messages_seen": 100, "race_statuses": 10},
        },
        "vision": {"frames_seen": 10, "detector_detections": 8},
        "policy_trace": {
            "inference_ticks": 1,
            "deployment_manifest": manifest,
            "samples": [{
                "official_active_gate_index": 3,
                "observation": observation,
                "normalized_action": observation[:4],
            }],
        },
    }

    result = verify.verify_shadow_report(
        report,
        checkpoint,
        model=FakeModel(),
        hybrid_prefix_confidence=True,
    )

    assert "tail_reserved_abi_mismatch_0" in result["blockers"]
