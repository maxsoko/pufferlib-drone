import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "drone_course_controller.py"
SPEC = importlib.util.spec_from_file_location("drone_course_controller", MODULE_PATH)
course = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = course
SPEC.loader.exec_module(course)


def pose(*, range_m=10.0, yaw=0.0, down=0.0, image_y=180.0):
    return SimpleNamespace(
        range_camera_m=range_m,
        yaw_error_rad=yaw,
        body_vector_ned_m=(range_m, range_m * yaw, down),
        image_center_px=(320.0, image_y),
    )


def guidance(*, heading=0.1, confidence=0.8):
    return course.GuidanceDetection(
        center_x_norm=0.2,
        heading_error_rad=heading,
        confidence=confidence,
        supporting_rows=30,
        pixel_count=500,
    )


def test_guidance_commands_small_forward_pitch_and_bounded_yaw():
    controller = course.CourseController()
    command = controller.update(
        now_s=1.0,
        measured_yaw_rad=0.2,
        official_gate_index=0,
        guidance=guidance(heading=0.5),
    )

    assert command.phase == "follow_guidance"
    assert command.pitch < 0.0
    assert command.yaw == pytest.approx(0.0)
    assert command.thrust == pytest.approx(controller.config.hover_thrust)


def test_guidance_loss_can_coast_straight_without_blind_yaw_search():
    controller = course.CourseController(
        config=course.CourseControllerConfig(guidance_lost_pitch_rad=-0.02)
    )
    controller.update(
        now_s=0.0,
        measured_yaw_rad=0.3,
        official_gate_index=0,
        guidance=guidance(heading=0.0),
    )

    command = controller.update(
        now_s=0.1,
        measured_yaw_rad=0.31,
        official_gate_index=0,
    )

    assert command.phase == "guidance_coast"
    assert command.pitch == pytest.approx(-0.02)
    assert command.yaw == pytest.approx(0.3)


def test_gate_right_of_camera_commands_negative_yaw_step():
    controller = course.CourseController()

    command = controller.update(
        now_s=0.0,
        measured_yaw_rad=0.2,
        official_gate_index=0,
        gate_pose=pose(range_m=12.0, yaw=0.1),
        gate_confidence=0.8,
    )

    assert command.phase == "align_gate"
    assert command.yaw < 0.2


def test_first_gate_can_disable_yaw_without_disabling_later_gates():
    controller = course.CourseController(
        config=course.CourseControllerConfig(
            gate_yaw_gain=0.2,
            first_gate_yaw_gain=0.0,
        )
    )
    controller.last_official_gate_index = 0
    assert controller._active_gate_yaw_gain() == pytest.approx(0.0)
    controller.last_official_gate_index = 1
    assert controller._active_gate_yaw_gain() == pytest.approx(0.2)


def test_first_gate_can_use_slower_approach_pitch():
    controller = course.CourseController(
        config=course.CourseControllerConfig(
            approach_pitch_rad=-0.04,
            first_gate_approach_pitch_rad=-0.02,
        )
    )
    controller.last_official_gate_index = 0
    assert controller._active_approach_pitch_rad() == pytest.approx(-0.02)
    controller.last_official_gate_index = 1
    assert controller._active_approach_pitch_rad() == pytest.approx(-0.04)


def test_later_gates_can_require_tighter_yaw_alignment():
    controller = course.CourseController(
        config=course.CourseControllerConfig(
            align_yaw_tolerance_rad=0.12,
            later_gate_align_yaw_tolerance_rad=0.05,
        )
    )
    controller.last_official_gate_index = 0
    assert controller._active_align_yaw_tolerance_rad() == pytest.approx(0.12)
    controller.last_official_gate_index = 1
    assert controller._active_align_yaw_tolerance_rad() == pytest.approx(0.05)


def test_later_gate_alignment_brakes_while_turning():
    controller = course.CourseController(
        config=course.CourseControllerConfig(
            later_gate_align_pitch_rad=0.04,
            later_gate_align_yaw_tolerance_rad=0.07,
        )
    )
    controller.last_official_gate_index = 1

    command = controller.update(
        now_s=0.0,
        measured_yaw_rad=0.0,
        official_gate_index=1,
        gate_pose=pose(range_m=20.0, yaw=0.15),
        gate_confidence=0.8,
    )

    assert command.phase == "align_gate"
    assert command.pitch == pytest.approx(0.04)


def test_later_gates_can_use_gentler_image_vertical_gain():
    controller = course.CourseController(
        config=course.CourseControllerConfig(
            vertical_image_kp=0.05,
            later_gate_vertical_image_kp=0.02,
        )
    )
    controller.last_official_gate_index = 0
    assert controller._active_vertical_image_kp() == pytest.approx(0.05)
    controller.last_official_gate_index = 1
    assert controller._active_vertical_image_kp() == pytest.approx(0.02)


def test_later_gate_roll_moves_toward_camera_right_gate():
    controller = course.CourseController(
        config=course.CourseControllerConfig(later_gate_roll_gain=0.06)
    )
    controller.last_official_gate_index = 0
    assert controller._active_gate_roll_rad(pose(image_y=180.0)) == pytest.approx(0.0)
    controller.last_official_gate_index = 1
    right_pose = pose()
    right_pose.image_center_px = (480.0, 180.0)
    assert controller._active_gate_roll_rad(right_pose) == pytest.approx(0.03)


def test_first_gate_can_reject_far_downstream_gate():
    controller = course.CourseController(
        config=course.CourseControllerConfig(
            gate_max_acquire_range_m=100.0,
            first_gate_max_acquire_range_m=35.0,
        )
    )
    controller.last_official_gate_index = 0
    assert controller._gate_is_usable(pose(range_m=75.0), 0.8) is False
    controller.last_official_gate_index = 1
    assert controller._gate_is_usable(pose(range_m=75.0), 0.8) is True


def test_gate_requires_alignment_before_approach_and_commit():
    config = course.CourseControllerConfig(align_consecutive_ticks=2, commit_range_m=4.0)
    controller = course.CourseController(config=config)

    first = controller.update(
        now_s=0.0,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=12.0, yaw=0.3),
        gate_confidence=0.8,
    )
    assert first.phase == "align_gate"
    assert first.pitch == pytest.approx(0.0)

    second = controller.update(
        now_s=0.02,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=12.0),
        gate_confidence=0.8,
    )
    third = controller.update(
        now_s=0.04,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=12.0),
        gate_confidence=0.8,
    )
    assert second.phase == "align_gate"
    assert third.phase == "approach_gate"
    assert third.pitch < 0.0

    commit = controller.update(
        now_s=0.06,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=3.8),
        gate_confidence=0.8,
    )
    assert commit.phase == "commit_gate"
    assert controller.phase == course.CoursePhase.COMMIT_GATE


def test_commit_continues_bounded_yaw_correction_while_gate_visible():
    config = course.CourseControllerConfig(
        align_consecutive_ticks=1,
        commit_range_m=4.0,
    )
    controller = course.CourseController(config=config)
    controller.update(
        now_s=0.0,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=3.0),
        gate_confidence=0.8,
    )

    command = controller.update(
        now_s=0.1,
        measured_yaw_rad=0.2,
        official_gate_index=0,
        gate_pose=pose(range_m=2.5, yaw=0.1),
        gate_confidence=0.8,
    )

    assert command.phase == "commit_gate"
    assert command.yaw < 0.2


def test_official_gate_advance_immediately_brakes_without_turning():
    controller = course.CourseController()
    controller.update(
        now_s=0.0,
        measured_yaw_rad=0.4,
        official_gate_index=0,
        guidance=guidance(),
    )

    brake = controller.update(
        now_s=1.0,
        measured_yaw_rad=0.45,
        official_gate_index=1,
        guidance=guidance(heading=-0.4),
    )

    assert brake.phase == "brake"
    assert brake.pitch > 0.0
    assert brake.yaw == pytest.approx(0.45)
    assert controller.phase == course.CoursePhase.BRAKE
    assert controller.transitions[-1]["reason"] == "official_gate_advanced"


def test_brake_settles_before_returning_to_guidance():
    config = course.CourseControllerConfig(brake_duration_s=1.0, brake_settle_s=0.5)
    controller = course.CourseController(config=config)
    controller.update(now_s=0.0, measured_yaw_rad=0.0, official_gate_index=0)
    controller.update(now_s=0.1, measured_yaw_rad=0.2, official_gate_index=1)

    braking = controller.update(now_s=0.9, measured_yaw_rad=0.3, official_gate_index=1)
    settling = controller.update(now_s=1.2, measured_yaw_rad=0.3, official_gate_index=1)
    following = controller.update(
        now_s=1.7,
        measured_yaw_rad=0.3,
        official_gate_index=1,
        guidance=guidance(heading=0.0),
    )

    assert braking.pitch > 0.0
    assert settling.pitch == pytest.approx(0.0)
    assert following.phase == "follow_guidance"


def test_commit_timeout_brakes_even_without_official_progress():
    config = course.CourseControllerConfig(
        align_consecutive_ticks=1,
        commit_range_m=4.0,
        commit_duration_s=0.5,
    )
    controller = course.CourseController(config=config)
    controller.update(
        now_s=0.0,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=3.0),
        gate_confidence=0.8,
    )
    assert controller.phase == course.CoursePhase.COMMIT_GATE

    command = controller.update(
        now_s=0.6,
        measured_yaw_rad=0.1,
        official_gate_index=0,
    )
    assert command.phase == "brake"
    assert command.pitch > 0.0
    assert controller.transitions[-1]["reason"] == "commit_timeout"


def test_vertical_thrust_filter_rejects_single_range_sign_flip():
    config = course.CourseControllerConfig(
        hover_thrust=0.27,
        min_thrust=0.18,
        max_thrust=0.42,
        vertical_kp=0.08,
        vertical_filter_alpha=0.1,
    )
    controller = course.CourseController(config=config)

    climb = controller._thrust_for_gate(pose(down=-2.0))
    after_single_flip = controller._thrust_for_gate(pose(down=2.0))

    assert climb == pytest.approx(0.42)
    assert after_single_flip > config.hover_thrust
    assert controller.filtered_vertical_error_m == pytest.approx(-1.6)


def test_fixed_approach_thrust_switches_to_hover_at_commit():
    config = course.CourseControllerConfig(
        align_consecutive_ticks=1,
        align_vertical_tolerance_m=10.0,
        commit_range_m=4.0,
        hover_thrust=0.27,
        approach_thrust=0.32,
    )
    controller = course.CourseController(config=config)

    approach = controller.update(
        now_s=0.0,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=6.0, down=-2.0),
        gate_confidence=0.8,
    )
    commit = controller.update(
        now_s=0.1,
        measured_yaw_rad=0.0,
        official_gate_index=0,
        gate_pose=pose(range_m=3.0, down=2.0),
        gate_confidence=0.8,
    )

    assert approach.phase == "approach_gate"
    assert approach.thrust == pytest.approx(0.32)
    assert commit.phase == "commit_gate"
    assert commit.thrust == pytest.approx(0.27)


def test_image_vertical_control_descends_toward_gate_below_camera():
    config = course.CourseControllerConfig(
        hover_thrust=0.27,
        approach_thrust=0.28,
        vertical_image_kp=0.05,
    )
    controller = course.CourseController(config=config)
    controller.phase = course.CoursePhase.APPROACH_GATE

    gate_below = controller._thrust_for_gate(pose(image_y=360.0))
    gate_above = controller._thrust_for_gate(pose(image_y=0.0))

    assert gate_below == pytest.approx(0.23)
    assert gate_above == pytest.approx(0.33)


def test_image_vertical_control_waits_until_gate_is_near():
    config = course.CourseControllerConfig(
        hover_thrust=0.27,
        approach_thrust=0.28,
        vertical_image_kp=0.05,
        vertical_image_control_range_m=20.0,
    )
    controller = course.CourseController(config=config)
    controller.phase = course.CoursePhase.APPROACH_GATE

    far_gate = controller._thrust_for_gate(pose(range_m=75.0, image_y=360.0))
    near_gate = controller._thrust_for_gate(pose(range_m=15.0, image_y=360.0))

    assert far_gate == pytest.approx(0.28)
    assert near_gate == pytest.approx(0.23)


def test_far_gate_can_use_hover_instead_of_climb_bias():
    config = course.CourseControllerConfig(
        hover_thrust=0.27,
        approach_thrust=0.28,
        vertical_image_kp=0.05,
        vertical_image_control_range_m=20.0,
        far_gate_thrust=0.27,
    )
    controller = course.CourseController(config=config)
    controller.phase = course.CoursePhase.APPROACH_GATE

    thrust = controller._thrust_for_gate(pose(range_m=75.0, image_y=360.0))

    assert thrust == pytest.approx(0.27)


@pytest.mark.skipif(course.cv2 is None or course.np is None, reason="opencv/numpy unavailable")
def test_cyan_guidance_detector_finds_synthetic_corridor():
    image = course.np.zeros((360, 640, 3), dtype=course.np.uint8)
    course.cv2.line(image, (100, 359), (300, 100), (255, 255, 0), 8)
    course.cv2.line(image, (540, 359), (340, 100), (255, 255, 0), 8)

    detector = course.CyanGuidanceDetector()
    detection = detector.detect_bgr(image)

    assert detection is not None
    assert abs(detection.center_x_norm) < 0.08
    assert abs(detection.heading_error_rad) < 0.08
    assert detection.confidence > 0.0
