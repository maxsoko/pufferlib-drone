from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_full_policy_teacher.c"


def test_phase_reset_tail_is_default_off_and_resets_each_selected_policy():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert 'env_int("PHASE_RESET_TAIL", 0)' in source
    assert 'getenv("PHASE_RESET_GATE4_CHECKPOINT")' in source
    assert 'getenv("PHASE_RESET_GATE5_CHECKPOINT")' in source
    assert "agent->current_gate != phase_reset_previous_gate" in source
    assert "memset(selected->mingru->state, 0" in source
    assert "student probability 1" in source


def test_phase_transition_trace_is_default_off_and_reports_full_entry_state():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert 'getenv("PHASE_TRANSITION_TRACE") != NULL' in source
    assert '"phase_transition from=%d to=%d step=%d "' in source
    assert '"pos=(%.9g,%.9g,%.9g) vel=(%.9g,%.9g,%.9g) "' in source
    assert '"quat=(%.9g,%.9g,%.9g,%.9g) "' in source
    assert '"euler=(%.9g,%.9g,%.9g) omega=(%.9g,%.9g,%.9g)\\n"' in source


def test_phase_action_roll_bias_is_default_off_scoped_and_clamped():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert 'env_int("PHASE_ACTION_BIAS_GATE", -1)' in source
    assert 'env_float("PHASE_ACTION_PITCH_BIAS", 0.0f)' in source
    assert 'env_float("PHASE_ACTION_ROLL_BIAS", 0.0f)' in source
    assert 'env_float("PHASE_ACTION_THRUST_BIAS", 0.0f)' in source
    assert "agent->current_gate == phase_action_bias_gate" in source
    assert "executed[0] + phase_action_pitch_bias, -1.0f, 1.0f" in source
    assert "executed[1] + phase_action_roll_bias, -1.0f, 1.0f" in source
    assert "executed[2] + phase_action_thrust_bias, -1.0f, 1.0f" in source


def test_final_phase_action_bias_is_independent_default_off_and_clamped():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_ACTION_PITCH_BIAS", 0.0f' in source
    assert '"PHASE_FINAL_ACTION_THRUST_BIAS", 0.0f' in source
    assert "agent->current_gate == env.num_gates - 1" in source
    assert "executed[0] + phase_final_action_pitch_bias" in source
    assert "executed[2] + phase_final_action_thrust_bias" in source


def test_final_observable_pd_is_default_off_visible_and_uses_only_policy_observation():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_OBSERVABLE_ROLL_KP", 0.0f' in source
    assert '"PHASE_FINAL_OBSERVABLE_ROLL_KD", 0.0f' in source
    assert '"PHASE_FINAL_OBSERVABLE_ROLL_KI", 0.0f' in source
    assert '"PHASE_FINAL_OBSERVABLE_ROLL_INTEGRAL_LIMIT", 5.0f' in source
    assert '"PHASE_FINAL_OBSERVABLE_THRUST_KP", 0.0f' in source
    assert '"PHASE_FINAL_OBSERVABLE_THRUST_KD", 0.0f' in source
    assert '"PHASE_FINAL_OBSERVABLE_ROLL_OVERRIDE", 0' in source
    assert "if (env.observations[10] > 0.5f)" in source
    assert "env.observations[12], 0.2f" in source
    assert "env.observations[13], 0.2f" in source
    assert "env.observations[1], 0.3333333333f" in source
    assert "env.observations[2], 0.3333333333f" in source
    assert "phase_final_observable_roll_integral" in source
    assert "+ observable_right * env.dt" in source
    assert "* phase_final_observable_roll_integral" in source
    assert "phase_final_observable_roll_override" in source
    assert ": executed[1] + roll_residual" in source
    assert "executed[2] + thrust_residual, -1.0f, 1.0f" in source


def test_final_observable_intercept_is_default_off_and_uses_observation_only():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_OBSERVABLE_INTERCEPT_ROLL", 0' in source
    assert '"PHASE_FINAL_INTERCEPT_TIME_FLOOR_S", 0.5f' in source
    assert '"PHASE_FINAL_INTERCEPT_FORWARD_SPEED_FLOOR_M_S", 4.1875f' in source
    assert '"PHASE_FINAL_INTERCEPT_LATERAL_ACCEL_SCALE", 1.0f' in source
    assert "observable_unscale_tanh(env.observations[11], 0.1f)" in source
    assert "observable_unscale_tanh(env.observations[0], 0.2f)" in source
    assert "observable_relative_world.y" in source
    assert "observable_velocity_world.y * time_to_plane" in source
    assert "phase_final_intercept_lateral_accel_scale * 2.0f" in source
    assert "desired_roll / env.sitl_max_roll_rad" in source
    assert '"PHASE_FINAL_OBSERVABLE_INTERCEPT_THRUST", 0' in source
    assert '"PHASE_FINAL_INTERCEPT_VERTICAL_ACCEL_SCALE", 1.0f' in source
    assert "observable_relative_world.z" in source
    assert "observable_velocity_world.z * time_to_plane" in source
    assert "env.sitl_vertical_accel_per_thrust" in source


def test_final_lateral_brake_is_default_off_visible_and_observation_scoped():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_LATERAL_BRAKE_PITCH_BIAS", 0.0f' in source
    assert '"PHASE_FINAL_LATERAL_BRAKE_RELEASE_RIGHT_M", 0.0f' in source
    assert "fabsf(observable_right)" in source
    assert "phase_final_lateral_brake_release_right_m" in source
    assert "executed[0] + phase_final_lateral_brake_pitch_bias" in source


def test_final_projected_miss_brake_is_default_off_and_observable():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_PROJECTED_MISS_BRAKE_PITCH_BIAS", 0.0f' in source
    assert '"PHASE_FINAL_PROJECTED_MISS_BRAKE_THRESHOLD_M", 0.75f' in source
    assert "observable_forward_rate" in source
    assert "observable_right_rate * projected_time_to_plane" in source
    assert "projected_right_at_plane" in source
    assert "phase_final_projected_miss_brake_pitch_bias" in source


def test_final_observable_course_controller_preserves_policy_pitch_and_yaw():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_OBSERVABLE_COURSE", 0' in source
    assert "phase_final_observable_course_controller" in source
    assert "use_full_course_origin ? 0.0f" in source
    assert "!phase_final_observable_course_active" in source
    assert "memcpy(course_action, executed, sizeof(course_action))" in source
    assert "executed[1] = course_action[1]" in source
    assert "executed[2] = course_action[2]" in source
    assert "executed[0] = course_action[0]" not in source
    assert "executed[3] = course_action[3]" not in source


def test_observable_course_teacher_can_use_full_course_origin():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert '"TEACHER_OBSERVABLE_COURSE_FULL_ORIGIN", 0' in source
    assert "teacher_observable_course_full_origin))" in source


def test_observable_course_teacher_can_own_pitch_from_observable_speed():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"TEACHER_OBSERVABLE_PITCH_SPEED_CONTROL", 0' in source
    assert '"TEACHER_OBSERVABLE_PITCH_SPEED_TARGET_M_S", 4.2f' in source
    assert "observable_course_teacher.debug_velocity_world.x" in source


def test_final_bang_bang_roll_is_default_off_and_camera_forward_scoped():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_BANG_BANG_ROLL", 0' in source
    assert '"PHASE_FINAL_ROLL_SWITCH_FORWARD_M", 0.0f' in source
    assert '"PHASE_FINAL_ROLL_ACCEL_ACTION", -1.0f' in source
    assert '"PHASE_FINAL_ROLL_BRAKE_ACTION", 0.0f' in source
    assert '"PHASE_FINAL_BANG_BANG_ROLL_PD_RESIDUAL", 0' in source
    assert '"PHASE_FINAL_BANG_BANG_ROLL_DIRECTIONAL", 0' in source
    assert "env.observations[11], 0.1f" in source
    assert "bang_bang_forward > phase_final_roll_switch_forward_m" in source
    assert "bang_bang_roll += roll_residual" in source
    assert "observable_right >= 0.0f ? 1.0f : -1.0f" in source


def test_final_deterministic_attitude_is_default_off_and_clamped():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_DETERMINISTIC_ATTITUDE", 0' in source
    assert '"PHASE_FINAL_DETERMINISTIC_PITCH_ACTION", 0.15f' in source
    assert '"PHASE_FINAL_DETERMINISTIC_YAW_ACTION", 0.0f' in source
    assert "phase_final_deterministic_pitch_action, -1.0f, 1.0f" in source
    assert "phase_final_deterministic_yaw_action, -1.0f, 1.0f" in source


def test_final_observable_pitch_speed_control_is_default_off_and_rate_based():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"PHASE_FINAL_OBSERVABLE_PITCH_SPEED_CONTROL", 0' in source
    assert '"PHASE_FINAL_OBSERVABLE_PITCH_SPEED_TARGET_M_S", 5.5f' in source
    assert "-observable_forward_rate" in source
    assert "phase_final_observable_pitch_speed_target_m_s" in source
    assert "* 0.35f" in source


def test_counter_roll_is_observable_latched_phase_scoped_and_default_off():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert 'env_int("PHASE_COUNTER_ROLL_GATE", -1)' in source
    assert '"PHASE_COUNTER_ROLL_FORWARD_M", 0.0f' in source
    assert '"PHASE_COUNTER_ROLL_ACTION", 0.0f' in source
    assert "env.observations[10] > 0.5f" in source
    assert "observable_unscale_tanh(\n                    env.observations[11], 0.1f)" in source
    assert "phase_counter_roll_latched = 1" in source
    assert "phase_counter_roll_latched = 0" in source


def test_collector_reports_conditioned_terminal_crossing_geometry():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert '"terminal_crossings=%.0f terminal_crossing_radial=%.6f "' in source
    assert '"terminal_crossing_right=%.6f terminal_crossing_vertical=%.6f\\n"' in source
    assert source.count("/ env.log.terminal_crossing_sampled") == 3
    assert '"terminal_gate=%d sampled=%.0f radial=%.6f right=%.6f vertical=%.6f\\n"' in source
    assert "env.log.terminal_gate_radial[gate] / sampled" in source


def test_collector_defaults_match_the_phase32_v3385_training_contract():
    source = COLLECTOR.read_text(encoding="utf-8")

    required_defaults = (
        'env_int("OBSERVABLE_GATE_INDEX", 1)',
        'env_int("OBSERVABLE_GATE_PROGRESS", 1)',
        'env.start_gate_index = env_int("START_GATE_INDEX", 0);',
        '"START_ELAPSED_TIME_JITTER", 0.0f',
        'env.custom_start_pos_jitter[0] = env_float("START_X_JITTER", 0.0f);',
        'env.custom_start_vel_jitter[0] = env_float("START_VX_JITTER", 0.0f);',
        '"START_ROLL_JITTER_RAD", 0.0f',
        'env.custom_start_omega_jitter[0] = env_float("START_WX_JITTER", 0.0f);',
        'env.custom_start_vel[2] = env_float("START_VZ", 0.0f);',
        '? 2.60794f : strtof(horizontal_scale, NULL)',
        '? 2.60794f : strtof(braking_scale, NULL)',
        '? 0.107491f : strtof(lateral_scale, NULL)',
        '? 10.0f : strtof(transition_speed, NULL)',
        '"SITL_VERTICAL_ACCEL_PER_THRUST", 32.81f',
        'env_float("SITL_GRAVITY_M_S2", 8.69f)',
        'env.crash_height = -25.0f;',
        'env.num_gates > 4 ? 220.0f : 140.0f',
        'env.safety_altitude = -23.0f;',
        '"SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS", 1',
        '"SITL_GATE_MOTION_PREDICT_DROPOUT", 0',
        '"SITL_GATE_MOTION_CONTROL_ACCEL_GAIN", 0.0f',
    )
    for expected in required_defaults:
        assert expected in source


def test_collector_spline_can_use_full_course_origin_for_segment_dagger():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert '"TEACHER_SPLINE_FULL_COURSE_ORIGIN", 0' in source
    assert "spline_x[0] = env.gates[0].pos.x - env.start_offset;" in source
    assert "spline_y[0] = 0.0f;" in source
    assert "spline_z[0] = 0.0f;" in source


def test_collector_supports_six_gate_progress_curricula():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert "if (env.num_gates > 4) env.num_gates = 4;" not in source
    assert "env.num_gates > DRONE_RACE_MAX_GATES" in source
    assert '"GATE%d_%c"' in source
    assert "{148.0f, 4.0f, -17.0f}" in source
    assert "env.observable_gate_progress" in source
    assert 'env_int(\n        "OBSERVABLE_GATE_PHASE_ONEHOT", 0)' in source


def test_collector_loads_the_declared_checkpoint_layout_directly():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert 'env_int(\n        "CHECKPOINT_LAYOUT_PRECISION_BYTES", 2)' in source
    assert "load_weights_with_alignment(" in source
    assert "checkpoint_alignment_elements" in source


def test_policy_feedback_can_be_scoped_to_one_gate():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert '"TEACHER_FEEDBACK_UNTIL_GATE", env.num_gates' in source
    assert source.count(
        "agent->current_gate < teacher_feedback_until_gate"
    ) == 2


def test_policy_action_bias_adapter_can_be_scoped_to_one_gate():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert '"TEACHER_POST_ACTION_OFFSET_FROM_GATE"' in source
    assert '"TEACHER_POST_ACTION_OFFSET_UNTIL_GATE"' in source
    assert '"TEACHER_POST_THRUST_OFFSET_FROM_GATE", env.num_gates' in source
    assert '"TEACHER_POST_THRUST_OFFSET_UNTIL_GATE", env.num_gates' in source
    assert '"TEACHER_POST_ROLL_OFFSET", 0.0f' in source
    assert '"TEACHER_POST_THRUST_OFFSET", 0.0f' in source
    assert '"TEACHER_GATE%d_%s_BIAS"' in source
    assert '"TEACHER_GATE%d_BIAS_VISIBLE_ONLY"' in source
    assert '"TEACHER_GATE%d_BIAS_MIN_ALIGNMENT"' in source
    assert '"TEACHER_GATE%d_BIAS_MIN_FORWARD_M"' in source
    assert "teacher_gate_action_bias[agent->current_gate][action]" in source
    assert "env.observations[10] > 0.5f" in source
    assert "env.observations[17]" in source
    assert "env.observations[11]" in source


def test_collector_can_keep_successful_executed_action_episodes_for_elite_imitation():
    source = COLLECTOR.read_text(encoding="utf-8")

    assert 'env_int("TEACHER_RECORD_EXECUTED_ACTION", 0)' in source
    assert '"TEACHER_KEEP_SUCCESSFUL_EPISODES_ONLY", 0' in source
    assert 'env_int("TEACHER_ACTION_NOISE_FROM_GATE", 0)' in source
    assert "record_executed_action ? executed : teacher" in source
    assert "env.log.success_rate > successes_before_step" in source
    assert "keep_successful_episodes_only && episode_success" in source
    assert '"kept_episodes=%d student_fraction=%.6f "' in source
