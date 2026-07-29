#define DRONE_RACE_BINDING
#include <stdio.h>
#include "drone_race.c"

#define OBS_SIZE DRONE_RACE_OBS_SIZE
#define NUM_ATNS DRONE_RACE_NUM_ATNS
#define ACT_SIZES {1, 1, 1, 1}
#ifndef OBS_TENSOR_T
#define OBS_TENSOR_T FloatTensor
#endif
// Keep headroom for diagnostic-only additions; vecenv grows this dictionary,
// but a realistic initial capacity avoids churn in every native evaluation.
#define ENV_LOG_DICT_CAPACITY 160

#define Env DroneRace
#include "vecenv.h"

static int get_int(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (int)item->value;
}

static float get_float(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (float)item->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = get_int(kwargs, "num_drones", 64);
    env->dt = get_float(kwargs, "dt", 0.02f);
    env->num_gates = get_int(kwargs, "num_gates", 4);
    env->num_gates_per_env_randomize = get_int(
        kwargs, "num_gates_per_env_randomize", 0);
    env->num_gates_per_env_min = get_int(
        kwargs, "num_gates_per_env_min", env->num_gates);
    env->num_gates_per_env_max = get_int(
        kwargs, "num_gates_per_env_max", env->num_gates);
    env->num_gates_per_env_seed = (unsigned int)get_int(
        kwargs, "num_gates_per_env_seed", 0);
    env->num_gates = select_num_gates_for_env(
        env->num_gates,
        env->num_gates_per_env_randomize,
        env->num_gates_per_env_min,
        env->num_gates_per_env_max,
        env->rng,
        env->num_gates_per_env_seed);
    env->gate_spacing = get_float(kwargs, "gate_spacing", 5.0f);
    env->gate_radius = get_float(kwargs, "gate_radius", 1.0f);
    env->gate_radius_randomize = get_int(kwargs, "gate_radius_randomize", 0);
    env->gate_radius_randomize_gate_index = get_int(
        kwargs, "gate_radius_randomize_gate_index", -1);
    env->gate_radius_min = get_float(kwargs, "gate_radius_min", env->gate_radius);
    env->gate_radius_max = get_float(kwargs, "gate_radius_max", env->gate_radius);
    env->gate_radius_profile_mix = get_int(
        kwargs, "gate_radius_profile_mix", 0);
    env->gate_radius_profile_mix_probability = get_float(
        kwargs, "gate_radius_profile_mix_probability", 0.5f);
    env->gate_radius_profile_mix_target = get_float(
        kwargs, "gate_radius_profile_mix_target", env->gate_radius);
    env->gate_altitude = get_float(kwargs, "gate_altitude", 1.0f);
    env->gate_lateral_amplitude = get_float(kwargs, "gate_lateral_amplitude", 1.5f);
    env->course_geometry_scale = get_float(kwargs, "course_geometry_scale", 1.0f);
    env->course_geometry_scale_randomize = get_int(
        kwargs, "course_geometry_scale_randomize", 0);
    env->course_geometry_scale_min = get_float(
        kwargs, "course_geometry_scale_min", 0.0f);
    env->course_geometry_scale_max = get_float(
        kwargs, "course_geometry_scale_max", 1.0f);
    env->start_offset = get_float(kwargs, "start_offset", 2.0f);
    env->crash_height = get_float(kwargs, "crash_height", 0.0f);
    env->pos_bound = get_float(kwargs, "pos_bound", 20.0f);
    env->plane_cross_tolerance = get_float(kwargs, "plane_cross_tolerance", 1e-3f);
    env->miss_tolerance = get_float(kwargs, "miss_tolerance", 0.5f);
    env->direction_min = get_float(kwargs, "direction_min", 0.05f);
    env->strict_missed_gate = get_int(kwargs, "strict_missed_gate", 1);
    env->max_steps = get_int(kwargs, "max_steps", 1500);
    env->time_limit_seconds = get_float(kwargs, "time_limit_seconds", 30.0f);
    env->safety_altitude = get_float(kwargs, "safety_altitude", 0.0f);
    env->w_progress = get_float(kwargs, "w_progress", 20.0f);
    env->w_gate = get_float(kwargs, "w_gate", 3.0f);
    env->w_ordered_gate = get_float(kwargs, "w_ordered_gate", 0.0f);
    env->w_finish = get_float(kwargs, "w_finish", 30.0f);
    env->w_time = get_float(kwargs, "w_time", 1.0f);
    env->w_ctrl = get_float(kwargs, "w_ctrl", 0.01f);
    env->w_body_rate = get_float(kwargs, "w_body_rate", 0.0f);
    env->w_cross_track = get_float(kwargs, "w_cross_track", 0.0f);
    env->cross_track_vertical_weight = get_float(
        kwargs, "cross_track_vertical_weight", 1.0f);
    env->cross_track_from_gate_index = get_int(
        kwargs, "cross_track_from_gate_index", 0);
    env->w_gate_camera_alignment = get_float(
        kwargs, "w_gate_camera_alignment", 0.0f);
    env->gate_camera_alignment_from_gate_index = get_int(
        kwargs, "gate_camera_alignment_from_gate_index", 0);
    env->w_gate_crossing_error = get_float(
        kwargs, "w_gate_crossing_error", 0.0f);
    env->gate_crossing_error_from_gate_index = get_int(
        kwargs, "gate_crossing_error_from_gate_index", 0);
    env->w_gate_exit_lateral_velocity = get_float(
        kwargs, "w_gate_exit_lateral_velocity", 0.0f);
    env->w_gate_exit_velocity = get_float(
        kwargs, "w_gate_exit_velocity", 0.0f);
    env->gate_exit_target_forward_speed = get_float(
        kwargs, "gate_exit_target_forward_speed", 10.0f);
    env->gate_exit_reward_lookahead_m = get_float(
        kwargs, "gate_exit_reward_lookahead_m", 10.0f);
    env->gate_exit_reward_skip_randomized_next_gate = get_int(
        kwargs, "gate_exit_reward_skip_randomized_next_gate", 0);
    env->w_forward_speed_excess = get_float(
        kwargs, "w_forward_speed_excess", 0.0f);
    env->target_forward_speed = get_float(
        kwargs, "target_forward_speed", 10.0f);
    env->w_altitude_floor = get_float(kwargs, "w_altitude_floor", 0.0f);
    env->w_descent_floor = get_float(kwargs, "w_descent_floor", 0.0f);
    env->w_action_teacher = get_float(kwargs, "w_action_teacher", 0.0f);
    env->teacher_action_blend = get_float(kwargs, "teacher_action_blend", 0.0f);
    env->teacher_action_blend_from_step = get_int(
        kwargs, "teacher_action_blend_from_step", 0);
    env->teacher_course_spline = get_int(kwargs, "teacher_course_spline", 0);
    env->teacher_segment_minimum_jerk = get_int(
        kwargs, "teacher_segment_minimum_jerk", 0);
    env->teacher_alignment_governor = get_int(
        kwargs, "teacher_alignment_governor", 0);
    env->teacher_from_gate_index = get_int(kwargs, "teacher_from_gate_index", 0);
    env->teacher_pitch_from_gate_index = get_int(
        kwargs, "teacher_pitch_from_gate_index", env->teacher_from_gate_index);
    env->teacher_roll_from_gate_index = get_int(
        kwargs, "teacher_roll_from_gate_index", env->teacher_from_gate_index);
    env->teacher_roll_until_gate_index = get_int(
        kwargs, "teacher_roll_until_gate_index", DRONE_RACE_MAX_GATES);
    env->teacher_thrust_from_gate_index = get_int(
        kwargs, "teacher_thrust_from_gate_index", env->teacher_from_gate_index);
    env->teacher_yaw_control = get_int(kwargs, "teacher_yaw_control", 0);
    env->teacher_yaw_from_gate_index = get_int(
        kwargs, "teacher_yaw_from_gate_index", env->teacher_from_gate_index);
    env->teacher_pitch_action = get_float(kwargs, "teacher_pitch_action", -0.5f);
    env->teacher_yaw_action = get_float(kwargs, "teacher_yaw_action", 0.0f);
    env->teacher_pitch_speed_control = get_int(
        kwargs, "teacher_pitch_speed_control", 0);
    env->teacher_pitch_speed_target_m_s = get_float(
        kwargs, "teacher_pitch_speed_target_m_s", 2.2f);
    env->teacher_pitch_speed_gain = get_float(
        kwargs, "teacher_pitch_speed_gain", 0.35f);
    env->teacher_pitch_speed_scale = get_float(
        kwargs, "teacher_pitch_speed_scale", 0.24f);
    env->teacher_roll_bias = get_float(kwargs, "teacher_roll_bias", 0.0f);
    env->teacher_roll_per_m = get_float(kwargs, "teacher_roll_per_m", 0.5f);
    env->teacher_roll_rate_per_m_s = get_float(
        kwargs, "teacher_roll_rate_per_m_s", 0.0f);
    env->teacher_true_velocity_damping = get_int(
        kwargs, "teacher_true_velocity_damping", 0);
    env->teacher_thrust_bias = get_float(kwargs, "teacher_thrust_bias", 0.0f);
    env->teacher_thrust_per_m = get_float(kwargs, "teacher_thrust_per_m", 0.0f);
    env->teacher_thrust_rate_per_m_s = get_float(
        kwargs, "teacher_thrust_rate_per_m_s", 0.0f);
    env->teacher_thrust_world_frame = get_int(
        kwargs, "teacher_thrust_world_frame", 0);
    env->teacher_pitch_weight = get_float(kwargs, "teacher_pitch_weight", 1.0f);
    env->teacher_roll_weight = get_float(kwargs, "teacher_roll_weight", 1.0f);
    env->teacher_thrust_weight = get_float(kwargs, "teacher_thrust_weight", 1.0f);
    env->teacher_yaw_weight = get_float(kwargs, "teacher_yaw_weight", 1.0f);
    env->invalid_penalty = get_float(kwargs, "invalid_penalty", 40.0f);
    env->late_invalid_penalty = get_float(
        kwargs, "late_invalid_penalty", env->invalid_penalty);
    env->late_invalid_penalty_from_gate_index = get_int(
        kwargs, "late_invalid_penalty_from_gate_index", DRONE_RACE_MAX_GATES);
    env->interface_mode = get_int(kwargs, "interface_mode", DRONE_RACE_INTERFACE_NATIVE_MOTOR);
    env->visual_observation = get_int(
        kwargs,
        "visual_observation",
        DRONE_RACE_OBS_SIZE == DRONE_RACE_VISUAL_OBS_SIZE ? 1 : 0);
    if (env->visual_observation
            && DRONE_RACE_OBS_SIZE != DRONE_RACE_VISUAL_OBS_SIZE) {
        fprintf(
            stderr,
            "visual_observation requires the drone_race_vision binding "
            "(%d floats, current binding has %d)\n",
            DRONE_RACE_VISUAL_OBS_SIZE,
            DRONE_RACE_OBS_SIZE);
        abort();
    }
    env->visual_camera_hz = get_float(kwargs, "visual_camera_hz", 30.0f);
    env->visual_focal_x_px = get_float(kwargs, "visual_focal_x_px", 32.0f);
    env->visual_focal_y_px = get_float(kwargs, "visual_focal_y_px", 32.0f);
    env->visual_camera_roll_rad = get_float(
        kwargs, "visual_camera_roll_rad", 0.0f);
    env->visual_camera_pitch_rad = get_float(
        kwargs, "visual_camera_pitch_rad", 0.0f);
    env->visual_camera_yaw_rad = get_float(
        kwargs, "visual_camera_yaw_rad", 0.0f);
    env->visual_camera_roll_jitter_rad = get_float(
        kwargs, "visual_camera_roll_jitter_rad", 0.0f);
    env->visual_camera_pitch_jitter_rad = get_float(
        kwargs, "visual_camera_pitch_jitter_rad", 0.0f);
    env->visual_camera_yaw_jitter_rad = get_float(
        kwargs, "visual_camera_yaw_jitter_rad", 0.0f);
    env->visual_camera_dropout_prob = get_float(
        kwargs, "visual_camera_dropout_prob", 0.0f);
    env->visual_edge_dropout_prob = get_float(
        kwargs, "visual_edge_dropout_prob", 0.0f);
    env->visual_edge_corrupt_prob = get_float(
        kwargs, "visual_edge_corrupt_prob", 0.10f);
    env->visual_false_segments = get_int(
        kwargs, "visual_false_segments", 0);
    env->visual_line_thickness_px = get_float(
        kwargs, "visual_line_thickness_px", 1.0f);
    env->visual_rolling_shutter_s = get_float(
        kwargs, "visual_rolling_shutter_s", 0.0f);
    env->max_cmd_forward = get_float(kwargs, "max_cmd_forward", 2.0f);
    env->max_cmd_lateral = get_float(kwargs, "max_cmd_lateral", 1.0f);
    env->max_cmd_vertical = get_float(kwargs, "max_cmd_vertical", 0.8f);
    env->max_cmd_yaw_rate = get_float(kwargs, "max_cmd_yaw_rate", 1.0f);
    env->setpoint_vel_kp = get_float(kwargs, "setpoint_vel_kp", 0.20f);
    env->setpoint_att_kp = get_float(kwargs, "setpoint_att_kp", 0.20f);
    env->setpoint_rate_kd = get_float(kwargs, "setpoint_rate_kd", 0.04f);
    env->setpoint_yaw_rate_kp = get_float(kwargs, "setpoint_yaw_rate_kp", 0.08f);
    env->setpoint_thrust_vel_kp = get_float(kwargs, "setpoint_thrust_vel_kp", 0.20f);
    env->setpoint_max_tilt = get_float(kwargs, "setpoint_max_tilt", 0.35f);
    env->setpoint_max_motor_delta = get_float(kwargs, "setpoint_max_motor_delta", 0.35f);
    env->telemetry_velocity_response_tau_s = get_float(
        kwargs, "telemetry_velocity_response_tau_s", 0.25f);
    env->telemetry_velocity_max_accel_m_s2 = get_float(
        kwargs, "telemetry_velocity_max_accel_m_s2", 8.0f);
    env->telemetry_velocity_teacher_speed_m_s = get_float(
        kwargs, "telemetry_velocity_teacher_speed_m_s", 8.0f);
    env->sitl_rate_gain_roll = get_float(kwargs, "sitl_rate_gain_roll", 0.0f);
    env->sitl_rate_gain_pitch = get_float(kwargs, "sitl_rate_gain_pitch", 0.0f);
    env->sitl_rate_gain_yaw = get_float(kwargs, "sitl_rate_gain_yaw", 0.0f);
    env->sitl_hover_thrust = get_float(kwargs, "sitl_hover_thrust", 0.0f);
    env->sitl_min_thrust = get_float(kwargs, "sitl_min_thrust", 0.0f);
    env->sitl_max_thrust = get_float(kwargs, "sitl_max_thrust", 0.0f);
    env->sitl_vertical_accel_per_thrust = get_float(kwargs, "sitl_vertical_accel_per_thrust", 0.0f);
    env->sitl_horizontal_accel_scale = get_float(kwargs, "sitl_horizontal_accel_scale", 1.0f);
    env->sitl_braking_accel_scale = get_float(kwargs, "sitl_braking_accel_scale", 1.0f);
    env->sitl_lateral_accel_scale = get_float(kwargs, "sitl_lateral_accel_scale", 1.0f);
    env->sitl_gate_transition_min_forward_speed = get_float(
        kwargs, "sitl_gate_transition_min_forward_speed", 0.0f);
    env->sitl_gate_transition_min_forward_speed_randomize = get_int(
        kwargs, "sitl_gate_transition_min_forward_speed_randomize", 0);
    env->sitl_gate_transition_min_forward_speed_min = get_float(
        kwargs,
        "sitl_gate_transition_min_forward_speed_min",
        env->sitl_gate_transition_min_forward_speed);
    env->sitl_gate_transition_min_forward_speed_max = get_float(
        kwargs,
        "sitl_gate_transition_min_forward_speed_max",
        env->sitl_gate_transition_min_forward_speed);
    env->sitl_gate_transition_from_gate_index = get_int(
        kwargs, "sitl_gate_transition_from_gate_index", 1);
    env->sitl_gate_transition_max_accel_m_s2 = get_float(
        kwargs, "sitl_gate_transition_max_accel_m_s2", 0.0f);
    env->sitl_gate_transition_preserve_velocity_direction = get_int(
        kwargs, "sitl_gate_transition_preserve_velocity_direction", 0);
    env->sitl_gate_transition_delay_steps = get_int(
        kwargs, "sitl_gate_transition_delay_steps", 0);
    env->sitl_gravity_m_s2 = get_float(kwargs, "sitl_gravity_m_s2", 0.0f);
    env->sitl_linear_drag = get_float(kwargs, "sitl_linear_drag", 0.0f);
    env->sitl_rate_lag_tau_s = get_float(kwargs, "sitl_rate_lag_tau_s", 0.0f);
    env->sitl_max_roll_rad = get_float(kwargs, "sitl_max_roll_rad", 0.0f);
    env->sitl_max_pitch_rad = get_float(kwargs, "sitl_max_pitch_rad", 0.0f);
    env->sitl_max_yaw_rad = get_float(kwargs, "sitl_max_yaw_rad", 0.0f);
    env->sitl_lock_hover_thrust = get_int(kwargs, "sitl_lock_hover_thrust", 0);
    env->sitl_lock_roll = get_int(kwargs, "sitl_lock_roll", 0);
    env->sitl_lock_yaw = get_int(kwargs, "sitl_lock_yaw", 0);
    env->sitl_attitude_kp = get_float(kwargs, "sitl_attitude_kp", 0.0f);
    env->sitl_body_rate_min_command_rad_s = get_float(
        kwargs, "sitl_body_rate_min_command_rad_s", 0.0f);
    env->sitl_attitude_error_deadband_rad = get_float(
        kwargs, "sitl_attitude_error_deadband_rad", 0.0f);
    env->sitl_max_body_rate_command_rad_s = get_float(
        kwargs, "sitl_max_body_rate_command_rad_s", 0.0f);
    env->sitl_plant_domain_randomize = get_int(kwargs, "sitl_plant_domain_randomize", 0);
    env->sitl_rate_gain_jitter_frac = get_float(kwargs, "sitl_rate_gain_jitter_frac", 0.0f);
    env->sitl_hover_thrust_jitter = get_float(kwargs, "sitl_hover_thrust_jitter", 0.0f);
    env->sitl_rate_lag_jitter_frac = get_float(kwargs, "sitl_rate_lag_jitter_frac", 0.0f);
    env->sitl_linear_drag_jitter_frac = get_float(kwargs, "sitl_linear_drag_jitter_frac", 0.0f);
    env->sitl_gate_obs_dropout_prob = get_float(kwargs, "sitl_gate_obs_dropout_prob", 0.0f);
    env->sitl_gate_obs_dropout_range_m = get_float(
        kwargs, "sitl_gate_obs_dropout_range_m", 0.0f);
    env->sitl_gate_obs_dropout_from_index = get_int(
        kwargs, "sitl_gate_obs_dropout_from_index", DRONE_RACE_MAX_GATES);
    env->sitl_gate_obs_sample_interval_steps = get_int(
        kwargs, "sitl_gate_obs_sample_interval_steps", 1);
    env->sitl_gate_motion_predict_dropout = get_int(
        kwargs, "sitl_gate_motion_predict_dropout", 0);
    env->sitl_gate_motion_control_accel_gain = get_float(
        kwargs, "sitl_gate_motion_control_accel_gain", 0.0f);
    env->sitl_gate_motion_initial_vx = get_float(
        kwargs, "sitl_gate_motion_initial_vx", 0.0f);
    env->sitl_gate_motion_initial_vy = get_float(
        kwargs, "sitl_gate_motion_initial_vy", 0.0f);
    env->sitl_gate_motion_initial_vz = get_float(
        kwargs, "sitl_gate_motion_initial_vz", 0.0f);
    env->sitl_gate_range_bias_m = get_float(kwargs, "sitl_gate_range_bias_m", 0.0f);
    env->gate_position_domain_randomize = get_int(
        kwargs, "gate_position_domain_randomize", 0);
    env->gate_position_domain_randomize_probability = get_float(
        kwargs, "gate_position_domain_randomize_probability", 1.0f);
    env->gate_position_randomized_radius = get_float(
        kwargs, "gate_position_randomized_radius", 0.0f);
    env->gate_position_randomize_from_index = get_int(
        kwargs, "gate_position_randomize_from_index", 0);
    env->gate_position_jitter_x = get_float(kwargs, "gate_position_jitter_x", 0.0f);
    env->gate_position_jitter_y = get_float(kwargs, "gate_position_jitter_y", 0.0f);
    env->gate_position_jitter_z = get_float(kwargs, "gate_position_jitter_z", 0.0f);
    env->gate_position_require_valid_course = get_int(
        kwargs, "gate_position_require_valid_course", 0);
    env->gate_position_min_forward_gap_m = get_float(
        kwargs, "gate_position_min_forward_gap_m", 0.0f);
    env->gate_position_max_segment_distance_m = get_float(
        kwargs, "gate_position_max_segment_distance_m", 0.0f);
    env->gate_position_resample_attempts = get_int(
        kwargs, "gate_position_resample_attempts", 64);
    env->observable_gate_index = get_int(kwargs, "observable_gate_index", 0);
    env->observable_gate_index_denominator = get_float(
        kwargs, "observable_gate_index_denominator", 0.0f);
    env->observable_gate_progress = get_int(
        kwargs, "observable_gate_progress", 0);
    env->observable_gate_phase_onehot = get_int(
        kwargs, "observable_gate_phase_onehot", 0);
    env->evaluation_episode_limit = get_int(
        kwargs, "evaluation_episode_limit", 0);
    env->evaluation_episode_offset = get_int(
        kwargs, "evaluation_episode_offset", 0);
    env->use_custom_gate_layout = get_int(kwargs, "use_custom_gate_layout", 0);
    env->use_custom_start = get_int(kwargs, "use_custom_start", 0);
    env->start_gate_index = get_int(kwargs, "start_gate_index", 0);
    env->start_elapsed_time = get_float(kwargs, "start_elapsed_time", 0.0f);
    env->start_elapsed_time_jitter = get_float(
        kwargs, "start_elapsed_time_jitter", 0.0f);
    env->mixed_start_curriculum = get_int(kwargs, "mixed_start_curriculum", 0);
    env->segment_start_probability = get_float(
        kwargs, "segment_start_probability", 0.5f);
    env->gate_local_start_curriculum = get_int(
        kwargs, "gate_local_start_curriculum", 0);
    env->gate_local_start_probability = get_float(
        kwargs, "gate_local_start_probability", 0.7f);
    env->gate_local_start_offset_min = get_float(
        kwargs, "gate_local_start_offset_min", 2.0f);
    env->gate_local_start_offset_max = get_float(
        kwargs, "gate_local_start_offset_max", 4.0f);
    env->custom_start_pos[0] = get_float(kwargs, "start_x", 0.0f);
    env->custom_start_pos[1] = get_float(kwargs, "start_y", 0.0f);
    env->custom_start_pos[2] = get_float(kwargs, "start_z", 0.0f);
    env->custom_start_pos_jitter[0] = get_float(kwargs, "start_x_jitter", 0.0f);
    env->custom_start_pos_jitter[1] = get_float(kwargs, "start_y_jitter", 0.0f);
    env->custom_start_pos_jitter[2] = get_float(kwargs, "start_z_jitter", 0.0f);
    env->reset_position_noise_xy = get_float(
        kwargs, "reset_position_noise_xy", 0.1f);
    env->reset_position_noise_z = get_float(
        kwargs, "reset_position_noise_z", 0.05f);
    env->custom_start_vel[0] = get_float(kwargs, "start_vx", 0.0f);
    env->custom_start_vel[1] = get_float(kwargs, "start_vy", 0.0f);
    env->custom_start_vel[2] = get_float(kwargs, "start_vz", 0.0f);
    env->custom_start_vel_jitter[0] = get_float(kwargs, "start_vx_jitter", 0.0f);
    env->custom_start_vel_jitter[1] = get_float(kwargs, "start_vy_jitter", 0.0f);
    env->custom_start_vel_jitter[2] = get_float(kwargs, "start_vz_jitter", 0.0f);
    env->custom_start_quat[0] = get_float(kwargs, "start_qw", 1.0f);
    env->custom_start_quat[1] = get_float(kwargs, "start_qx", 0.0f);
    env->custom_start_quat[2] = get_float(kwargs, "start_qy", 0.0f);
    env->custom_start_quat[3] = get_float(kwargs, "start_qz", 0.0f);
    env->custom_start_attitude_jitter[0] = get_float(
        kwargs, "start_roll_jitter_rad", 0.0f);
    env->custom_start_attitude_jitter[1] = get_float(
        kwargs, "start_pitch_jitter_rad", 0.0f);
    env->custom_start_attitude_jitter[2] = get_float(
        kwargs, "start_yaw_jitter_rad", 0.0f);
    env->custom_start_omega[0] = get_float(kwargs, "start_wx", 0.0f);
    env->custom_start_omega[1] = get_float(kwargs, "start_wy", 0.0f);
    env->custom_start_omega[2] = get_float(kwargs, "start_wz", 0.0f);
    env->custom_start_omega_jitter[0] = get_float(kwargs, "start_wx_jitter", 0.0f);
    env->custom_start_omega_jitter[1] = get_float(kwargs, "start_wy_jitter", 0.0f);
    env->custom_start_omega_jitter[2] = get_float(kwargs, "start_wz_jitter", 0.0f);
    for (int i = 0; i < DRONE_RACE_MAX_GATES; i++) {
        char key_x[16];
        char key_y[16];
        char key_z[16];
        snprintf(key_x, sizeof(key_x), "gate%d_x", i);
        snprintf(key_y, sizeof(key_y), "gate%d_y", i);
        snprintf(key_z, sizeof(key_z), "gate%d_z", i);
        char key_radius[24];
        snprintf(key_radius, sizeof(key_radius), "gate%d_radius", i);
        env->custom_gate_pos[i][0] = get_float(kwargs, key_x, 0.0f);
        env->custom_gate_pos[i][1] = get_float(kwargs, key_y, 0.0f);
        env->custom_gate_pos[i][2] = get_float(kwargs, key_z, 0.0f);
        env->custom_gate_radius[i] = get_float(kwargs, key_radius, 0.0f);
    }
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "valid_run_rate", log->valid_run_rate);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "gates_passed", log->gates_passed);
    static const char* gate_count_episode_keys[DRONE_RACE_MAX_GATES] = {
        "gate_count1_episode", "gate_count2_episode",
        "gate_count3_episode", "gate_count4_episode",
        "gate_count5_episode", "gate_count6_episode",
        "gate_count7_episode", "gate_count8_episode",
        "gate_count9_episode", "gate_count10_episode",
        "gate_count11_episode", "gate_count12_episode",
        "gate_count13_episode", "gate_count14_episode",
        "gate_count15_episode", "gate_count16_episode",
    };
    static const char* gate_count_success_keys[DRONE_RACE_MAX_GATES] = {
        "gate_count1_success", "gate_count2_success",
        "gate_count3_success", "gate_count4_success",
        "gate_count5_success", "gate_count6_success",
        "gate_count7_success", "gate_count8_success",
        "gate_count9_success", "gate_count10_success",
        "gate_count11_success", "gate_count12_success",
        "gate_count13_success", "gate_count14_success",
        "gate_count15_success", "gate_count16_success",
    };
    for (int gate_count = 0; gate_count < DRONE_RACE_MAX_GATES; gate_count++) {
        dict_set(
            out,
            gate_count_episode_keys[gate_count],
            log->gate_count_episode[gate_count]);
        dict_set(
            out,
            gate_count_success_keys[gate_count],
            log->gate_count_success[gate_count]);
    }
    dict_set(out, "completion_time", log->completion_time);
    dict_set(out, "final_progress", log->final_progress);
    dict_set(out, "max_progress", log->max_progress);
    dict_set(out, "closest_gate_range", log->closest_gate_range);
    dict_set(out, "final_x", log->final_x);
    dict_set(out, "final_y", log->final_y);
    dict_set(out, "final_z", log->final_z);
    dict_set(out, "gate_local_start_rate", log->gate_local_start_rate);
    dict_set(out, "gate_local_reset_count", log->gate_local_reset_count);
    dict_set(out, "reset_count", log->reset_count);
    dict_set(out, "crash", log->crash);
    dict_set(out, "crash_low", log->crash_low);
    dict_set(out, "crash_high", log->crash_high);
    dict_set(out, "crash_xy", log->crash_xy);
    dict_set(out, "crash_low_z", log->crash_low_z);
    dict_set(out, "crash_low_vz", log->crash_low_vz);
    dict_set(out, "crash_low_progress", log->crash_low_progress);
    dict_set(out, "crash_low_time", log->crash_low_time);
    dict_set(out, "crash_low_next_gate0", log->crash_low_next_gate0);
    dict_set(out, "crash_low_next_gate1", log->crash_low_next_gate1);
    dict_set(out, "crash_low_next_gate2", log->crash_low_next_gate2);
    dict_set(out, "crash_low_next_gate3plus", log->crash_low_next_gate3plus);
    dict_set(out, "crash_low_floor_margin_pre", log->crash_low_floor_margin_pre);
    dict_set(out, "crash_low_ttf_pre", log->crash_low_ttf_pre);
    dict_set(out, "crash_low_stop_margin_pre", log->crash_low_stop_margin_pre);
    dict_set(out, "crash_low_max_up_accel_pre", log->crash_low_max_up_accel_pre);
    dict_set(out, "floor_impact_risk", log->floor_impact_risk);
    dict_set(out, "floor_stop_violation", log->floor_stop_violation);
    dict_set(out, "floor_risk_steps", log->floor_risk_steps);
    dict_set(out, "floor_stop_violation_steps", log->floor_stop_violation_steps);
    dict_set(out, "floor_risk_sampled", log->floor_risk_sampled);
    dict_set(out, "min_floor_ttf", log->min_floor_ttf);
    dict_set(out, "min_floor_stop_margin", log->min_floor_stop_margin);
    dict_set(out, "terminal_crossing_sampled", log->terminal_crossing_sampled);
    dict_set(out, "terminal_crossing_radial", log->terminal_crossing_radial);
    dict_set(out, "terminal_crossing_right", log->terminal_crossing_right);
    dict_set(out, "terminal_crossing_vertical", log->terminal_crossing_vertical);
    dict_set(out, "terminal_crossing_abs_right", log->terminal_crossing_abs_right);
    dict_set(out, "terminal_crossing_abs_vertical", log->terminal_crossing_abs_vertical);
    static const char* sampled_keys[DRONE_RACE_MAX_GATES] = {
        "terminal_gate0_sampled", "terminal_gate1_sampled",
        "terminal_gate2_sampled", "terminal_gate3_sampled",
        "terminal_gate4_sampled", "terminal_gate5_sampled",
        "terminal_gate6_sampled", "terminal_gate7_sampled",
        "terminal_gate8_sampled", "terminal_gate9_sampled",
        "terminal_gate10_sampled", "terminal_gate11_sampled",
        "terminal_gate12_sampled", "terminal_gate13_sampled",
        "terminal_gate14_sampled", "terminal_gate15_sampled",
    };
    static const char* radial_keys[DRONE_RACE_MAX_GATES] = {
        "terminal_gate0_radial", "terminal_gate1_radial",
        "terminal_gate2_radial", "terminal_gate3_radial",
        "terminal_gate4_radial", "terminal_gate5_radial",
        "terminal_gate6_radial", "terminal_gate7_radial",
        "terminal_gate8_radial", "terminal_gate9_radial",
        "terminal_gate10_radial", "terminal_gate11_radial",
        "terminal_gate12_radial", "terminal_gate13_radial",
        "terminal_gate14_radial", "terminal_gate15_radial",
    };
    static const char* right_keys[DRONE_RACE_MAX_GATES] = {
        "terminal_gate0_right", "terminal_gate1_right",
        "terminal_gate2_right", "terminal_gate3_right",
        "terminal_gate4_right", "terminal_gate5_right",
        "terminal_gate6_right", "terminal_gate7_right",
        "terminal_gate8_right", "terminal_gate9_right",
        "terminal_gate10_right", "terminal_gate11_right",
        "terminal_gate12_right", "terminal_gate13_right",
        "terminal_gate14_right", "terminal_gate15_right",
    };
    static const char* vertical_keys[DRONE_RACE_MAX_GATES] = {
        "terminal_gate0_vertical", "terminal_gate1_vertical",
        "terminal_gate2_vertical", "terminal_gate3_vertical",
        "terminal_gate4_vertical", "terminal_gate5_vertical",
        "terminal_gate6_vertical", "terminal_gate7_vertical",
        "terminal_gate8_vertical", "terminal_gate9_vertical",
        "terminal_gate10_vertical", "terminal_gate11_vertical",
        "terminal_gate12_vertical", "terminal_gate13_vertical",
        "terminal_gate14_vertical", "terminal_gate15_vertical",
    };
    for (int gate = 0; gate < DRONE_RACE_MAX_GATES; gate++) {
        dict_set(out, sampled_keys[gate], log->terminal_gate_sampled[gate]);
        dict_set(out, radial_keys[gate], log->terminal_gate_radial[gate]);
        dict_set(out, right_keys[gate], log->terminal_gate_right[gate]);
        dict_set(out, vertical_keys[gate], log->terminal_gate_vertical[gate]);
    }
    static const char* ordered_sampled_keys[DRONE_RACE_MAX_GATES] = {
        "ordered_gate0_sampled", "ordered_gate1_sampled",
        "ordered_gate2_sampled", "ordered_gate3_sampled",
        "ordered_gate4_sampled", "ordered_gate5_sampled",
        "ordered_gate6_sampled", "ordered_gate7_sampled",
        "ordered_gate8_sampled", "ordered_gate9_sampled",
        "ordered_gate10_sampled", "ordered_gate11_sampled",
        "ordered_gate12_sampled", "ordered_gate13_sampled",
        "ordered_gate14_sampled", "ordered_gate15_sampled",
    };
    static const char* ordered_radial_keys[DRONE_RACE_MAX_GATES] = {
        "ordered_gate0_radial", "ordered_gate1_radial",
        "ordered_gate2_radial", "ordered_gate3_radial",
        "ordered_gate4_radial", "ordered_gate5_radial",
        "ordered_gate6_radial", "ordered_gate7_radial",
        "ordered_gate8_radial", "ordered_gate9_radial",
        "ordered_gate10_radial", "ordered_gate11_radial",
        "ordered_gate12_radial", "ordered_gate13_radial",
        "ordered_gate14_radial", "ordered_gate15_radial",
    };
    static const char* ordered_right_keys[DRONE_RACE_MAX_GATES] = {
        "ordered_gate0_right", "ordered_gate1_right",
        "ordered_gate2_right", "ordered_gate3_right",
        "ordered_gate4_right", "ordered_gate5_right",
        "ordered_gate6_right", "ordered_gate7_right",
        "ordered_gate8_right", "ordered_gate9_right",
        "ordered_gate10_right", "ordered_gate11_right",
        "ordered_gate12_right", "ordered_gate13_right",
        "ordered_gate14_right", "ordered_gate15_right",
    };
    static const char* ordered_vertical_keys[DRONE_RACE_MAX_GATES] = {
        "ordered_gate0_vertical", "ordered_gate1_vertical",
        "ordered_gate2_vertical", "ordered_gate3_vertical",
        "ordered_gate4_vertical", "ordered_gate5_vertical",
        "ordered_gate6_vertical", "ordered_gate7_vertical",
        "ordered_gate8_vertical", "ordered_gate9_vertical",
        "ordered_gate10_vertical", "ordered_gate11_vertical",
        "ordered_gate12_vertical", "ordered_gate13_vertical",
        "ordered_gate14_vertical", "ordered_gate15_vertical",
    };
    for (int gate = 0; gate < DRONE_RACE_MAX_GATES; gate++) {
        dict_set(out, ordered_sampled_keys[gate], log->ordered_gate_sampled[gate]);
        dict_set(out, ordered_radial_keys[gate], log->ordered_gate_radial[gate]);
        dict_set(out, ordered_right_keys[gate], log->ordered_gate_right[gate]);
        dict_set(out, ordered_vertical_keys[gate], log->ordered_gate_vertical[gate]);
    }
    dict_set(out, "crossing_margin_violation", log->crossing_margin_violation);
    dict_set(out, "action_envelope_violation", log->action_envelope_violation);
    dict_set(out, "wire_rate_envelope_violation", log->wire_rate_envelope_violation);
    dict_set(out, "thrust_envelope_violation", log->thrust_envelope_violation);
    dict_set(out, "gate2_terminal_crossing_sampled", log->gate2_terminal_crossing_sampled);
    dict_set(out, "gate2_terminal_crossing_radial", log->gate2_terminal_crossing_radial);
    dict_set(out, "gate2_terminal_crossing_right", log->gate2_terminal_crossing_right);
    dict_set(out, "gate2_terminal_crossing_vertical", log->gate2_terminal_crossing_vertical);
    dict_set(out, "gate2_crossing_sampled", log->gate2_crossing_sampled);
    dict_set(out, "gate2_crossing_radial", log->gate2_crossing_radial);
    dict_set(out, "gate2_crossing_right", log->gate2_crossing_right);
    dict_set(out, "gate2_crossing_vertical", log->gate2_crossing_vertical);
    dict_set(out, "gate2_exit_vx", log->gate2_exit_vx);
    dict_set(out, "gate2_exit_vy", log->gate2_exit_vy);
    dict_set(out, "gate2_exit_vz", log->gate2_exit_vz);
    dict_set(out, "gate2_completion_sampled", log->gate2_completion_sampled);
    dict_set(out, "gate2_completion_crossing_radial", log->gate2_completion_crossing_radial);
    dict_set(out, "gate2_completion_crossing_right", log->gate2_completion_crossing_right);
    dict_set(out, "gate2_completion_crossing_vertical", log->gate2_completion_crossing_vertical);
    dict_set(out, "gate2_completion_exit_vx", log->gate2_completion_exit_vx);
    dict_set(out, "gate2_completion_exit_vy", log->gate2_completion_exit_vy);
    dict_set(out, "gate2_completion_exit_vz", log->gate2_completion_exit_vz);
    dict_set(out, "gate2_downstream_failure_sampled", log->gate2_downstream_failure_sampled);
    dict_set(out, "gate2_downstream_failure_crossing_radial", log->gate2_downstream_failure_crossing_radial);
    dict_set(out, "gate2_downstream_failure_crossing_right", log->gate2_downstream_failure_crossing_right);
    dict_set(out, "gate2_downstream_failure_crossing_vertical", log->gate2_downstream_failure_crossing_vertical);
    dict_set(out, "gate2_downstream_failure_exit_vx", log->gate2_downstream_failure_exit_vx);
    dict_set(out, "gate2_downstream_failure_exit_vy", log->gate2_downstream_failure_exit_vy);
    dict_set(out, "gate2_downstream_failure_exit_vz", log->gate2_downstream_failure_exit_vz);
    dict_set(out, "gate0_exit_sampled", log->gate0_exit_sampled);
    dict_set(out, "gate0_exit_vx", log->gate0_exit_vx);
    dict_set(out, "gate0_exit_vy", log->gate0_exit_vy);
    dict_set(out, "gate0_exit_vz", log->gate0_exit_vz);
    dict_set(out, "gate0_action_steps", log->gate0_action_steps);
    dict_set(out, "gate0_action_pitch", log->gate0_action_pitch);
    dict_set(out, "gate1_action_steps", log->gate1_action_steps);
    dict_set(out, "gate1_action_pitch", log->gate1_action_pitch);
    dict_set(out, "gate1_action_roll", log->gate1_action_roll);
    dict_set(out, "gate1_action_thrust", log->gate1_action_thrust);
    dict_set(out, "gate1_action_yaw", log->gate1_action_yaw);
    dict_set(out, "gate1_early_action_steps", log->gate1_early_action_steps);
    dict_set(out, "gate1_early_action_pitch", log->gate1_early_action_pitch);
    dict_set(out, "gate1_early_action_roll", log->gate1_early_action_roll);
    dict_set(out, "gate1_early_action_thrust", log->gate1_early_action_thrust);
    dict_set(out, "gate1_early_action_yaw", log->gate1_early_action_yaw);
    dict_set(out, "gate2_action_steps", log->gate2_action_steps);
    dict_set(out, "gate2_action_pitch", log->gate2_action_pitch);
    dict_set(out, "gate2_action_roll", log->gate2_action_roll);
    dict_set(out, "gate2_action_thrust", log->gate2_action_thrust);
    dict_set(out, "gate2_action_yaw", log->gate2_action_yaw);
    dict_set(out, "gate3_action_steps", log->gate3_action_steps);
    dict_set(out, "gate3_action_pitch", log->gate3_action_pitch);
    dict_set(out, "gate3_action_roll", log->gate3_action_roll);
    dict_set(out, "gate3_action_thrust", log->gate3_action_thrust);
    dict_set(out, "gate3_action_yaw", log->gate3_action_yaw);
    dict_set(out, "out_of_order", log->out_of_order);
    dict_set(out, "missed_gate", log->missed_gate);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "episode_slot0_n", log->episode_slot0_n);
    dict_set(out, "episode_slot0_success", log->episode_slot0_success);
    dict_set(out, "episode_slot1_n", log->episode_slot1_n);
    dict_set(out, "episode_slot1_success", log->episode_slot1_success);
    dict_set(out, "episode_slot2_n", log->episode_slot2_n);
    dict_set(out, "episode_slot2_success", log->episode_slot2_success);
    dict_set(out, "episode_slot3_n", log->episode_slot3_n);
    dict_set(out, "episode_slot3_success", log->episode_slot3_success);
}
