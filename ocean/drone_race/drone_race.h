#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// The generic static-vector logger normally ignores environments that have
// not completed an episode. Reset-sampler diagnostics must include those
// environments to avoid completion-length bias.
#define STATIC_VEC_LOG_RESET_COUNTERS 1

#define Log DroneBaseLog
#include "../drone/dronelib.h"
#undef Log

#define DRONE_RACE_VISUAL_WIDTH 64
#define DRONE_RACE_VISUAL_HEIGHT 64
#define DRONE_RACE_VISUAL_MASK_SIZE \
    (DRONE_RACE_VISUAL_WIDTH * DRONE_RACE_VISUAL_HEIGHT)
#define DRONE_RACE_VISUAL_BODY_RATE_OFFSET DRONE_RACE_VISUAL_MASK_SIZE
#define DRONE_RACE_VISUAL_MOTOR_OFFSET \
    (DRONE_RACE_VISUAL_BODY_RATE_OFFSET + 3)
#define DRONE_RACE_VISUAL_ACTION_HISTORY_OFFSET \
    (DRONE_RACE_VISUAL_MOTOR_OFFSET + 4)
#define DRONE_RACE_VISUAL_NEW_FRAME_OFFSET \
    (DRONE_RACE_VISUAL_ACTION_HISTORY_OFFSET + 12)
#define DRONE_RACE_VISUAL_FRAME_AGE_OFFSET \
    (DRONE_RACE_VISUAL_NEW_FRAME_OFFSET + 1)
#define DRONE_RACE_VISUAL_DT_OFFSET \
    (DRONE_RACE_VISUAL_FRAME_AGE_OFFSET + 1)
#define DRONE_RACE_VISUAL_LEGAL_OBS_SIZE \
    (DRONE_RACE_VISUAL_DT_OFFSET + 1)
#define DRONE_RACE_VISUAL_PRIVILEGED_OFFSET \
    DRONE_RACE_VISUAL_LEGAL_OBS_SIZE
#define DRONE_RACE_VISUAL_PRIVILEGED_SIZE 34
#define DRONE_RACE_VISUAL_OBS_SIZE \
    (DRONE_RACE_VISUAL_LEGAL_OBS_SIZE + DRONE_RACE_VISUAL_PRIVILEGED_SIZE)

// The historical environment has a fixed 32-float observation. A separate
// drone_race_vision binding overrides this macro at compile time so retained
// checkpoints and their ABI remain byte-for-byte untouched.
#ifndef DRONE_RACE_OBS_SIZE
#define DRONE_RACE_OBS_SIZE 32
#endif
#define DRONE_RACE_NUM_ATNS 4
#define DRONE_RACE_MAX_GATES 32
#define DRONE_RACE_INTERFACE_NATIVE_MOTOR 0
#define DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW 1
#define DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT 2
#define DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY 3

// Training-only course profile. It is used to compute an action-shaped reward,
// never appended to the policy observation.
typedef struct {
    int points;
    float x[DRONE_RACE_MAX_GATES + 1];
    float a[DRONE_RACE_MAX_GATES];
    float b[DRONE_RACE_MAX_GATES];
    float c[DRONE_RACE_MAX_GATES];
    float d[DRONE_RACE_MAX_GATES];
} DroneRaceCourseSpline;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float valid_run_rate;
    float success_rate;
    float gates_passed;
    // Diagnostic-only distribution for fixed-per-vector-instance gate counts.
    // Indexed by num_gates - 1 through the engine cap. Values are aggregated
    // as episode and success rates and never enter the actor observation.
    float gate_count_episode[DRONE_RACE_MAX_GATES];
    float gate_count_success[DRONE_RACE_MAX_GATES];
    float completion_time;
    float final_progress;
    float max_progress;
    float closest_gate_range;
    float final_x;
    float final_y;
    float final_z;
    // Training diagnostic only: fraction of completed episodes initialized
    // from the randomized gate-local discovery curriculum.
    float gate_local_start_rate;
    // Unbiased reset-sampler diagnostics. Static vector logging divides both
    // by the same completed-episode count, so their ratio remains exact.
    float gate_local_reset_count;
    float reset_count;
    float crash;
    float crash_low;
    float crash_high;
    float crash_xy;
    float crash_low_z;
    float crash_low_vz;
    float crash_low_progress;
    float crash_low_time;
    float crash_low_next_gate0;
    float crash_low_next_gate1;
    float crash_low_next_gate2;
    float crash_low_next_gate3plus;
    float crash_low_floor_margin_pre;
    float crash_low_ttf_pre;
    float crash_low_stop_margin_pre;
    float crash_low_max_up_accel_pre;
    float floor_impact_risk;
    float floor_stop_violation;
    float floor_risk_steps;
    float floor_stop_violation_steps;
    float floor_risk_sampled;
    float min_floor_ttf;
    float min_floor_stop_margin;
    float terminal_crossing_sampled;
    float terminal_crossing_radial;
    float terminal_crossing_right;
    float terminal_crossing_vertical;
    float terminal_crossing_abs_right;
    float terminal_crossing_abs_vertical;
    float terminal_gate_sampled[DRONE_RACE_MAX_GATES];
    float terminal_gate_radial[DRONE_RACE_MAX_GATES];
    float terminal_gate_right[DRONE_RACE_MAX_GATES];
    float terminal_gate_vertical[DRONE_RACE_MAX_GATES];
    float ordered_gate_sampled[DRONE_RACE_MAX_GATES];
    float ordered_gate_radial[DRONE_RACE_MAX_GATES];
    float ordered_gate_right[DRONE_RACE_MAX_GATES];
    float ordered_gate_vertical[DRONE_RACE_MAX_GATES];
    float crossing_margin_violation;
    float action_envelope_violation;
    float wire_rate_envelope_violation;
    float thrust_envelope_violation;
    float gate2_terminal_crossing_sampled;
    float gate2_terminal_crossing_radial;
    float gate2_terminal_crossing_right;
    float gate2_terminal_crossing_vertical;
    float gate2_crossing_sampled;
    float gate2_crossing_radial;
    float gate2_crossing_right;
    float gate2_crossing_vertical;
    float gate2_exit_vx;
    float gate2_exit_vy;
    float gate2_exit_vz;
    float gate2_completion_sampled;
    float gate2_completion_crossing_radial;
    float gate2_completion_crossing_right;
    float gate2_completion_crossing_vertical;
    float gate2_completion_exit_vx;
    float gate2_completion_exit_vy;
    float gate2_completion_exit_vz;
    float gate2_downstream_failure_sampled;
    float gate2_downstream_failure_crossing_radial;
    float gate2_downstream_failure_crossing_right;
    float gate2_downstream_failure_crossing_vertical;
    float gate2_downstream_failure_exit_vx;
    float gate2_downstream_failure_exit_vy;
    float gate2_downstream_failure_exit_vz;
    float gate0_exit_sampled;
    float gate0_exit_vx;
    float gate0_exit_vy;
    float gate0_exit_vz;
    float gate0_action_steps;
    float gate0_action_pitch;
    float gate1_action_steps;
    float gate1_action_pitch;
    float gate1_action_roll;
    float gate1_action_thrust;
    float gate1_action_yaw;
    float gate1_early_action_steps;
    float gate1_early_action_pitch;
    float gate1_early_action_roll;
    float gate1_early_action_thrust;
    float gate1_early_action_yaw;
    float gate2_action_steps;
    float gate2_action_pitch;
    float gate2_action_roll;
    float gate2_action_thrust;
    float gate2_action_yaw;
    float gate3_action_steps;
    float gate3_action_pitch;
    float gate3_action_roll;
    float gate3_action_thrust;
    float gate3_action_yaw;
    float out_of_order;
    float missed_gate;
    float timeout;
    float episode_slot0_n;
    float episode_slot0_success;
    float episode_slot1_n;
    float episode_slot1_success;
    float episode_slot2_n;
    float episode_slot2_success;
    float episode_slot3_n;
    float episode_slot3_success;
    float n;
};

typedef struct Client Client;
struct Client {
    int unused;
};

typedef struct {
    Drone drone;
    Target target;
    float last_action[4];
    // Legal visual-policy memory input. Index 0 is the most recently executed
    // action; indices 1 and 2 are the two preceding actions.
    float last_action_history[3][4];
    unsigned int visual_rng;
    float visual_camera_roll_rad;
    float visual_camera_pitch_rad;
    float visual_camera_yaw_rad;
    float visual_camera_accumulator_s;
    float visual_frame_age_s;
    int visual_frame_count;
    Vec3 gate_motion_raw_world;
    Vec3 gate_motion_filtered_world;
    Vec3 gate_motion_rate_world;
    int gate_motion_valid;
    int gate_motion_gate_index;
    float progress;
    float max_progress;
    float closest_gate_range;
    float elapsed_time;
    float episode_return;
    float crash_low_floor_margin_pre;
    float crash_low_ttf_pre;
    float crash_low_stop_margin_pre;
    float crash_low_max_up_accel_pre;
    float floor_risk_steps;
    float floor_stop_violation_steps;
    float min_floor_ttf;
    float min_floor_stop_margin;
    float terminal_crossing_radial;
    float terminal_crossing_right;
    float terminal_crossing_vertical;
    int ordered_gate_sampled[DRONE_RACE_MAX_GATES];
    float ordered_gate_radial[DRONE_RACE_MAX_GATES];
    float ordered_gate_right[DRONE_RACE_MAX_GATES];
    float ordered_gate_vertical[DRONE_RACE_MAX_GATES];
    int crossing_margin_violation;
    int action_envelope_violation;
    int wire_rate_envelope_violation;
    int thrust_envelope_violation;
    float gate0_exit_vx;
    float gate0_exit_vy;
    float gate0_exit_vz;
    float gate0_action_steps;
    float gate0_action_pitch;
    float gate1_action_steps;
    float gate1_action_pitch;
    float gate1_action_roll;
    float gate1_action_thrust;
    float gate1_action_yaw;
    float gate1_early_action_steps;
    float gate1_early_action_pitch;
    float gate1_early_action_roll;
    float gate1_early_action_thrust;
    float gate1_early_action_yaw;
    float gate2_crossing_radial;
    float gate2_crossing_right;
    float gate2_crossing_vertical;
    float gate2_exit_vx;
    float gate2_exit_vy;
    float gate2_exit_vz;
    float gate2_action_steps;
    float gate2_action_pitch;
    float gate2_action_roll;
    float gate2_action_thrust;
    float gate2_action_yaw;
    float gate3_action_steps;
    float gate3_action_pitch;
    float gate3_action_roll;
    float gate3_action_thrust;
    float gate3_action_yaw;
    int step_count;
    int current_gate;
    int valid_run;
    int floor_impact_risk;
    int floor_stop_violation;
    int floor_risk_sampled;
    int terminal_crossing_sampled;
    int terminal_crossing_gate_index;
    int gate0_exit_sampled;
    int gate2_crossing_sampled;
    int transition_speed_floor_delay_steps_remaining;
    int episode_ordinal;
    int out_of_order;
    int missed_gate;
    int crash;
    // Visual Dreamer needs the actual terminal observation in replay. When
    // set, the following vector step performs a reset-only no-op.
    int pending_reset;
    int crash_low;
    int crash_high;
    int crash_xy;
    int timeout;
    // Reset provenance is never appended to the legal actor observation.
    int gate_local_start_sampled;
    // True only when this episode sampled late-gate position jitter. Course
    // randomization can otherwise be enabled globally while an anchor episode
    // still uses the nominal gate positions.
    int gate_positions_randomized;
    float randomized_course_geometry_scale;
    float randomized_sitl_gate_transition_min_forward_speed;
    Target randomized_gates[DRONE_RACE_MAX_GATES];
    float randomized_between_gate_distance[DRONE_RACE_MAX_GATES];
} DroneRaceAgent;

typedef struct {
    Log log;
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    float dt;
    int num_gates;
    // Default-off variable-course contract. The binding chooses one count at
    // construction from the vector instance index; resets never resample it.
    int num_gates_per_env_randomize;
    int num_gates_per_env_min;
    int num_gates_per_env_max;
    unsigned int num_gates_per_env_seed;
    float gate_spacing;
    float gate_radius;
    int gate_radius_randomize;
    // -1 randomizes every gate (legacy behavior); otherwise only this index.
    int gate_radius_randomize_gate_index;
    float gate_radius_min;
    float gate_radius_max;
    // Episode-level exact profile mixture. Anchor episodes preserve the fixed
    // per-gate radii; target episodes replace every aperture with target_radius.
    int gate_radius_profile_mix;
    float gate_radius_profile_mix_probability;
    float gate_radius_profile_mix_target;
    float gate_altitude;
    float gate_lateral_amplitude;
    float course_geometry_scale;
    int course_geometry_scale_randomize;
    float course_geometry_scale_min;
    float course_geometry_scale_max;
    float start_offset;
    float crash_height;
    float pos_bound;
    float plane_cross_tolerance;
    float miss_tolerance;
    float direction_min;
    int strict_missed_gate;
    int max_steps;
    float time_limit_seconds;
    float safety_altitude;
    float w_progress;
    float w_gate;
    // Optional one-time reward for a legal ordered crossing, scaled from one
    // at gate center to zero at the aperture rim. Kept separate from the
    // historical post-first-gate progress multiplier for checkpoint parity.
    float w_ordered_gate;
    float w_finish;
    float w_time;
    float w_ctrl;
    // SkyDreamer body-rate penalty, adapted to env->dt while preserving the
    // paper's frequency-normalized exponential form. Default zero preserves
    // all historical curricula.
    float w_body_rate;
    float w_cross_track;
    float cross_track_vertical_weight;
    int cross_track_from_gate_index;
    float w_gate_camera_alignment;
    int gate_camera_alignment_from_gate_index;
    float w_gate_crossing_error;
    int gate_crossing_error_from_gate_index;
    float w_gate_exit_lateral_velocity;
    float w_gate_exit_velocity;
    float gate_exit_target_forward_speed;
    float gate_exit_reward_lookahead_m;
    // Avoid reward targets derived from a randomized next gate that is not
    // present in the policy observation yet. Disabled by default for backward
    // compatibility with retained checkpoints and curricula.
    int gate_exit_reward_skip_randomized_next_gate;
    float w_forward_speed_excess;
    float target_forward_speed;
    float w_altitude_floor;
    float w_descent_floor;
    float w_action_teacher;
    // Training-only DAgger-style intervention for attitude-setpoint control.
    // Zero preserves historical dynamics; one executes the teacher on active
    // channels while the raw policy remains the imitation-loss target.
    float teacher_action_blend;
    // Training-only delay before DAgger intervention. Zero preserves the
    // historical teacher behavior and every deployment leaves blending off.
    int teacher_action_blend_from_step;
    int teacher_course_spline;
    // Training-only smooth segment oracle. The default zero preserves every
    // historical controller and deployment path.
    int teacher_segment_minimum_jerk;
    // Training-only, memoryless align-then-go oracle. The default zero keeps
    // this privileged state feedback out of every historical/runtime path.
    int teacher_alignment_governor;
    int teacher_from_gate_index;
    int teacher_pitch_from_gate_index;
    int teacher_roll_from_gate_index;
    int teacher_roll_until_gate_index;
    int teacher_thrust_from_gate_index;
    int teacher_yaw_control;
    int teacher_yaw_from_gate_index;
    float teacher_pitch_action;
    float teacher_yaw_action;
    // Optional training-only speed controller for the pitch teacher. This
    // replaces the fixed pitch target when enabled and is never observed by
    // or compiled into the deployed recurrent policy.
    int teacher_pitch_speed_control;
    float teacher_pitch_speed_target_m_s;
    float teacher_pitch_speed_gain;
    float teacher_pitch_speed_scale;
    float teacher_roll_bias;
    float teacher_roll_per_m;
    float teacher_roll_rate_per_m_s;
    // Default-off privileged oracle mode. When enabled, derivative feedback
    // uses true target-minus-vehicle velocity instead of the optional public
    // gate-motion estimate. It is training infrastructure only.
    int teacher_true_velocity_damping;
    float teacher_thrust_bias;
    float teacher_thrust_per_m;
    float teacher_thrust_rate_per_m_s;
    // Training-only altitude targets may use world-up error so a large bank
    // cannot rotate lateral gate displacement into the thrust command.
    int teacher_thrust_world_frame;
    float teacher_pitch_weight;
    float teacher_roll_weight;
    float teacher_thrust_weight;
    float teacher_yaw_weight;
    float invalid_penalty;
    float late_invalid_penalty;
    int late_invalid_penalty_from_gate_index;
    int interface_mode;
    // Informed-Dreamer visual ABI. The first 4118 floats are legal deployment
    // observations and the final 34 are training-only decoder targets. Only
    // the drone_race_vision binding has sufficient observation capacity.
    int visual_observation;
    // Default-off native PPO boundary. The policy-visible tensor retains its
    // historical width for the CUDA engine, but every training-only tail value
    // is zero and the legal step-dt slot carries public active_gate_index / 6.
    int visual_policy_legal_only;
    float visual_camera_hz;
    float visual_focal_x_px;
    float visual_focal_y_px;
    float visual_camera_roll_rad;
    float visual_camera_pitch_rad;
    float visual_camera_yaw_rad;
    float visual_camera_roll_jitter_rad;
    float visual_camera_pitch_jitter_rad;
    float visual_camera_yaw_jitter_rad;
    float visual_camera_dropout_prob;
    float visual_edge_dropout_prob;
    float visual_edge_corrupt_prob;
    int visual_false_segments;
    float visual_line_thickness_px;
    float visual_rolling_shutter_s;
    float max_cmd_forward;
    float max_cmd_lateral;
    float max_cmd_vertical;
    float max_cmd_yaw_rate;
    float setpoint_vel_kp;
    float setpoint_att_kp;
    float setpoint_rate_kd;
    float setpoint_yaw_rate_kp;
    float setpoint_thrust_vel_kp;
    float setpoint_max_tilt;
    float setpoint_max_motor_delta;
    // VQ1 v3391 telemetry-enabled plant. Actions are normalized local-course
    // velocity commands and the plant models the simulator's position-target
    // response with a first-order lag plus an acceleration cap.
    float telemetry_velocity_response_tau_s;
    float telemetry_velocity_max_accel_m_s2;
    float telemetry_velocity_teacher_speed_m_s;
    // v3379 SITL plant (logs/sitl/plant_model.json, July 2026 live probes).
    float sitl_rate_gain_roll;
    float sitl_rate_gain_pitch;
    float sitl_rate_gain_yaw;
    float sitl_hover_thrust;
    float sitl_min_thrust;
    float sitl_max_thrust;
    float sitl_vertical_accel_per_thrust;
    float sitl_horizontal_accel_scale;
    float sitl_braking_accel_scale;
    float sitl_lateral_accel_scale;
    float sitl_gate_transition_min_forward_speed;
    int sitl_gate_transition_min_forward_speed_randomize;
    float sitl_gate_transition_min_forward_speed_min;
    float sitl_gate_transition_min_forward_speed_max;
    int sitl_gate_transition_from_gate_index;
    float sitl_gate_transition_max_accel_m_s2;
    int sitl_gate_transition_preserve_velocity_direction;
    int sitl_gate_transition_delay_steps;
    float sitl_gravity_m_s2;
    float sitl_linear_drag;
    float sitl_rate_lag_tau_s;
    float sitl_max_roll_rad;
    float sitl_max_pitch_rad;
    float sitl_max_yaw_rad;
    int sitl_lock_hover_thrust;
    int sitl_lock_roll;
    int sitl_lock_yaw;
    float sitl_attitude_kp;
    float sitl_body_rate_min_command_rad_s;
    float sitl_attitude_error_deadband_rad;
    float sitl_max_body_rate_command_rad_s;
    int sitl_plant_domain_randomize;
    float sitl_rate_gain_jitter_frac;
    float sitl_hover_thrust_jitter;
    float sitl_rate_lag_jitter_frac;
    float sitl_linear_drag_jitter_frac;
    float sitl_gate_obs_dropout_prob;
    float sitl_gate_obs_dropout_range_m;
    int sitl_gate_obs_dropout_from_index;
    int sitl_gate_obs_sample_interval_steps;
    int sitl_gate_motion_predict_dropout;
    float sitl_gate_motion_control_accel_gain;
    float sitl_gate_motion_initial_vx;
    float sitl_gate_motion_initial_vy;
    float sitl_gate_motion_initial_vz;
    float sitl_gate_range_bias_m;
    int gate_position_domain_randomize;
    // Probability that an episode receives position jitter when domain
    // randomization is enabled. A value below one mixes exact nominal-course
    // anchor episodes into the same PPO batch.
    float gate_position_domain_randomize_probability;
    // Optional teaching aperture applied only to position-randomized gates in
    // randomized episodes. Zero preserves each gate's nominal radius.
    float gate_position_randomized_radius;
    int gate_position_randomize_from_index;
    float gate_position_jitter_x;
    float gate_position_jitter_y;
    float gate_position_jitter_z;
    // Default-off validation/rejection layer over the existing randomized
    // course draws. This guarantees ordered, non-overlapping, reachable
    // segments without changing historical randomization streams.
    int gate_position_require_valid_course;
    float gate_position_min_forward_gap_m;
    float gate_position_max_segment_distance_m;
    int gate_position_resample_attempts;
    int observable_gate_index;
    float observable_gate_index_denominator;
    // Default-off long-course ABI: preserve the fixed public index/scale
    // value above one instead of saturating it. Official completion never
    // derives from this scalar; it remains an observable progress feature.
    int observable_gate_progress_unbounded;
    // New six-gate policy contract: observation 23 is normalized official
    // progress and observations 24..31 are reserved zeros. The legacy
    // three-phase adapter remains unchanged when this flag is disabled.
    int observable_gate_progress;
    // Optional deployable capacity extension: observations 24..29 are a
    // six-way one-hot encoding of the same official gate index used by
    // observation 23. Observations 30..31 remain zero. Default off preserves
    // the retained six-gate ABI exactly.
    int observable_gate_phase_onehot;
    // Evaluation-only cap on completed episodes retained by each vector env.
    // Zero leaves training and ordinary evaluation behavior unchanged.
    int evaluation_episode_limit;
    // Evaluation-only number of reset samples to consume before the first
    // scored episode. This supports representative deterministic strata
    // without running every easier preceding episode.
    int evaluation_episode_offset;
    int evaluation_episode_offset_applied;
    float sitl_nominal_rate_gain_roll;
    float sitl_nominal_rate_gain_pitch;
    float sitl_nominal_rate_gain_yaw;
    float sitl_nominal_hover_thrust;
    float sitl_nominal_rate_lag_tau_s;
    float sitl_nominal_linear_drag;

    int use_custom_gate_layout;
    float custom_gate_pos[DRONE_RACE_MAX_GATES][3];
    float custom_gate_radius[DRONE_RACE_MAX_GATES];
    int use_custom_start;
    int start_gate_index;
    float start_elapsed_time;
    // Optional symmetric reset distribution around an empirical segment
    // entry state. Zero amplitudes preserve the historical fixed reset and
    // its RNG stream exactly.
    float start_elapsed_time_jitter;
    int mixed_start_curriculum;
    float segment_start_probability;
    // SkyDreamer-style discovery curriculum. This is independent from the
    // empirical single-segment reset above: a sampled episode begins behind a
    // uniformly selected active gate, while the actor receives no gate index.
    int gate_local_start_curriculum;
    float gate_local_start_probability;
    float gate_local_start_offset_min;
    float gate_local_start_offset_max;
    // Training-only active-gate sampling range [min, max_exclusive). A
    // nonpositive maximum preserves the historical full [0, num_gates)
    // draw and its RNG stream exactly.
    int gate_local_start_gate_min;
    int gate_local_start_gate_max_exclusive;
    float custom_start_pos[3];
    float custom_start_pos_jitter[3];
    // Historical spawn noise is separate from the explicit segment-start
    // distribution so measured-state replay can disable it without changing
    // ordinary training defaults.
    float reset_position_noise_xy;
    float reset_position_noise_z;
    float custom_start_vel[3];
    float custom_start_vel_jitter[3];
    float custom_start_quat[4];
    float custom_start_attitude_jitter[3];
    float custom_start_omega[3];
    float custom_start_omega_jitter[3];

    Target gates[DRONE_RACE_MAX_GATES];
    float between_gate_distance[DRONE_RACE_MAX_GATES];
    DroneRaceCourseSpline teacher_course_lateral_spline;
    DroneRaceCourseSpline teacher_course_vertical_spline;
    int teacher_course_spline_valid;
    DroneRaceAgent* agents;
} DroneRace;

void init(DroneRace* env);
void c_reset(DroneRace* env);
void c_step(DroneRace* env);
void c_render(DroneRace* env);
void c_close(DroneRace* env);
