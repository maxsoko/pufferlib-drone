#include "drone_race.h"
#include <stdio.h>
#include <time.h>

static inline float race_dist3(Vec3 a, Vec3 b) {
    return norm3(sub3(a, b));
}

typedef struct {
    float floor_margin;
    float time_to_floor;
    float stop_margin;
    float max_up_accel;
} FloorRisk;

static const float FLOOR_TTF_SENTINEL = 999.0f;
static const float FLOOR_RISK_TTF_SECONDS = 0.25f;
static const float TS002_CAMERA_UPTILT_RAD = 0.0f;
static const float TS002_CAMERA_HALF_FOV_RAD = 0.7853981634f;
static const float TS002_GATE_INNER_WIDTH_M = 1.5f;

typedef struct {
    float roll;
    float pitch;
    float yaw;
} EulerAngles;

typedef struct {
    float pitch;
    float roll;
    float thrust;
    float yaw;
    bool pitch_active;
    bool roll_active;
    bool thrust_active;
    bool yaw_active;
    bool valid;
} ActionTeacherTarget;

static EulerAngles quat_to_euler(Quat q) {
    float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);

    return (EulerAngles){
        .roll = atan2f(sinr_cosp, cosr_cosp),
        .pitch = asinf(clampf(sinp, -1.0f, 1.0f)),
        .yaw = atan2f(siny_cosp, cosy_cosp),
    };
}

static Quat euler_to_quat(EulerAngles e) {
    float cr = cosf(e.roll * 0.5f);
    float sr = sinf(e.roll * 0.5f);
    float cp = cosf(e.pitch * 0.5f);
    float sp = sinf(e.pitch * 0.5f);
    float cy = cosf(e.yaw * 0.5f);
    float sy = sinf(e.yaw * 0.5f);
    return (Quat){
        .w = cr * cp * cy + sr * sp * sy,
        .x = sr * cp * cy - cr * sp * sy,
        .y = cr * sp * cy + sr * cp * sy,
        .z = cr * cp * sy - sr * sp * cy,
    };
}

static float symmetric_reset_jitter(float amplitude, unsigned int* rng) {
    amplitude = fabsf(amplitude);
    // Avoid consuming RNG when disabled. Existing fixed-start curricula depend
    // on their historical reset stream for exact checkpoint comparisons.
    return amplitude > 0.0f ? rndf(-amplitude, amplitude, rng) : 0.0f;
}

static void apply_interface_defaults(DroneRace* env) {
    if (env->interface_mode != DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW
            && env->interface_mode != DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT
            && env->interface_mode != DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY) {
        env->interface_mode = DRONE_RACE_INTERFACE_NATIVE_MOTOR;
    }
    if (env->max_cmd_forward <= 0.0f) env->max_cmd_forward = 2.0f;
    if (env->max_cmd_lateral <= 0.0f) env->max_cmd_lateral = 1.0f;
    if (env->max_cmd_vertical <= 0.0f) env->max_cmd_vertical = 0.8f;
    if (env->max_cmd_yaw_rate <= 0.0f) env->max_cmd_yaw_rate = 1.0f;
    if (env->setpoint_vel_kp <= 0.0f) env->setpoint_vel_kp = 0.20f;
    if (env->setpoint_att_kp <= 0.0f) env->setpoint_att_kp = 0.20f;
    if (env->setpoint_rate_kd <= 0.0f) env->setpoint_rate_kd = 0.04f;
    if (env->setpoint_yaw_rate_kp <= 0.0f) env->setpoint_yaw_rate_kp = 0.08f;
    if (env->setpoint_thrust_vel_kp <= 0.0f) env->setpoint_thrust_vel_kp = 0.20f;
    if (env->setpoint_max_tilt <= 0.0f) env->setpoint_max_tilt = 0.35f;
    if (env->setpoint_max_motor_delta <= 0.0f) env->setpoint_max_motor_delta = 0.35f;
    if (env->telemetry_velocity_response_tau_s <= 0.0f) {
        env->telemetry_velocity_response_tau_s = 0.25f;
    }
    if (env->telemetry_velocity_max_accel_m_s2 <= 0.0f) {
        env->telemetry_velocity_max_accel_m_s2 = 8.0f;
    }
    if (env->telemetry_velocity_teacher_speed_m_s <= 0.0f) {
        env->telemetry_velocity_teacher_speed_m_s = 8.0f;
    }
    if (env->teacher_pitch_weight == 0.0f
            && env->teacher_roll_weight == 0.0f
            && env->teacher_thrust_weight == 0.0f) {
        env->teacher_pitch_weight = 1.0f;
        env->teacher_roll_weight = 1.0f;
        env->teacher_thrust_weight = 1.0f;
    }
    if (env->teacher_roll_until_gate_index <= 0) {
        env->teacher_roll_until_gate_index = DRONE_RACE_MAX_GATES;
    }
    env->teacher_action_blend = clampf(env->teacher_action_blend, 0.0f, 1.0f);
    if (env->sitl_rate_gain_roll == 0.0f) env->sitl_rate_gain_roll = 2.46f;
    if (env->sitl_rate_gain_pitch == 0.0f) env->sitl_rate_gain_pitch = -2.42f;
    if (env->sitl_rate_gain_yaw == 0.0f) env->sitl_rate_gain_yaw = 2.213f;
    if (env->sitl_hover_thrust <= 0.0f) env->sitl_hover_thrust = 0.27f;
    if (env->sitl_min_thrust <= 0.0f) env->sitl_min_thrust = 0.18f;
    if (env->sitl_max_thrust <= env->sitl_hover_thrust) env->sitl_max_thrust = 0.42f;
    if (env->sitl_vertical_accel_per_thrust <= 0.0f) env->sitl_vertical_accel_per_thrust = 36.32f;
    if (env->sitl_horizontal_accel_scale <= 0.0f) env->sitl_horizontal_accel_scale = 1.0f;
    if (env->sitl_braking_accel_scale <= 0.0f) env->sitl_braking_accel_scale = 1.0f;
    if (env->sitl_lateral_accel_scale <= 0.0f) env->sitl_lateral_accel_scale = 1.0f;
    if (env->sitl_gravity_m_s2 <= 0.0f) env->sitl_gravity_m_s2 = 9.80665f;
    if (env->sitl_rate_lag_tau_s <= 0.0f) env->sitl_rate_lag_tau_s = 0.12f;
    if (env->sitl_max_roll_rad <= 0.0f) env->sitl_max_roll_rad = 0.5f;
    if (env->sitl_max_pitch_rad <= 0.0f) env->sitl_max_pitch_rad = 0.5f;
    if (env->sitl_max_yaw_rad <= 0.0f) env->sitl_max_yaw_rad = 3.1415926535f;
    if (env->sitl_attitude_kp <= 0.0f) env->sitl_attitude_kp = 1.5f;
    if (env->sitl_body_rate_min_command_rad_s <= 0.0f) {
        env->sitl_body_rate_min_command_rad_s = 0.05f;
    }
    if (env->sitl_attitude_error_deadband_rad <= 0.0f) {
        env->sitl_attitude_error_deadband_rad = 0.004f;
    }
    if (env->sitl_max_body_rate_command_rad_s <= 0.0f) {
        env->sitl_max_body_rate_command_rad_s = 0.4f;
    }
    if (env->visual_camera_hz < 0.0f) env->visual_camera_hz = 0.0f;
    if (env->visual_focal_x_px <= 0.0f) env->visual_focal_x_px = 32.0f;
    if (env->visual_focal_y_px <= 0.0f) env->visual_focal_y_px = 32.0f;
    if (env->visual_line_thickness_px <= 0.0f) {
        env->visual_line_thickness_px = 1.0f;
    }
    env->visual_camera_dropout_prob = clampf(
        env->visual_camera_dropout_prob, 0.0f, 1.0f);
    env->visual_edge_dropout_prob = clampf(
        env->visual_edge_dropout_prob, 0.0f, 1.0f);
    env->visual_edge_corrupt_prob = clampf(
        env->visual_edge_corrupt_prob, 0.0f, 1.0f);
    if (env->visual_false_segments < 0) env->visual_false_segments = 0;
    if (env->sitl_linear_drag <= 0.0f && env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT) {
        env->sitl_linear_drag = 0.05f;
    }
}

static void snapshot_sitl_nominals(DroneRace* env) {
    env->sitl_nominal_rate_gain_roll = env->sitl_rate_gain_roll;
    env->sitl_nominal_rate_gain_pitch = env->sitl_rate_gain_pitch;
    env->sitl_nominal_rate_gain_yaw = env->sitl_rate_gain_yaw;
    env->sitl_nominal_hover_thrust = env->sitl_hover_thrust;
    env->sitl_nominal_rate_lag_tau_s = env->sitl_rate_lag_tau_s;
    env->sitl_nominal_linear_drag = env->sitl_linear_drag;
}

static void apply_sitl_domain_randomize(DroneRace* env) {
    if (!env->sitl_plant_domain_randomize) {
        return;
    }
    float rate_j = env->sitl_rate_gain_jitter_frac;
    if (rate_j > 0.0f) {
        env->sitl_rate_gain_roll = env->sitl_nominal_rate_gain_roll
            * (1.0f + rndf(-rate_j, rate_j, &env->rng));
        env->sitl_rate_gain_pitch = env->sitl_nominal_rate_gain_pitch
            * (1.0f + rndf(-rate_j, rate_j, &env->rng));
        env->sitl_rate_gain_yaw = env->sitl_nominal_rate_gain_yaw
            * (1.0f + rndf(-rate_j, rate_j, &env->rng));
    }
    if (env->sitl_hover_thrust_jitter > 0.0f) {
        env->sitl_hover_thrust = clampf(
            env->sitl_nominal_hover_thrust + rndf(
                -env->sitl_hover_thrust_jitter,
                env->sitl_hover_thrust_jitter,
                &env->rng),
            0.05f,
            0.95f);
    }
    float lag_j = env->sitl_rate_lag_jitter_frac;
    if (lag_j > 0.0f) {
        env->sitl_rate_lag_tau_s = fmaxf(
            0.02f,
            env->sitl_nominal_rate_lag_tau_s
                * (1.0f + rndf(-lag_j, lag_j, &env->rng)));
    }
    float drag_j = env->sitl_linear_drag_jitter_frac;
    if (drag_j > 0.0f) {
        env->sitl_linear_drag = fmaxf(
            0.0f,
            env->sitl_nominal_linear_drag
                * (1.0f + rndf(-drag_j, drag_j, &env->rng)));
    }
}

static float max_vertical_accel(const Drone* drone) {
    float max_thrust = 4.0f * drone->params.k_thrust * drone->params.max_rpm * drone->params.max_rpm;
    Vec3 force_world = quat_rotate(drone->state.quat, (Vec3){0.0f, 0.0f, max_thrust});
    return force_world.z / drone->params.mass - drone->params.gravity;
}

static FloorRisk compute_floor_risk(DroneRace* env, DroneRaceAgent* agent) {
    float floor_margin = agent->drone.state.pos.z - env->crash_height;
    float descent_speed = fmaxf(-agent->drone.state.vel.z, 0.0f);
    float max_up_accel = max_vertical_accel(&agent->drone);
    float time_to_floor = FLOOR_TTF_SENTINEL;
    float stop_margin = floor_margin;

    if (floor_margin <= 0.0f) {
        time_to_floor = 0.0f;
    } else if (descent_speed > 1e-4f) {
        time_to_floor = floor_margin / descent_speed;
        if (max_up_accel > 1e-4f) {
            float stopping_distance = descent_speed * descent_speed / (2.0f * max_up_accel);
            stop_margin = floor_margin - stopping_distance;
        } else {
            stop_margin = -FLOOR_TTF_SENTINEL;
        }
    }

    return (FloorRisk){
        .floor_margin = floor_margin,
        .time_to_floor = time_to_floor,
        .stop_margin = stop_margin,
        .max_up_accel = max_up_accel,
    };
}

static void update_floor_risk_stats(DroneRaceAgent* agent, FloorRisk risk) {
    if (risk.floor_margin < 0.0f || risk.time_to_floor >= FLOOR_TTF_SENTINEL) {
        return;
    }

    agent->floor_risk_sampled = 1;
    if (risk.time_to_floor < agent->min_floor_ttf) {
        agent->min_floor_ttf = risk.time_to_floor;
    }
    if (risk.stop_margin < agent->min_floor_stop_margin) {
        agent->min_floor_stop_margin = risk.stop_margin;
    }
    if (risk.time_to_floor < FLOOR_RISK_TTF_SECONDS) {
        agent->floor_impact_risk = 1;
        agent->floor_risk_steps += 1.0f;
    }
    if (risk.stop_margin < 0.0f) {
        agent->floor_stop_violation = 1;
        agent->floor_stop_violation_steps += 1.0f;
    }
}

static int select_num_gates_for_env(
        int configured_num_gates,
        int randomize_per_env,
        int minimum,
        int maximum,
        unsigned int env_index,
        unsigned int seed) {
    configured_num_gates = configured_num_gates < 1
        ? 1
        : configured_num_gates;
    configured_num_gates = configured_num_gates > DRONE_RACE_MAX_GATES
        ? DRONE_RACE_MAX_GATES
        : configured_num_gates;
    if (!randomize_per_env) return configured_num_gates;

    minimum = minimum < 1 ? 1 : minimum;
    maximum = maximum < 1 ? 1 : maximum;
    minimum = minimum > DRONE_RACE_MAX_GATES
        ? DRONE_RACE_MAX_GATES
        : minimum;
    maximum = maximum > DRONE_RACE_MAX_GATES
        ? DRONE_RACE_MAX_GATES
        : maximum;
    if (minimum > maximum) {
        int swap = minimum;
        minimum = maximum;
        maximum = swap;
    }
    unsigned int span = (unsigned int)(maximum - minimum + 1);
    // Cyclic stratification is exactly uniform for vectors whose size is a
    // multiple of span. The independent seed only rotates assignment; it does
    // not consume or perturb the historical vehicle/course RNG stream.
    unsigned int slot = (env_index + seed % span) % span;
    return minimum + (int)slot;
}

static Vec3 course_gate_position(
        DroneRace* env, int gate_index, float geometry_scale) {
    float x;
    float y;
    float z;
    if (env->use_custom_gate_layout) {
        x = env->custom_gate_pos[gate_index][0];
        y = env->custom_gate_pos[gate_index][1];
        z = env->custom_gate_pos[gate_index][2];
    } else {
        y = (gate_index % 2 == 0)
            ? env->gate_lateral_amplitude
            : -env->gate_lateral_amplitude;
        x = gate_index * env->gate_spacing;
        z = env->gate_altitude;
    }

    // Geometry curricula preserve the official longitudinal timing and morph
    // only the lateral/vertical displacement around the configured start.
    geometry_scale = clampf(geometry_scale, 0.0f, 1.0f);
    float origin_y = env->use_custom_start ? env->custom_start_pos[1] : 0.0f;
    float origin_z = env->use_custom_start ? env->custom_start_pos[2] : 0.0f;
    return (Vec3){
        x,
        origin_y + geometry_scale * (y - origin_y),
        origin_z + geometry_scale * (z - origin_z),
    };
}

static void build_course(DroneRace* env) {
    if (env->num_gates < 1) env->num_gates = 1;
    if (env->num_gates > DRONE_RACE_MAX_GATES) env->num_gates = DRONE_RACE_MAX_GATES;

    for (int i = 0; i < env->num_gates; i++) {
        // Training can introduce the real lateral/vertical course shape
        // continuously while retaining the official start, x spacing, gate
        // count, recurrent history, and observable relative-pose contract.
        // Scale 0 is a straight level course; scale 1 is nominal geometry.
        env->gates[i] = (Target){
            .pos = course_gate_position(env, i, env->course_geometry_scale),
            .vel = {0.0f, 0.0f, 0.0f},
            .orientation = {1.0f, 0.0f, 0.0f, 0.0f},
            .normal = {1.0f, 0.0f, 0.0f},
            .radius = env->custom_gate_radius[i] > 0.0f
                ? env->custom_gate_radius[i]
                : env->gate_radius,
        };
        env->between_gate_distance[i] = 0.0f;
    }

    for (int i = 0; i < env->num_gates - 1; i++) {
        env->between_gate_distance[i] = race_dist3(env->gates[i + 1].pos, env->gates[i].pos);
    }
}

static int build_course_natural_spline(
        DroneRaceCourseSpline* spline,
        const float* x,
        const float* values,
        int points) {
    if (points < 2 || points > DRONE_RACE_MAX_GATES + 1) return 0;
    float h[DRONE_RACE_MAX_GATES] = {0};
    float alpha[DRONE_RACE_MAX_GATES + 1] = {0};
    float l[DRONE_RACE_MAX_GATES + 1] = {0};
    float mu[DRONE_RACE_MAX_GATES + 1] = {0};
    float z[DRONE_RACE_MAX_GATES + 1] = {0};
    float c_knots[DRONE_RACE_MAX_GATES + 1] = {0};

    for (int i = 0; i < points; i++) spline->x[i] = x[i];
    for (int i = 0; i < points - 1; i++) {
        h[i] = x[i + 1] - x[i];
        if (h[i] <= 1e-4f) return 0;
    }
    for (int i = 1; i < points - 1; i++) {
        alpha[i] = 3.0f * (values[i + 1] - values[i]) / h[i]
            - 3.0f * (values[i] - values[i - 1]) / h[i - 1];
    }
    l[0] = 1.0f;
    for (int i = 1; i < points - 1; i++) {
        l[i] = 2.0f * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        if (fabsf(l[i]) <= 1e-6f) return 0;
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[points - 1] = 1.0f;
    for (int j = points - 2; j >= 0; j--) {
        c_knots[j] = z[j] - mu[j] * c_knots[j + 1];
        spline->a[j] = values[j];
        spline->b[j] = (values[j + 1] - values[j]) / h[j]
            - h[j] * (c_knots[j + 1] + 2.0f * c_knots[j]) / 3.0f;
        spline->c[j] = c_knots[j];
        spline->d[j] = (c_knots[j + 1] - c_knots[j]) / (3.0f * h[j]);
    }
    spline->points = points;
    return 1;
}

static int build_course_exit_spline(
        DroneRaceCourseSpline* spline,
        const float* x,
        const float* values,
        int points,
        int exit_tangent_from_point) {
    DroneRaceCourseSpline natural = {0};
    if (!build_course_natural_spline(&natural, x, values, points)) return 0;
    float slopes[DRONE_RACE_MAX_GATES + 1] = {0};

    for (int i = 0; i < points; i++) spline->x[i] = x[i];
    for (int i = 0; i < points - 1; i++) {
        float h = x[i + 1] - x[i];
        if (h <= 1e-4f) return 0;
        slopes[i] = i >= exit_tangent_from_point
            ? (values[i + 1] - values[i]) / h
            : natural.b[i];
    }
    if (exit_tangent_from_point < points - 1) {
        slopes[points - 1] = slopes[points - 2];
    } else {
        int final_segment = points - 2;
        float dx = x[points - 1] - x[final_segment];
        slopes[points - 1] = natural.b[final_segment]
            + 2.0f * natural.c[final_segment] * dx
            + 3.0f * natural.d[final_segment] * dx * dx;
    }
    for (int i = 0; i < points - 1; i++) {
        float h = x[i + 1] - x[i];
        float secant = (values[i + 1] - values[i]) / h;
        spline->a[i] = values[i];
        spline->b[i] = slopes[i];
        spline->c[i] = (3.0f * secant - 2.0f * slopes[i] - slopes[i + 1]) / h;
        spline->d[i] = (slopes[i] + slopes[i + 1] - 2.0f * secant) / (h * h);
    }
    spline->points = points;
    return 1;
}

static void evaluate_course_spline(
        const DroneRaceCourseSpline* spline,
        float x,
        float* value,
        float* first_derivative,
        float* second_derivative) {
    int segment = 0;
    while (segment + 1 < spline->points - 1
            && x > spline->x[segment + 1]) {
        segment++;
    }
    float dx = x - spline->x[segment];
    *value = spline->a[segment]
        + spline->b[segment] * dx
        + spline->c[segment] * dx * dx
        + spline->d[segment] * dx * dx * dx;
    *first_derivative = spline->b[segment]
        + 2.0f * spline->c[segment] * dx
        + 3.0f * spline->d[segment] * dx * dx;
    *second_derivative = 2.0f * spline->c[segment]
        + 6.0f * spline->d[segment] * dx;
}

static Target* agent_gate(DroneRace* env, DroneRaceAgent* agent, int gate_idx);

static void minimum_jerk_basis(
        float s, float* value, float* first, float* second) {
    s = clampf(s, 0.0f, 1.0f);
    float s2 = s * s;
    float one_minus_s = 1.0f - s;
    *value = s2 * s * (10.0f + s * (-15.0f + 6.0f * s));
    *first = 30.0f * s2 * one_minus_s * one_minus_s;
    *second = 60.0f * s * one_minus_s * (1.0f - 2.0f * s);
}

static void build_course_teacher_profile(DroneRace* env) {
    env->teacher_course_spline_valid = 0;
    if (!env->teacher_course_spline
            || env->gate_position_domain_randomize
            || env->course_geometry_scale_randomize) return;

    int points = env->num_gates + 1;
    float x[DRONE_RACE_MAX_GATES + 1] = {0};
    float y[DRONE_RACE_MAX_GATES + 1] = {0};
    float z[DRONE_RACE_MAX_GATES + 1] = {0};

    // The custom v3385 profile starts at the official origin. Synthetic
    // courses retain their existing gate-aligned reset geometry.
    if (env->use_custom_gate_layout) {
        x[0] = env->gates[0].pos.x - env->start_offset;
        y[0] = 0.0f;
        z[0] = 0.0f;
    } else {
        Vec3 course_start = sub3(
            env->gates[0].pos,
            scalmul3(env->gates[0].normal, env->start_offset));
        x[0] = course_start.x;
        y[0] = course_start.y;
        z[0] = course_start.z;
    }
    for (int gate = 0; gate < env->num_gates; gate++) {
        x[gate + 1] = env->gates[gate].pos.x;
        y[gate + 1] = env->gates[gate].pos.y;
        z[gate + 1] = env->gates[gate].pos.z;
    }

    // From gate 2 onward, use the outgoing segment tangent. This is the
    // validated v3385 racing-line profile and makes exit velocity part of the
    // dense action target instead of waiting for sparse downstream failure.
    const int exit_tangent_from_point = 3;
    env->teacher_course_spline_valid = build_course_exit_spline(
            &env->teacher_course_lateral_spline,
            x, y, points, exit_tangent_from_point)
        && build_course_exit_spline(
            &env->teacher_course_vertical_spline,
            x, z, points, exit_tangent_from_point);
}

static void course_spline_teacher_action(
        DroneRace* env,
        DroneRaceAgent* agent,
        float* teacher_roll,
        float* teacher_thrust) {
    if (!env->teacher_course_spline_valid) return;

    float desired_y;
    float desired_dy_dx;
    float desired_d2y_dx2;
    float desired_z;
    float desired_dz_dx;
    float desired_d2z_dx2;
    float course_x = agent->drone.state.pos.x;
    evaluate_course_spline(
        &env->teacher_course_lateral_spline,
        course_x,
        &desired_y,
        &desired_dy_dx,
        &desired_d2y_dx2);
    evaluate_course_spline(
        &env->teacher_course_vertical_spline,
        course_x,
        &desired_z,
        &desired_dz_dx,
        &desired_d2z_dx2);

    // The observable controller's validated 6.1 m derivative lookahead begins
    // at gate 2. No future-gate value is added to the policy observation.
    if (agent->current_gate >= 2) {
        float lookahead_x = fminf(
            course_x + 6.10f,
            env->teacher_course_lateral_spline.x[
                env->teacher_course_lateral_spline.points - 1]);
        float ignored;
        evaluate_course_spline(
            &env->teacher_course_lateral_spline,
            lookahead_x,
            &ignored,
            &desired_dy_dx,
            &desired_d2y_dx2);
    }

    Target* active_gate = &env->gates[agent->current_gate];
    float distance_to_gate = fmaxf(active_gate->pos.x - course_x, 0.0f);
    float close_blend = clampf(1.0f - distance_to_gate / 10.0f, 0.0f, 1.0f);
    float forward_speed = fmaxf(agent->drone.state.vel.x, 0.0f);
    float lateral_position_kp = 1.5f + 5.0f * close_blend;
    float accel_y = desired_d2y_dx2 * forward_speed * forward_speed
        + lateral_position_kp * (desired_y - agent->drone.state.pos.y)
        + 2.7f * (desired_dy_dx * forward_speed - agent->drone.state.vel.y);
    float accel_z = desired_d2z_dx2 * forward_speed * forward_speed
        + 2.2f * (desired_z - agent->drone.state.pos.z)
        + 2.0f * (desired_dz_dx * forward_speed - agent->drone.state.vel.z);

    float force_y = accel_y + env->sitl_linear_drag * agent->drone.state.vel.y;
    float force_z = env->sitl_gravity_m_s2 + accel_z
        + env->sitl_linear_drag * agent->drone.state.vel.z;
    EulerAngles attitude = quat_to_euler(agent->drone.state.quat);
    float desired_roll = atan2f(
        -force_y * cosf(attitude.pitch),
        fmaxf(force_z * env->sitl_lateral_accel_scale, 1e-3f));
    desired_roll = clampf(
        desired_roll, -env->sitl_max_roll_rad, env->sitl_max_roll_rad);
    *teacher_roll = desired_roll / fmaxf(env->sitl_max_roll_rad, 1e-4f);

    float vertical_fraction = fmaxf(
        cosf(desired_roll) * cosf(attitude.pitch), 0.50f);
    float cmd_thrust = force_z
        / fmaxf(env->sitl_vertical_accel_per_thrust * vertical_fraction, 1e-4f);
    if (cmd_thrust >= env->sitl_hover_thrust) {
        *teacher_thrust = (cmd_thrust - env->sitl_hover_thrust)
            / fmaxf(env->sitl_max_thrust - env->sitl_hover_thrust, 1e-4f);
    } else {
        *teacher_thrust = (cmd_thrust - env->sitl_hover_thrust)
            / fmaxf(env->sitl_hover_thrust - env->sitl_min_thrust, 1e-4f);
    }
    *teacher_roll = clampf(*teacher_roll, -1.0f, 1.0f);
    *teacher_thrust = clampf(*teacher_thrust, -1.0f, 1.0f);
}

static void course_segment_minimum_jerk_teacher_action(
        DroneRace* env,
        DroneRaceAgent* agent,
        float* teacher_roll,
        float* teacher_thrust) {
    int gate_index = agent->current_gate;
    if (gate_index < 0 || gate_index >= env->num_gates) return;

    Target* active_gate = agent_gate(env, agent, gate_index);
    Vec3 start;
    if (gate_index > 0) {
        start = agent_gate(env, agent, gate_index - 1)->pos;
    } else if (env->use_custom_start) {
        start = (Vec3){
            env->custom_start_pos[0],
            env->custom_start_pos[1],
            env->custom_start_pos[2],
        };
    } else {
        start = sub3(
            active_gate->pos,
            scalmul3(active_gate->normal, env->start_offset));
    }

    float segment_x = active_gate->pos.x - start.x;
    if (segment_x <= 1e-4f) return;
    float basis;
    float basis_ds;
    float basis_d2s;
    minimum_jerk_basis(
        (agent->drone.state.pos.x - start.x) / segment_x,
        &basis, &basis_ds, &basis_d2s);
    float inv_segment_x = 1.0f / segment_x;
    float delta_y = active_gate->pos.y - start.y;
    float delta_z = active_gate->pos.z - start.z;
    float desired_y = start.y + delta_y * basis;
    float desired_dy_dx = delta_y * basis_ds * inv_segment_x;
    float desired_d2y_dx2 = delta_y * basis_d2s
        * inv_segment_x * inv_segment_x;
    float desired_z = start.z + delta_z * basis;
    float desired_dz_dx = delta_z * basis_ds * inv_segment_x;
    float desired_d2z_dx2 = delta_z * basis_d2s
        * inv_segment_x * inv_segment_x;

    float distance_to_gate = fmaxf(active_gate->pos.x
        - agent->drone.state.pos.x, 0.0f);
    float close_blend = clampf(1.0f - distance_to_gate / 10.0f, 0.0f, 1.0f);
    float forward_speed = fmaxf(agent->drone.state.vel.x, 0.0f);
    float lateral_position_kp = 1.5f + 5.0f * close_blend;
    float accel_y = desired_d2y_dx2 * forward_speed * forward_speed
        + lateral_position_kp * (desired_y - agent->drone.state.pos.y)
        + 2.7f * (desired_dy_dx * forward_speed - agent->drone.state.vel.y);
    float accel_z = desired_d2z_dx2 * forward_speed * forward_speed
        + 2.2f * (desired_z - agent->drone.state.pos.z)
        + 2.0f * (desired_dz_dx * forward_speed - agent->drone.state.vel.z);

    float force_y = accel_y + env->sitl_linear_drag * agent->drone.state.vel.y;
    float force_z = env->sitl_gravity_m_s2 + accel_z
        + env->sitl_linear_drag * agent->drone.state.vel.z;
    EulerAngles attitude = quat_to_euler(agent->drone.state.quat);
    float desired_roll = atan2f(
        -force_y * cosf(attitude.pitch),
        fmaxf(force_z * env->sitl_lateral_accel_scale, 1e-3f));
    desired_roll = clampf(
        desired_roll, -env->sitl_max_roll_rad, env->sitl_max_roll_rad);
    *teacher_roll = desired_roll / fmaxf(env->sitl_max_roll_rad, 1e-4f);

    float vertical_fraction = fmaxf(
        cosf(desired_roll) * cosf(attitude.pitch), 0.50f);
    float cmd_thrust = force_z
        / fmaxf(env->sitl_vertical_accel_per_thrust * vertical_fraction, 1e-4f);
    if (cmd_thrust >= env->sitl_hover_thrust) {
        *teacher_thrust = (cmd_thrust - env->sitl_hover_thrust)
            / fmaxf(env->sitl_max_thrust - env->sitl_hover_thrust, 1e-4f);
    } else {
        *teacher_thrust = (cmd_thrust - env->sitl_hover_thrust)
            / fmaxf(env->sitl_hover_thrust - env->sitl_min_thrust, 1e-4f);
    }
    *teacher_roll = clampf(*teacher_roll, -1.0f, 1.0f);
    *teacher_thrust = clampf(*teacher_thrust, -1.0f, 1.0f);
}

static void course_alignment_governed_teacher_action(
        DroneRace* env,
        DroneRaceAgent* agent,
        float* teacher_pitch,
        float* teacher_roll,
        float* teacher_thrust) {
    int gate_index = agent->current_gate;
    if (gate_index < 0 || gate_index >= env->num_gates) return;

    Target* active_gate = agent_gate(env, agent, gate_index);
    Vec3 relative = sub3(active_gate->pos, agent->drone.state.pos);
    float radial_error = sqrtf(
        relative.y * relative.y + relative.z * relative.z);
    float alignment = clampf(1.0f - radial_error / 1.0f, 0.0f, 1.0f);
    float approach_speed = fmaxf(
        env->teacher_pitch_speed_target_m_s, 0.20f);
    float target_forward_speed = 0.20f
        + alignment * (approach_speed - 0.20f);
    *teacher_pitch = clampf(
        0.45f * (agent->drone.state.vel.x - target_forward_speed),
        -0.80f,
        0.80f);

    float accel_y = 0.40f * relative.y
        - 1.20f * agent->drone.state.vel.y;
    float accel_z = 0.80f * relative.z
        - 1.50f * agent->drone.state.vel.z;
    float force_y = accel_y + env->sitl_linear_drag * agent->drone.state.vel.y;
    float force_z = env->sitl_gravity_m_s2 + accel_z
        + env->sitl_linear_drag * agent->drone.state.vel.z;
    EulerAngles attitude = quat_to_euler(agent->drone.state.quat);
    float desired_roll = atan2f(
        -force_y * cosf(attitude.pitch),
        fmaxf(force_z * env->sitl_lateral_accel_scale, 1e-3f));
    desired_roll = clampf(
        desired_roll, -env->sitl_max_roll_rad, env->sitl_max_roll_rad);
    *teacher_roll = desired_roll / fmaxf(env->sitl_max_roll_rad, 1e-4f);

    float vertical_fraction = fmaxf(
        cosf(desired_roll) * cosf(attitude.pitch), 0.50f);
    float cmd_thrust = force_z
        / fmaxf(env->sitl_vertical_accel_per_thrust * vertical_fraction, 1e-4f);
    if (cmd_thrust >= env->sitl_hover_thrust) {
        *teacher_thrust = (cmd_thrust - env->sitl_hover_thrust)
            / fmaxf(env->sitl_max_thrust - env->sitl_hover_thrust, 1e-4f);
    } else {
        *teacher_thrust = (cmd_thrust - env->sitl_hover_thrust)
            / fmaxf(env->sitl_hover_thrust - env->sitl_min_thrust, 1e-4f);
    }
    *teacher_roll = clampf(*teacher_roll, -1.0f, 1.0f);
    *teacher_thrust = clampf(*teacher_thrust, -1.0f, 1.0f);
}

static ActionTeacherTarget action_teacher_target(
        DroneRace* env, DroneRaceAgent* agent, int gate_index) {
    ActionTeacherTarget target = {
        .pitch = env->teacher_pitch_action,
        .roll = 0.0f,
        .thrust = 0.0f,
        .yaw = env->teacher_yaw_action,
        .pitch_active = gate_index >= env->teacher_pitch_from_gate_index,
        .roll_active = gate_index >= env->teacher_roll_from_gate_index
            && gate_index < env->teacher_roll_until_gate_index,
        .thrust_active = gate_index >= env->teacher_thrust_from_gate_index,
        .yaw_active = env->teacher_yaw_control
            && gate_index >= env->teacher_yaw_from_gate_index,
        .valid = gate_index >= 0 && gate_index < env->num_gates,
    };
    if (!target.valid) return target;

    Quat q_inv = quat_inverse(agent->drone.state.quat);
    Target* gate = agent_gate(env, agent, gate_index);
    Vec3 rel_world = sub3(gate->pos, agent->drone.state.pos);
    Vec3 rel_body = quat_rotate(q_inv, rel_world);
    if (env->teacher_pitch_speed_control) {
        target.pitch = clampf(
            env->teacher_pitch_speed_scale * clampf(
                (agent->drone.state.vel.x
                    - env->teacher_pitch_speed_target_m_s)
                    * env->teacher_pitch_speed_gain,
                -0.60f,
                0.60f),
            -1.0f,
            1.0f);
    }
    Vec3 gate_rate_body = {0.0f, 0.0f, 0.0f};
    if (env->teacher_true_velocity_damping) {
        Vec3 relative_rate_world = sub3(gate->vel, agent->drone.state.vel);
        gate_rate_body = quat_rotate(q_inv, relative_rate_world);
    } else if (agent->gate_motion_valid
            && agent->gate_motion_gate_index == gate_index) {
        gate_rate_body = quat_rotate(q_inv, agent->gate_motion_rate_world);
    }
    // Verified v3385 mode-2 convention: positive roll accelerates left
    // (negative body-right), so a gate to the right needs negative roll.
    target.roll = clampf(
        env->teacher_roll_bias
            - rel_body.y * env->teacher_roll_per_m
            - gate_rate_body.y * env->teacher_roll_rate_per_m_s,
        -1.0f,
        1.0f);
    float thrust_position_error = env->teacher_thrust_world_frame
        ? rel_world.z
        : rel_body.z;
    float thrust_rate_error = env->teacher_thrust_world_frame
        ? agent->gate_motion_rate_world.z
        : gate_rate_body.z;
    target.thrust = clampf(
        env->teacher_thrust_bias
            + thrust_position_error * env->teacher_thrust_per_m
            + thrust_rate_error * env->teacher_thrust_rate_per_m_s,
        -1.0f,
        1.0f);
    if (env->teacher_alignment_governor
            && gate_index >= env->teacher_from_gate_index) {
        course_alignment_governed_teacher_action(
            env, agent, &target.pitch, &target.roll, &target.thrust);
    } else if (env->teacher_segment_minimum_jerk
            && gate_index >= env->teacher_from_gate_index) {
        course_segment_minimum_jerk_teacher_action(
            env, agent, &target.roll, &target.thrust);
    } else if (env->teacher_course_spline
            && env->teacher_course_spline_valid
            && gate_index >= env->teacher_from_gate_index) {
        // agent->current_gate equals gate_index when called before the step.
        course_spline_teacher_action(
            env, agent, &target.roll, &target.thrust);
    }
    return target;
}

static bool agent_course_randomized(DroneRace* env) {
    return env->gate_position_domain_randomize
        || env->course_geometry_scale_randomize
        || env->gate_radius_randomize
        || env->gate_radius_profile_mix;
}

static Target* agent_gate(DroneRace* env, DroneRaceAgent* agent, int gate_idx) {
    if (agent_course_randomized(env)) {
        return &agent->randomized_gates[gate_idx];
    }
    return &env->gates[gate_idx];
}

#if DRONE_RACE_OBS_SIZE == DRONE_RACE_VISUAL_OBS_SIZE
typedef struct {
    float x;
    float y;
    int valid;
} DroneRaceVisualPoint;

static void visual_mask_splat(
        float* mask, float x, float y, float thickness, float value) {
    float radius = fmaxf(thickness, 0.25f) + 0.75f;
    int min_x = (int)floorf(x - radius);
    int max_x = (int)ceilf(x + radius);
    int min_y = (int)floorf(y - radius);
    int max_y = (int)ceilf(y + radius);
    for (int py = min_y; py <= max_y; py++) {
        if (py < 0 || py >= DRONE_RACE_VISUAL_HEIGHT) continue;
        for (int px = min_x; px <= max_x; px++) {
            if (px < 0 || px >= DRONE_RACE_VISUAL_WIDTH) continue;
            float dx = ((float)px + 0.5f) - x;
            float dy = ((float)py + 0.5f) - y;
            float distance = sqrtf(dx * dx + dy * dy);
            float confidence = value * clampf(1.0f - distance / radius, 0.0f, 1.0f);
            int index = py * DRONE_RACE_VISUAL_WIDTH + px;
            mask[index] = fmaxf(mask[index], confidence);
        }
    }
}

static void visual_mask_line(
        float* mask,
        DroneRaceVisualPoint a,
        DroneRaceVisualPoint b,
        float thickness,
        float value) {
    if (!a.valid || !b.valid) return;
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    int steps = (int)ceilf(fmaxf(fabsf(dx), fabsf(dy)) * 2.0f);
    if (steps < 1) steps = 1;
    if (steps > 512) steps = 512;
    for (int step = 0; step <= steps; step++) {
        float t = (float)step / (float)steps;
        visual_mask_splat(
            mask,
            a.x + t * dx,
            a.y + t * dy,
            thickness,
            value);
    }
}

static DroneRaceVisualPoint visual_project_world_point(
        const DroneRace* env,
        const DroneRaceAgent* agent,
        Quat body_inverse,
        Quat camera_inverse,
        Vec3 world_point,
        Vec3 camera_rates) {
    Vec3 relative_world = sub3(world_point, agent->drone.state.pos);
    Vec3 relative_body = quat_rotate(body_inverse, relative_world);
    Vec3 relative_camera = quat_rotate(camera_inverse, relative_body);
    if (relative_camera.x <= 0.05f) {
        return (DroneRaceVisualPoint){0.0f, 0.0f, 0};
    }

    float center_x = 0.5f * (float)DRONE_RACE_VISUAL_WIDTH;
    float center_y = 0.5f * (float)DRONE_RACE_VISUAL_HEIGHT;
    float x = center_x
        + env->visual_focal_x_px * relative_camera.y / relative_camera.x;
    float y = center_y
        - env->visual_focal_y_px * relative_camera.z / relative_camera.x;

    // SkyDreamer's rolling-shutter approximation: yaw shears rows and pitch
    // scales them around the principal point. Zero preserves an ideal frame.
    float shutter = fmaxf(env->visual_rolling_shutter_s, 0.0f);
    if (shutter > 0.0f) {
        x -= shutter * camera_rates.z * (y - center_y);
        y = center_y + (1.0f + shutter * camera_rates.y) * (y - center_y);
    }
    if (!isfinite(x) || !isfinite(y)
            || fabsf(x) > 4.0f * DRONE_RACE_VISUAL_WIDTH
            || fabsf(y) > 4.0f * DRONE_RACE_VISUAL_HEIGHT) {
        return (DroneRaceVisualPoint){0.0f, 0.0f, 0};
    }
    return (DroneRaceVisualPoint){x, y, 1};
}

static void visual_render_edge_segments(
        DroneRace* env,
        DroneRaceAgent* agent,
        float* mask,
        DroneRaceVisualPoint a,
        DroneRaceVisualPoint b) {
    const int subdivisions = 5;
    for (int segment = 0; segment < subdivisions; segment++) {
        if (rndf(0.0f, 1.0f, &agent->visual_rng)
                < clampf(env->visual_edge_dropout_prob, 0.0f, 1.0f)) {
            continue;
        }
        float t0 = (float)segment / (float)subdivisions;
        float t1 = (float)(segment + 1) / (float)subdivisions;
        DroneRaceVisualPoint p0 = {
            a.x + t0 * (b.x - a.x),
            a.y + t0 * (b.y - a.y),
            a.valid && b.valid,
        };
        DroneRaceVisualPoint p1 = {
            a.x + t1 * (b.x - a.x),
            a.y + t1 * (b.y - a.y),
            a.valid && b.valid,
        };
        if (rndf(0.0f, 1.0f, &agent->visual_rng)
                < clampf(env->visual_edge_corrupt_prob, 0.0f, 1.0f)) {
            // Geles et al. randomize a fraction of projected edge segments.
            // Keep corruption in pixel space so it cannot leak gate state.
            p0.x = rndf(0.0f, DRONE_RACE_VISUAL_WIDTH - 1.0f, &agent->visual_rng);
            p0.y = rndf(0.0f, DRONE_RACE_VISUAL_HEIGHT - 1.0f, &agent->visual_rng);
            p1.x = rndf(0.0f, DRONE_RACE_VISUAL_WIDTH - 1.0f, &agent->visual_rng);
            p1.y = rndf(0.0f, DRONE_RACE_VISUAL_HEIGHT - 1.0f, &agent->visual_rng);
            p0.valid = 1;
            p1.valid = 1;
        }
        visual_mask_line(
            mask,
            p0,
            p1,
            fmaxf(env->visual_line_thickness_px, 0.25f),
            1.0f);
    }
}

static void visual_render_mask(
        DroneRace* env, DroneRaceAgent* agent, float* mask) {
    memset(mask, 0, DRONE_RACE_VISUAL_MASK_SIZE * sizeof(float));
    Quat body_inverse = quat_inverse(agent->drone.state.quat);
    Quat camera_orientation = euler_to_quat((EulerAngles){
        .roll = agent->visual_camera_roll_rad,
        .pitch = agent->visual_camera_pitch_rad,
        .yaw = agent->visual_camera_yaw_rad,
    });
    Quat camera_inverse = quat_inverse(camera_orientation);
    Vec3 camera_rates = quat_rotate(camera_inverse, agent->drone.state.omega);
    const int edge_start[4] = {0, 1, 3, 2};
    const int edge_end[4] = {1, 3, 2, 0};

    // Render every physical gate that is in view. The actor receives no active
    // gate index; recurrent course phase must disambiguate repeated views.
    for (int gate_index = 0; gate_index < env->num_gates; gate_index++) {
        Target* gate = agent_gate(env, agent, gate_index);
        float radius = fmaxf(gate->radius, 1e-3f);
        Vec3 local_corners[4] = {
            {0.0f, -radius, -radius},
            {0.0f, radius, -radius},
            {0.0f, -radius, radius},
            {0.0f, radius, radius},
        };
        DroneRaceVisualPoint projected[4];
        for (int corner = 0; corner < 4; corner++) {
            Vec3 world_corner = add3(
                gate->pos,
                quat_rotate(gate->orientation, local_corners[corner]));
            projected[corner] = visual_project_world_point(
                env,
                agent,
                body_inverse,
                camera_inverse,
                world_corner,
                camera_rates);
        }
        for (int edge = 0; edge < 4; edge++) {
            visual_render_edge_segments(
                env,
                agent,
                mask,
                projected[edge_start[edge]],
                projected[edge_end[edge]]);
        }
    }

    for (int segment = 0; segment < env->visual_false_segments; segment++) {
        DroneRaceVisualPoint a = {
            rndf(0.0f, DRONE_RACE_VISUAL_WIDTH - 1.0f, &agent->visual_rng),
            rndf(0.0f, DRONE_RACE_VISUAL_HEIGHT - 1.0f, &agent->visual_rng),
            1,
        };
        DroneRaceVisualPoint b = {
            rndf(0.0f, DRONE_RACE_VISUAL_WIDTH - 1.0f, &agent->visual_rng),
            rndf(0.0f, DRONE_RACE_VISUAL_HEIGHT - 1.0f, &agent->visual_rng),
            1,
        };
        visual_mask_line(
            mask,
            a,
            b,
            fmaxf(env->visual_line_thickness_px, 0.25f),
            rndf(0.25f, 1.0f, &agent->visual_rng));
    }
}

static float visual_log_ratio(float value, float reference) {
    value = fmaxf(value, 1e-12f);
    reference = fmaxf(reference, 1e-12f);
    return tanhf(logf(value / reference));
}

static void compute_visual_observation(DroneRace* env, int agent_index) {
    DroneRaceAgent* agent = &env->agents[agent_index];
    float* obs = &env->observations[agent_index * DRONE_RACE_OBS_SIZE];
    float frame_interval = env->visual_camera_hz > 0.0f
        ? 1.0f / env->visual_camera_hz
        : 0.0f;
    int new_frame = agent->step_count == 0;
    if (!new_frame) {
        agent->visual_camera_accumulator_s += env->dt;
        if (frame_interval <= 0.0f
                || agent->visual_camera_accumulator_s + 1e-7f >= frame_interval) {
            new_frame = 1;
            if (frame_interval > 0.0f) {
                agent->visual_camera_accumulator_s -= frame_interval;
                if (agent->visual_camera_accumulator_s >= frame_interval) {
                    agent->visual_camera_accumulator_s = fmodf(
                        agent->visual_camera_accumulator_s, frame_interval);
                }
            }
        }
    }
    if (new_frame && agent->step_count > 0
            && rndf(0.0f, 1.0f, &agent->visual_rng)
                < clampf(env->visual_camera_dropout_prob, 0.0f, 1.0f)) {
        new_frame = 0;
    }
    if (new_frame) {
        visual_render_mask(env, agent, obs);
        agent->visual_frame_age_s = 0.0f;
        agent->visual_frame_count += 1;
    } else {
        agent->visual_frame_age_s += env->dt;
    }

    obs[DRONE_RACE_VISUAL_BODY_RATE_OFFSET] =
        agent->drone.state.omega.x / fmaxf(agent->drone.params.max_omega, 1e-3f);
    obs[DRONE_RACE_VISUAL_BODY_RATE_OFFSET + 1] =
        agent->drone.state.omega.y / fmaxf(agent->drone.params.max_omega, 1e-3f);
    obs[DRONE_RACE_VISUAL_BODY_RATE_OFFSET + 2] =
        agent->drone.state.omega.z / fmaxf(agent->drone.params.max_omega, 1e-3f);
    for (int motor = 0; motor < 4; motor++) {
        obs[DRONE_RACE_VISUAL_MOTOR_OFFSET + motor] =
            agent->drone.state.rpms[motor]
            / fmaxf(agent->drone.params.max_rpm, 1e-3f);
    }
    for (int history = 0; history < 3; history++) {
        for (int action = 0; action < 4; action++) {
            obs[DRONE_RACE_VISUAL_ACTION_HISTORY_OFFSET + 4 * history + action] =
                agent->last_action_history[history][action];
        }
    }
    obs[DRONE_RACE_VISUAL_NEW_FRAME_OFFSET] = new_frame ? 1.0f : 0.0f;
    obs[DRONE_RACE_VISUAL_FRAME_AGE_OFFSET] = clampf(
        agent->visual_frame_age_s / 0.25f, 0.0f, 1.0f);
    obs[DRONE_RACE_VISUAL_DT_OFFSET] = clampf(env->dt / 0.02f, 0.0f, 1.0f);

    // Training-only informed targets. The actor implementation must slice at
    // DRONE_RACE_VISUAL_LEGAL_OBS_SIZE before any encoder or recurrent update.
    int idx = DRONE_RACE_VISUAL_PRIVILEGED_OFFSET;
    State* state = &agent->drone.state;
    Quat body_inverse = quat_inverse(state->quat);
    Target* active_gate = agent_gate(
        env,
        agent,
        agent->current_gate < env->num_gates
            ? agent->current_gate
            : env->num_gates - 1);
    Vec3 gate_relative_body = quat_rotate(
        body_inverse, sub3(active_gate->pos, state->pos));
    Vec3 velocity_body = quat_rotate(body_inverse, state->vel);
    obs[idx++] = tanhf(state->pos.x / 20.0f);
    obs[idx++] = tanhf(state->pos.y / 20.0f);
    obs[idx++] = tanhf(state->pos.z / 20.0f);
    obs[idx++] = tanhf(gate_relative_body.x / 10.0f);
    obs[idx++] = tanhf(gate_relative_body.y / 10.0f);
    obs[idx++] = tanhf(gate_relative_body.z / 10.0f);
    obs[idx++] = state->vel.x / fmaxf(agent->drone.params.max_vel, 1e-3f);
    obs[idx++] = state->vel.y / fmaxf(agent->drone.params.max_vel, 1e-3f);
    obs[idx++] = state->vel.z / fmaxf(agent->drone.params.max_vel, 1e-3f);
    obs[idx++] = velocity_body.x / fmaxf(agent->drone.params.max_vel, 1e-3f);
    obs[idx++] = velocity_body.y / fmaxf(agent->drone.params.max_vel, 1e-3f);
    obs[idx++] = velocity_body.z / fmaxf(agent->drone.params.max_vel, 1e-3f);
    obs[idx++] = state->quat.w;
    obs[idx++] = state->quat.x;
    obs[idx++] = state->quat.y;
    obs[idx++] = state->quat.z;
    obs[idx++] = state->omega.x / fmaxf(agent->drone.params.max_omega, 1e-3f);
    obs[idx++] = state->omega.y / fmaxf(agent->drone.params.max_omega, 1e-3f);
    obs[idx++] = state->omega.z / fmaxf(agent->drone.params.max_omega, 1e-3f);
    for (int motor = 0; motor < 4; motor++) {
        obs[idx++] = state->rpms[motor]
            / fmaxf(agent->drone.params.max_rpm, 1e-3f);
    }
    obs[idx++] = agent->visual_camera_roll_rad / 3.14159265358979323846f;
    obs[idx++] = agent->visual_camera_pitch_rad / 3.14159265358979323846f;
    obs[idx++] = agent->visual_camera_yaw_rad / 3.14159265358979323846f;
    if (env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT) {
        // Decode parameters that actually govern the deployable CTBR surrogate.
        // Native-motor constants are irrelevant in this mode and previously
        // taught the informed decoder targets that did not affect the plant.
        obs[idx++] = visual_log_ratio(env->sitl_hover_thrust, 0.27f);
        obs[idx++] = visual_log_ratio(
            env->sitl_vertical_accel_per_thrust, 32.81f);
        obs[idx++] = visual_log_ratio(env->sitl_max_thrust, 0.42f);
        obs[idx++] = visual_log_ratio(env->sitl_rate_lag_tau_s, 0.52f);
        obs[idx++] = visual_log_ratio(env->sitl_gravity_m_s2, 8.69f);
        obs[idx++] = tanhf(env->sitl_linear_drag);
    } else {
        obs[idx++] = visual_log_ratio(agent->drone.params.mass, BASE_MASS);
        obs[idx++] = visual_log_ratio(agent->drone.params.k_thrust, BASE_K_THRUST);
        obs[idx++] = visual_log_ratio(agent->drone.params.max_rpm, BASE_MAX_RPM);
        obs[idx++] = visual_log_ratio(agent->drone.params.k_mot, BASE_K_MOT);
        obs[idx++] = visual_log_ratio(agent->drone.params.gravity, BASE_GRAVITY);
        obs[idx++] = tanhf(agent->drone.params.b_drag);
    }
    // Training-only progress follows the same fixed-denominator contract as
    // the public runtime scalar when explicitly configured. The historical
    // count-normalized decoder target remains byte-exact by default.
    float visual_gate_phase_denominator =
        env->observable_gate_index_denominator > 0.0f
            ? env->observable_gate_index_denominator
            : (float)fmaxf(env->num_gates, 1);
    float visual_gate_phase =
        (float)agent->current_gate / visual_gate_phase_denominator;
    obs[idx++] = env->observable_gate_progress_unbounded
        ? fmaxf(visual_gate_phase, 0.0f)
        : clampf(visual_gate_phase, 0.0f, 1.0f);
    obs[idx++] = clampf(active_gate->radius / 3.0f, 0.0f, 1.0f);
    if (idx != DRONE_RACE_VISUAL_OBS_SIZE) {
        abort();
    }
    int unbounded_visual_phase_index =
        DRONE_RACE_VISUAL_PRIVILEGED_OFFSET + 32;
    for (int value = 0; value < DRONE_RACE_VISUAL_OBS_SIZE; value++) {
        if (env->observable_gate_progress_unbounded
                && value == unbounded_visual_phase_index) {
            continue;
        }
        obs[value] = clampf(obs[value], -1.0f, 1.0f);
    }
}
#endif

static float agent_between_gate_distance(
        DroneRace* env, DroneRaceAgent* agent, int gate_idx) {
    if (agent_course_randomized(env)) {
        return agent->randomized_between_gate_distance[gate_idx];
    }
    return env->between_gate_distance[gate_idx];
}

static bool telemetry_velocity_teacher_action(
        DroneRace* env,
        DroneRaceAgent* agent,
        float out[DRONE_RACE_NUM_ATNS]) {
    if (agent->current_gate < 0 || agent->current_gate >= env->num_gates) {
        memset(out, 0, DRONE_RACE_NUM_ATNS * sizeof(float));
        return false;
    }
    Vec3 rel = sub3(
        agent_gate(env, agent, agent->current_gate)->pos,
        agent->drone.state.pos);
    float range = norm3(rel);
    if (range <= 1e-5f) {
        memset(out, 0, DRONE_RACE_NUM_ATNS * sizeof(float));
        return true;
    }
    float speed = fminf(env->telemetry_velocity_teacher_speed_m_s, range * 2.0f);
    Vec3 desired = scalmul3(rel, speed / range);
    out[0] = clampf(desired.x / env->max_cmd_forward, -1.0f, 1.0f);
    out[1] = clampf(desired.y / env->max_cmd_lateral, -1.0f, 1.0f);
    out[2] = clampf(-desired.z / env->max_cmd_vertical, -1.0f, 1.0f);
    out[3] = 0.0f;
    return true;
}

static bool randomized_course_gate_valid(
        const DroneRace* env,
        const DroneRaceAgent* agent,
        int gate_index,
        const Target* candidate) {
    if (!env->gate_position_require_valid_course) return true;
    if (!isfinite(candidate->pos.x)
            || !isfinite(candidate->pos.y)
            || !isfinite(candidate->pos.z)
            || !isfinite(candidate->radius)
            || candidate->radius <= 0.0f) {
        return false;
    }
    if (gate_index <= 0) return true;

    const Target* previous = &agent->randomized_gates[gate_index - 1];
    float aperture_gap = previous->radius + candidate->radius + 1e-3f;
    float minimum_forward_gap = env->gate_position_min_forward_gap_m > 0.0f
        ? env->gate_position_min_forward_gap_m
        : aperture_gap;
    float forward_gap = candidate->pos.x - previous->pos.x;
    float segment_distance = race_dist3(candidate->pos, previous->pos);
    if (forward_gap < minimum_forward_gap
            || segment_distance < aperture_gap) {
        return false;
    }
    if (env->gate_position_max_segment_distance_m > 0.0f
            && segment_distance > env->gate_position_max_segment_distance_m) {
        return false;
    }
    for (int prior_index = 0; prior_index < gate_index - 1; prior_index++) {
        const Target* prior = &agent->randomized_gates[prior_index];
        float separation = race_dist3(candidate->pos, prior->pos);
        if (separation < prior->radius + candidate->radius + 1e-3f) {
            return false;
        }
    }
    return true;
}

static void build_randomized_agent_course(DroneRace* env, DroneRaceAgent* agent) {
    if (!agent_course_randomized(env)) {
        return;
    }

    float geometry_scale = clampf(env->course_geometry_scale, 0.0f, 1.0f);
    if (env->course_geometry_scale_randomize) {
        float scale_min = clampf(env->course_geometry_scale_min, 0.0f, 1.0f);
        float scale_max = clampf(env->course_geometry_scale_max, 0.0f, 1.0f);
        if (scale_min > scale_max) {
            float swap = scale_min;
            scale_min = scale_max;
            scale_max = swap;
        }
        geometry_scale = rndf(scale_min, scale_max, &env->rng);
    }
    agent->randomized_course_geometry_scale = geometry_scale;

    float gate_radius = env->gate_radius;
    if (env->gate_radius_randomize) {
        float radius_min = fmaxf(env->gate_radius_min, 1e-3f);
        float radius_max = fmaxf(env->gate_radius_max, 1e-3f);
        if (radius_min > radius_max) {
            float swap = radius_min;
            radius_min = radius_max;
            radius_max = swap;
        }
        gate_radius = rndf(radius_min, radius_max, &env->rng);
    }

    // Sample the aperture profile from a course-only stream. Enabling the
    // mixture must not change vehicle initialization for a paired episode.
    unsigned int aperture_rng = env->rng ^ 0x85ebca6bu;
    float aperture_profile_probability = clampf(
        env->gate_radius_profile_mix_probability, 0.0f, 1.0f);
    bool use_target_aperture_profile = env->gate_radius_profile_mix
        && (aperture_profile_probability >= 1.0f
            || (aperture_profile_probability > 0.0f
                && rndf(0.0f, 1.0f, &aperture_rng)
                    < aperture_profile_probability));
    float target_aperture_radius = fmaxf(
        env->gate_radius_profile_mix_target, 1e-3f);

    int first = env->gate_position_randomize_from_index;
    if (first < 0) first = 0;
    if (first > env->num_gates) first = env->num_gates;
    // Position-domain samples must not consume the vehicle/start-state RNG.
    // Otherwise enabling jitter only for late gates silently changes the
    // already-solved prefix before those gates can affect the trajectory.
    unsigned int position_rng = env->rng ^ 0x9e3779b9u;
    float position_randomize_probability = clampf(
        env->gate_position_domain_randomize_probability, 0.0f, 1.0f);
    bool randomize_positions = env->gate_position_domain_randomize
        && (position_randomize_probability >= 1.0f
            || (position_randomize_probability > 0.0f
                && rndf(0.0f, 1.0f, &position_rng)
                    < position_randomize_probability));
    agent->gate_positions_randomized = randomize_positions ? 1 : 0;

    for (int i = 0; i < env->num_gates; i++) {
        Target base_gate = env->gates[i];
        if (env->gate_radius_randomize
                && (env->gate_radius_randomize_gate_index < 0
                    || env->gate_radius_randomize_gate_index == i)) {
            base_gate.radius = gate_radius;
        }
        if (use_target_aperture_profile) {
            base_gate.radius = target_aperture_radius;
        }
        if (env->course_geometry_scale_randomize) {
            base_gate.pos = course_gate_position(
                env, i, geometry_scale);
        }
        if (randomize_positions && i >= first
                && env->gate_position_randomized_radius > 0.0f) {
            base_gate.radius = env->gate_position_randomized_radius;
        }

        int attempts = env->gate_position_require_valid_course
            ? env->gate_position_resample_attempts
            : 1;
        if (attempts < 1) attempts = 1;
        bool accepted = false;
        Target candidate = base_gate;
        for (int attempt = 0; attempt < attempts; attempt++) {
            candidate = base_gate;
            if (randomize_positions && i >= first) {
                candidate.pos.x += rndf(
                    -fabsf(env->gate_position_jitter_x),
                    fabsf(env->gate_position_jitter_x),
                    &position_rng);
                candidate.pos.y += rndf(
                    -fabsf(env->gate_position_jitter_y),
                    fabsf(env->gate_position_jitter_y),
                    &position_rng);
                candidate.pos.z += rndf(
                    -fabsf(env->gate_position_jitter_z),
                    fabsf(env->gate_position_jitter_z),
                    &position_rng);
            }
            if (randomized_course_gate_valid(env, agent, i, &candidate)) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            fprintf(
                stderr,
                "failed to sample a valid randomized course gate %d after %d attempts\n",
                i,
                attempts);
            abort();
        }
        agent->randomized_gates[i] = candidate;
        /*
         * The default-off path above consumes the same three position draws,
         * in the same order, as the historical code. Additional draws occur
         * only after an explicitly enabled validity rejection.
         */
        agent->randomized_between_gate_distance[i] = 0.0f;
    }
    for (int i = 0; i < env->num_gates - 1; i++) {
        agent->randomized_between_gate_distance[i] = race_dist3(
            agent->randomized_gates[i + 1].pos,
            agent->randomized_gates[i].pos);
    }
}

static float remaining_distance(DroneRace* env, DroneRaceAgent* agent) {
    if (agent->current_gate >= env->num_gates) {
        return 0.0f;
    }

    float rem = race_dist3(
        agent->drone.state.pos,
        agent_gate(env, agent, agent->current_gate)->pos);
    for (int i = agent->current_gate; i < env->num_gates - 1; i++) {
        rem += agent_between_gate_distance(env, agent, i);
    }
    return rem;
}

static float gate_exit_lateral_velocity_error_sq(
        DroneRace* env, DroneRaceAgent* agent, int gate_idx) {
    if (gate_idx < 0 || gate_idx + 1 >= env->num_gates) {
        return 0.0f;
    }

    Target* gate = agent_gate(env, agent, gate_idx);
    Target* next_gate = agent_gate(env, agent, gate_idx + 1);
    Vec3 segment = sub3(next_gate->pos, gate->pos);
    float segment_forward = dot3(segment, gate->normal);
    if (fabsf(segment_forward) < 1e-4f) {
        return 0.0f;
    }

    float forward_speed = fmaxf(dot3(agent->drone.state.vel, gate->normal), 0.0f);
    float target_lateral_speed = forward_speed * segment.y / segment_forward;
    float lateral_error = agent->drone.state.vel.y - target_lateral_speed;
    return lateral_error * lateral_error;
}

static float gate_exit_velocity_error_sq(
        DroneRace* env, DroneRaceAgent* agent, int gate_idx) {
    if (gate_idx < 0 || gate_idx + 1 >= env->num_gates) {
        return 0.0f;
    }

    Target* gate = agent_gate(env, agent, gate_idx);
    Target* next_gate = agent_gate(env, agent, gate_idx + 1);
    Vec3 segment = sub3(next_gate->pos, gate->pos);
    float segment_forward = dot3(segment, gate->normal);
    if (fabsf(segment_forward) < 1e-4f) {
        return 0.0f;
    }

    float target_forward = fmaxf(env->gate_exit_target_forward_speed, 0.0f);
    Vec3 target_velocity = scalmul3(
        segment, target_forward / segment_forward);
    Vec3 error = sub3(agent->drone.state.vel, target_velocity);
    return dot3(error, error);
}

static bool gate_exit_reward_active(
        DroneRace* env, DroneRaceAgent* agent, int gate_idx) {
    if (gate_idx < 0 || gate_idx + 1 >= env->num_gates) {
        return false;
    }
    if (!env->gate_exit_reward_skip_randomized_next_gate
            || !agent->gate_positions_randomized) {
        return true;
    }

    int first_randomized_gate = env->gate_position_randomize_from_index;
    if (first_randomized_gate < 0) first_randomized_gate = 0;
    if (first_randomized_gate > env->num_gates) {
        first_randomized_gate = env->num_gates;
    }
    return gate_idx + 1 < first_randomized_gate;
}

static float forward_speed_excess_error_sq(
        DroneRace* env, DroneRaceAgent* agent) {
    if (agent->current_gate < 0 || agent->current_gate >= env->num_gates) {
        return 0.0f;
    }
    Target* gate = agent_gate(env, agent, agent->current_gate);
    float forward_speed = dot3(agent->drone.state.vel, gate->normal);
    float excess = fmaxf(
        forward_speed - fmaxf(env->target_forward_speed, 0.0f), 0.0f);
    return excess * excess;
}

static float cross_track_penalty(
        DroneRace* env, DroneRaceAgent* agent, Vec3 pos) {
    if (env->w_cross_track <= 0.0f
            || agent->current_gate < env->cross_track_from_gate_index
            || agent->current_gate >= env->num_gates) {
        return 0.0f;
    }
    Target* gate = agent_gate(env, agent, agent->current_gate);
    Vec3 rel = sub3(gate->pos, pos);
    float along = dot3(rel, gate->normal);
    Vec3 radial_vec = sub3(rel, scalmul3(gate->normal, along));
    float proximity = expf(-fabsf(along) / 10.0f);
    float weighted_error_sq = radial_vec.x * radial_vec.x
        + radial_vec.y * radial_vec.y
        + fmaxf(env->cross_track_vertical_weight, 0.0f)
            * radial_vec.z * radial_vec.z;
    return env->w_cross_track * env->dt * proximity
        * weighted_error_sq;
}

static float gate_crossing_error_penalty(
        DroneRace* env, int gate_index, float radial) {
    if (env->w_gate_crossing_error <= 0.0f
            || gate_index < env->gate_crossing_error_from_gate_index
            || gate_index >= env->num_gates) {
        return 0.0f;
    }
    return env->w_gate_crossing_error * radial * radial;
}

static float gate_camera_alignment_error_sq(
        DroneRace* env, DroneRaceAgent* agent) {
    if (env->w_gate_camera_alignment <= 0.0f
            || agent->current_gate < env->gate_camera_alignment_from_gate_index
            || agent->current_gate >= env->num_gates) {
        return 0.0f;
    }
    Vec3 rel_world = sub3(
        agent_gate(env, agent, agent->current_gate)->pos,
        agent->drone.state.pos);
    Vec3 rel_body = quat_rotate(
        quat_inverse(agent->drone.state.quat), rel_world);
    float yaw_error = atan2f(rel_body.y, fmaxf(rel_body.x, 1e-4f));
    float elevation = atan2f(-rel_body.z, fmaxf(rel_body.x, 1e-4f));
    float pitch_error = elevation - TS002_CAMERA_UPTILT_RAD;
    float yaw_normalized = yaw_error / TS002_CAMERA_HALF_FOV_RAD;
    float pitch_normalized = pitch_error / TS002_CAMERA_HALF_FOV_RAD;
    return yaw_normalized * yaw_normalized
        + pitch_normalized * pitch_normalized;
}

static float invalid_run_penalty(DroneRace* env, int gate_index) {
    if (gate_index >= env->late_invalid_penalty_from_gate_index) {
        return env->late_invalid_penalty;
    }
    return env->invalid_penalty;
}

static float paper_body_rate_penalty(
        const DroneRace* env, const State* state) {
    if (env->w_body_rate <= 0.0f) return 0.0f;
    float rate_l1 = fabsf(state->omega.x)
        + fabsf(state->omega.y)
        + fabsf(state->omega.z);
    // Paper: (exp(min(||Omega||_1, 17)) - 1) / (2 * f_c * 1e5).
    // Since dt = 1/f_c, this is exactly dt * expm1(...) / (2e5).
    return env->w_body_rate * env->dt * 5.0e-6f
        * expm1f(fminf(rate_l1, 17.0f));
}

static bool gate_plane_crossing_details(
        Vec3 prev_pos,
        Vec3 pos,
        Target* gate,
        float plane_tol,
        float direction_min,
        float* radial_out,
        Vec3* radial_vector_out) {
    Vec3 move = sub3(pos, prev_pos);
    float move_norm = norm3(move);
    if (move_norm < 1e-8f) return false;

    Vec3 move_dir = scalmul3(move, 1.0f / move_norm);
    Vec3 prev_rel = sub3(prev_pos, gate->pos);
    Vec3 curr_rel = sub3(pos, gate->pos);
    float d_prev = dot3(prev_rel, gate->normal);
    float d_curr = dot3(curr_rel, gate->normal);

    // A crossing is a directed sign change through the plane. Requiring the
    // vehicle to clear the tolerance band on both sides in one 120 Hz step
    // silently drops valid low-speed crossings that enter the band gradually.
    // Keep plane_tol in the ABI for compatibility; direction and radial checks
    // below provide the actual rejection criteria.
    (void)plane_tol;
    if (!(d_prev <= 0.0f && d_curr >= 0.0f)) return false;
    float denom = d_prev - d_curr;
    if (fabsf(denom) < 1e-8f) return false;

    float t = clampf(d_prev / denom, 0.0f, 1.0f);
    Vec3 hit = add3(prev_pos, scalmul3(move, t));
    Vec3 radial = sub3(hit, gate->pos);
    float normal_part = dot3(radial, gate->normal);
    radial = sub3(radial, scalmul3(gate->normal, normal_part));

    float radial_dist = norm3(radial);
    if (radial_out != NULL) *radial_out = radial_dist;
    if (radial_vector_out != NULL) *radial_vector_out = radial;
    return dot3(move_dir, gate->normal) > direction_min;
}

static bool gate_crossing_details(
        Vec3 prev_pos,
        Vec3 pos,
        Target* gate,
        float plane_tol,
        float direction_min,
        float* radial_out,
        Vec3* radial_vector_out) {
    float radial_dist = 0.0f;
    if (!gate_plane_crossing_details(
            prev_pos, pos, gate, plane_tol, direction_min,
            &radial_dist, radial_vector_out)) {
        if (radial_out != NULL) *radial_out = radial_dist;
        return false;
    }
    if (radial_out != NULL) *radial_out = radial_dist;
    return radial_dist <= gate->radius;
}

static bool gate_crossing(
        Vec3 prev_pos,
        Vec3 pos,
        Target* gate,
        float plane_tol,
        float direction_min,
        float* radial_out) {
    return gate_crossing_details(
        prev_pos, pos, gate, plane_tol, direction_min, radial_out, NULL);
}

static void record_terminal_crossing(
        DroneRaceAgent* agent,
        int gate_index,
        float radial,
        Vec3 radial_vector) {
    agent->terminal_crossing_sampled = 1;
    agent->terminal_crossing_gate_index = gate_index;
    agent->terminal_crossing_radial = radial;
    agent->terminal_crossing_right = radial_vector.y;
    agent->terminal_crossing_vertical = radial_vector.z;
}

static void set_agent_target(DroneRace* env, DroneRaceAgent* agent) {
    int gate_idx = agent->current_gate < env->num_gates ? agent->current_gate : env->num_gates - 1;
    agent->target = *agent_gate(env, agent, gate_idx);
    agent->drone.target = &agent->target;
}

static void move_drone_for_dt(Drone* drone, float* actions, float dt) {
    clamp4(actions, -1.0f, 1.0f);
    int substeps = 5;
    float step_dt = dt / (float)substeps;
    for (int s = 0; s < substeps; s++) {
        rk4_step(&drone->state, &drone->params, actions, step_dt);
        clamp3(&drone->state.vel, -drone->params.max_vel, drone->params.max_vel);
        clamp3(&drone->state.omega, -drone->params.max_omega, drone->params.max_omega);
        for (int i = 0; i < 4; i++) {
            drone->state.rpms[i] = clampf(drone->state.rpms[i], 0.0f, drone->params.max_rpm);
        }
    }
}

static void move_drone_sitl_plant(
        DroneRace* env,
        DroneRaceAgent* agent,
        const float* policy_action) {
    State* state = &agent->drone.state;
    float dt = env->dt;

    float desired_pitch = clampf(policy_action[0], -1.0f, 1.0f) * env->sitl_max_pitch_rad;
    float desired_roll = env->sitl_lock_roll
        ? 0.0f
        : clampf(policy_action[1], -1.0f, 1.0f) * env->sitl_max_roll_rad;
    float desired_yaw = env->sitl_lock_yaw
        ? 0.0f
        : clampf(policy_action[3], -1.0f, 1.0f) * env->sitl_max_yaw_rad;
    float thrust_action = env->sitl_lock_hover_thrust
        ? 0.0f
        : clampf(policy_action[2], -1.0f, 1.0f);
    float thrust_span = thrust_action >= 0.0f
        ? env->sitl_max_thrust - env->sitl_hover_thrust
        : env->sitl_hover_thrust - env->sitl_min_thrust;
    float cmd_thrust = env->sitl_hover_thrust + thrust_action * thrust_span;
    if (!isfinite(cmd_thrust)
            || cmd_thrust < env->sitl_min_thrust - 1e-6f
            || cmd_thrust > env->sitl_max_thrust + 1e-6f) {
        agent->thrust_envelope_violation = 1;
    }

    EulerAngles attitude = quat_to_euler(state->quat);
    float desired[3] = {desired_roll, desired_pitch, desired_yaw};
    float measured[3] = {attitude.roll, attitude.pitch, attitude.yaw};
    float plant_gain[3] = {
        env->sitl_rate_gain_roll,
        env->sitl_rate_gain_pitch,
        env->sitl_rate_gain_yaw,
    };
    float omega_target[3] = {0.0f, 0.0f, 0.0f};
    for (int axis = 0; axis < 3; axis++) {
        float error = atan2f(
            sinf(desired[axis] - measured[axis]),
            cosf(desired[axis] - measured[axis]));
        if (fabsf(error) <= env->sitl_attitude_error_deadband_rad) continue;
        float wire_command = env->sitl_attitude_kp * error / plant_gain[axis];
        if (fabsf(wire_command) < env->sitl_body_rate_min_command_rad_s) {
            wire_command = copysignf(env->sitl_body_rate_min_command_rad_s, wire_command);
        }
        wire_command = clampf(
            wire_command,
            -env->sitl_max_body_rate_command_rad_s,
            env->sitl_max_body_rate_command_rad_s);
        if (!isfinite(wire_command)
                || fabsf(wire_command)
                    > env->sitl_max_body_rate_command_rad_s + 1e-6f) {
            agent->wire_rate_envelope_violation = 1;
        }
        omega_target[axis] = plant_gain[axis] * wire_command;
    }
    float alpha = dt / fmaxf(env->sitl_rate_lag_tau_s, dt);
    state->omega.x += alpha * (omega_target[0] - state->omega.x);
    state->omega.y += alpha * (omega_target[1] - state->omega.y);
    state->omega.z += alpha * (omega_target[2] - state->omega.z);

    attitude.roll += state->omega.x * dt;
    attitude.pitch += state->omega.y * dt;
    attitude.yaw += state->omega.z * dt;
    state->quat = euler_to_quat(attitude);

    if (env->sitl_lock_hover_thrust) {
        float vertical_fraction = fmaxf(
            cosf(attitude.roll) * cosf(attitude.pitch), 0.5f);
        cmd_thrust = clampf(
            env->sitl_hover_thrust / vertical_fraction,
            env->sitl_min_thrust,
            env->sitl_max_thrust);
    }

    float thrust_accel = env->sitl_vertical_accel_per_thrust * cmd_thrust;
    Vec3 thrust_body = {0.0f, 0.0f, thrust_accel};
    Vec3 thrust_world = quat_rotate(state->quat, thrust_body);
    // Match official v3385 evidence: nose-down (negative pitch) accelerates
    // forward, while positive roll accelerates toward negative world Y. The
    // compact native plant stores world Z as up, so only X needs inversion.
    thrust_world.x = -thrust_world.x;
    thrust_world.x *= thrust_world.x >= 0.0f
        ? env->sitl_horizontal_accel_scale
        : env->sitl_braking_accel_scale;
    thrust_world.y *= env->sitl_lateral_accel_scale;
    Vec3 gravity = {0.0f, 0.0f, -env->sitl_gravity_m_s2};
    Vec3 accel = add3(thrust_world, gravity);
    if (env->sitl_linear_drag > 0.0f) {
        accel.x -= env->sitl_linear_drag * state->vel.x;
        accel.y -= env->sitl_linear_drag * state->vel.y;
        accel.z -= env->sitl_linear_drag * state->vel.z;
    }

    state->vel = add3(state->vel, scalmul3(accel, dt));
    state->pos = add3(state->pos, scalmul3(state->vel, dt));
    clamp3(&state->vel, -BASE_MAX_VEL, BASE_MAX_VEL);
    clamp3(&state->omega, -BASE_MAX_OMEGA, BASE_MAX_OMEGA);
}

static void move_drone_vq1_telemetry_velocity(
        DroneRace* env,
        DroneRaceAgent* agent,
        const float* policy_action) {
    State* state = &agent->drone.state;
    float dt = env->dt;
    Vec3 desired_velocity = {
        clampf(policy_action[0], -1.0f, 1.0f) * env->max_cmd_forward,
        clampf(policy_action[1], -1.0f, 1.0f) * env->max_cmd_lateral,
        -clampf(policy_action[2], -1.0f, 1.0f) * env->max_cmd_vertical,
    };
    float alpha = 1.0f - expf(
        -dt / fmaxf(env->telemetry_velocity_response_tau_s, dt));
    float max_step = env->telemetry_velocity_max_accel_m_s2 * dt;
    Vec3 velocity_step = scalmul3(sub3(desired_velocity, state->vel), alpha);
    clamp3(&velocity_step, -max_step, max_step);
    state->vel = add3(state->vel, velocity_step);
    state->pos = add3(state->pos, scalmul3(state->vel, dt));

    // Yaw is not needed to interpret local-course velocity actions, but keep a
    // bounded state so the fourth action retains a deployable yaw-rate meaning.
    float yaw_rate = clampf(policy_action[3], -1.0f, 1.0f)
        * env->max_cmd_yaw_rate;
    EulerAngles attitude = quat_to_euler(state->quat);
    attitude.roll = 0.0f;
    attitude.pitch = 0.0f;
    attitude.yaw += yaw_rate * dt;
    state->quat = euler_to_quat(attitude);
    state->omega = (Vec3){0.0f, 0.0f, yaw_rate};
}

static void policy_to_motor_actions(
        DroneRace* env,
        DroneRaceAgent* agent,
        const float* policy_action,
        float* motor_action) {
    if (env->interface_mode == DRONE_RACE_INTERFACE_NATIVE_MOTOR) {
        for (int i = 0; i < DRONE_RACE_NUM_ATNS; i++) {
            motor_action[i] = policy_action[i];
        }
        return;
    }

    Quat q_inv = quat_inverse(agent->drone.state.quat);
    Vec3 vel_body = quat_rotate(q_inv, agent->drone.state.vel);
    EulerAngles attitude = quat_to_euler(agent->drone.state.quat);

    float cmd_forward = clampf(policy_action[0], -1.0f, 1.0f) * env->max_cmd_forward;
    float cmd_right = clampf(policy_action[1], -1.0f, 1.0f) * env->max_cmd_lateral;
    float cmd_down = clampf(policy_action[2], -1.0f, 1.0f) * env->max_cmd_vertical;
    float cmd_yaw_rate = clampf(policy_action[3], -1.0f, 1.0f) * env->max_cmd_yaw_rate;

    float forward_error = cmd_forward - vel_body.x;
    float right_error = cmd_right - vel_body.y;
    float up_error = -cmd_down - vel_body.z;
    float desired_pitch = clampf(
        -env->setpoint_vel_kp * forward_error,
        -env->setpoint_max_tilt,
        env->setpoint_max_tilt);
    float desired_roll = clampf(
        env->setpoint_vel_kp * right_error,
        -env->setpoint_max_tilt,
        env->setpoint_max_tilt);

    float roll_effort = env->setpoint_att_kp * (desired_roll - attitude.roll)
        - env->setpoint_rate_kd * agent->drone.state.omega.x;
    float pitch_effort = env->setpoint_att_kp * (desired_pitch - attitude.pitch)
        - env->setpoint_rate_kd * agent->drone.state.omega.y;
    float yaw_effort = env->setpoint_yaw_rate_kp * (cmd_yaw_rate - agent->drone.state.omega.z);
    float thrust_effort = env->setpoint_thrust_vel_kp * up_error;

    float limit = env->setpoint_max_motor_delta;
    roll_effort = clampf(roll_effort, -limit, limit);
    pitch_effort = clampf(pitch_effort, -limit, limit);
    yaw_effort = clampf(yaw_effort, -limit, limit);
    thrust_effort = clampf(thrust_effort, -limit, limit);

    motor_action[0] = thrust_effort - roll_effort - pitch_effort - yaw_effort;
    motor_action[1] = thrust_effort - roll_effort + pitch_effort + yaw_effort;
    motor_action[2] = thrust_effort + roll_effort + pitch_effort - yaw_effort;
    motor_action[3] = thrust_effort + roll_effort - pitch_effort + yaw_effort;
    clamp4(motor_action, -1.0f, 1.0f);
}

static void reset_agent(DroneRace* env, DroneRaceAgent* agent) {
    int episode_ordinal = agent->episode_ordinal;
    memset(agent, 0, sizeof(*agent));
    agent->episode_ordinal = episode_ordinal;
    int agent_index = env->agents != NULL ? (int)(agent - env->agents) : 0;
    agent->visual_rng = env->rng
        ^ (0x9e3779b9u * (unsigned int)(agent_index + 1))
        ^ (0x85ebca6bu * (unsigned int)(episode_ordinal + 1));
    agent->visual_camera_roll_rad = env->visual_camera_roll_rad
        + symmetric_reset_jitter(
            env->visual_camera_roll_jitter_rad, &agent->visual_rng);
    agent->visual_camera_pitch_rad = env->visual_camera_pitch_rad
        + symmetric_reset_jitter(
            env->visual_camera_pitch_jitter_rad, &agent->visual_rng);
    agent->visual_camera_yaw_rad = env->visual_camera_yaw_rad
        + symmetric_reset_jitter(
            env->visual_camera_yaw_jitter_rad, &agent->visual_rng);
    build_randomized_agent_course(env, agent);
    float transition_speed = fmaxf(
        env->sitl_gate_transition_min_forward_speed, 0.0f);
    if (env->sitl_gate_transition_min_forward_speed_randomize) {
        float speed_min = fmaxf(
            env->sitl_gate_transition_min_forward_speed_min, 0.0f);
        float speed_max = fmaxf(
            env->sitl_gate_transition_min_forward_speed_max, 0.0f);
        if (speed_min > speed_max) {
            float swap = speed_min;
            speed_min = speed_max;
            speed_max = swap;
        }
        transition_speed = rndf(speed_min, speed_max, &env->rng);
    }
    agent->randomized_sitl_gate_transition_min_forward_speed = transition_speed;
    agent->valid_run = 1;
    agent->drone.target = &agent->target;
    init_drone(&agent->drone, &env->rng, 0.05f);

    bool use_gate_local_start = false;
    if (env->gate_local_start_curriculum) {
        float probability = clampf(
            env->gate_local_start_probability, 0.0f, 1.0f);
        use_gate_local_start = probability >= 1.0f
            || (probability > 0.0f
                && rndf(0.0f, 1.0f, &env->rng) < probability);
    }

    int use_segment_start = env->use_custom_start && !use_gate_local_start;
    if (env->mixed_start_curriculum
            && !use_gate_local_start
            && env->use_custom_start
            && env->start_gate_index > 0) {
        float probability = clampf(
            env->segment_start_probability, 0.0f, 1.0f);
        use_segment_start = rndf(0.0f, 1.0f, &env->rng) < probability;
    }

    if (use_gate_local_start) {
        int gate_min = env->gate_local_start_gate_min;
        if (gate_min < 0) gate_min = 0;
        if (gate_min >= env->num_gates) gate_min = env->num_gates - 1;
        int gate_max_exclusive = env->gate_local_start_gate_max_exclusive;
        if (gate_max_exclusive <= 0 || gate_max_exclusive > env->num_gates) {
            gate_max_exclusive = env->num_gates;
        }
        if (gate_max_exclusive <= gate_min) gate_max_exclusive = gate_min + 1;
        int sampled_gate = (int)rndf(
            (float)gate_min, (float)gate_max_exclusive, &env->rng);
        if (sampled_gate >= gate_max_exclusive) {
            sampled_gate = gate_max_exclusive - 1;
        }
        agent->current_gate = sampled_gate;
        agent->gate_local_start_sampled = 1;
    } else {
        agent->current_gate = use_segment_start ? env->start_gate_index : 0;
    }
    env->log.reset_count += 1.0f;
    env->log.gate_local_reset_count += use_gate_local_start ? 1.0f : 0.0f;
    if (agent->current_gate < 0) agent->current_gate = 0;
    if (agent->current_gate >= env->num_gates) {
        agent->current_gate = env->num_gates - 1;
    }
    agent->elapsed_time = use_segment_start
        ? fmaxf(
            env->start_elapsed_time + symmetric_reset_jitter(
                env->start_elapsed_time_jitter, &env->rng),
            0.0f)
        : 0.0f;

    Target* first_gate = agent_gate(env, agent, agent->current_gate);
    if (use_gate_local_start) {
        float offset_min = fmaxf(env->gate_local_start_offset_min, 0.0f);
        float offset_max = fmaxf(env->gate_local_start_offset_max, 0.0f);
        if (offset_min > offset_max) {
            float swap = offset_min;
            offset_min = offset_max;
            offset_max = swap;
        }
        float offset = rndf(offset_min, offset_max, &env->rng);
        agent->drone.state.pos = sub3(
            first_gate->pos,
            scalmul3(first_gate->normal, offset));
    } else if (use_segment_start) {
        agent->drone.state.pos = (Vec3){
            env->custom_start_pos[0] + symmetric_reset_jitter(
                env->custom_start_pos_jitter[0], &env->rng),
            env->custom_start_pos[1] + symmetric_reset_jitter(
                env->custom_start_pos_jitter[1], &env->rng),
            env->custom_start_pos[2] + symmetric_reset_jitter(
                env->custom_start_pos_jitter[2], &env->rng),
        };
    } else if (env->use_custom_gate_layout) {
        agent->drone.state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    } else {
        agent->drone.state.pos = sub3(
            first_gate->pos,
            scalmul3(first_gate->normal, env->start_offset));
    }
    if (env->reset_position_noise_xy > 0.0f) {
        agent->drone.state.pos.x += rndf(
            -env->reset_position_noise_xy,
            env->reset_position_noise_xy,
            &env->rng);
        agent->drone.state.pos.y += rndf(
            -env->reset_position_noise_xy,
            env->reset_position_noise_xy,
            &env->rng);
    }
    if (env->reset_position_noise_z > 0.0f) {
        agent->drone.state.pos.z += rndf(
            -env->reset_position_noise_z,
            env->reset_position_noise_z,
            &env->rng);
    }
    agent->drone.prev_pos = agent->drone.state.pos;
    set_agent_target(env, agent);
    agent->closest_gate_range = remaining_distance(env, agent);
    agent->min_floor_ttf = FLOOR_TTF_SENTINEL;
    agent->min_floor_stop_margin = FLOOR_TTF_SENTINEL;
    if (env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT
            || env->interface_mode == DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY) {
        agent->drone.state.quat = use_segment_start
            ? (Quat){
                env->custom_start_quat[0],
                env->custom_start_quat[1],
                env->custom_start_quat[2],
                env->custom_start_quat[3],
            }
            : (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        quat_normalize(&agent->drone.state.quat);
        if (use_segment_start) {
            EulerAngles attitude = quat_to_euler(agent->drone.state.quat);
            attitude.roll += symmetric_reset_jitter(
                env->custom_start_attitude_jitter[0], &env->rng);
            attitude.pitch += symmetric_reset_jitter(
                env->custom_start_attitude_jitter[1], &env->rng);
            attitude.yaw += symmetric_reset_jitter(
                env->custom_start_attitude_jitter[2], &env->rng);
            agent->drone.state.quat = euler_to_quat(attitude);
            quat_normalize(&agent->drone.state.quat);
        }
        agent->drone.state.omega = use_segment_start
            ? (Vec3){
                env->custom_start_omega[0] + symmetric_reset_jitter(
                    env->custom_start_omega_jitter[0], &env->rng),
                env->custom_start_omega[1] + symmetric_reset_jitter(
                    env->custom_start_omega_jitter[1], &env->rng),
                env->custom_start_omega[2] + symmetric_reset_jitter(
                    env->custom_start_omega_jitter[2], &env->rng),
            }
            : (Vec3){0.0f, 0.0f, 0.0f};
        agent->drone.state.vel = use_segment_start
            ? (Vec3){
                env->custom_start_vel[0] + symmetric_reset_jitter(
                    env->custom_start_vel_jitter[0], &env->rng),
                env->custom_start_vel[1] + symmetric_reset_jitter(
                    env->custom_start_vel_jitter[1], &env->rng),
                env->custom_start_vel[2] + symmetric_reset_jitter(
                    env->custom_start_vel_jitter[2], &env->rng),
            }
            : (Vec3){0.0f, 0.0f, 0.0f};
    }
}

static void add_log(DroneRace* env, DroneRaceAgent* agent, bool success) {
    env->log.perf += success ? 1.0f : 0.0f;
    env->log.score += (float)agent->current_gate;
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += (float)agent->step_count;
    env->log.valid_run_rate += agent->valid_run ? 1.0f : 0.0f;
    env->log.success_rate += success ? 1.0f : 0.0f;
    env->log.gates_passed += (float)agent->current_gate;
    int gate_count_index = env->num_gates - 1;
    if (gate_count_index >= 0 && gate_count_index < DRONE_RACE_MAX_GATES) {
        env->log.gate_count_episode[gate_count_index] += 1.0f;
        env->log.gate_count_success[gate_count_index] += success ? 1.0f : 0.0f;
    }
    env->log.completion_time += success ? agent->elapsed_time : 0.0f;
    env->log.final_progress += agent->progress;
    env->log.max_progress += agent->max_progress;
    env->log.closest_gate_range += agent->closest_gate_range;
    env->log.final_x += agent->drone.state.pos.x;
    env->log.final_y += agent->drone.state.pos.y;
    env->log.final_z += agent->drone.state.pos.z;
    env->log.gate_local_start_rate += agent->gate_local_start_sampled
        ? 1.0f : 0.0f;
    env->log.crash += agent->crash ? 1.0f : 0.0f;
    env->log.crash_low += agent->crash_low ? 1.0f : 0.0f;
    env->log.crash_high += agent->crash_high ? 1.0f : 0.0f;
    env->log.crash_xy += agent->crash_xy ? 1.0f : 0.0f;
    env->log.crash_low_z += agent->crash_low ? agent->drone.state.pos.z : 0.0f;
    env->log.crash_low_vz += agent->crash_low ? agent->drone.state.vel.z : 0.0f;
    env->log.crash_low_progress += agent->crash_low ? agent->progress : 0.0f;
    env->log.crash_low_time += agent->crash_low ? agent->elapsed_time : 0.0f;
    if (agent->crash_low) {
        if (agent->current_gate <= 0) {
            env->log.crash_low_next_gate0 += 1.0f;
        } else if (agent->current_gate == 1) {
            env->log.crash_low_next_gate1 += 1.0f;
        } else if (agent->current_gate == 2) {
            env->log.crash_low_next_gate2 += 1.0f;
        } else {
            env->log.crash_low_next_gate3plus += 1.0f;
        }
    }
    env->log.crash_low_floor_margin_pre += agent->crash_low ? agent->crash_low_floor_margin_pre : 0.0f;
    env->log.crash_low_ttf_pre += agent->crash_low ? agent->crash_low_ttf_pre : 0.0f;
    env->log.crash_low_stop_margin_pre += agent->crash_low ? agent->crash_low_stop_margin_pre : 0.0f;
    env->log.crash_low_max_up_accel_pre += agent->crash_low ? agent->crash_low_max_up_accel_pre : 0.0f;
    env->log.floor_impact_risk += agent->floor_impact_risk ? 1.0f : 0.0f;
    env->log.floor_stop_violation += agent->floor_stop_violation ? 1.0f : 0.0f;
    env->log.floor_risk_steps += agent->floor_risk_steps;
    env->log.floor_stop_violation_steps += agent->floor_stop_violation_steps;
    env->log.floor_risk_sampled += agent->floor_risk_sampled ? 1.0f : 0.0f;
    env->log.min_floor_ttf += agent->floor_risk_sampled ? agent->min_floor_ttf : 0.0f;
    env->log.min_floor_stop_margin += agent->floor_risk_sampled ? agent->min_floor_stop_margin : 0.0f;
    env->log.gate0_exit_sampled += agent->gate0_exit_sampled ? 1.0f : 0.0f;
    env->log.gate0_exit_vx += agent->gate0_exit_sampled ? agent->gate0_exit_vx : 0.0f;
    env->log.gate0_exit_vy += agent->gate0_exit_sampled ? agent->gate0_exit_vy : 0.0f;
    env->log.gate0_exit_vz += agent->gate0_exit_sampled ? agent->gate0_exit_vz : 0.0f;
    env->log.gate0_action_steps += agent->gate0_action_steps;
    env->log.gate0_action_pitch += agent->gate0_action_pitch;
    env->log.gate1_action_steps += agent->gate1_action_steps;
    env->log.gate1_action_pitch += agent->gate1_action_pitch;
    env->log.gate1_action_roll += agent->gate1_action_roll;
    env->log.gate1_action_thrust += agent->gate1_action_thrust;
    env->log.gate1_action_yaw += agent->gate1_action_yaw;
    env->log.gate1_early_action_steps += agent->gate1_early_action_steps;
    env->log.gate1_early_action_pitch += agent->gate1_early_action_pitch;
    env->log.gate1_early_action_roll += agent->gate1_early_action_roll;
    env->log.gate1_early_action_thrust += agent->gate1_early_action_thrust;
    env->log.gate1_early_action_yaw += agent->gate1_early_action_yaw;
    env->log.gate2_crossing_sampled += agent->gate2_crossing_sampled ? 1.0f : 0.0f;
    env->log.gate2_crossing_radial += agent->gate2_crossing_sampled
        ? agent->gate2_crossing_radial : 0.0f;
    env->log.gate2_crossing_right += agent->gate2_crossing_sampled
        ? agent->gate2_crossing_right : 0.0f;
    env->log.gate2_crossing_vertical += agent->gate2_crossing_sampled
        ? agent->gate2_crossing_vertical : 0.0f;
    env->log.gate2_exit_vx += agent->gate2_crossing_sampled ? agent->gate2_exit_vx : 0.0f;
    env->log.gate2_exit_vy += agent->gate2_crossing_sampled ? agent->gate2_exit_vy : 0.0f;
    env->log.gate2_exit_vz += agent->gate2_crossing_sampled ? agent->gate2_exit_vz : 0.0f;
    bool gate2_completion = agent->gate2_crossing_sampled && success;
    bool gate2_downstream_failure = agent->gate2_crossing_sampled && !success;
    env->log.gate2_completion_sampled += gate2_completion ? 1.0f : 0.0f;
    env->log.gate2_completion_crossing_radial += gate2_completion
        ? agent->gate2_crossing_radial : 0.0f;
    env->log.gate2_completion_crossing_right += gate2_completion
        ? agent->gate2_crossing_right : 0.0f;
    env->log.gate2_completion_crossing_vertical += gate2_completion
        ? agent->gate2_crossing_vertical : 0.0f;
    env->log.gate2_completion_exit_vx += gate2_completion ? agent->gate2_exit_vx : 0.0f;
    env->log.gate2_completion_exit_vy += gate2_completion ? agent->gate2_exit_vy : 0.0f;
    env->log.gate2_completion_exit_vz += gate2_completion ? agent->gate2_exit_vz : 0.0f;
    env->log.gate2_downstream_failure_sampled += gate2_downstream_failure ? 1.0f : 0.0f;
    env->log.gate2_downstream_failure_crossing_radial += gate2_downstream_failure
        ? agent->gate2_crossing_radial : 0.0f;
    env->log.gate2_downstream_failure_crossing_right += gate2_downstream_failure
        ? agent->gate2_crossing_right : 0.0f;
    env->log.gate2_downstream_failure_crossing_vertical += gate2_downstream_failure
        ? agent->gate2_crossing_vertical : 0.0f;
    env->log.gate2_downstream_failure_exit_vx += gate2_downstream_failure
        ? agent->gate2_exit_vx : 0.0f;
    env->log.gate2_downstream_failure_exit_vy += gate2_downstream_failure
        ? agent->gate2_exit_vy : 0.0f;
    env->log.gate2_downstream_failure_exit_vz += gate2_downstream_failure
        ? agent->gate2_exit_vz : 0.0f;
    env->log.gate2_action_steps += agent->gate2_action_steps;
    env->log.gate2_action_pitch += agent->gate2_action_pitch;
    env->log.gate2_action_roll += agent->gate2_action_roll;
    env->log.gate2_action_thrust += agent->gate2_action_thrust;
    env->log.gate2_action_yaw += agent->gate2_action_yaw;
    env->log.gate3_action_steps += agent->gate3_action_steps;
    env->log.gate3_action_pitch += agent->gate3_action_pitch;
    env->log.gate3_action_roll += agent->gate3_action_roll;
    env->log.gate3_action_thrust += agent->gate3_action_thrust;
    env->log.gate3_action_yaw += agent->gate3_action_yaw;
    env->log.terminal_crossing_sampled += agent->terminal_crossing_sampled ? 1.0f : 0.0f;
    env->log.terminal_crossing_radial += agent->terminal_crossing_sampled
        ? agent->terminal_crossing_radial : 0.0f;
    env->log.terminal_crossing_right += agent->terminal_crossing_sampled
        ? agent->terminal_crossing_right : 0.0f;
    env->log.terminal_crossing_vertical += agent->terminal_crossing_sampled
        ? agent->terminal_crossing_vertical : 0.0f;
    env->log.terminal_crossing_abs_right += agent->terminal_crossing_sampled
        ? fabsf(agent->terminal_crossing_right) : 0.0f;
    env->log.terminal_crossing_abs_vertical += agent->terminal_crossing_sampled
        ? fabsf(agent->terminal_crossing_vertical) : 0.0f;
    for (int gate = 0; gate < DRONE_RACE_MAX_GATES; gate++) {
        env->log.ordered_gate_sampled[gate] += agent->ordered_gate_sampled[gate]
            ? 1.0f : 0.0f;
        env->log.ordered_gate_radial[gate] += agent->ordered_gate_sampled[gate]
            ? agent->ordered_gate_radial[gate] : 0.0f;
        env->log.ordered_gate_right[gate] += agent->ordered_gate_sampled[gate]
            ? agent->ordered_gate_right[gate] : 0.0f;
        env->log.ordered_gate_vertical[gate] += agent->ordered_gate_sampled[gate]
            ? agent->ordered_gate_vertical[gate] : 0.0f;
    }
    env->log.crossing_margin_violation += agent->crossing_margin_violation
        ? 1.0f : 0.0f;
    env->log.action_envelope_violation += agent->action_envelope_violation
        ? 1.0f : 0.0f;
    env->log.wire_rate_envelope_violation += agent->wire_rate_envelope_violation
        ? 1.0f : 0.0f;
    env->log.thrust_envelope_violation += agent->thrust_envelope_violation
        ? 1.0f : 0.0f;
    if (agent->terminal_crossing_sampled
            && agent->terminal_crossing_gate_index >= 0
            && agent->terminal_crossing_gate_index < DRONE_RACE_MAX_GATES) {
        int gate = agent->terminal_crossing_gate_index;
        env->log.terminal_gate_sampled[gate] += 1.0f;
        env->log.terminal_gate_radial[gate] += agent->terminal_crossing_radial;
        env->log.terminal_gate_right[gate] += agent->terminal_crossing_right;
        env->log.terminal_gate_vertical[gate] += agent->terminal_crossing_vertical;
    }
    bool gate2_crossing = agent->terminal_crossing_sampled
        && agent->terminal_crossing_gate_index == 2;
    env->log.gate2_terminal_crossing_sampled += gate2_crossing ? 1.0f : 0.0f;
    env->log.gate2_terminal_crossing_radial += gate2_crossing
        ? agent->terminal_crossing_radial : 0.0f;
    env->log.gate2_terminal_crossing_right += gate2_crossing
        ? agent->terminal_crossing_right : 0.0f;
    env->log.gate2_terminal_crossing_vertical += gate2_crossing
        ? agent->terminal_crossing_vertical : 0.0f;
    env->log.out_of_order += agent->out_of_order ? 1.0f : 0.0f;
    env->log.missed_gate += agent->missed_gate ? 1.0f : 0.0f;
    env->log.timeout += agent->timeout ? 1.0f : 0.0f;
    if (agent->episode_ordinal == 0) {
        env->log.episode_slot0_n += 1.0f;
        env->log.episode_slot0_success += success ? 1.0f : 0.0f;
    } else if (agent->episode_ordinal == 1) {
        env->log.episode_slot1_n += 1.0f;
        env->log.episode_slot1_success += success ? 1.0f : 0.0f;
    } else if (agent->episode_ordinal == 2) {
        env->log.episode_slot2_n += 1.0f;
        env->log.episode_slot2_success += success ? 1.0f : 0.0f;
    } else if (agent->episode_ordinal == 3) {
        env->log.episode_slot3_n += 1.0f;
        env->log.episode_slot3_success += success ? 1.0f : 0.0f;
    }
    agent->episode_ordinal += 1;
    env->log.n += 1.0f;
}

static void compute_one_observation(DroneRace* env, int i) {
    DroneRaceAgent* agent = &env->agents[i];
    float* obs = &env->observations[i * DRONE_RACE_OBS_SIZE];
#if DRONE_RACE_OBS_SIZE == DRONE_RACE_VISUAL_OBS_SIZE
    if (env->visual_observation) {
        compute_visual_observation(env, i);
        return;
    }
#endif
    if (env->interface_mode == DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY) {
        memset(obs, 0, DRONE_RACE_OBS_SIZE * sizeof(float));
        int idx = 0;
        // Stationary-gate relative rates in the fixed local course frame. This
        // is exactly reproducible from LOCAL_POSITION_NED velocity.
        obs[idx++] = tanhf(-agent->drone.state.vel.x * 0.2f);
        obs[idx++] = tanhf(-agent->drone.state.vel.y * 0.3333333333f);
        obs[idx++] = tanhf(agent->drone.state.vel.z * 0.3333333333f);
        idx += 3; // no body-rate dependency in the local-velocity plant
        obs[idx++] = 1.0f; // fixed local-course attitude quaternion
        idx += 3;

        if (agent->current_gate < env->num_gates) {
            Vec3 rel = sub3(
                agent_gate(env, agent, agent->current_gate)->pos,
                agent->drone.state.pos);
            float forward = rel.x;
            float right = rel.y;
            float down = -rel.z;
            float range = fmaxf(norm3(rel), 1e-3f);
            float yaw_error = atan2f(right, fmaxf(forward, 1e-4f));
            float elevation = atan2f(-down, fmaxf(forward, 1e-4f));
            obs[idx++] = 1.0f;
            obs[idx++] = tanhf(forward * 0.1f);
            obs[idx++] = tanhf(right * 0.2f);
            obs[idx++] = tanhf(down * 0.2f);
            obs[idx++] = yaw_error / TS002_CAMERA_HALF_FOV_RAD;
            obs[idx++] = elevation / TS002_CAMERA_HALF_FOV_RAD;
            obs[idx++] = clampf(TS002_GATE_INNER_WIDTH_M / range, 0.0f, 1.0f);
            float yaw_score = 1.0f - fabsf(yaw_error) / TS002_CAMERA_HALF_FOV_RAD;
            float pitch_score = 1.0f - fabsf(elevation) / TS002_CAMERA_HALF_FOV_RAD;
            obs[idx++] = clampf(0.5f * (yaw_score + pitch_score), 0.0f, 1.0f);
        } else {
            idx += 8;
        }

        obs[idx++] = env->time_limit_seconds > 0.0f
            ? clampf(agent->elapsed_time / env->time_limit_seconds, 0.0f, 1.0f)
            : 0.0f;
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            obs[idx++] = agent->last_action[k];
        }
        float denominator = env->observable_gate_index_denominator > 0.0f
            ? env->observable_gate_index_denominator
            : (float)fmaxf(env->num_gates, 1);
        float gate_progress = (float)agent->current_gate / denominator;
        obs[idx++] = env->observable_gate_progress_unbounded
            ? fmaxf(gate_progress, 0.0f)
            : clampf(gate_progress, 0.0f, 1.0f);
        for (int gate_phase = 0; gate_phase < 6; gate_phase++) {
            obs[idx++] = env->observable_gate_phase_onehot
                    && agent->current_gate == gate_phase
                ? 1.0f
                : 0.0f;
        }
        idx += 2;
        for (int j = 0; j < DRONE_RACE_OBS_SIZE; j++) {
            if (env->observable_gate_progress_unbounded && j == 23) continue;
            obs[j] = clampf(obs[j], -1.0f, 1.0f);
        }
        return;
    }
    if (env->interface_mode == DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW
            || env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT) {
        memset(obs, 0, DRONE_RACE_OBS_SIZE * sizeof(float));
        Quat q = agent->drone.state.quat;
        Quat q_inv = quat_inverse(q);

        int idx = 0;
        // Camera-derived gate motion. Cascaded continuous-time low-pass filters
        // reproduce the live 15 Hz detector contract without exposing native
        // translation state or amplifying pixel-quantized range changes.
        int gate_motion_offset = idx;
        obs[idx++] = 0.0f; // gate forward rate
        obs[idx++] = 0.0f; // gate right rate
        obs[idx++] = 0.0f; // gate down rate
        obs[idx++] = agent->drone.state.omega.x / agent->drone.params.max_omega;
        obs[idx++] = agent->drone.state.omega.y / agent->drone.params.max_omega;
        obs[idx++] = agent->drone.state.omega.z / agent->drone.params.max_omega;
        obs[idx++] = q.w;
        obs[idx++] = q.x;
        obs[idx++] = q.y;
        obs[idx++] = q.z;

        if (agent->current_gate < env->num_gates) {
            Vec3 rel_world = sub3(
                agent_gate(env, agent, agent->current_gate)->pos,
                agent->drone.state.pos);
            Vec3 rel_body = quat_rotate(q_inv, rel_world);
            float forward = rel_body.x;
            float right = rel_body.y;
            float down = -rel_body.z;
            float range = fmaxf(norm3(rel_body), 1e-3f);
            float yaw_error = atan2f(right, fmaxf(forward, 1e-4f));
            float elevation = atan2f(-down, fmaxf(forward, 1e-4f));
            float camera_pitch_error = elevation - TS002_CAMERA_UPTILT_RAD;
            bool visible = forward > 0.05f
                && fabsf(yaw_error) <= TS002_CAMERA_HALF_FOV_RAD
                && fabsf(camera_pitch_error) <= TS002_CAMERA_HALF_FOV_RAD;

            if (visible && env->sitl_gate_obs_dropout_range_m > 0.0f
                    && agent->current_gate >= env->sitl_gate_obs_dropout_from_index
                    && range <= env->sitl_gate_obs_dropout_range_m) {
                visible = false;
            }
            if (visible && env->sitl_gate_obs_dropout_prob > 0.0f
                    && rndf(0.0f, 1.0f, &env->rng) < env->sitl_gate_obs_dropout_prob) {
                visible = false;
            }
            if (visible && env->sitl_gate_obs_sample_interval_steps > 1
                    && agent->step_count % env->sitl_gate_obs_sample_interval_steps != 0) {
                // The official camera is about 15 Hz while policy inference is
                // about 60 Hz. Opt-in sampling exposes one fresh geometric
                // measurement per interval and exercises the same prediction
                // path on intervening control ticks.
                visible = false;
            }
            if (visible && !agent->gate_motion_valid && range > 35.0f) {
                visible = false;
            }

            if (visible && env->dt > 0.0f) {
                if (!agent->gate_motion_valid
                        || agent->gate_motion_gate_index != agent->current_gate) {
                    agent->gate_motion_raw_world = rel_world;
                    agent->gate_motion_filtered_world = rel_world;
                    agent->gate_motion_rate_world = (Vec3){
                        env->sitl_gate_motion_initial_vx,
                        env->sitl_gate_motion_initial_vy,
                        env->sitl_gate_motion_initial_vz,
                    };
                    agent->gate_motion_valid = 1;
                    agent->gate_motion_gate_index = agent->current_gate;
                } else {
                    Vec3 innovation = sub3(rel_world, agent->gate_motion_raw_world);
                    float innovation_limit = fmaxf(3.0f, 30.0f * env->dt);
                    if (norm3(innovation) <= innovation_limit) {
                        float position_alpha = 1.0f - expf(-env->dt / 0.30f);
                        float rate_alpha = 1.0f - expf(-env->dt / 0.60f);
                        Vec3 previous_filtered = agent->gate_motion_filtered_world;
                        agent->gate_motion_filtered_world = add3(
                            previous_filtered,
                            scalmul3(sub3(rel_world, previous_filtered), position_alpha));
                        Vec3 measured_rate = scalmul3(
                            sub3(agent->gate_motion_filtered_world, previous_filtered),
                            1.0f / env->dt);
                        agent->gate_motion_rate_world = add3(
                            agent->gate_motion_rate_world,
                            scalmul3(
                                sub3(measured_rate, agent->gate_motion_rate_world),
                                rate_alpha));
                        agent->gate_motion_raw_world = rel_world;
                    }
                }
                Vec3 gate_rate = quat_rotate(q_inv, agent->gate_motion_rate_world);
                obs[gate_motion_offset] = tanhf(gate_rate.x * 0.2f);
                obs[gate_motion_offset + 1] = tanhf(gate_rate.y * 0.3333333333f);
                obs[gate_motion_offset + 2] = tanhf(-gate_rate.z * 0.3333333333f);

                // Use the last accepted raw pose for zero-lag temporal data
                // association. The filtered pose is reserved for deriving a
                // stable rate: using it directly delayed gate-plane crossings.
                // A rejected far-gate/structure frame therefore cannot steer
                // the policy, while a valid pose is not phase-shifted.
                rel_body = quat_rotate(q_inv, agent->gate_motion_raw_world);
                forward = rel_body.x;
                right = rel_body.y;
                down = -rel_body.z;
                if (env->sitl_gate_range_bias_m != 0.0f) {
                    forward += env->sitl_gate_range_bias_m;
                }
                range = fmaxf(norm3((Vec3){forward, right, down}), 1e-3f);
                yaw_error = atan2f(right, fmaxf(forward, 1e-4f));
                elevation = atan2f(-down, fmaxf(forward, 1e-4f));
                camera_pitch_error = elevation - TS002_CAMERA_UPTILT_RAD;
            } else if (agent->gate_motion_valid
                    && agent->gate_motion_gate_index == agent->current_gate) {
                // Default behavior holds the last associated gate. The opt-in
                // predictor advances that observable pose with the filtered
                // world-frame gate rate. The optional control-aware term uses
                // only observable roll to integrate the fitted lateral
                // acceleration during dropout; zero preserves constant-rate
                // N218 behavior.
                if (env->sitl_gate_motion_predict_dropout) {
                    Vec3 previous_rate = agent->gate_motion_rate_world;
                    if (env->sitl_gate_motion_control_accel_gain != 0.0f) {
                        EulerAngles attitude = quat_to_euler(q);
                        float bounded_roll = clampf(attitude.roll, -1.2f, 1.2f);
                        agent->gate_motion_rate_world.y +=
                            env->sitl_gate_motion_control_accel_gain
                            * tanf(bounded_roll) * env->dt;
                    }
                    Vec3 delta = scalmul3(
                        add3(previous_rate, agent->gate_motion_rate_world),
                        0.5f * env->dt);
                    agent->gate_motion_raw_world = add3(
                        agent->gate_motion_raw_world, delta);
                    agent->gate_motion_filtered_world = add3(
                        agent->gate_motion_filtered_world, delta);
                }
                Vec3 gate_rate = quat_rotate(q_inv, agent->gate_motion_rate_world);
                obs[gate_motion_offset] = tanhf(gate_rate.x * 0.2f);
                obs[gate_motion_offset + 1] = tanhf(gate_rate.y * 0.3333333333f);
                obs[gate_motion_offset + 2] = tanhf(-gate_rate.z * 0.3333333333f);
                rel_body = quat_rotate(q_inv, agent->gate_motion_raw_world);
                forward = rel_body.x;
                right = rel_body.y;
                down = -rel_body.z;
                range = fmaxf(norm3(rel_body), 1e-3f);
                yaw_error = atan2f(right, fmaxf(forward, 1e-4f));
                elevation = atan2f(-down, fmaxf(forward, 1e-4f));
                camera_pitch_error = elevation - TS002_CAMERA_UPTILT_RAD;
                visible = true;
            } else {
                agent->gate_motion_valid = 0;
                agent->gate_motion_rate_world = (Vec3){0.0f, 0.0f, 0.0f};
            }

            obs[idx++] = visible ? 1.0f : 0.0f;
            if (visible) {
                obs[idx++] = tanhf(forward * 0.1f);
                obs[idx++] = tanhf(right * 0.2f);
                obs[idx++] = tanhf(down * 0.2f);
                obs[idx++] = yaw_error / TS002_CAMERA_HALF_FOV_RAD;
                obs[idx++] = camera_pitch_error / TS002_CAMERA_HALF_FOV_RAD;
                obs[idx++] = clampf(TS002_GATE_INNER_WIDTH_M / range, 0.0f, 1.0f);
                float yaw_score = 1.0f - fabsf(yaw_error) / TS002_CAMERA_HALF_FOV_RAD;
                float pitch_score = 1.0f - fabsf(camera_pitch_error) / TS002_CAMERA_HALF_FOV_RAD;
                // Unlike apparent size, this supplies nonredundant alignment
                // quality and is computed entirely from official camera pose.
                obs[idx++] = clampf(0.5f * (yaw_score + pitch_score), 0.0f, 1.0f);
            } else {
                idx += 7;
            }
        } else {
            idx += 8;
        }

        obs[idx++] = env->time_limit_seconds > 0.0f
            ? clampf(agent->elapsed_time / env->time_limit_seconds, 0.0f, 1.0f)
            : 0.0f;
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            obs[idx++] = agent->last_action[k];
        }
        if (env->observable_gate_progress) {
            float denominator = env->observable_gate_index_denominator > 0.0f
                ? env->observable_gate_index_denominator
                : (float)fmaxf(env->num_gates, 1);
            float gate_progress = (float)agent->current_gate / denominator;
            obs[idx++] = env->observable_gate_progress_unbounded
                ? fmaxf(gate_progress, 0.0f)
                : clampf(gate_progress, 0.0f, 1.0f);
            for (int gate_phase = 0; gate_phase < 6; gate_phase++) {
                obs[idx++] = env->observable_gate_phase_onehot
                        && agent->current_gate == gate_phase
                    ? 1.0f
                    : 0.0f;
            }
            // Preserve two reserved values for future observable-only ABI
            // additions. memset above guarantees both remain exact zero.
            idx += 2;
        } else {
            for (int gate_phase = 1; gate_phase <= 3; gate_phase++) {
                obs[idx++] = env->observable_gate_index
                        && agent->current_gate == gate_phase
                    ? 1.0f
                    : 0.0f;
            }
            bool gate_two_active = env->observable_gate_index
                && agent->current_gate == 2;
            bool gate_three_active = env->observable_gate_index
                && agent->current_gate == 3;
            obs[idx++] = gate_two_active ? obs[12] : 0.0f;
            obs[idx++] = gate_two_active ? obs[1] : 0.0f;
            obs[idx++] = gate_two_active ? obs[13] : 0.0f;
            obs[idx++] = gate_two_active ? obs[2] : 0.0f;
            obs[idx++] = gate_three_active ? obs[12] : 0.0f;
            obs[idx++] = gate_three_active ? obs[1] : 0.0f;
        }
    } else {
        compute_drone_observations(&agent->drone, obs);
    }
    for (int j = 0; j < DRONE_RACE_OBS_SIZE; j++) {
        if (env->observable_gate_progress_unbounded
                && env->observable_gate_progress
                && j == 23) {
            continue;
        }
        obs[j] = clampf(obs[j], -1.0f, 1.0f);
    }
}

static void compute_observations(DroneRace* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_one_observation(env, i);
    }
}

void init(DroneRace* env) {
    apply_interface_defaults(env);
    snapshot_sitl_nominals(env);
    build_course(env);
    build_course_teacher_profile(env);
    env->agents = (DroneRaceAgent*)calloc(env->num_agents, sizeof(DroneRaceAgent));
    memset(&env->log, 0, sizeof(Log));
    env->evaluation_episode_offset_applied = 0;
}

static float transition_speed_floor(
        const DroneRace* env, const DroneRaceAgent* agent) {
    return env->sitl_gate_transition_min_forward_speed_randomize
        ? agent->randomized_sitl_gate_transition_min_forward_speed
        : env->sitl_gate_transition_min_forward_speed;
}

static bool transition_speed_floor_active(
        const DroneRace* env, const DroneRaceAgent* agent) {
    int from_gate_index = env->sitl_gate_transition_from_gate_index > 0
        ? env->sitl_gate_transition_from_gate_index
        : 1;
    return agent->current_gate >= from_gate_index
        && agent->current_gate < env->num_gates;
}

static void raise_transition_forward_speed(
        DroneRace* env, DroneRaceAgent* agent, float target_forward_speed) {
    Vec3 direction = agent_gate(env, agent, agent->current_gate)->normal;
    float forward_speed = dot3(agent->drone.state.vel, direction);
    if (target_forward_speed <= forward_speed) return;

    if (env->sitl_gate_transition_preserve_velocity_direction
            && forward_speed > 1e-4f) {
        agent->drone.state.vel = scalmul3(
            agent->drone.state.vel,
            target_forward_speed / forward_speed);
    } else {
        agent->drone.state.vel = add3(
            agent->drone.state.vel,
            scalmul3(direction, target_forward_speed - forward_speed));
    }
}

static void apply_pending_transition_speed_floor(
        DroneRace* env, DroneRaceAgent* agent) {
    if (agent->transition_speed_floor_delay_steps_remaining <= 0) return;
    agent->transition_speed_floor_delay_steps_remaining--;
    if (agent->transition_speed_floor_delay_steps_remaining == 0
            && transition_speed_floor_active(env, agent)) {
        raise_transition_forward_speed(
            env, agent, transition_speed_floor(env, agent));
    }
}

static void apply_transition_speed_ramp(
        DroneRace* env, DroneRaceAgent* agent) {
    if (env->interface_mode != DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT
            || env->sitl_gate_transition_max_accel_m_s2 <= 0.0f
            || env->dt <= 0.0f
            || !transition_speed_floor_active(env, agent)) {
        return;
    }
    float speed_floor = transition_speed_floor(env, agent);
    if (speed_floor <= 0.0f) return;
    Vec3 direction = agent_gate(env, agent, agent->current_gate)->normal;
    float forward_speed = dot3(agent->drone.state.vel, direction);
    float speed_step = fminf(
        speed_floor - forward_speed,
        env->sitl_gate_transition_max_accel_m_s2 * env->dt);
    if (speed_step > 0.0f) {
        raise_transition_forward_speed(env, agent, forward_speed + speed_step);
    }
}

void c_reset(DroneRace* env) {
    apply_sitl_domain_randomize(env);
    if (!env->evaluation_episode_offset_applied) {
        int offset = env->evaluation_episode_offset;
        if (offset < 0) offset = 0;
        for (int skipped = 0; skipped < offset; skipped++) {
            for (int i = 0; i < env->num_agents; i++) {
                reset_agent(env, &env->agents[i]);
                env->agents[i].episode_ordinal += 1;
            }
        }
        env->evaluation_episode_offset_applied = 1;
    }
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, &env->agents[i]);
    }
    compute_observations(env);
}

void c_step(DroneRace* env) {
    if (env->evaluation_episode_limit > 0
            && env->log.n >= (float)env->evaluation_episode_limit) {
        // Exact-count evaluation freezes an env after its assigned quota. The
        // aggregate then contains exactly quota * vector-env-count episodes,
        // independent of differences in episode duration. This path is never
        // enabled during training or ordinary development screens.
        return;
    }
    for (int i = 0; i < env->num_agents; i++) {
        DroneRaceAgent* agent = &env->agents[i];
        if (env->visual_observation && agent->pending_reset) {
            // Return the true terminal visual observation on the preceding
            // step. This reset-only step is marked invalid by the dedicated
            // Dreamer collector and prevents an action inferred from terminal
            // pixels from driving a freshly reset vehicle.
            reset_agent(env, agent);
            env->rewards[i] = 0.0f;
            env->terminals[i] = 0.0f;
            continue;
        }
        int reward_gate_index = agent->current_gate;
        float* policy_action = &env->actions[i * DRONE_RACE_NUM_ATNS];
        float motor_action[DRONE_RACE_NUM_ATNS] = {0};
        float executed_action[DRONE_RACE_NUM_ATNS];
        float telemetry_teacher[DRONE_RACE_NUM_ATNS] = {0};
        bool telemetry_teacher_valid = false;
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            executed_action[k] = policy_action[k];
        }
        ActionTeacherTarget teacher = action_teacher_target(
            env, agent, reward_gate_index);
        if (env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT
                && env->teacher_action_blend > 0.0f
                && agent->step_count >= env->teacher_action_blend_from_step
                && teacher.valid) {
            float blend = env->teacher_action_blend;
            if (teacher.pitch_active) {
                executed_action[0] = (1.0f - blend) * policy_action[0]
                    + blend * teacher.pitch;
            }
            if (teacher.roll_active) {
                executed_action[1] = (1.0f - blend) * policy_action[1]
                    + blend * teacher.roll;
            }
            if (teacher.thrust_active) {
                executed_action[2] = (1.0f - blend) * policy_action[2]
                    + blend * teacher.thrust;
            }
            if (teacher.yaw_active) {
                executed_action[3] = (1.0f - blend) * policy_action[3]
                    + blend * teacher.yaw;
            }
        } else if (env->interface_mode
                    == DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY) {
            telemetry_teacher_valid = telemetry_velocity_teacher_action(
                env, agent, telemetry_teacher);
            if (telemetry_teacher_valid
                    && env->teacher_action_blend > 0.0f
                    && agent->step_count >= env->teacher_action_blend_from_step) {
                float blend = env->teacher_action_blend;
                for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
                    executed_action[k] = (1.0f - blend) * policy_action[k]
                        + blend * telemetry_teacher[k];
                }
            }
        }

        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            if (!isfinite(executed_action[k])
                    || executed_action[k] < -1.0f - 1e-6f
                    || executed_action[k] > 1.0f + 1e-6f) {
                agent->action_envelope_violation = 1;
            }
        }

        if (reward_gate_index == 0) {
            agent->gate0_action_steps += 1.0f;
            agent->gate0_action_pitch += policy_action[0];
        }
        if (reward_gate_index == 1) {
            bool early = agent->gate1_action_steps < 60.0f;
            agent->gate1_action_steps += 1.0f;
            agent->gate1_action_pitch += policy_action[0];
            agent->gate1_action_roll += policy_action[1];
            agent->gate1_action_thrust += policy_action[2];
            agent->gate1_action_yaw += policy_action[3];
            if (early) {
                agent->gate1_early_action_steps += 1.0f;
                agent->gate1_early_action_pitch += policy_action[0];
                agent->gate1_early_action_roll += policy_action[1];
                agent->gate1_early_action_thrust += policy_action[2];
                agent->gate1_early_action_yaw += policy_action[3];
            }
        }
        if (reward_gate_index == 2) {
            agent->gate2_action_steps += 1.0f;
            agent->gate2_action_pitch += policy_action[0];
            agent->gate2_action_roll += policy_action[1];
            agent->gate2_action_thrust += policy_action[2];
            agent->gate2_action_yaw += policy_action[3];
        }
        if (reward_gate_index == 3) {
            agent->gate3_action_steps += 1.0f;
            agent->gate3_action_pitch += policy_action[0];
            agent->gate3_action_roll += policy_action[1];
            agent->gate3_action_thrust += policy_action[2];
            agent->gate3_action_yaw += policy_action[3];
        }

        agent->drone.prev_pos = agent->drone.state.pos;
        float prev_remaining = remaining_distance(env, agent);
        for (int history = 2; history > 0; history--) {
            for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
                agent->last_action_history[history][k] =
                    agent->last_action_history[history - 1][k];
            }
        }
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            // Observations report the action that actually drove the plant.
            // At deployment blend is zero, so this is the raw policy action.
            agent->last_action[k] = executed_action[k];
            agent->last_action_history[0][k] = executed_action[k];
        }
        FloorRisk pre_floor_risk = compute_floor_risk(env, agent);
        update_floor_risk_stats(agent, pre_floor_risk);

        apply_pending_transition_speed_floor(env, agent);
        apply_transition_speed_ramp(env, agent);

        if (env->interface_mode == DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY) {
            move_drone_vq1_telemetry_velocity(env, agent, executed_action);
        } else if (env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT) {
            move_drone_sitl_plant(env, agent, executed_action);
        } else {
            policy_to_motor_actions(env, agent, executed_action, motor_action);
            move_drone_for_dt(&agent->drone, motor_action, env->dt);
        }
        agent->elapsed_time += env->dt;
        agent->step_count += 1;

        bool success = false;
        bool active_gate_plane_crossed = false;
        bool ordered_gate_passed = false;
        float radial = 0.0f;
        Vec3 crossing_radial_vector = {0.0f, 0.0f, 0.0f};
        int crossing_gate_index = agent->current_gate;
        if (agent->current_gate < env->num_gates && gate_crossing_details(
                agent->drone.prev_pos, agent->drone.state.pos,
                agent_gate(env, agent, agent->current_gate),
                env->plane_cross_tolerance,
                env->direction_min, &radial, &crossing_radial_vector)) {
            active_gate_plane_crossed = true;
            ordered_gate_passed = true;
            agent->ordered_gate_sampled[crossing_gate_index] = 1;
            agent->ordered_gate_radial[crossing_gate_index] = radial;
            agent->ordered_gate_right[crossing_gate_index] =
                crossing_radial_vector.y;
            agent->ordered_gate_vertical[crossing_gate_index] =
                crossing_radial_vector.z;
            if (radial > 0.50f) {
                agent->crossing_margin_violation = 1;
            }
            if (crossing_gate_index == 0) {
                agent->gate0_exit_sampled = 1;
                agent->gate0_exit_vx = agent->drone.state.vel.x;
                agent->gate0_exit_vy = agent->drone.state.vel.y;
                agent->gate0_exit_vz = agent->drone.state.vel.z;
            }
            if (crossing_gate_index == 2) {
                agent->gate2_crossing_sampled = 1;
                agent->gate2_crossing_radial = radial;
                agent->gate2_crossing_right = crossing_radial_vector.y;
                agent->gate2_crossing_vertical = crossing_radial_vector.z;
                agent->gate2_exit_vx = agent->drone.state.vel.x;
                agent->gate2_exit_vy = agent->drone.state.vel.y;
                agent->gate2_exit_vz = agent->drone.state.vel.z;
            }
            if (crossing_gate_index == env->num_gates - 1) {
                record_terminal_crossing(
                    agent, crossing_gate_index, radial, crossing_radial_vector);
            }
            agent->current_gate += 1;
            float transition_speed = transition_speed_floor(env, agent);
            if (env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT
                    && transition_speed_floor_active(env, agent)
                    && transition_speed > 0.0f
                    && env->sitl_gate_transition_max_accel_m_s2 <= 0.0f) {
                // Official v3385 traces show gate 2 closing near 10 m/s after
                // gate 1, while the compact plant otherwise exits near 5 m/s.
                // Reproduce that observable transition state without adding a
                // privileged policy input or any runtime flight arbitration.
                if (env->sitl_gate_transition_delay_steps > 0) {
                    agent->transition_speed_floor_delay_steps_remaining =
                        env->sitl_gate_transition_delay_steps;
                } else {
                    raise_transition_forward_speed(env, agent, transition_speed);
                }
            }
            set_agent_target(env, agent);
            if (agent->current_gate >= env->num_gates) {
                success = true;
            }
        } else if (agent->current_gate < env->num_gates && env->strict_missed_gate) {
            float miss_radial = 0.0f;
            Vec3 miss_radial_vector = {0.0f, 0.0f, 0.0f};
            bool crossed_plane = gate_plane_crossing_details(
                agent->drone.prev_pos, agent->drone.state.pos,
                agent_gate(env, agent, agent->current_gate),
                env->plane_cross_tolerance,
                env->direction_min,
                &miss_radial,
                &miss_radial_vector);
            // Once the vehicle crosses the active gate plane it cannot recover
            // an ordered pass. End any out-of-aperture crossing immediately;
            // allowing a near miss to continue created a reward exploit where
            // policies flew to the course boundary after missing gate 2.
            if (crossed_plane
                    && miss_radial > agent_gate(env, agent, agent->current_gate)->radius) {
                // This crossing terminates the episode even when it occurs on
                // an early gate. Preserve its signed geometry so full-course
                // curricula can rank two candidates stuck on the same prefix.
                record_terminal_crossing(
                    agent,
                    agent->current_gate,
                    miss_radial,
                    miss_radial_vector);
                active_gate_plane_crossed = true;
                radial = miss_radial;
                agent->missed_gate = 1;
                agent->valid_run = 0;
            }
        }

        for (int gate = agent->current_gate + 1; gate < env->num_gates; gate++) {
            if (gate_crossing(
                    agent->drone.prev_pos,
                    agent->drone.state.pos,
                    agent_gate(env, agent, gate),
                    env->plane_cross_tolerance, env->direction_min, NULL)) {
                agent->out_of_order = 1;
                agent->valid_run = 0;
                break;
            }
        }

        float current_remaining = remaining_distance(env, agent);
        float progress_delta = prev_remaining - current_remaining;
        float initial_remaining = fmaxf(env->start_offset + (float)(env->num_gates - 1) * env->gate_spacing, 1.0f);
        agent->progress = clampf(1.0f - current_remaining / initial_remaining, 0.0f, 1.0f);
        agent->max_progress = fmaxf(agent->max_progress, agent->progress);
        agent->closest_gate_range = fminf(agent->closest_gate_range, current_remaining);

        Vec3 pos = agent->drone.state.pos;
        bool crash_xy = fabsf(pos.x) > env->pos_bound || fabsf(pos.y) > env->pos_bound;
        bool crash_low = pos.z < env->crash_height;
        bool crash_high = pos.z > env->pos_bound;
        if (crash_xy || crash_low || crash_high) {
            agent->crash = 1;
            agent->crash_xy = crash_xy ? 1 : 0;
            agent->crash_low = crash_low ? 1 : 0;
            agent->crash_high = crash_high ? 1 : 0;
            if (crash_low) {
                agent->crash_low_floor_margin_pre = pre_floor_risk.floor_margin;
                agent->crash_low_ttf_pre = pre_floor_risk.time_to_floor;
                agent->crash_low_stop_margin_pre = pre_floor_risk.stop_margin;
                agent->crash_low_max_up_accel_pre = pre_floor_risk.max_up_accel;
            }
            agent->valid_run = 0;
        }
        if (agent->step_count >= env->max_steps || agent->elapsed_time >= env->time_limit_seconds) {
            agent->timeout = 1;
        }

        float ctrl = 0.0f;
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) ctrl += policy_action[k] * policy_action[k];
        float reward = env->w_progress * progress_delta - env->w_time * env->dt - env->w_ctrl * ctrl;
        reward -= paper_body_rate_penalty(env, &agent->drone.state);
        reward -= cross_track_penalty(env, agent, pos);
        reward -= env->w_gate_camera_alignment * env->dt
            * gate_camera_alignment_error_sq(env, agent);
        reward -= env->w_forward_speed_excess * env->dt
            * forward_speed_excess_error_sq(env, agent);
        if (active_gate_plane_crossed) {
            // A wide curriculum aperture should bootstrap terminal credit, not
            // make a rim-skimming trajectory equivalent to a centered pass.
            // This training-only terminal term uses the same crossing geometry
            // as the judge and adds no privileged policy observation.
            reward -= gate_crossing_error_penalty(
                env, reward_gate_index, radial);
        }
        if ((env->w_gate_exit_lateral_velocity > 0.0f
                    || env->w_gate_exit_velocity > 0.0f)
                && gate_exit_reward_active(env, agent, reward_gate_index)) {
            Target* exit_gate = agent_gate(env, agent, reward_gate_index);
            Vec3 to_exit_gate = sub3(exit_gate->pos, pos);
            float distance_to_plane = fabsf(dot3(to_exit_gate, exit_gate->normal));
            float lookahead = fmaxf(env->gate_exit_reward_lookahead_m, 1e-3f);
            float proximity = expf(-distance_to_plane / lookahead);
            reward -= env->w_gate_exit_lateral_velocity * env->dt * proximity
                * gate_exit_lateral_velocity_error_sq(env, agent, reward_gate_index);
            reward -= env->w_gate_exit_velocity * env->dt * proximity
                * gate_exit_velocity_error_sq(env, agent, reward_gate_index);
        }
        if (env->interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT
                && env->w_action_teacher > 0.0f
                && teacher.valid
                && (teacher.pitch_active
                    || teacher.roll_active
                    || teacher.thrust_active
                    || teacher.yaw_active)) {
            // Compare the raw sampled policy action to the pre-step teacher.
            // This remains aligned even on a gate-crossing transition.
            float pitch_error = policy_action[0] - teacher.pitch;
            float roll_error = policy_action[1] - teacher.roll;
            float thrust_error = policy_action[2] - teacher.thrust;
            float yaw_error = policy_action[3] - teacher.yaw;
            reward -= env->w_action_teacher * env->dt
                * ((teacher.pitch_active
                        ? env->teacher_pitch_weight * pitch_error * pitch_error : 0.0f)
                    + (teacher.roll_active
                        ? env->teacher_roll_weight * roll_error * roll_error : 0.0f)
                    + (teacher.thrust_active
                        ? env->teacher_thrust_weight * thrust_error * thrust_error : 0.0f)
                    + (teacher.yaw_active
                        ? env->teacher_yaw_weight * yaw_error * yaw_error : 0.0f));
        }
        if (env->interface_mode == DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY
                && env->w_action_teacher > 0.0f
                && telemetry_teacher_valid) {
            float teacher_error_sq = 0.0f;
            for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
                float error = policy_action[k] - telemetry_teacher[k];
                teacher_error_sq += error * error;
            }
            reward -= env->w_action_teacher * env->dt * teacher_error_sq;
        }
        float altitude_deficit = fmaxf(env->safety_altitude - pos.z, 0.0f);
        if (env->w_altitude_floor > 0.0f && altitude_deficit > 0.0f) {
            reward -= env->w_altitude_floor * altitude_deficit * altitude_deficit;
        }
        if (env->w_descent_floor > 0.0f && altitude_deficit > 0.0f && agent->drone.state.vel.z < 0.0f) {
            reward -= env->w_descent_floor * -agent->drone.state.vel.z;
        }
        if (agent->current_gate > 0 && progress_delta > 0.0f) reward += env->w_gate * progress_delta;
        if (ordered_gate_passed && env->w_ordered_gate > 0.0f) {
            float aperture = fmaxf(
                agent_gate(env, agent, reward_gate_index)->radius, 1e-4f);
            float centered_crossing = clampf(1.0f - radial / aperture, 0.0f, 1.0f);
            reward += env->w_ordered_gate * centered_crossing;
        }
        if (success) reward += env->w_finish;
        if (!agent->valid_run) {
            reward -= invalid_run_penalty(env, reward_gate_index);
        }

        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool done = success || !agent->valid_run || agent->timeout;
        env->terminals[i] = done ? 1.0f : 0.0f;
        if (done) {
            add_log(env, agent, success && agent->valid_run);
            if (env->visual_observation) {
                agent->pending_reset = 1;
            } else {
                reset_agent(env, agent);
            }
        }
    }
    compute_observations(env);
}

void c_render(DroneRace* env) {
    (void)env;
}

void c_close(DroneRace* env) {
    free(env->agents);
    env->agents = NULL;
}

#ifndef DRONE_RACE_BINDING
int main(void) {
    srand((unsigned int)time(NULL));
    DroneRace env = {0};
    env.num_agents = 1;
    env.dt = 1.0f / 120.0f;
    env.num_gates = 4;
    env.gate_spacing = 5.0f;
    env.gate_radius = 1.0f;
    env.gate_altitude = 1.0f;
    env.gate_lateral_amplitude = 1.5f;
    env.start_offset = 2.0f;
    env.crash_height = 0.0f;
    env.pos_bound = 20.0f;
    env.plane_cross_tolerance = 1e-3f;
    env.miss_tolerance = 0.5f;
    env.direction_min = 0.05f;
    env.strict_missed_gate = 1;
    env.max_steps = 57600;
    env.time_limit_seconds = 480.0f;
    env.safety_altitude = 0.0f;
    env.w_progress = 20.0f;
    env.w_gate = 3.0f;
    env.w_finish = 30.0f;
    env.w_time = 1.0f;
    env.w_ctrl = 0.01f;
    env.w_altitude_floor = 0.0f;
    env.w_descent_floor = 0.0f;
    env.invalid_penalty = 40.0f;
    env.observations = (float*)calloc(DRONE_RACE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(DRONE_RACE_NUM_ATNS, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    init(&env);
    c_reset(&env);
    for (int i = 0; i < 10; i++) {
        c_step(&env);
    }
    printf("drone_race smoke reward=%f terminal=%f\n", env.rewards[0], env.terminals[0]);
    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
#endif
