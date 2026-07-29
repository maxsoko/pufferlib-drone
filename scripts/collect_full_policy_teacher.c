#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DRONE_RACE_BINDING
#include "ocean/drone_race/drone_race.c"
#include "src/puffernet.h"

static float teacher_clamp(float value, float low, float high) {
    return fmaxf(low, fminf(high, value));
}

static float env_float(const char* name, float fallback) {
    const char* value = getenv(name);
    return value == NULL ? fallback : strtof(value, NULL);
}

static int env_int(const char* name, int fallback) {
    const char* value = getenv(name);
    return value == NULL ? fallback : atoi(value);
}

typedef struct TeacherSpline TeacherSpline;
struct TeacherSpline {
    int points;
    float x[DRONE_RACE_MAX_GATES + 1];
    float a[DRONE_RACE_MAX_GATES];
    float b[DRONE_RACE_MAX_GATES];
    float c[DRONE_RACE_MAX_GATES];
    float d[DRONE_RACE_MAX_GATES];
};

// Stateful course controller whose runtime input is exactly the policy
// observation.  The fixed-course spline is a training prior, analogous to
// knowledge stored in policy weights; act() never reads DroneRaceAgent,
// env->gates, position, or native velocity.  Velocity is reconstructed from
// successive observable active-gate poses and retained through gate changes.
typedef struct ObservableCourseTeacher ObservableCourseTeacher;
struct ObservableCourseTeacher {
    TeacherSpline lateral_spline;
    TeacherSpline vertical_spline;
    float gate_x[DRONE_RACE_MAX_GATES];
    float gate_y[DRONE_RACE_MAX_GATES];
    float gate_z[DRONE_RACE_MAX_GATES];
    Vec3 previous_rel_world;
    Vec3 velocity_world;
    Vec3 debug_velocity_world;
    Vec3 debug_desired_position;
    Vec3 debug_acceleration;
    int num_gates;
    int previous_gate;
    int pose_valid;
    int velocity_valid;
};

static int build_teacher_spline(
        TeacherSpline* spline,
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

static int build_teacher_exit_spline(
        TeacherSpline* spline,
        const float* x,
        const float* values,
        int points,
        int exit_tangent_from_point,
        float exit_slope_scale) {
    if (points < 2 || points > DRONE_RACE_MAX_GATES + 1) return 0;
    TeacherSpline natural = {0};
    if (!build_teacher_spline(&natural, x, values, points)) return 0;
    float slopes[DRONE_RACE_MAX_GATES + 1] = {0};
    for (int i = 0; i < points; i++) spline->x[i] = x[i];
    for (int i = 0; i < points - 1; i++) {
        float h = x[i + 1] - x[i];
        if (h <= 1e-4f) return 0;
        if (i >= exit_tangent_from_point) {
            slopes[i] = exit_slope_scale * (values[i + 1] - values[i]) / h;
        } else {
            slopes[i] = natural.b[i];
        }
    }
    if (exit_tangent_from_point < points - 1) {
        slopes[points - 1] = slopes[points - 2];
    } else {
        int final_segment = points - 2;
        float final_dx = x[points - 1] - x[final_segment];
        slopes[points - 1] = natural.b[final_segment]
            + 2.0f * natural.c[final_segment] * final_dx
            + 3.0f * natural.d[final_segment] * final_dx * final_dx;
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

static void evaluate_teacher_spline(
        const TeacherSpline* spline,
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

static float observable_unscale_tanh(float value, float scale) {
    value = teacher_clamp(value, -0.9999f, 0.9999f);
    return atanhf(value) / scale;
}

static int observable_gate_phase(
        const float* observation,
        int gate_progress_observation,
        float gate_progress_denominator) {
    if (gate_progress_observation) {
        float denominator = fmaxf(gate_progress_denominator, 1.0f);
        return (int)lroundf(
            teacher_clamp(observation[23], 0.0f, 1.0f) * denominator);
    }
    if (observation[25] > 0.5f) return 3;
    if (observation[24] > 0.5f) return 2;
    if (observation[23] > 0.5f) return 1;
    return 0;
}

static int init_observable_course_teacher(
        ObservableCourseTeacher* controller,
        const DroneRace* env,
        int exit_tangent_from_gate,
        float exit_slope_scale,
        int use_full_course_origin) {
    int num_gates = env->num_gates;
    if (num_gates < 1 || num_gates > DRONE_RACE_MAX_GATES) return 0;
    float course_x[DRONE_RACE_MAX_GATES + 1] = {0};
    float course_y[DRONE_RACE_MAX_GATES + 1] = {0};
    float course_z[DRONE_RACE_MAX_GATES + 1] = {0};
    course_x[0] = use_full_course_origin ? 0.0f : env->custom_start_pos[0];
    course_y[0] = use_full_course_origin ? 0.0f : env->custom_start_pos[1];
    course_z[0] = use_full_course_origin ? 0.0f : env->custom_start_pos[2];
    memset(controller, 0, sizeof(*controller));
    controller->num_gates = num_gates;
    controller->previous_gate = -1;
    for (int gate = 0; gate < num_gates; gate++) {
        course_x[gate + 1] = env->gates[gate].pos.x;
        course_y[gate + 1] = env->gates[gate].pos.y;
        course_z[gate + 1] = env->gates[gate].pos.z;
        controller->gate_x[gate] = course_x[gate + 1];
        controller->gate_y[gate] = course_y[gate + 1];
        controller->gate_z[gate] = course_z[gate + 1];
    }
    return build_teacher_exit_spline(
            &controller->lateral_spline,
            course_x,
            course_y,
            num_gates + 1,
            exit_tangent_from_gate + 1,
            exit_slope_scale)
        && build_teacher_exit_spline(
            &controller->vertical_spline,
            course_x,
            course_z,
            num_gates + 1,
            exit_tangent_from_gate + 1,
            exit_slope_scale);
}

static void observable_course_teacher_action(
        ObservableCourseTeacher* controller,
        const float* observation,
        float dt,
        int reset_state,
        float velocity_alpha,
        float motion_rate_blend,
        float derivative_lookahead_m,
        float vertical_derivative_lookahead_m,
        int derivative_lookahead_from_gate,
        float lateral_position_kp,
        float close_lateral_position_kp,
        float close_position_range_m,
        float lateral_velocity_kd,
        float vertical_position_kp,
        float vertical_velocity_kd,
        float curvature_scale,
        float linear_drag,
        float lateral_accel_scale,
        float gravity_m_s2,
        float vertical_accel_per_thrust,
        float hover_thrust,
        float min_thrust,
        float max_thrust,
        float max_roll_rad,
        int gate_progress_observation,
        float gate_progress_denominator,
        float* action) {
    int gate = observable_gate_phase(
        observation, gate_progress_observation, gate_progress_denominator);
    if (gate >= controller->num_gates) gate = controller->num_gates - 1;

    Quat q = {
        observation[6], observation[7], observation[8], observation[9],
    };
    float q_norm = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (q_norm > 1e-6f) {
        q.w /= q_norm;
        q.x /= q_norm;
        q.y /= q_norm;
        q.z /= q_norm;
    } else {
        q = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    }

    Vec3 rel_body = {
        observable_unscale_tanh(observation[11], 0.1f),
        observable_unscale_tanh(observation[12], 0.2f),
        -observable_unscale_tanh(observation[13], 0.2f),
    };
    Vec3 rel_world = quat_rotate(q, rel_body);
    Vec3 gate_rate_body = {
        observable_unscale_tanh(observation[0], 0.2f),
        observable_unscale_tanh(observation[1], 0.3333333333f),
        -observable_unscale_tanh(observation[2], 0.3333333333f),
    };
    Vec3 motion_velocity = scalmul3(
        quat_rotate(q, gate_rate_body), -1.0f);

    if (reset_state) {
        controller->pose_valid = 0;
        controller->velocity_valid = 0;
        controller->previous_gate = -1;
    }
    if (!controller->velocity_valid) {
        controller->velocity_world = motion_velocity;
        controller->velocity_valid = 1;
    }
    if (controller->pose_valid && controller->previous_gate == gate && dt > 0.0f) {
        Vec3 rel_delta = sub3(rel_world, controller->previous_rel_world);
        float displacement = norm3(rel_delta);
        // A held pose represents the detector's close-range blind interval.
        // Preserve the last estimate instead of falsely estimating zero speed.
        if (displacement > 1e-5f && displacement < 1.0f) {
            Vec3 finite_difference_velocity = scalmul3(rel_delta, -1.0f / dt);
            float alpha = teacher_clamp(velocity_alpha, 0.0f, 1.0f);
            controller->velocity_world = add3(
                controller->velocity_world,
                scalmul3(
                    sub3(finite_difference_velocity, controller->velocity_world),
                    alpha));
        }
    }
    float rate_blend = teacher_clamp(motion_rate_blend, 0.0f, 1.0f);
    Vec3 velocity_world = add3(
        scalmul3(controller->velocity_world, 1.0f - rate_blend),
        scalmul3(motion_velocity, rate_blend));
    controller->previous_rel_world = rel_world;
    controller->previous_gate = gate;
    controller->pose_valid = 1;

    // Phase plus relative pose identifies position along the fixed-course
    // spline without observing global vehicle coordinates.
    float course_x = controller->gate_x[gate] - rel_world.x;
    float course_y = controller->gate_y[gate] - rel_world.y;
    float course_z = controller->gate_z[gate] - rel_world.z;
    float desired_y;
    float desired_dy_dx;
    float desired_d2y_dx2;
    float desired_z;
    float desired_dz_dx;
    float desired_d2z_dx2;
    evaluate_teacher_spline(
        &controller->lateral_spline,
        course_x,
        &desired_y,
        &desired_dy_dx,
        &desired_d2y_dx2);
    evaluate_teacher_spline(
        &controller->vertical_spline,
        course_x,
        &desired_z,
        &desired_dz_dx,
        &desired_d2z_dx2);
    if (gate >= derivative_lookahead_from_gate && derivative_lookahead_m > 0.0f) {
        float lookahead_x = fminf(
            course_x + derivative_lookahead_m,
            controller->lateral_spline.x[controller->lateral_spline.points - 1]);
        float ignored_value;
        evaluate_teacher_spline(
            &controller->lateral_spline,
            lookahead_x,
            &ignored_value,
            &desired_dy_dx,
            &desired_d2y_dx2);
    }
    if (gate >= derivative_lookahead_from_gate
            && vertical_derivative_lookahead_m > 0.0f) {
        float lookahead_x = fminf(
            course_x + vertical_derivative_lookahead_m,
            controller->vertical_spline.x[controller->vertical_spline.points - 1]);
        float ignored_value;
        evaluate_teacher_spline(
            &controller->vertical_spline,
            lookahead_x,
            &ignored_value,
            &desired_dz_dx,
            &desired_d2z_dx2);
    }

    float forward_speed = fmaxf(velocity_world.x, 0.0f);
    float close_position_blend = close_position_range_m > 0.0f
        ? teacher_clamp(
            1.0f - fmaxf(rel_world.x, 0.0f) / close_position_range_m,
            0.0f,
            1.0f)
        : 0.0f;
    float position_kp = lateral_position_kp
        + close_position_blend * close_lateral_position_kp;
    float accel_y = curvature_scale * desired_d2y_dx2
            * forward_speed * forward_speed
        + position_kp * (desired_y - course_y)
        + lateral_velocity_kd
            * (desired_dy_dx * forward_speed - velocity_world.y);
    float accel_z = curvature_scale * desired_d2z_dx2
            * forward_speed * forward_speed
        + vertical_position_kp * (desired_z - course_z)
        + vertical_velocity_kd
            * (desired_dz_dx * forward_speed - velocity_world.z);
    controller->debug_velocity_world = velocity_world;
    controller->debug_desired_position = (Vec3){course_x, desired_y, desired_z};
    controller->debug_acceleration = (Vec3){0.0f, accel_y, accel_z};

    float force_y = accel_y + linear_drag * velocity_world.y;
    float force_z = gravity_m_s2 + accel_z + linear_drag * velocity_world.z;
    EulerAngles attitude = quat_to_euler(q);
    float desired_roll = atan2f(
        -force_y * cosf(attitude.pitch),
        fmaxf(force_z * lateral_accel_scale, 1e-3f));
    desired_roll = teacher_clamp(desired_roll, -max_roll_rad, max_roll_rad);
    action[1] = desired_roll / max_roll_rad;

    float vertical_fraction = fmaxf(
        cosf(desired_roll) * cosf(attitude.pitch), 0.50f);
    float cmd_thrust = force_z
        / (vertical_accel_per_thrust * vertical_fraction);
    if (cmd_thrust >= hover_thrust) {
        action[2] = (cmd_thrust - hover_thrust)
            / fmaxf(max_thrust - hover_thrust, 1e-4f);
    } else {
        action[2] = (cmd_thrust - hover_thrust)
            / fmaxf(hover_thrust - min_thrust, 1e-4f);
    }
    action[1] = teacher_clamp(action[1], -1.0f, 1.0f);
    action[2] = teacher_clamp(action[2], -1.0f, 1.0f);
}

static float round_bf16(float value) {
    union { float f; uint32_t u; } bits = {.f = value};
    uint32_t rounding = 0x7FFFu + ((bits.u >> 16) & 1u);
    bits.u = (bits.u + rounding) & 0xFFFF0000u;
    return bits.f;
}

static void quantize_values(float* values, int count) {
    for (int i = 0; i < count; i++) values[i] = round_bf16(values[i]);
}

static void quantize_policy_weights(PufferNet* policy) {
    quantize_values(policy->encoder->weights,
        policy->encoder->input_dim * policy->encoder->output_dim);
    quantize_values(policy->decoder->weights,
        policy->decoder->input_dim * policy->decoder->output_dim);
    for (int layer = 0; layer < policy->mingru->num_layers; layer++) {
        Linear* projection = policy->mingru->proj[layer];
        quantize_values(projection->weights,
            projection->input_dim * projection->output_dim);
    }
}

static void linear_bf16(Linear* layer, const float* input) {
    for (int output = 0; output < layer->output_dim; output++) {
        float sum = 0.0f;
        for (int input_index = 0; input_index < layer->input_dim; input_index++) {
            sum += input[input_index]
                * layer->weights[output * layer->input_dim + input_index];
        }
        layer->output[output] = round_bf16(sum);
    }
}

static float cuda_fast_tanh(float value) {
    float v1 = fminf(fmaxf(value, -9.0f), 9.0f);
    float v2 = v1 * v1;
    float p = v2 * -2.76076847742355e-16f + 2.00018790482477e-13f;
    p = v2 * p + -8.60467152213735e-11f;
    p = v2 * p + 5.12229709037114e-08f;
    p = v2 * p + 1.48572235717979e-05f;
    p = v2 * p + 6.37261928875436e-04f;
    p = v2 * p + 4.89352455891786e-03f;
    p = v1 * p;
    float q = v2 * 1.19825839466702e-06f + 1.18534705686654e-04f;
    q = v2 * q + 2.26843463243900e-03f;
    q = v2 * q + 4.89352518554385e-03f;
    return p / q;
}

static float cuda_fast_sigmoid(float value) {
    return teacher_clamp(
        0.5f * (cuda_fast_tanh(0.5f * value) + 1.0f), 0.0f, 1.0f);
}

static float cuda_lerp(float a, float b, float weight) {
    float difference = b - a;
    return fabsf(weight) < 0.5f
        ? a + weight * difference
        : b - difference * (1.0f - weight);
}

static void forward_puffernet_bf16(
        PufferNet* policy, const float* observations, float* actions) {
    float quantized_observation[DRONE_RACE_OBS_SIZE];
    for (int i = 0; i < DRONE_RACE_OBS_SIZE; i++) {
        quantized_observation[i] = round_bf16(observations[i]);
    }
    linear_bf16(policy->encoder, quantized_observation);
    float* input = policy->encoder->output;
    int hidden_size = policy->mingru->hidden_size;
    for (int layer = 0; layer < policy->mingru->num_layers; layer++) {
        Linear* projection = policy->mingru->proj[layer];
        linear_bf16(projection, input);
        float* state = policy->mingru->state + layer * hidden_size;
        for (int hidden_index = 0; hidden_index < hidden_size; hidden_index++) {
            float hidden = projection->output[hidden_index];
            float gate = projection->output[hidden_size + hidden_index];
            float highway = projection->output[2 * hidden_size + hidden_index];
            float gate_value = _sigmoid(gate);
            float candidate = hidden >= 0.0f
                ? hidden + 0.5f : cuda_fast_sigmoid(hidden);
            float recurrent = cuda_lerp(
                state[hidden_index], candidate, gate_value);
            float highway_value = _sigmoid(highway);
            policy->mingru->output[hidden_index] = round_bf16(
                highway_value * recurrent
                + (1.0f - highway_value) * input[hidden_index]);
            state[hidden_index] = round_bf16(recurrent);
        }
        input = policy->mingru->output;
    }
    linear_bf16(policy->decoder, input);
    for (int action = 0; action < DRONE_RACE_NUM_ATNS; action++) {
        actions[action] = teacher_clamp(policy->decoder->output[action], -1.0f, 1.0f);
    }
}

static void forward_decoder_head(
        PufferNet* head, float* hidden, float* actions, int use_bf16) {
    if (use_bf16) {
        linear_bf16(head->decoder, hidden);
    } else {
        linear(head->decoder, hidden);
    }
    for (int action = 0; action < DRONE_RACE_NUM_ATNS; action++) {
        actions[action] = teacher_clamp(
            head->decoder->output[action], -1.0f, 1.0f);
    }
}

static float default_course_gate_coordinate(int gate, int axis) {
    // Gates 0..3 retain the measured development proxy. Gates 4..5 are
    // deliberately synthetic anchors for domain-randomized generic control;
    // they are not asserted to be official positions and are never deployed.
    static const float course[6][3] = {
        {22.35f, 0.19f, 0.30f},
        {43.77f, 2.04f, -3.45f},
        {67.53733f, -1.24074f, -10.23080f},
        {95.63180f, 5.10711f, -18.24322f},
        {122.0f, -3.0f, -18.0f},
        {148.0f, 4.0f, -17.0f},
    };
    if (gate < 6) return course[gate][axis];
    if (axis == 0) return course[5][0] + 26.0f * (float)(gate - 5);
    if (axis == 1) return gate % 2 == 0 ? -4.0f : 4.0f;
    return -17.0f;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 7) {
        fprintf(stderr,
            "usage: %s OUTPUT.bin [EPISODES] [CHECKPOINT.bin] "
            "[STUDENT_PROBABILITY] [ACTION_NOISE] [BF16]\n", argv[0]);
        return 2;
    }
    int target_episodes = argc == 3 ? atoi(argv[2]) : 128;
    if (argc >= 3) target_episodes = atoi(argv[2]);
    if (target_episodes < 1) return 2;
    const char* checkpoint = argc >= 4 ? argv[3] : NULL;
    float student_probability = argc >= 5 ? strtof(argv[4], NULL) : 0.0f;
    float action_noise = argc >= 6 ? strtof(argv[5], NULL) : 0.08f;
    int use_bf16 = argc >= 7 ? atoi(argv[6]) : 0;
    int record_executed_action = env_int("TEACHER_RECORD_EXECUTED_ACTION", 0);
    int keep_successful_episodes_only = env_int(
        "TEACHER_KEEP_SUCCESSFUL_EPISODES_ONLY", 0);
    int action_noise_from_gate = env_int("TEACHER_ACTION_NOISE_FROM_GATE", 0);
    int checkpoint_layout_precision_bytes = env_int(
        "CHECKPOINT_LAYOUT_PRECISION_BYTES", 2);
    if (checkpoint_layout_precision_bytes != 2
            && checkpoint_layout_precision_bytes != 4) {
        fprintf(stderr,
            "CHECKPOINT_LAYOUT_PRECISION_BYTES must be 2 (BF16) or 4 (FP32)\n");
        return 2;
    }
    int checkpoint_alignment_elements = 16 / checkpoint_layout_precision_bytes;
    if (student_probability < 0.0f || student_probability > 1.0f || action_noise < 0.0f) {
        fprintf(stderr, "student probability must be in [0,1] and noise must be nonnegative\n");
        return 2;
    }
    FILE* out = fopen(argv[1], "wb");
    if (out == NULL) {
        perror("fopen");
        return 2;
    }

    DroneRace env = {0};
    env.num_agents = 1;
    int episode_seed_offset = env_int("SEED", 3385);
    int episode_seed_sequence = env_int("EPISODE_SEED_SEQUENCE", 0);
    env.rng = (unsigned int)episode_seed_offset;
    const char* policy_hz = getenv("POLICY_HZ");
    float command_hz = policy_hz == NULL ? 60.0f : strtof(policy_hz, NULL);
    if (command_hz <= 0.0f) command_hz = 60.0f;
    env.dt = 1.0f / command_hz;
    const char* num_gates_value = getenv("NUM_GATES");
    env.num_gates = num_gates_value == NULL ? 2 : atoi(num_gates_value);
    if (env.num_gates < 2) env.num_gates = 2;
    if (env.num_gates > DRONE_RACE_MAX_GATES) {
        env.num_gates = DRONE_RACE_MAX_GATES;
    }
    env.use_custom_start = 1;
    env.start_gate_index = env_int("START_GATE_INDEX", 0);
    env.start_elapsed_time = env_float("START_ELAPSED_TIME", 0.0f);
    env.start_elapsed_time_jitter = env_float(
        "START_ELAPSED_TIME_JITTER", 0.0f);
    env.custom_start_pos[0] = env_float("START_X", 0.0f);
    env.custom_start_pos[1] = env_float("START_Y", 0.0f);
    env.custom_start_pos[2] = env_float("START_Z", 0.0f);
    env.custom_start_pos_jitter[0] = env_float("START_X_JITTER", 0.0f);
    env.custom_start_pos_jitter[1] = env_float("START_Y_JITTER", 0.0f);
    env.custom_start_pos_jitter[2] = env_float("START_Z_JITTER", 0.0f);
    env.custom_start_vel[0] = env_float("START_VX", 0.0f);
    env.custom_start_vel[1] = env_float("START_VY", 0.0f);
    env.custom_start_vel[2] = env_float("START_VZ", 0.0f);
    env.custom_start_vel_jitter[0] = env_float("START_VX_JITTER", 0.0f);
    env.custom_start_vel_jitter[1] = env_float("START_VY_JITTER", 0.0f);
    env.custom_start_vel_jitter[2] = env_float("START_VZ_JITTER", 0.0f);
    env.custom_start_quat[0] = env_float("START_QW", 1.0f);
    env.custom_start_quat[1] = env_float("START_QX", 0.0f);
    env.custom_start_quat[2] = env_float("START_QY", 0.0f);
    env.custom_start_quat[3] = env_float("START_QZ", 0.0f);
    env.custom_start_attitude_jitter[0] = env_float(
        "START_ROLL_JITTER_RAD", 0.0f);
    env.custom_start_attitude_jitter[1] = env_float(
        "START_PITCH_JITTER_RAD", 0.0f);
    env.custom_start_attitude_jitter[2] = env_float(
        "START_YAW_JITTER_RAD", 0.0f);
    env.custom_start_omega[0] = env_float("START_WX", 0.0f);
    env.custom_start_omega[1] = env_float("START_WY", 0.0f);
    env.custom_start_omega[2] = env_float("START_WZ", 0.0f);
    env.custom_start_omega_jitter[0] = env_float("START_WX_JITTER", 0.0f);
    env.custom_start_omega_jitter[1] = env_float("START_WY_JITTER", 0.0f);
    env.custom_start_omega_jitter[2] = env_float("START_WZ_JITTER", 0.0f);
    env.use_custom_gate_layout = 1;
    for (int gate = 0; gate < env.num_gates; gate++) {
        for (int axis = 0; axis < 3; axis++) {
            char key[24];
            const char axis_name[3] = {'X', 'Y', 'Z'};
            snprintf(key, sizeof(key), "GATE%d_%c", gate, axis_name[axis]);
            env.custom_gate_pos[gate][axis] = env_float(
                key, default_course_gate_coordinate(gate, axis));
        }
    }
    env.course_geometry_scale = env_float("COURSE_GEOMETRY_SCALE", 1.0f);
    env.gate_position_domain_randomize = env_int(
        "GATE_POSITION_DOMAIN_RANDOMIZE", 0);
    env.gate_position_domain_randomize_probability = env_float(
        "GATE_POSITION_DOMAIN_RANDOMIZE_PROBABILITY", 1.0f);
    env.gate_position_randomized_radius = env_float(
        "GATE_POSITION_RANDOMIZED_RADIUS", 0.0f);
    env.gate_position_randomize_from_index = env_int(
        "GATE_POSITION_RANDOMIZE_FROM_INDEX", 2);
    env.gate_position_jitter_x = env_float("GATE_POSITION_JITTER_X", 0.0f);
    env.gate_position_jitter_y = env_float("GATE_POSITION_JITTER_Y", 0.0f);
    env.gate_position_jitter_z = env_float("GATE_POSITION_JITTER_Z", 0.0f);
    // The retained full-policy contract exposes official race phase and the
    // phase-gated camera adapters in observations 23..31. Default the teacher
    // collector to that deployment contract so an omitted environment variable
    // cannot silently produce a legacy all-zero phase dataset.
    env.observable_gate_index = env_int("OBSERVABLE_GATE_INDEX", 1);
    env.observable_gate_index_denominator = env_float(
        "OBSERVABLE_GATE_INDEX_DENOMINATOR", (float)env.num_gates);
    env.observable_gate_progress = env_int("OBSERVABLE_GATE_PROGRESS", 1);
    env.observable_gate_phase_onehot = env_int(
        "OBSERVABLE_GATE_PHASE_ONEHOT", 0);
    env.gate_radius = env_float("GATE_RADIUS", 0.75f);
    env.gate_radius_profile_mix = env_int("GATE_RADIUS_PROFILE_MIX", 0);
    env.gate_radius_profile_mix_probability = env_float(
        "GATE_RADIUS_PROFILE_MIX_PROBABILITY", 0.5f);
    env.gate_radius_profile_mix_target = env_float(
        "GATE_RADIUS_PROFILE_MIX_TARGET", env.gate_radius);
    for (int gate = 0; gate < env.num_gates; gate++) {
        char key[32];
        snprintf(key, sizeof(key), "GATE%d_RADIUS", gate);
        env.custom_gate_radius[gate] = env_float(key, 0.0f);
    }
    env.gate_spacing = env_float("GATE_SPACING", 21.82f);
    env.start_offset = 22.35f;
    env.crash_height = -25.0f;
    env.pos_bound = env_float(
        "POS_BOUND", env.num_gates > 4 ? 220.0f : 140.0f);
    env.plane_cross_tolerance = 0.001f;
    env.miss_tolerance = 0.5f;
    env.direction_min = 0.05f;
    env.strict_missed_gate = 1;
    env.time_limit_seconds = env_float(
        "TIME_LIMIT_SECONDS", env.num_gates > 4 ? 45.0f : 30.0f);
    env.max_steps = (int)lroundf(env.time_limit_seconds / env.dt);
    env.safety_altitude = -23.0f;
    env.w_progress = 35.0f;
    env.w_gate = 15.0f;
    env.w_finish = 180.0f;
    env.w_time = 0.15f;
    env.w_ctrl = 0.01f;
    env.invalid_penalty = 600.0f;
    env.interface_mode = DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT;
    env.sitl_rate_gain_roll = -2.416f;
    env.sitl_rate_gain_pitch = -2.418f;
    env.sitl_rate_gain_yaw = -2.189f;
    env.sitl_hover_thrust = 0.27f;
    env.sitl_min_thrust = 0.18f;
    env.sitl_max_thrust = 0.42f;
    env.sitl_vertical_accel_per_thrust = env_float(
        "SITL_VERTICAL_ACCEL_PER_THRUST", 32.81f);
    const char* horizontal_scale = getenv("SITL_HORIZONTAL_ACCEL_SCALE");
    env.sitl_horizontal_accel_scale = horizontal_scale == NULL
        ? 2.60794f : strtof(horizontal_scale, NULL);
    const char* braking_scale = getenv("SITL_BRAKING_ACCEL_SCALE");
    env.sitl_braking_accel_scale = braking_scale == NULL
        ? 2.60794f : strtof(braking_scale, NULL);
    const char* lateral_scale = getenv("SITL_LATERAL_ACCEL_SCALE");
    env.sitl_lateral_accel_scale = lateral_scale == NULL
        ? 0.107491f : strtof(lateral_scale, NULL);
    const char* transition_speed = getenv("SITL_GATE_TRANSITION_MIN_FORWARD_SPEED");
    env.sitl_gate_transition_min_forward_speed = transition_speed == NULL
        ? 10.0f : strtof(transition_speed, NULL);
    env.sitl_gravity_m_s2 = env_float("SITL_GRAVITY_M_S2", 8.69f);
    env.sitl_linear_drag = 0.14f;
    env.sitl_rate_lag_tau_s = 0.52f;
    env.sitl_max_roll_rad = 0.50f;
    env.sitl_max_pitch_rad = 0.50f;
    env.sitl_max_yaw_rad = (float)M_PI;
    env.sitl_lock_hover_thrust = 0;
    env.sitl_lock_roll = 0;
    env.sitl_lock_yaw = 0;
    env.sitl_attitude_kp = 1.5f;
    env.sitl_body_rate_min_command_rad_s = 0.05f;
    env.sitl_attitude_error_deadband_rad = 0.004f;
    env.sitl_max_body_rate_command_rad_s = 0.4f;
    env.sitl_gate_obs_sample_interval_steps = env_int(
        "SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS", 1);
    env.sitl_gate_motion_predict_dropout = env_int(
        "SITL_GATE_MOTION_PREDICT_DROPOUT", 0);
    env.sitl_gate_motion_control_accel_gain = env_float(
        "SITL_GATE_MOTION_CONTROL_ACCEL_GAIN", 0.0f);
    env.observations = calloc(DRONE_RACE_OBS_SIZE, sizeof(float));
    env.actions = calloc(DRONE_RACE_NUM_ATNS, sizeof(float));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(float));
    init(&env);
    c_reset(&env);

    Weights* policy_weights = NULL;
    PufferNet* policy = NULL;
    Weights* previous_weights = NULL;
    PufferNet* previous_policy = NULL;
    Weights* residual_weights = NULL;
    PufferNet* residual_policy = NULL;
    Weights* final_weights = NULL;
    PufferNet* final_policy = NULL;
    Weights* residual_low_weights = NULL;
    PufferNet* residual_low_policy = NULL;
    Weights* final_low_weights = NULL;
    PufferNet* final_low_policy = NULL;
    Weights* phase_reset_gate4_weights = NULL;
    PufferNet* phase_reset_gate4_policy = NULL;
    Weights* phase_reset_gate5_weights = NULL;
    PufferNet* phase_reset_gate5_policy = NULL;
    if (checkpoint != NULL && checkpoint[0] != '\0' && strcmp(checkpoint, "-") != 0) {
        policy_weights = load_weights_with_alignment(
            checkpoint, checkpoint_alignment_elements);
        if (policy_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        policy = make_puffernet(
            policy_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) quantize_policy_weights(policy);
    } else if (student_probability > 0.0f) {
        fprintf(stderr, "student probability requires a checkpoint\n");
        return 2;
    }
    int phase_reset_tail = env_int("PHASE_RESET_TAIL", 0);
    if (phase_reset_tail) {
        const char* gate4_checkpoint = getenv("PHASE_RESET_GATE4_CHECKPOINT");
        const char* gate5_checkpoint = getenv("PHASE_RESET_GATE5_CHECKPOINT");
        if (policy == NULL
                || gate4_checkpoint == NULL || gate4_checkpoint[0] == '\0'
                || gate5_checkpoint == NULL || gate5_checkpoint[0] == '\0'
                || student_probability != 1.0f) {
            fprintf(stderr,
                "PHASE_RESET_TAIL requires a base Gate-3 checkpoint, "
                "Gate-4/Gate-5 checkpoints, and student probability 1\n");
            return 2;
        }
        phase_reset_gate4_weights = load_weights_with_alignment(
            gate4_checkpoint, checkpoint_alignment_elements);
        phase_reset_gate5_weights = load_weights_with_alignment(
            gate5_checkpoint, checkpoint_alignment_elements);
        if (phase_reset_gate4_weights == NULL
                || phase_reset_gate5_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        phase_reset_gate4_policy = make_puffernet(
            phase_reset_gate4_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        phase_reset_gate5_policy = make_puffernet(
            phase_reset_gate5_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) {
            quantize_policy_weights(phase_reset_gate4_policy);
            quantize_policy_weights(phase_reset_gate5_policy);
        }
    }
    int policy_teacher = getenv("POLICY_TEACHER") != NULL;
    if (policy_teacher && policy == NULL) {
        fprintf(stderr, "POLICY_TEACHER requires a checkpoint\n");
        return 2;
    }
    const char* previous_checkpoint = getenv("PHASE_PREVIOUS_CHECKPOINT");
    if (previous_checkpoint != NULL && previous_checkpoint[0] != '\0') {
        if (policy == NULL || !policy_teacher) {
            fprintf(stderr, "PHASE_PREVIOUS_CHECKPOINT requires POLICY_TEACHER and a base checkpoint\n");
            return 2;
        }
        previous_weights = load_weights_with_alignment(
            previous_checkpoint, checkpoint_alignment_elements);
        if (previous_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        previous_policy = make_puffernet(
            previous_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) quantize_policy_weights(previous_policy);
    }
    const char* residual_checkpoint = getenv("PHASE_RESIDUAL_CHECKPOINT");
    if (residual_checkpoint != NULL && residual_checkpoint[0] != '\0') {
        if (policy == NULL || !policy_teacher) {
            fprintf(stderr, "PHASE_RESIDUAL_CHECKPOINT requires POLICY_TEACHER and a base checkpoint\n");
            return 2;
        }
        residual_weights = load_weights_with_alignment(
            residual_checkpoint, checkpoint_alignment_elements);
        if (residual_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        residual_policy = make_puffernet(
            residual_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) quantize_policy_weights(residual_policy);
    }
    const char* final_checkpoint = getenv("PHASE_FINAL_CHECKPOINT");
    if (final_checkpoint != NULL && final_checkpoint[0] != '\0') {
        if (residual_policy == NULL) {
            fprintf(stderr, "PHASE_FINAL_CHECKPOINT requires PHASE_RESIDUAL_CHECKPOINT\n");
            return 2;
        }
        final_weights = load_weights_with_alignment(
            final_checkpoint, checkpoint_alignment_elements);
        if (final_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        final_policy = make_puffernet(
            final_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) quantize_policy_weights(final_policy);
    }
    const char* residual_low_checkpoint = getenv(
        "PHASE_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT");
    if (residual_low_checkpoint != NULL && residual_low_checkpoint[0] != '\0') {
        if (residual_policy == NULL) {
            fprintf(stderr, "low-confidence residual requires PHASE_RESIDUAL_CHECKPOINT\n");
            return 2;
        }
        residual_low_weights = load_weights_with_alignment(
            residual_low_checkpoint, checkpoint_alignment_elements);
        if (residual_low_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        residual_low_policy = make_puffernet(
            residual_low_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) quantize_policy_weights(residual_low_policy);
    }
    const char* final_low_checkpoint = getenv(
        "PHASE_FINAL_LOW_CONFIDENCE_CHECKPOINT");
    if (final_low_checkpoint != NULL && final_low_checkpoint[0] != '\0') {
        if (final_policy == NULL) {
            fprintf(stderr, "low-confidence final requires PHASE_FINAL_CHECKPOINT\n");
            return 2;
        }
        final_low_weights = load_weights_with_alignment(
            final_low_checkpoint, checkpoint_alignment_elements);
        if (final_low_weights == NULL) return 2;
        int logit_sizes[DRONE_RACE_NUM_ATNS] = {1, 1, 1, 1};
        final_low_policy = make_puffernet(
            final_low_weights, 1, DRONE_RACE_OBS_SIZE, 128, 3,
            logit_sizes, DRONE_RACE_NUM_ATNS);
        if (use_bf16) quantize_policy_weights(final_low_policy);
    }
    int restore_phase_yaw = previous_policy != NULL || residual_policy != NULL
        || env_int("POLICY_RESTORE_PHASE_YAW", 0);
    int policy_legacy_confidence = env_int("POLICY_LEGACY_CONFIDENCE", 0);
    float phase_residual_start = env_float(
        "PHASE_RESIDUAL_START", 2.0f / 3.0f);
    float phase_residual_full = env_float(
        "PHASE_RESIDUAL_FULL", 1.0f);
    float phase_final_start = env_float("PHASE_FINAL_START", 1.0f);
    float phase_residual_min_confidence = env_float(
        "PHASE_RESIDUAL_MIN_CONFIDENCE", 0.0f);
    float phase_final_min_confidence = env_float(
        "PHASE_FINAL_MIN_CONFIDENCE", 0.0f);
    if (final_policy == NULL && phase_residual_full <= phase_residual_start) {
        fprintf(stderr, "PHASE_RESIDUAL_FULL must exceed PHASE_RESIDUAL_START\n");
        return 2;
    }
    const char* roll_scale_value = getenv("TEACHER_ROLL_SCALE");
    float teacher_roll_scale = roll_scale_value == NULL
        ? 1.0f : strtof(roll_scale_value, NULL);
    const char* pitch_scale_value = getenv("TEACHER_PITCH_SCALE");
    float teacher_pitch_scale = pitch_scale_value == NULL
        ? 1.0f : strtof(pitch_scale_value, NULL);
    const char* pitch_offset_value = getenv("TEACHER_PITCH_OFFSET");
    float teacher_pitch_offset = pitch_offset_value == NULL
        ? 0.0f : strtof(pitch_offset_value, NULL);
    const char* roll_scale_gate_value = getenv("TEACHER_ROLL_SCALE_FROM_GATE");
    int teacher_roll_scale_from_gate = roll_scale_gate_value == NULL
        ? env.num_gates : atoi(roll_scale_gate_value);
    const char* thrust_scale_value = getenv("TEACHER_THRUST_SCALE");
    float teacher_thrust_scale = thrust_scale_value == NULL
        ? 1.0f : strtof(thrust_scale_value, NULL);
    const char* thrust_offset_value = getenv("TEACHER_THRUST_OFFSET");
    float teacher_thrust_offset = thrust_offset_value == NULL
        ? 0.0f : strtof(thrust_offset_value, NULL);
    float teacher_gate_dropout_range = env_float(
        "TEACHER_GATE_DROPOUT_RANGE", 0.0f);
    int teacher_gate_dropout_from_gate = env_int(
        "TEACHER_GATE_DROPOUT_FROM_GATE", env.num_gates);
    env.sitl_gate_obs_dropout_range_m = teacher_gate_dropout_range;
    env.sitl_gate_obs_dropout_from_index = teacher_gate_dropout_from_gate;
    float teacher_close_action_range = env_float(
        "TEACHER_CLOSE_ACTION_RANGE", 0.0f);
    int teacher_close_action_from_gate = env_int(
        "TEACHER_CLOSE_ACTION_FROM_GATE", env.num_gates);
    float teacher_close_pitch_offset = env_float(
        "TEACHER_CLOSE_PITCH_OFFSET", 0.0f);
    float teacher_close_roll_offset = env_float(
        "TEACHER_CLOSE_ROLL_OFFSET", 0.0f);
    float teacher_close_thrust_offset = env_float(
        "TEACHER_CLOSE_THRUST_OFFSET", 0.0f);
    int teacher_feedback_from_gate = env_int(
        "TEACHER_FEEDBACK_FROM_GATE", env.num_gates);
    int teacher_feedback_until_gate = env_int(
        "TEACHER_FEEDBACK_UNTIL_GATE", env.num_gates);
    int teacher_feedback_roll_from_gate = env_int(
        "TEACHER_FEEDBACK_ROLL_FROM_GATE", teacher_feedback_from_gate);
    int teacher_feedback_thrust_from_gate = env_int(
        "TEACHER_FEEDBACK_THRUST_FROM_GATE", teacher_feedback_from_gate);
    float teacher_feedback_roll_kp = env_float(
        "TEACHER_FEEDBACK_ROLL_KP", 0.0f);
    float teacher_feedback_roll_kd = env_float(
        "TEACHER_FEEDBACK_ROLL_KD", 0.0f);
    int teacher_feedback_roll_residual = env_int(
        "TEACHER_FEEDBACK_ROLL_RESIDUAL", 0);
    float teacher_feedback_thrust_kp = env_float(
        "TEACHER_FEEDBACK_THRUST_KP", 0.0f);
    float teacher_feedback_thrust_kd = env_float(
        "TEACHER_FEEDBACK_THRUST_KD", 0.0f);
    float teacher_feedback_thrust_bias = env_float(
        "TEACHER_FEEDBACK_THRUST_BIAS", 0.0f);
    int teacher_feedback_thrust_residual = env_int(
        "TEACHER_FEEDBACK_THRUST_RESIDUAL", 0);
    int teacher_analytic_label_from_gate = env_int(
        "TEACHER_ANALYTIC_LABEL_FROM_GATE", env.num_gates);
    int teacher_ballistic_label_from_gate = env_int(
        "TEACHER_BALLISTIC_LABEL_FROM_GATE", env.num_gates);
    float teacher_ballistic_time_floor = env_float(
        "TEACHER_BALLISTIC_TIME_FLOOR", 0.50f);
    float teacher_ballistic_roll_scale = env_float(
        "TEACHER_BALLISTIC_ROLL_SCALE", 1.0f);
    float teacher_ballistic_thrust_scale = env_float(
        "TEACHER_BALLISTIC_THRUST_SCALE", 1.0f);
    float teacher_ballistic_exit_velocity_scale = env_float(
        "TEACHER_BALLISTIC_EXIT_VELOCITY_SCALE", 0.0f);
    float teacher_ballistic_lateral_aperture_offset_m = env_float(
        "TEACHER_BALLISTIC_LATERAL_APERTURE_OFFSET_M", 0.0f);
    int teacher_ballistic_lateral_aperture_offset_from_gate = env_int(
        "TEACHER_BALLISTIC_LATERAL_APERTURE_OFFSET_FROM_GATE", env.num_gates);
    int teacher_ballistic_lateral_aperture_offset_until_gate = env_int(
        "TEACHER_BALLISTIC_LATERAL_APERTURE_OFFSET_UNTIL_GATE", env.num_gates);
    int teacher_spline_label_from_gate = env_int(
        "TEACHER_SPLINE_LABEL_FROM_GATE", env.num_gates);
    int teacher_spline_label_until_gate = env_int(
        "TEACHER_SPLINE_LABEL_UNTIL_GATE", env.num_gates);
    float teacher_spline_lateral_position_kp = env_float(
        "TEACHER_SPLINE_LATERAL_POSITION_KP", 1.0f);
    float teacher_spline_close_lateral_position_kp = env_float(
        "TEACHER_SPLINE_CLOSE_LATERAL_POSITION_KP", 0.0f);
    float teacher_spline_lateral_velocity_kd = env_float(
        "TEACHER_SPLINE_LATERAL_VELOCITY_KD", 1.5f);
    float teacher_spline_vertical_position_kp = env_float(
        "TEACHER_SPLINE_VERTICAL_POSITION_KP", 1.0f);
    float teacher_spline_vertical_velocity_kd = env_float(
        "TEACHER_SPLINE_VERTICAL_VELOCITY_KD", 1.5f);
    float teacher_spline_close_position_range_m = env_float(
        "TEACHER_SPLINE_CLOSE_POSITION_RANGE_M", 10.0f);
    float teacher_spline_recovery_trigger_m = env_float(
        "TEACHER_SPLINE_RECOVERY_TRIGGER_M", 0.0f);
    float teacher_spline_recovery_time_floor_s = env_float(
        "TEACHER_SPLINE_RECOVERY_TIME_FLOOR_S", 0.50f);
    float teacher_spline_curvature_scale = env_float(
        "TEACHER_SPLINE_CURVATURE_SCALE", 1.0f);
    float teacher_spline_derivative_lookahead_m = env_float(
        "TEACHER_SPLINE_DERIVATIVE_LOOKAHEAD_M", 0.0f);
    float teacher_spline_vertical_derivative_lookahead_m = env_float(
        "TEACHER_SPLINE_VERTICAL_DERIVATIVE_LOOKAHEAD_M", 0.0f);
    int teacher_spline_derivative_lookahead_from_gate = env_int(
        "TEACHER_SPLINE_DERIVATIVE_LOOKAHEAD_FROM_GATE", 0);
    int teacher_spline_exit_tangents = env_int(
        "TEACHER_SPLINE_EXIT_TANGENTS", 1);
    int teacher_spline_full_course_origin = env_int(
        "TEACHER_SPLINE_FULL_COURSE_ORIGIN", 0);
    int teacher_spline_exit_tangent_from_gate = env_int(
        "TEACHER_SPLINE_EXIT_TANGENT_FROM_GATE", 2);
    float teacher_spline_exit_slope_scale = env_float(
        "TEACHER_SPLINE_EXIT_SLOPE_SCALE", 1.0f);
    int teacher_observable_course = env_int(
        "TEACHER_OBSERVABLE_COURSE", 0);
    int teacher_observable_course_full_origin = env_int(
        "TEACHER_OBSERVABLE_COURSE_FULL_ORIGIN", 0);
    int phase_final_observable_course = env_int(
        "PHASE_FINAL_OBSERVABLE_COURSE", 0);
    int teacher_observable_course_from_gate = env_int(
        "TEACHER_OBSERVABLE_COURSE_FROM_GATE", 1);
    float teacher_observable_velocity_alpha = env_float(
        "TEACHER_OBSERVABLE_VELOCITY_ALPHA", 0.35f);
    float teacher_observable_motion_rate_blend = env_float(
        "TEACHER_OBSERVABLE_MOTION_RATE_BLEND", 0.0f);
    int teacher_observable_pitch_speed_control = env_int(
        "TEACHER_OBSERVABLE_PITCH_SPEED_CONTROL", 0);
    float teacher_observable_pitch_speed_target_m_s = env_float(
        "TEACHER_OBSERVABLE_PITCH_SPEED_TARGET_M_S", 4.2f);
    ObservableCourseTeacher observable_course_teacher = {0};
    ObservableCourseTeacher phase_final_observable_course_controller = {0};
    if (teacher_observable_course
            && !init_observable_course_teacher(
                &observable_course_teacher,
                &env,
                teacher_spline_exit_tangent_from_gate,
                teacher_spline_exit_slope_scale,
                teacher_observable_course_full_origin)) {
        fprintf(stderr, "unable to construct observable course teacher\n");
        return 2;
    }
    if (phase_final_observable_course
            && !init_observable_course_teacher(
                &phase_final_observable_course_controller,
                &env,
                teacher_spline_exit_tangent_from_gate,
                teacher_spline_exit_slope_scale,
                1)) {
        fprintf(stderr, "unable to construct final observable course controller\n");
        return 2;
    }
    float teacher_analytic_label_roll_scale = env_float(
        "TEACHER_ANALYTIC_LABEL_ROLL_SCALE", 1.0f);
    float teacher_analytic_label_pitch_scale = env_float(
        "TEACHER_ANALYTIC_LABEL_PITCH_SCALE", 1.0f);
    int teacher_analytic_pitch_scale_from_gate = env_int(
        "TEACHER_ANALYTIC_PITCH_SCALE_FROM_GATE", env.num_gates);
    float teacher_analytic_pitch_scale_late = env_float(
        "TEACHER_ANALYTIC_PITCH_SCALE_LATE",
        teacher_analytic_label_pitch_scale);
    float teacher_analytic_label_roll_rate_scale = env_float(
        "TEACHER_ANALYTIC_LABEL_ROLL_RATE_SCALE", 1.0f);
    float teacher_analytic_label_thrust_offset = env_float(
        "TEACHER_ANALYTIC_LABEL_THRUST_OFFSET", 0.0f);
    int teacher_execute_label_from_gate = env_int(
        "TEACHER_EXECUTE_LABEL_FROM_GATE", env.num_gates);
    int teacher_analytic_pitch_from_gate = env_int(
        "TEACHER_ANALYTIC_PITCH_FROM_GATE", env.num_gates);
    float teacher_initial_forward_speed = env_float(
        "TEACHER_INITIAL_FORWARD_SPEED", 2.2f);
    float teacher_forward_speed = env_float(
        "TEACHER_FORWARD_SPEED", 1.5f);
    if (teacher_initial_forward_speed <= 0.0f || teacher_forward_speed <= 0.0f) {
        fprintf(stderr, "teacher forward speeds must be positive\n");
        return 2;
    }
    int teacher_post_thrust_offset_from_gate = env_int(
        "TEACHER_POST_THRUST_OFFSET_FROM_GATE", env.num_gates);
    int teacher_post_thrust_offset_until_gate = env_int(
        "TEACHER_POST_THRUST_OFFSET_UNTIL_GATE", env.num_gates);
    int teacher_post_action_offset_from_gate = env_int(
        "TEACHER_POST_ACTION_OFFSET_FROM_GATE",
        teacher_post_thrust_offset_from_gate);
    int teacher_post_action_offset_until_gate = env_int(
        "TEACHER_POST_ACTION_OFFSET_UNTIL_GATE",
        teacher_post_thrust_offset_until_gate);
    float teacher_post_roll_offset = env_float(
        "TEACHER_POST_ROLL_OFFSET", 0.0f);
    float teacher_post_thrust_offset = env_float(
        "TEACHER_POST_THRUST_OFFSET", 0.0f);
    float teacher_gate_action_bias[DRONE_RACE_MAX_GATES][DRONE_RACE_NUM_ATNS] = {0};
    int teacher_gate_action_bias_visible_only[DRONE_RACE_MAX_GATES] = {0};
    float teacher_gate_action_bias_min_alignment[DRONE_RACE_MAX_GATES] = {0};
    float teacher_gate_action_bias_min_forward_norm[DRONE_RACE_MAX_GATES] = {0};
    const char* action_names[DRONE_RACE_NUM_ATNS] = {
        "PITCH", "ROLL", "THRUST", "YAW",
    };
    for (int gate = 0; gate < env.num_gates; gate++) {
        char visible_key[48];
        snprintf(
            visible_key, sizeof(visible_key),
            "TEACHER_GATE%d_BIAS_VISIBLE_ONLY", gate);
        teacher_gate_action_bias_visible_only[gate] = env_int(visible_key, 0);
        char alignment_key[48];
        snprintf(
            alignment_key, sizeof(alignment_key),
            "TEACHER_GATE%d_BIAS_MIN_ALIGNMENT", gate);
        teacher_gate_action_bias_min_alignment[gate] = env_float(
            alignment_key, 0.0f);
        char forward_key[48];
        snprintf(
            forward_key, sizeof(forward_key),
            "TEACHER_GATE%d_BIAS_MIN_FORWARD_M", gate);
        teacher_gate_action_bias_min_forward_norm[gate] = tanhf(
            env_float(forward_key, 0.0f) * 0.1f);
        for (int action = 0; action < DRONE_RACE_NUM_ATNS; action++) {
            char key[48];
            snprintf(
                key, sizeof(key), "TEACHER_GATE%d_%s_BIAS",
                gate, action_names[action]);
            teacher_gate_action_bias[gate][action] = env_float(key, 0.0f);
        }
    }

    int episodes = 0;
    int reset_state = 1;
    int last_gate = 0;
    int gate_steps = 0;
    long records = 0;
    long attempted_records = 0;
    long kept_student_steps = 0;
    long episode_student_steps = 0;
    int kept_episodes = 0;
    long spline_recovery_steps = 0;
    const size_t record_width = DRONE_RACE_OBS_SIZE + DRONE_RACE_NUM_ATNS + 1;
    float* episode_records = NULL;
    long episode_record_count = 0;
    if (keep_successful_episodes_only) {
        episode_records = calloc(
            (size_t)env.max_steps * record_width, sizeof(float));
        if (episode_records == NULL) {
            fprintf(stderr, "unable to allocate success-filter episode buffer\n");
            return 2;
        }
    }
    float last_policy_yaw_action = 0.0f;
    TeacherSpline lateral_spline = {0};
    TeacherSpline vertical_spline = {0};
    int teacher_spline_valid = 0;
    int phase_reset_previous_gate = -1;
    int phase_action_bias_gate = env_int("PHASE_ACTION_BIAS_GATE", -1);
    float phase_action_pitch_bias = env_float("PHASE_ACTION_PITCH_BIAS", 0.0f);
    float phase_action_roll_bias = env_float("PHASE_ACTION_ROLL_BIAS", 0.0f);
    float phase_action_thrust_bias = env_float("PHASE_ACTION_THRUST_BIAS", 0.0f);
    float phase_final_action_pitch_bias = env_float(
        "PHASE_FINAL_ACTION_PITCH_BIAS", 0.0f);
    float phase_final_action_thrust_bias = env_float(
        "PHASE_FINAL_ACTION_THRUST_BIAS", 0.0f);
    float phase_final_observable_roll_kp = env_float(
        "PHASE_FINAL_OBSERVABLE_ROLL_KP", 0.0f);
    float phase_final_observable_roll_kd = env_float(
        "PHASE_FINAL_OBSERVABLE_ROLL_KD", 0.0f);
    float phase_final_observable_roll_ki = env_float(
        "PHASE_FINAL_OBSERVABLE_ROLL_KI", 0.0f);
    float phase_final_observable_roll_integral_limit = env_float(
        "PHASE_FINAL_OBSERVABLE_ROLL_INTEGRAL_LIMIT", 5.0f);
    float phase_final_observable_thrust_kp = env_float(
        "PHASE_FINAL_OBSERVABLE_THRUST_KP", 0.0f);
    float phase_final_observable_thrust_kd = env_float(
        "PHASE_FINAL_OBSERVABLE_THRUST_KD", 0.0f);
    int phase_final_observable_roll_override = env_int(
        "PHASE_FINAL_OBSERVABLE_ROLL_OVERRIDE", 0);
    int phase_final_observable_intercept_roll = env_int(
        "PHASE_FINAL_OBSERVABLE_INTERCEPT_ROLL", 0);
    int phase_final_observable_intercept_thrust = env_int(
        "PHASE_FINAL_OBSERVABLE_INTERCEPT_THRUST", 0);
    float phase_final_intercept_vertical_accel_scale = env_float(
        "PHASE_FINAL_INTERCEPT_VERTICAL_ACCEL_SCALE", 1.0f);
    float phase_final_intercept_time_floor_s = env_float(
        "PHASE_FINAL_INTERCEPT_TIME_FLOOR_S", 0.5f);
    float phase_final_intercept_forward_speed_floor_m_s = env_float(
        "PHASE_FINAL_INTERCEPT_FORWARD_SPEED_FLOOR_M_S", 4.1875f);
    float phase_final_intercept_lateral_accel_scale = env_float(
        "PHASE_FINAL_INTERCEPT_LATERAL_ACCEL_SCALE", 1.0f);
    float phase_final_lateral_brake_pitch_bias = env_float(
        "PHASE_FINAL_LATERAL_BRAKE_PITCH_BIAS", 0.0f);
    float phase_final_lateral_brake_release_right_m = env_float(
        "PHASE_FINAL_LATERAL_BRAKE_RELEASE_RIGHT_M", 0.0f);
    float phase_final_projected_miss_brake_pitch_bias = env_float(
        "PHASE_FINAL_PROJECTED_MISS_BRAKE_PITCH_BIAS", 0.0f);
    float phase_final_projected_miss_brake_threshold_m = env_float(
        "PHASE_FINAL_PROJECTED_MISS_BRAKE_THRESHOLD_M", 0.75f);
    int phase_final_bang_bang_roll = env_int(
        "PHASE_FINAL_BANG_BANG_ROLL", 0);
    float phase_final_roll_switch_forward_m = env_float(
        "PHASE_FINAL_ROLL_SWITCH_FORWARD_M", 0.0f);
    float phase_final_roll_accel_action = env_float(
        "PHASE_FINAL_ROLL_ACCEL_ACTION", -1.0f);
    float phase_final_roll_brake_action = env_float(
        "PHASE_FINAL_ROLL_BRAKE_ACTION", 0.0f);
    int phase_final_bang_bang_roll_pd_residual = env_int(
        "PHASE_FINAL_BANG_BANG_ROLL_PD_RESIDUAL", 0);
    int phase_final_bang_bang_roll_directional = env_int(
        "PHASE_FINAL_BANG_BANG_ROLL_DIRECTIONAL", 0);
    int phase_final_deterministic_attitude = env_int(
        "PHASE_FINAL_DETERMINISTIC_ATTITUDE", 0);
    float phase_final_deterministic_pitch_action = env_float(
        "PHASE_FINAL_DETERMINISTIC_PITCH_ACTION", 0.15f);
    float phase_final_deterministic_yaw_action = env_float(
        "PHASE_FINAL_DETERMINISTIC_YAW_ACTION", 0.0f);
    int phase_final_observable_pitch_speed_control = env_int(
        "PHASE_FINAL_OBSERVABLE_PITCH_SPEED_CONTROL", 0);
    float phase_final_observable_pitch_speed_target_m_s = env_float(
        "PHASE_FINAL_OBSERVABLE_PITCH_SPEED_TARGET_M_S", 5.5f);
    int phase_counter_roll_gate = env_int("PHASE_COUNTER_ROLL_GATE", -1);
    float phase_counter_roll_forward_m = env_float(
        "PHASE_COUNTER_ROLL_FORWARD_M", 0.0f);
    float phase_counter_roll_action = env_float(
        "PHASE_COUNTER_ROLL_ACTION", 0.0f);
    int phase_counter_roll_latched = 0;
    float phase_final_observable_roll_integral = 0.0f;
    int phase_final_observable_course_active = 0;
    float phase_final_bang_bang_roll_direction = 0.0f;
    while (episodes < target_episodes) {
        DroneRaceAgent* agent = &env.agents[0];
        if (reset_state) {
            last_gate = 0;
            gate_steps = 0;
            if (teacher_spline_label_from_gate < env.num_gates) {
                float spline_x[DRONE_RACE_MAX_GATES + 1] = {0};
                float spline_y[DRONE_RACE_MAX_GATES + 1] = {0};
                float spline_z[DRONE_RACE_MAX_GATES + 1] = {0};
                if (teacher_spline_full_course_origin) {
                    spline_x[0] = env.gates[0].pos.x - env.start_offset;
                    spline_y[0] = 0.0f;
                    spline_z[0] = 0.0f;
                } else {
                    spline_x[0] = agent->drone.state.pos.x;
                    spline_y[0] = agent->drone.state.pos.y;
                    spline_z[0] = agent->drone.state.pos.z;
                }
                for (int gate_index = 0; gate_index < env.num_gates; gate_index++) {
                    Target* spline_gate = agent_gate(&env, agent, gate_index);
                    spline_x[gate_index + 1] = spline_gate->pos.x;
                    spline_y[gate_index + 1] = spline_gate->pos.y;
                    spline_z[gate_index + 1] = spline_gate->pos.z;
                }
                if (teacher_spline_exit_tangents) {
                    teacher_spline_valid = build_teacher_exit_spline(
                            &lateral_spline, spline_x, spline_y,
                            env.num_gates + 1,
                            teacher_spline_exit_tangent_from_gate + 1,
                            teacher_spline_exit_slope_scale)
                        && build_teacher_exit_spline(
                            &vertical_spline, spline_x, spline_z,
                            env.num_gates + 1,
                            teacher_spline_exit_tangent_from_gate + 1,
                            teacher_spline_exit_slope_scale);
                } else {
                    teacher_spline_valid = build_teacher_spline(
                            &lateral_spline, spline_x, spline_y, env.num_gates + 1)
                        && build_teacher_spline(
                            &vertical_spline, spline_x, spline_z, env.num_gates + 1);
                }
                if (!teacher_spline_valid) {
                    fprintf(stderr, "unable to construct monotonic course spline\n");
                    return 2;
                }
            }
        } else if (agent->current_gate != last_gate) {
            if (getenv("TEACHER_TRACE") != NULL && episodes == 0) {
                Target* transition_gate = agent_gate(
                    &env, agent, agent->current_gate);
                Vec3 transition_world = sub3(
                    transition_gate->pos, agent->drone.state.pos);
                Vec3 transition_body = quat_rotate(
                    quat_inverse(agent->drone.state.quat), transition_world);
                fprintf(stderr,
                    "transition step=%d gate=%d pos=(%.6f,%.6f,%.6f) "
                    "vel=(%.6f,%.6f,%.6f) "
                    "quat=(%.6f,%.6f,%.6f,%.6f) "
                    "rel_body_frd=(%.6f,%.6f,%.6f)\n",
                    agent->step_count, agent->current_gate,
                    agent->drone.state.pos.x, agent->drone.state.pos.y,
                    agent->drone.state.pos.z,
                    agent->drone.state.vel.x, agent->drone.state.vel.y,
                    agent->drone.state.vel.z,
                    agent->drone.state.quat.w, agent->drone.state.quat.x,
                    agent->drone.state.quat.y, agent->drone.state.quat.z,
                    transition_body.x, transition_body.y, -transition_body.z);
            }
            last_gate = agent->current_gate;
            gate_steps = 0;
        } else {
            gate_steps++;
        }
        Target* gate = agent_gate(&env, agent, agent->current_gate);
        Vec3 rel_world = sub3(gate->pos, agent->drone.state.pos);
        Quat body_from_world = quat_inverse(agent->drone.state.quat);
        Vec3 rel_body = quat_rotate(body_from_world, rel_world);
        Vec3 velocity_body = quat_rotate(body_from_world, agent->drone.state.vel);
        float desired_forward_speed = agent->current_gate == 0
            ? teacher_initial_forward_speed
            : teacher_forward_speed;
        float teacher[4] = {
            0.24f * teacher_clamp(
                -(desired_forward_speed - agent->drone.state.vel.x) * 0.35f,
                -0.60f, 0.60f),
            -0.40f * teacher_clamp(
                rel_body.y * 0.08f - agent->drone.state.vel.y * 0.10f,
                -0.50f, 0.50f),
            teacher_clamp(rel_body.z * 0.08f - agent->drone.state.vel.z * 0.25f, -0.80f, 0.60f),
            0.0f,
        };
        float analytic_teacher[DRONE_RACE_NUM_ATNS];
        memcpy(analytic_teacher, teacher, sizeof(analytic_teacher));
        float unmodified_policy_action[DRONE_RACE_NUM_ATNS] = {0};
        if (policy_teacher) {
            float policy_observation[DRONE_RACE_OBS_SIZE];
            memcpy(policy_observation, env.observations, sizeof(policy_observation));
            if (policy_legacy_confidence) {
                // Domain-adaptation teacher only: before the official/native
                // parity correction, feature 17 was a centering score instead
                // of GatePoseEstimate's apparent-pixel-size confidence. Let a
                // proven parent execute with that legacy input while the
                // dataset still records the corrected official observation.
                policy_observation[17] = teacher_clamp(
                    1.0f - 0.5f * (
                        fabsf(policy_observation[14])
                        + fabsf(policy_observation[15])),
                    0.0f,
                    1.0f);
            }
            float race_phase = env.observable_gate_progress
                ? policy_observation[23]
                : policy_observation[23] / 3.0f
                    + policy_observation[24] * (2.0f / 3.0f)
                    + policy_observation[25];
            if (restore_phase_yaw) {
                policy_observation[22] = last_policy_yaw_action;
            }
            if (use_bf16) {
                forward_puffernet_bf16(policy, policy_observation, teacher);
            } else {
                forward_puffernet(policy, policy_observation, teacher);
            }
            if (residual_policy != NULL) {
                float previous_action[DRONE_RACE_NUM_ATNS] = {0};
                if (previous_policy != NULL) {
                    forward_decoder_head(
                        previous_policy, policy->mingru->output,
                        previous_action, use_bf16);
                }
                float residual_action[DRONE_RACE_NUM_ATNS] = {0};
                forward_decoder_head(
                    residual_policy, policy->mingru->output,
                    residual_action, use_bf16);
                if (final_policy != NULL) {
                    float final_action[DRONE_RACE_NUM_ATNS] = {0};
                    forward_decoder_head(
                        final_policy, policy->mingru->output,
                        final_action, use_bf16);
                    float residual_low_action[DRONE_RACE_NUM_ATNS] = {0};
                    if (residual_low_policy != NULL) {
                        forward_decoder_head(
                            residual_low_policy, policy->mingru->output,
                            residual_low_action, use_bf16);
                    }
                    float final_low_action[DRONE_RACE_NUM_ATNS] = {0};
                    if (final_low_policy != NULL) {
                        forward_decoder_head(
                            final_low_policy, policy->mingru->output,
                            final_low_action, use_bf16);
                    }
                    float confidence = policy_observation[17];
                    float* selected_action = teacher;
                    if (race_phase >= phase_final_start) {
                        if (confidence >= phase_final_min_confidence) {
                            selected_action = final_action;
                        } else if (final_low_policy != NULL) {
                            selected_action = final_low_action;
                        }
                    } else if (race_phase >= phase_residual_start
                            && race_phase < phase_final_start) {
                        if (confidence >= phase_residual_min_confidence) {
                            selected_action = residual_action;
                        } else if (residual_low_policy != NULL) {
                            selected_action = residual_low_action;
                        }
                    } else if (previous_policy != NULL
                            && race_phase >= 1.0f / 3.0f
                            && race_phase < phase_residual_start) {
                        selected_action = previous_action;
                    }
                    if (selected_action != teacher) {
                        memcpy(teacher, selected_action, sizeof(teacher));
                    }
                } else {
                    float blend = teacher_clamp(
                        (race_phase - phase_residual_start)
                            / (phase_residual_full - phase_residual_start),
                        0.0f,
                        1.0f);
                    for (int action = 0; action < DRONE_RACE_NUM_ATNS; action++) {
                        teacher[action] += blend
                            * (residual_action[action] - teacher[action]);
                    }
                }
            }
            memcpy(
                unmodified_policy_action,
                teacher,
                sizeof(unmodified_policy_action));
            if (agent->current_gate >= teacher_roll_scale_from_gate) {
                teacher[0] = teacher_clamp(
                    teacher[0] * teacher_pitch_scale + teacher_pitch_offset,
                    -1.0f,
                    1.0f);
                teacher[1] = teacher_clamp(
                    teacher[1] * teacher_roll_scale, -1.0f, 1.0f);
                teacher[2] = teacher_clamp(
                    teacher[2] * teacher_thrust_scale + teacher_thrust_offset,
                    -1.0f,
                    1.0f);
            }
            if (teacher_close_action_range > 0.0f
                    && agent->current_gate >= teacher_close_action_from_gate
                    && norm3(rel_body) <= teacher_close_action_range) {
                teacher[0] += teacher_close_pitch_offset;
                teacher[1] += teacher_close_roll_offset;
                teacher[2] += teacher_close_thrust_offset;
            }
            if (agent->current_gate >= teacher_feedback_roll_from_gate
                    && agent->current_gate < teacher_feedback_until_gate) {
                if (teacher_feedback_roll_kp != 0.0f
                        || teacher_feedback_roll_kd != 0.0f) {
                    teacher[1] = (teacher_feedback_roll_residual ? teacher[1] : 0.0f)
                        - teacher_feedback_roll_kp * rel_body.y
                        + teacher_feedback_roll_kd * velocity_body.y;
                }
            }
            if (agent->current_gate >= teacher_feedback_thrust_from_gate
                    && agent->current_gate < teacher_feedback_until_gate) {
                if (teacher_feedback_thrust_kp != 0.0f
                        || teacher_feedback_thrust_kd != 0.0f
                        || teacher_feedback_thrust_bias != 0.0f) {
                    teacher[2] = (teacher_feedback_thrust_residual ? teacher[2] : 0.0f)
                        + teacher_feedback_thrust_bias
                        + teacher_feedback_thrust_kp * rel_body.z
                        - teacher_feedback_thrust_kd * velocity_body.z;
                }
            }
            if (agent->current_gate >= teacher_ballistic_label_from_gate) {
                // Privileged training-only trajectory teacher. Recompute the
                // constant-acceleration solution to the active gate every step,
                // then convert the coupled lateral/vertical accelerations to
                // attitude and hover-centered thrust. No position, velocity, or
                // gate coordinates are written to the student observation.
                float forward_speed = fmaxf(agent->drone.state.vel.x, 3.0f);
                float time_to_gate = fmaxf(
                    rel_world.x / forward_speed,
                    fmaxf(teacher_ballistic_time_floor, env.dt));
                float inv_time_sq = 1.0f / (time_to_gate * time_to_gate);
                float relative_target_y = rel_world.y;
                float effective_gate_y = gate->pos.y;
                if (teacher_ballistic_lateral_aperture_offset_m > 0.0f
                        && agent->current_gate
                            >= teacher_ballistic_lateral_aperture_offset_from_gate
                        && agent->current_gate
                            < teacher_ballistic_lateral_aperture_offset_until_gate
                        && agent->current_gate + 1 < env.num_gates) {
                    Target* next_gate = agent_gate(
                        &env, agent, agent->current_gate + 1);
                    float offset = teacher_clamp(
                        next_gate->pos.y - gate->pos.y,
                        -teacher_ballistic_lateral_aperture_offset_m,
                        teacher_ballistic_lateral_aperture_offset_m);
                    relative_target_y += offset;
                    effective_gate_y += offset;
                }
                float accel_y;
                if (teacher_ballistic_exit_velocity_scale != 0.0f
                        && agent->current_gate + 1 < env.num_gates) {
                    // Follow a cubic lateral trajectory through the current
                    // gate with the velocity needed for the following leg.
                    // Center-only constant acceleration reaches alternating
                    // gates with the wrong lateral momentum when official
                    // lateral authority is low, making the next gate
                    // physically unreachable even though the current pass is
                    // valid. This is privileged teacher/reward information;
                    // neither the next gate nor native velocity is exposed to
                    // the student observation.
                    Target* next_gate = agent_gate(
                        &env, agent, agent->current_gate + 1);
                    float next_forward = next_gate->pos.x - gate->pos.x;
                    float target_exit_velocity_y = fabsf(next_forward) > 1e-4f
                        ? teacher_ballistic_exit_velocity_scale * forward_speed
                            * (next_gate->pos.y - effective_gate_y) / next_forward
                        : 0.0f;
                    accel_y = 6.0f * relative_target_y * inv_time_sq
                        - (4.0f * agent->drone.state.vel.y
                            + 2.0f * target_exit_velocity_y) / time_to_gate;
                } else {
                    accel_y = 2.0f
                        * (relative_target_y
                            - agent->drone.state.vel.y * time_to_gate)
                        * inv_time_sq;
                }
                float accel_z = 2.0f
                    * (rel_world.z - agent->drone.state.vel.z * time_to_gate)
                    * inv_time_sq;
                accel_y += env.sitl_linear_drag * agent->drone.state.vel.y;
                accel_z += env.sitl_linear_drag * agent->drone.state.vel.z;

                float nominal_thrust_accel = fmaxf(
                    env.sitl_vertical_accel_per_thrust * env.sitl_hover_thrust,
                    1.0f);
                // The compact plant can deliberately reduce lateral authority
                // to reproduce official v3385 traces.  Convert the requested
                // world-Y acceleration through that gain; otherwise the
                // privileged teacher under-commands roll by exactly the
                // transfer mismatch we are trying to teach the student.
                float lateral_thrust_accel = nominal_thrust_accel * fmaxf(
                    fabsf(env.sitl_lateral_accel_scale), 1e-3f);
                float desired_roll = asinf(teacher_clamp(
                    -teacher_ballistic_roll_scale * accel_y / lateral_thrust_accel,
                    -sinf(env.sitl_max_roll_rad),
                    sinf(env.sitl_max_roll_rad)));
                teacher[1] = teacher_clamp(
                    desired_roll / env.sitl_max_roll_rad, -1.0f, 1.0f);

                EulerAngles current_attitude = quat_to_euler(agent->drone.state.quat);
                float vertical_fraction = fmaxf(
                    cosf(desired_roll) * cosf(current_attitude.pitch), 0.50f);
                float cmd_thrust = (
                    env.sitl_gravity_m_s2
                    + teacher_ballistic_thrust_scale * accel_z)
                    / (env.sitl_vertical_accel_per_thrust * vertical_fraction);
                if (cmd_thrust >= env.sitl_hover_thrust) {
                    teacher[2] = (cmd_thrust - env.sitl_hover_thrust)
                        / fmaxf(env.sitl_max_thrust - env.sitl_hover_thrust, 1e-4f);
                } else {
                    teacher[2] = (cmd_thrust - env.sitl_hover_thrust)
                        / fmaxf(env.sitl_hover_thrust - env.sitl_min_thrust, 1e-4f);
                }
            }
            if (teacher_spline_valid
                    && agent->current_gate >= teacher_spline_label_from_gate
                    && agent->current_gate < teacher_spline_label_until_gate) {
                // Privileged training-only full-course teacher. A natural cubic
                // spline passes through the start and every gate, so its first
                // derivative supplies a continuous gate-entry/exit velocity and
                // its curvature supplies racing-line feed-forward acceleration.
                // The recurrent student still receives only the 32-value
                // official-observable contract recorded above.
                float desired_y;
                float desired_dy_dx;
                float desired_d2y_dx2;
                float desired_z;
                float desired_dz_dx;
                float desired_d2z_dx2;
                evaluate_teacher_spline(
                    &lateral_spline,
                    agent->drone.state.pos.x,
                    &desired_y,
                    &desired_dy_dx,
                    &desired_d2y_dx2);
                evaluate_teacher_spline(
                    &vertical_spline,
                    agent->drone.state.pos.x,
                    &desired_z,
                    &desired_dz_dx,
                    &desired_d2z_dx2);
                if (agent->current_gate
                            >= teacher_spline_derivative_lookahead_from_gate
                        && teacher_spline_derivative_lookahead_m > 0.0f) {
                    float lookahead_x = fminf(
                        agent->drone.state.pos.x
                            + teacher_spline_derivative_lookahead_m,
                        lateral_spline.x[lateral_spline.points - 1]);
                    float ignored_value;
                    evaluate_teacher_spline(
                        &lateral_spline,
                        lookahead_x,
                        &ignored_value,
                        &desired_dy_dx,
                        &desired_d2y_dx2);
                }
                if (agent->current_gate
                            >= teacher_spline_derivative_lookahead_from_gate
                        && teacher_spline_vertical_derivative_lookahead_m > 0.0f) {
                    float lookahead_x = fminf(
                        agent->drone.state.pos.x
                            + teacher_spline_vertical_derivative_lookahead_m,
                        vertical_spline.x[vertical_spline.points - 1]);
                    float ignored_value;
                    evaluate_teacher_spline(
                        &vertical_spline,
                        lookahead_x,
                        &ignored_value,
                        &desired_dz_dx,
                        &desired_d2z_dx2);
                }

                float forward_speed = fmaxf(agent->drone.state.vel.x, 0.0f);
                float desired_velocity_y = desired_dy_dx * forward_speed;
                float desired_velocity_z = desired_dz_dx * forward_speed;
                float close_position_blend = teacher_spline_close_position_range_m > 0.0f
                    ? teacher_clamp(
                        1.0f - fmaxf(rel_world.x, 0.0f)
                            / teacher_spline_close_position_range_m,
                        0.0f,
                        1.0f)
                    : 0.0f;
                float lateral_position_kp = teacher_spline_lateral_position_kp
                    + close_position_blend
                        * teacher_spline_close_lateral_position_kp;
                float accel_y = teacher_spline_curvature_scale
                        * desired_d2y_dx2 * forward_speed * forward_speed
                    + lateral_position_kp
                        * (desired_y - agent->drone.state.pos.y)
                    + teacher_spline_lateral_velocity_kd
                        * (desired_velocity_y - agent->drone.state.vel.y);
                float accel_z = teacher_spline_curvature_scale
                        * desired_d2z_dx2 * forward_speed * forward_speed
                    + teacher_spline_vertical_position_kp
                        * (desired_z - agent->drone.state.pos.z)
                    + teacher_spline_vertical_velocity_kd
                        * (desired_velocity_z - agent->drone.state.vel.z);

                if (teacher_spline_recovery_trigger_m > 0.0f) {
                    float time_to_gate = fmaxf(
                        rel_world.x / fmaxf(forward_speed, 3.0f),
                        fmaxf(teacher_spline_recovery_time_floor_s, env.dt));
                    float projected_error_y = rel_world.y
                        - agent->drone.state.vel.y * time_to_gate;
                    float projected_error_z = rel_world.z
                        - agent->drone.state.vel.z * time_to_gate;
                    float projected_miss = sqrtf(
                        projected_error_y * projected_error_y
                        + projected_error_z * projected_error_z);
                    if (projected_miss > teacher_spline_recovery_trigger_m) {
                        // Off-policy feasibility guard for DAgger. The nominal
                        // spline optimizes exit velocity, but a student already
                        // outside its corridor must first make the active
                        // aperture. This same constant-acceleration intercept is
                        // used at every gate and is never exposed at runtime.
                        float inv_time_sq = 1.0f / (time_to_gate * time_to_gate);
                        accel_y = 2.0f * projected_error_y * inv_time_sq;
                        accel_z = 2.0f * projected_error_z * inv_time_sq;
                        spline_recovery_steps++;
                    }
                }

                // Compensate the compact plant's linear drag, then solve the
                // coupled roll/thrust force instead of assuming roll and thrust
                // are independent. Positive roll produces negative world Y.
                float force_y = accel_y
                    + env.sitl_linear_drag * agent->drone.state.vel.y;
                float force_z = env.sitl_gravity_m_s2 + accel_z
                    + env.sitl_linear_drag * agent->drone.state.vel.z;
                EulerAngles current_attitude = quat_to_euler(agent->drone.state.quat);
                float desired_roll = atan2f(
                    -force_y * cosf(current_attitude.pitch),
                    fmaxf(force_z * env.sitl_lateral_accel_scale, 1e-3f));
                desired_roll = teacher_clamp(
                    desired_roll, -env.sitl_max_roll_rad, env.sitl_max_roll_rad);
                teacher[1] = desired_roll / env.sitl_max_roll_rad;

                float vertical_fraction = fmaxf(
                    cosf(desired_roll) * cosf(current_attitude.pitch), 0.50f);
                float cmd_thrust = force_z
                    / (env.sitl_vertical_accel_per_thrust * vertical_fraction);
                if (cmd_thrust >= env.sitl_hover_thrust) {
                    teacher[2] = (cmd_thrust - env.sitl_hover_thrust)
                        / fmaxf(env.sitl_max_thrust - env.sitl_hover_thrust, 1e-4f);
                } else {
                    teacher[2] = (cmd_thrust - env.sitl_hover_thrust)
                        / fmaxf(env.sitl_hover_thrust - env.sitl_min_thrust, 1e-4f);
                }
            }
            for (int action = 0; action < DRONE_RACE_NUM_ATNS; action++) {
                teacher[action] = teacher_clamp(teacher[action], -1.0f, 1.0f);
            }
        }
        if (policy_teacher
                && agent->current_gate >= teacher_analytic_label_from_gate) {
            memcpy(teacher, analytic_teacher, sizeof(teacher));
            teacher[0] = teacher_clamp(
                teacher[0] * teacher_analytic_label_pitch_scale,
                -1.0f, 1.0f);
            teacher[1] = -0.40f * teacher_analytic_label_roll_scale
                * teacher_clamp(
                    rel_body.y * 0.08f
                        - agent->drone.state.vel.y * 0.10f
                            * teacher_analytic_label_roll_rate_scale,
                    -0.50f, 0.50f);
            teacher[1] = teacher_clamp(teacher[1], -1.0f, 1.0f);
            teacher[2] = teacher_clamp(
                teacher[2] + teacher_analytic_label_thrust_offset,
                -1.0f, 1.0f);
        }
        if (teacher_observable_course
                && agent->current_gate >= teacher_observable_course_from_gate) {
            observable_course_teacher_action(
                &observable_course_teacher,
                env.observations,
                env.dt,
                reset_state,
                teacher_observable_velocity_alpha,
                teacher_observable_motion_rate_blend,
                teacher_spline_derivative_lookahead_m,
                teacher_spline_vertical_derivative_lookahead_m,
                teacher_spline_derivative_lookahead_from_gate,
                teacher_spline_lateral_position_kp,
                teacher_spline_close_lateral_position_kp,
                teacher_spline_close_position_range_m,
                teacher_spline_lateral_velocity_kd,
                teacher_spline_vertical_position_kp,
                teacher_spline_vertical_velocity_kd,
                teacher_spline_curvature_scale,
                env.sitl_linear_drag,
                env.sitl_lateral_accel_scale,
                env.sitl_gravity_m_s2,
                env.sitl_vertical_accel_per_thrust,
                env.sitl_hover_thrust,
                env.sitl_min_thrust,
                env.sitl_max_thrust,
                env.sitl_max_roll_rad,
                env.observable_gate_progress,
                env.observable_gate_index_denominator,
                teacher);
            if (teacher_observable_pitch_speed_control) {
                teacher[0] = teacher_clamp(
                    0.24f * teacher_clamp(
                        (observable_course_teacher.debug_velocity_world.x
                            - teacher_observable_pitch_speed_target_m_s)
                            * 0.35f,
                        -0.60f,
                        0.60f),
                    -1.0f,
                    1.0f);
            }
        }
        if (policy_teacher
                && agent->current_gate >= teacher_analytic_pitch_from_gate) {
            // Let the trajectory teacher own longitudinal speed as well as the
            // spline's roll/thrust channels. Retaining an off-policy student's
            // pitch can otherwise double forward speed after a correction and
            // make the following aperture unreachable.
            float analytic_pitch_scale = agent->current_gate
                    >= teacher_analytic_pitch_scale_from_gate
                ? teacher_analytic_pitch_scale_late
                : teacher_analytic_label_pitch_scale;
            teacher[0] = teacher_clamp(
                analytic_teacher[0] * analytic_pitch_scale,
                -1.0f,
                1.0f);
        }
        if (policy_teacher
                && agent->current_gate >= teacher_post_action_offset_from_gate
                && agent->current_gate < teacher_post_action_offset_until_gate) {
            // Final gate-indexed action adapter. Apply after every analytic,
            // ballistic, spline, or observable teacher so diagnostic execution
            // and DAgger labels use exactly the same bounded local residual.
            teacher[1] = teacher_clamp(
                teacher[1] + teacher_post_roll_offset, -1.0f, 1.0f);
            teacher[2] = teacher_clamp(
                teacher[2] + teacher_post_thrust_offset, -1.0f, 1.0f);
        }
        if (policy_teacher && agent->current_gate < env.num_gates
                && (!teacher_gate_action_bias_visible_only[agent->current_gate]
                    || env.observations[10] > 0.5f)
                && env.observations[17]
                    >= teacher_gate_action_bias_min_alignment[agent->current_gate]
                && env.observations[11]
                    >= teacher_gate_action_bias_min_forward_norm[agent->current_gate]) {
            for (int action = 0; action < DRONE_RACE_NUM_ATNS; action++) {
                teacher[action] = teacher_clamp(
                    teacher[action]
                        + teacher_gate_action_bias[agent->current_gate][action],
                    -1.0f,
                    1.0f);
            }
        }
        float executed[4];
        int execute_student = policy != NULL
            && rndf(0.0f, 1.0f, &env.rng) < student_probability;
        if (execute_student) {
            if (phase_reset_tail
                    && agent->current_gate >= 3
                    && agent->current_gate <= 5) {
                PufferNet* selected = policy;
                if (agent->current_gate == 4) {
                    selected = phase_reset_gate4_policy;
                } else if (agent->current_gate == 5) {
                    selected = phase_reset_gate5_policy;
                }
                if (agent->current_gate != phase_reset_previous_gate) {
                    memset(selected->mingru->state, 0,
                        (size_t)selected->mingru->num_layers
                            * selected->mingru->batch_size
                            * selected->mingru->hidden_size * sizeof(float));
                    phase_reset_previous_gate = agent->current_gate;
                }
                if (use_bf16) {
                    forward_puffernet_bf16(selected, env.observations, executed);
                } else {
                    forward_puffernet(selected, env.observations, executed);
                }
            } else if (policy_teacher) {
                memcpy(executed, unmodified_policy_action, sizeof(executed));
            } else if (use_bf16) {
                forward_puffernet_bf16(policy, env.observations, executed);
            } else {
                forward_puffernet(policy, env.observations, executed);
            }
        } else {
            for (int k = 0; k < 4; k++) {
                float noise = (k == 3 || agent->current_gate < action_noise_from_gate)
                    ? 0.0f : rndf(-action_noise, action_noise, &env.rng);
                executed[k] = teacher_clamp(teacher[k] + noise, -1.0f, 1.0f);
            }
        }
        if (agent->current_gate == phase_action_bias_gate) {
            if (phase_action_pitch_bias != 0.0f) {
                executed[0] = teacher_clamp(
                    executed[0] + phase_action_pitch_bias, -1.0f, 1.0f);
            }
            if (phase_action_roll_bias != 0.0f) {
                executed[1] = teacher_clamp(
                    executed[1] + phase_action_roll_bias, -1.0f, 1.0f);
            }
            if (phase_action_thrust_bias != 0.0f) {
                executed[2] = teacher_clamp(
                    executed[2] + phase_action_thrust_bias, -1.0f, 1.0f);
            }
        }
        if (reset_state || agent->current_gate != env.num_gates - 1) {
            phase_final_observable_roll_integral = 0.0f;
            phase_final_observable_course_active = 0;
            phase_final_bang_bang_roll_direction = 0.0f;
        }
        if (agent->current_gate == env.num_gates - 1) {
            if (phase_final_action_pitch_bias != 0.0f) {
                executed[0] = teacher_clamp(
                    executed[0] + phase_final_action_pitch_bias,
                    -1.0f,
                    1.0f);
            }
            if (phase_final_action_thrust_bias != 0.0f) {
                executed[2] = teacher_clamp(
                    executed[2] + phase_final_action_thrust_bias,
                    -1.0f,
                    1.0f);
            }
            if (env.observations[10] > 0.5f) {
                float observable_right = observable_unscale_tanh(
                    env.observations[12], 0.2f);
                float observable_down = observable_unscale_tanh(
                    env.observations[13], 0.2f);
                float observable_right_rate = observable_unscale_tanh(
                    env.observations[1], 0.3333333333f);
                float observable_down_rate = observable_unscale_tanh(
                    env.observations[2], 0.3333333333f);
                if (phase_final_lateral_brake_pitch_bias != 0.0f
                        && fabsf(observable_right)
                            > phase_final_lateral_brake_release_right_m) {
                    executed[0] = teacher_clamp(
                        executed[0] + phase_final_lateral_brake_pitch_bias,
                        -1.0f,
                        1.0f);
                }
                if (phase_final_projected_miss_brake_pitch_bias != 0.0f) {
                    float observable_forward = observable_unscale_tanh(
                        env.observations[11], 0.1f);
                    float observable_forward_rate = observable_unscale_tanh(
                        env.observations[0], 0.2f);
                    float projected_time_to_plane = fmaxf(
                        observable_forward / fmaxf(
                            -observable_forward_rate,
                            phase_final_intercept_forward_speed_floor_m_s),
                        env.dt);
                    float projected_right_at_plane = observable_right
                        + observable_right_rate * projected_time_to_plane;
                    if (projected_right_at_plane
                            > phase_final_projected_miss_brake_threshold_m) {
                        executed[0] = teacher_clamp(
                            executed[0]
                                + phase_final_projected_miss_brake_pitch_bias,
                            -1.0f,
                            1.0f);
                    }
                }
                phase_final_observable_roll_integral = teacher_clamp(
                    phase_final_observable_roll_integral
                        + observable_right * env.dt,
                    -fabsf(phase_final_observable_roll_integral_limit),
                    fabsf(phase_final_observable_roll_integral_limit));
                float roll_residual =
                    -phase_final_observable_roll_kp * observable_right
                    -phase_final_observable_roll_kd * observable_right_rate
                    -phase_final_observable_roll_ki
                        * phase_final_observable_roll_integral;
                float thrust_residual =
                    -phase_final_observable_thrust_kp * observable_down
                    -phase_final_observable_thrust_kd * observable_down_rate;
                executed[1] = teacher_clamp(
                    phase_final_observable_roll_override
                        ? roll_residual
                        : executed[1] + roll_residual,
                    -1.0f,
                    1.0f);
                executed[2] = teacher_clamp(
                    executed[2] + thrust_residual, -1.0f, 1.0f);
                if (phase_final_observable_intercept_roll) {
                    Quat observation_quat = {
                        env.observations[6], env.observations[7],
                        env.observations[8], env.observations[9],
                    };
                    quat_normalize(&observation_quat);
                    Vec3 observable_relative_body = {
                        observable_unscale_tanh(env.observations[11], 0.1f),
                        observable_right,
                        -observable_down,
                    };
                    Vec3 observable_gate_rate_body = {
                        observable_unscale_tanh(env.observations[0], 0.2f),
                        observable_right_rate,
                        -observable_down_rate,
                    };
                    Vec3 observable_relative_world = quat_rotate(
                        observation_quat, observable_relative_body);
                    Vec3 observable_velocity_world = scalmul3(
                        quat_rotate(observation_quat, observable_gate_rate_body),
                        -1.0f);
                    float observable_forward_speed = fmaxf(
                        observable_velocity_world.x,
                        fmaxf(
                            phase_final_intercept_forward_speed_floor_m_s,
                            0.1f));
                    float time_to_plane = fmaxf(
                        observable_relative_world.x / observable_forward_speed,
                        fmaxf(phase_final_intercept_time_floor_s, env.dt));
                    float desired_lateral_accel =
                        phase_final_intercept_lateral_accel_scale * 2.0f * (
                        observable_relative_world.y
                            - observable_velocity_world.y * time_to_plane)
                        / (time_to_plane * time_to_plane);
                    float lateral_force = desired_lateral_accel
                        + env.sitl_linear_drag * observable_velocity_world.y;
                    EulerAngles observable_attitude = quat_to_euler(
                        observation_quat);
                    float desired_roll = atan2f(
                        -lateral_force * cosf(observable_attitude.pitch),
                        fmaxf(
                            env.sitl_gravity_m_s2
                                * env.sitl_lateral_accel_scale,
                            1e-3f));
                    executed[1] = teacher_clamp(
                        desired_roll / env.sitl_max_roll_rad,
                        -1.0f,
                        1.0f);
                }
                if (phase_final_observable_intercept_thrust) {
                    Quat observation_quat = {
                        env.observations[6], env.observations[7],
                        env.observations[8], env.observations[9],
                    };
                    quat_normalize(&observation_quat);
                    Vec3 observable_relative_body = {
                        observable_unscale_tanh(env.observations[11], 0.1f),
                        observable_right,
                        -observable_down,
                    };
                    Vec3 observable_gate_rate_body = {
                        observable_unscale_tanh(env.observations[0], 0.2f),
                        observable_right_rate,
                        -observable_down_rate,
                    };
                    Vec3 observable_relative_world = quat_rotate(
                        observation_quat, observable_relative_body);
                    Vec3 observable_velocity_world = scalmul3(
                        quat_rotate(observation_quat, observable_gate_rate_body),
                        -1.0f);
                    float observable_forward_speed = fmaxf(
                        observable_velocity_world.x,
                        phase_final_intercept_forward_speed_floor_m_s);
                    float time_to_plane = fmaxf(
                        observable_relative_world.x / observable_forward_speed,
                        fmaxf(phase_final_intercept_time_floor_s, env.dt));
                    float desired_vertical_accel =
                        phase_final_intercept_vertical_accel_scale * 2.0f * (
                            observable_relative_world.z
                                - observable_velocity_world.z * time_to_plane)
                            / (time_to_plane * time_to_plane);
                    float vertical_force = env.sitl_gravity_m_s2
                        + desired_vertical_accel
                        + env.sitl_linear_drag * observable_velocity_world.z;
                    EulerAngles observable_attitude = quat_to_euler(
                        observation_quat);
                    float vertical_fraction = fmaxf(
                        cosf(observable_attitude.roll)
                            * cosf(observable_attitude.pitch),
                        0.50f);
                    float commanded_thrust = vertical_force
                        / (env.sitl_vertical_accel_per_thrust
                            * vertical_fraction);
                    if (commanded_thrust >= env.sitl_hover_thrust) {
                        executed[2] = (commanded_thrust - env.sitl_hover_thrust)
                            / fmaxf(
                                env.sitl_max_thrust - env.sitl_hover_thrust,
                                1e-4f);
                    } else {
                        executed[2] = (commanded_thrust - env.sitl_hover_thrust)
                            / fmaxf(
                                env.sitl_hover_thrust - env.sitl_min_thrust,
                                1e-4f);
                    }
                    executed[2] = teacher_clamp(executed[2], -1.0f, 1.0f);
                }
                if (phase_final_observable_course) {
                    float course_action[DRONE_RACE_NUM_ATNS];
                    memcpy(course_action, executed, sizeof(course_action));
                    observable_course_teacher_action(
                        &phase_final_observable_course_controller,
                        env.observations,
                        env.dt,
                        !phase_final_observable_course_active,
                        teacher_observable_velocity_alpha,
                        teacher_observable_motion_rate_blend,
                        teacher_spline_derivative_lookahead_m,
                        teacher_spline_vertical_derivative_lookahead_m,
                        teacher_spline_derivative_lookahead_from_gate,
                        teacher_spline_lateral_position_kp,
                        teacher_spline_close_lateral_position_kp,
                        teacher_spline_close_position_range_m,
                        teacher_spline_lateral_velocity_kd,
                        teacher_spline_vertical_position_kp,
                        teacher_spline_vertical_velocity_kd,
                        teacher_spline_curvature_scale,
                        env.sitl_linear_drag,
                        env.sitl_lateral_accel_scale,
                        env.sitl_gravity_m_s2,
                        env.sitl_vertical_accel_per_thrust,
                        env.sitl_hover_thrust,
                        env.sitl_min_thrust,
                        env.sitl_max_thrust,
                        env.sitl_max_roll_rad,
                        env.observable_gate_progress,
                        env.observable_gate_index_denominator,
                        course_action);
                    phase_final_observable_course_active = 1;
                    executed[1] = course_action[1];
                    executed[2] = course_action[2];
                }
                if (phase_final_bang_bang_roll) {
                    float bang_bang_forward = observable_unscale_tanh(
                        env.observations[11], 0.1f);
                    if (phase_final_bang_bang_roll_directional
                            && phase_final_bang_bang_roll_direction == 0.0f) {
                        phase_final_bang_bang_roll_direction =
                            observable_right >= 0.0f ? 1.0f : -1.0f;
                    }
                    float roll_direction = phase_final_bang_bang_roll_directional
                        ? phase_final_bang_bang_roll_direction
                        : 1.0f;
                    float bang_bang_roll = roll_direction * (
                        bang_bang_forward > phase_final_roll_switch_forward_m
                            ? phase_final_roll_accel_action
                            : phase_final_roll_brake_action);
                    if (phase_final_bang_bang_roll_pd_residual) {
                        bang_bang_roll += roll_residual;
                    }
                    executed[1] = teacher_clamp(
                        bang_bang_roll,
                        -1.0f,
                        1.0f);
                }
                if (phase_final_deterministic_attitude) {
                    executed[0] = teacher_clamp(
                        phase_final_deterministic_pitch_action, -1.0f, 1.0f);
                    executed[3] = teacher_clamp(
                        phase_final_deterministic_yaw_action, -1.0f, 1.0f);
                }
                if (phase_final_observable_pitch_speed_control) {
                    float observable_forward_rate = observable_unscale_tanh(
                        env.observations[0], 0.2f);
                    float observable_forward_speed = fmaxf(
                        -observable_forward_rate,
                        phase_final_intercept_forward_speed_floor_m_s);
                    executed[0] = teacher_clamp(
                        0.24f * teacher_clamp(
                            (observable_forward_speed
                                - phase_final_observable_pitch_speed_target_m_s)
                                * 0.35f,
                            -0.60f,
                            0.60f),
                        -1.0f,
                        1.0f);
                }
            }
        }
        if (agent->current_gate == phase_counter_roll_gate) {
            if (!phase_counter_roll_latched
                    && phase_counter_roll_forward_m > 0.0f
                    && env.observations[10] > 0.5f) {
                float observable_forward = observable_unscale_tanh(
                    env.observations[11], 0.1f);
                if (observable_forward > 0.0f
                        && observable_forward <= phase_counter_roll_forward_m) {
                    phase_counter_roll_latched = 1;
                }
            }
            if (phase_counter_roll_latched) {
                executed[1] = teacher_clamp(
                    phase_counter_roll_action, -1.0f, 1.0f);
            }
        }
        if (policy_teacher
                && agent->current_gate >= teacher_execute_label_from_gate) {
            memcpy(executed, teacher, sizeof(executed));
        }
        float record[DRONE_RACE_OBS_SIZE + DRONE_RACE_NUM_ATNS + 1];
        memcpy(record, env.observations, DRONE_RACE_OBS_SIZE * sizeof(float));
        memcpy(
            record + DRONE_RACE_OBS_SIZE,
            record_executed_action ? executed : teacher,
            DRONE_RACE_NUM_ATNS * sizeof(float));
        record[DRONE_RACE_OBS_SIZE + DRONE_RACE_NUM_ATNS] =
            reset_state ? 1.0f : 0.0f;
        attempted_records++;
        if (execute_student) episode_student_steps++;
        if (keep_successful_episodes_only) {
            if (episode_record_count >= env.max_steps) {
                fprintf(stderr, "success-filter episode exceeded max_steps\n");
                return 2;
            }
            memcpy(
                episode_records + (size_t)episode_record_count * record_width,
                record, sizeof(record));
            episode_record_count++;
        } else {
            if (fwrite(record, sizeof(record), 1, out) != 1) {
                perror("fwrite");
                return 2;
            }
            records++;
            if (execute_student) kept_student_steps++;
        }
        if (getenv("TEACHER_TRACE") != NULL && episodes == 0 && agent->step_count % 120 == 0) {
            fprintf(stderr,
                "trace step=%d gate=%d pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f) "
                "rate_obs=(%.3f,%.3f,%.3f) rel_obs=(%.3f,%.3f,%.3f) "
                "action=(%.3f,%.3f,%.3f,%.3f) teacher=(%.3f,%.3f,%.3f,%.3f)\n",
                agent->step_count, agent->current_gate,
                agent->drone.state.pos.x, agent->drone.state.pos.y, agent->drone.state.pos.z,
                agent->drone.state.vel.x, agent->drone.state.vel.y, agent->drone.state.vel.z,
                env.observations[0], env.observations[1], env.observations[2],
                env.observations[11], env.observations[12], env.observations[13],
                executed[0], executed[1], executed[2], executed[3],
                teacher[0], teacher[1], teacher[2], teacher[3]);
            if (teacher_observable_course) {
                fprintf(stderr,
                    "observable velocity=(%.3f,%.3f,%.3f) "
                    "desired_yz=(%.3f,%.3f) accel_yz=(%.3f,%.3f)\n",
                    observable_course_teacher.debug_velocity_world.x,
                    observable_course_teacher.debug_velocity_world.y,
                    observable_course_teacher.debug_velocity_world.z,
                    observable_course_teacher.debug_desired_position.y,
                    observable_course_teacher.debug_desired_position.z,
                    observable_course_teacher.debug_acceleration.y,
                    observable_course_teacher.debug_acceleration.z);
            }
        }
        memcpy(env.actions, executed, sizeof(executed));
        last_policy_yaw_action = executed[3];
        float successes_before_step = env.log.success_rate;
        int gate_before_step = agent->current_gate;
        c_step(&env);
        if (getenv("PHASE_TRANSITION_TRACE") != NULL
                && !env.terminals[0]
                && agent->current_gate != gate_before_step) {
            EulerAngles transition_attitude = quat_to_euler(agent->drone.state.quat);
            fprintf(stderr,
                "phase_transition from=%d to=%d step=%d "
                "pos=(%.9g,%.9g,%.9g) vel=(%.9g,%.9g,%.9g) "
                "quat=(%.9g,%.9g,%.9g,%.9g) "
                "euler=(%.9g,%.9g,%.9g) omega=(%.9g,%.9g,%.9g)\n",
                gate_before_step, agent->current_gate, agent->step_count,
                agent->drone.state.pos.x, agent->drone.state.pos.y,
                agent->drone.state.pos.z, agent->drone.state.vel.x,
                agent->drone.state.vel.y, agent->drone.state.vel.z,
                agent->drone.state.quat.w, agent->drone.state.quat.x,
                agent->drone.state.quat.y, agent->drone.state.quat.z,
                transition_attitude.roll, transition_attitude.pitch,
                transition_attitude.yaw, agent->drone.state.omega.x,
                agent->drone.state.omega.y, agent->drone.state.omega.z);
        }
        reset_state = env.terminals[0] > 0.5f;
        if (reset_state) {
            bool episode_success = env.log.success_rate > successes_before_step;
            if (keep_successful_episodes_only && episode_success) {
                if (fwrite(
                        episode_records,
                        record_width * sizeof(float),
                        (size_t)episode_record_count,
                        out) != (size_t)episode_record_count) {
                    perror("fwrite");
                    return 2;
                }
                records += episode_record_count;
                kept_student_steps += episode_student_steps;
                kept_episodes++;
            }
            if (!keep_successful_episodes_only) kept_episodes++;
            episode_record_count = 0;
            episode_student_steps = 0;
            episodes++;
            phase_reset_previous_gate = -1;
            phase_counter_roll_latched = 0;
            if (policy != NULL) {
                memset(policy->mingru->state, 0,
                    (size_t)policy->mingru->num_layers * policy->mingru->batch_size
                        * policy->mingru->hidden_size * sizeof(float));
            }
            if (residual_policy != NULL) {
                memset(residual_policy->mingru->state, 0,
                    (size_t)residual_policy->mingru->num_layers
                        * residual_policy->mingru->batch_size
                        * residual_policy->mingru->hidden_size * sizeof(float));
            }
            if (previous_policy != NULL) {
                memset(previous_policy->mingru->state, 0,
                    (size_t)previous_policy->mingru->num_layers
                        * previous_policy->mingru->batch_size
                        * previous_policy->mingru->hidden_size * sizeof(float));
            }
            if (final_policy != NULL) {
                memset(final_policy->mingru->state, 0,
                    (size_t)final_policy->mingru->num_layers
                        * final_policy->mingru->batch_size
                        * final_policy->mingru->hidden_size * sizeof(float));
            }
            if (residual_low_policy != NULL) {
                memset(residual_low_policy->mingru->state, 0,
                    (size_t)residual_low_policy->mingru->num_layers
                        * residual_low_policy->mingru->batch_size
                        * residual_low_policy->mingru->hidden_size * sizeof(float));
            }
            if (final_low_policy != NULL) {
                memset(final_low_policy->mingru->state, 0,
                    (size_t)final_low_policy->mingru->num_layers
                        * final_low_policy->mingru->batch_size
                        * final_low_policy->mingru->hidden_size * sizeof(float));
            }
            if (phase_reset_gate4_policy != NULL) {
                memset(phase_reset_gate4_policy->mingru->state, 0,
                    (size_t)phase_reset_gate4_policy->mingru->num_layers
                        * phase_reset_gate4_policy->mingru->batch_size
                        * phase_reset_gate4_policy->mingru->hidden_size
                        * sizeof(float));
            }
            if (phase_reset_gate5_policy != NULL) {
                memset(phase_reset_gate5_policy->mingru->state, 0,
                    (size_t)phase_reset_gate5_policy->mingru->num_layers
                        * phase_reset_gate5_policy->mingru->batch_size
                        * phase_reset_gate5_policy->mingru->hidden_size
                        * sizeof(float));
            }
            last_policy_yaw_action = 0.0f;
            if (episode_seed_sequence) {
                // Exact native evaluation assigns seed i to vector env i and
                // collects one episode from each. Rebuild the just-reset
                // episode with that same seed schedule so a serial teacher
                // dataset covers the identical initial-condition population.
                env.rng = (unsigned int)(episode_seed_offset + episodes);
                reset_agent(&env, &env.agents[0]);
                compute_observations(&env);
            }
        }
    }

    fclose(out);
    printf(
        "teacher_dataset records=%ld attempted_records=%ld episodes=%d "
        "kept_episodes=%d student_fraction=%.6f "
        "success_rate=%.6f crash_rate=%.6f "
        "gates_per_episode=%.6f missed_rate=%.6f timeout_rate=%.6f "
        "spline_recovery_fraction=%.6f "
        "final_xyz=(%.3f,%.3f,%.3f) closest=%.3f "
        "terminal_crossings=%.0f terminal_crossing_radial=%.6f "
        "terminal_crossing_right=%.6f terminal_crossing_vertical=%.6f\n",
        records, attempted_records, episodes, kept_episodes,
        records > 0 ? (float)kept_student_steps / records : 0.0f,
        env.log.n > 0.0f ? env.log.success_rate / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.crash / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.gates_passed / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.missed_gate / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.timeout / env.log.n : 0.0f,
        records > 0 ? (float)spline_recovery_steps / records : 0.0f,
        env.log.n > 0.0f ? env.log.final_x / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.final_y / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.final_z / env.log.n : 0.0f,
        env.log.n > 0.0f ? env.log.closest_gate_range / env.log.n : 0.0f,
        env.log.terminal_crossing_sampled,
        env.log.terminal_crossing_sampled > 0.0f
            ? env.log.terminal_crossing_radial
                / env.log.terminal_crossing_sampled
            : 0.0f,
        env.log.terminal_crossing_sampled > 0.0f
            ? env.log.terminal_crossing_right
                / env.log.terminal_crossing_sampled
            : 0.0f,
        env.log.terminal_crossing_sampled > 0.0f
            ? env.log.terminal_crossing_vertical
                / env.log.terminal_crossing_sampled
            : 0.0f);
    for (int gate = 0; gate < env.num_gates; gate++) {
        float sampled = env.log.terminal_gate_sampled[gate];
        if (sampled <= 0.0f) continue;
        printf(
            "terminal_gate=%d sampled=%.0f radial=%.6f right=%.6f vertical=%.6f\n",
            gate,
            sampled,
            env.log.terminal_gate_radial[gate] / sampled,
            env.log.terminal_gate_right[gate] / sampled,
            env.log.terminal_gate_vertical[gate] / sampled);
    }
    c_close(&env);
    free(episode_records);
    if (policy != NULL) free_puffernet(policy);
    if (previous_policy != NULL) free_puffernet(previous_policy);
    if (residual_policy != NULL) free_puffernet(residual_policy);
    if (final_policy != NULL) free_puffernet(final_policy);
    if (residual_low_policy != NULL) free_puffernet(residual_low_policy);
    if (final_low_policy != NULL) free_puffernet(final_low_policy);
    if (phase_reset_gate4_policy != NULL) free_puffernet(phase_reset_gate4_policy);
    if (phase_reset_gate5_policy != NULL) free_puffernet(phase_reset_gate5_policy);
    free(policy_weights);
    free(previous_weights);
    free(residual_weights);
    free(final_weights);
    free(residual_low_weights);
    free(final_low_weights);
    free(phase_reset_gate4_weights);
    free(phase_reset_gate5_weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
