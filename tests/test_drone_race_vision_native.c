#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DRONE_RACE_OBS_SIZE 4152
#define DRONE_RACE_BINDING
#include "ocean/drone_race/drone_race.c"

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

static DroneRace make_visual_env(void) {
    DroneRace env = {0};
    env.num_agents = 1;
    env.rng = 7;
    env.dt = 1.0f / 64.0f;
    env.num_gates = 2;
    env.use_custom_gate_layout = 1;
    env.custom_gate_pos[0][0] = 5.0f;
    env.custom_gate_pos[1][0] = 10.0f;
    env.custom_gate_pos[1][1] = 2.0f;
    env.gate_radius = 1.0f;
    env.start_offset = 5.0f;
    env.crash_height = -10.0f;
    env.pos_bound = 100.0f;
    env.direction_min = 0.05f;
    env.strict_missed_gate = 1;
    env.max_steps = 6400;
    env.time_limit_seconds = 100.0f;
    env.w_progress = 1.0f;
    env.w_gate = 1.0f;
    env.w_finish = 1.0f;
    env.invalid_penalty = 1.0f;
    env.interface_mode = DRONE_RACE_INTERFACE_NATIVE_MOTOR;
    env.visual_observation = 1;
    env.visual_camera_hz = 30.0f;
    env.visual_focal_x_px = 32.0f;
    env.visual_focal_y_px = 32.0f;
    env.visual_line_thickness_px = 1.0f;
    env.visual_edge_corrupt_prob = 0.0f;
    env.observations = calloc(DRONE_RACE_OBS_SIZE, sizeof(float));
    env.actions = calloc(DRONE_RACE_NUM_ATNS, sizeof(float));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(float));
    init(&env);
    c_reset(&env);
    return env;
}

static void free_visual_env(DroneRace* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

static int test_visual_schema_and_mask(void) {
    CHECK(DRONE_RACE_VISUAL_MASK_SIZE == 4096,
        "visual mask must remain 64x64");
    CHECK(DRONE_RACE_VISUAL_LEGAL_OBS_SIZE == 4118,
        "legal observation ABI changed unexpectedly");
    CHECK(DRONE_RACE_VISUAL_PRIVILEGED_SIZE == 34,
        "privileged decoder ABI changed unexpectedly");
    CHECK(DRONE_RACE_VISUAL_OBS_SIZE == DRONE_RACE_OBS_SIZE,
        "vision binding must expose legal plus privileged values");

    DroneRace env = make_visual_env();
    float mask_sum = 0.0f;
    int fractional_pixels = 0;
    for (int i = 0; i < DRONE_RACE_VISUAL_MASK_SIZE; i++) {
        float value = env.observations[i];
        CHECK(value >= 0.0f && value <= 1.0f,
            "mask confidence must be bounded");
        mask_sum += value;
        fractional_pixels += value > 0.0f && value < 1.0f;
    }
    CHECK(mask_sum > 1.0f, "a centered gate should render visible inner edges");
    CHECK(fractional_pixels > 0,
        "anti-aliased edge mask should preserve continuous confidence");
    CHECK(env.observations[DRONE_RACE_VISUAL_NEW_FRAME_OFFSET] == 1.0f,
        "reset must produce a fresh visual frame");
    CHECK(env.observations[DRONE_RACE_VISUAL_FRAME_AGE_OFFSET] == 0.0f,
        "fresh reset frame must have zero age");
    free_visual_env(&env);
    return 0;
}

static int test_legal_visual_observation_is_count_agnostic(void) {
    DroneRace five = make_visual_env();
    DroneRace twelve = make_visual_env();
    five.num_gates = 5;
    twelve.num_gates = 12;
    for (int gate = 2; gate < 12; gate++) {
        Target hidden = {
            .pos = {-100.0f - (float)gate, 0.0f, 0.0f},
            .orientation = {1.0f, 0.0f, 0.0f, 0.0f},
            .normal = {1.0f, 0.0f, 0.0f},
            .radius = 1.0f,
        };
        twelve.gates[gate] = hidden;
        if (gate < 5) five.gates[gate] = hidden;
    }
    five.agents[0].visual_rng = 12345u;
    twelve.agents[0].visual_rng = 12345u;
    five.agents[0].current_gate = 1;
    twelve.agents[0].current_gate = 1;
    five.agents[0].step_count = 0;
    twelve.agents[0].step_count = 0;
    memset(five.observations, 0, DRONE_RACE_OBS_SIZE * sizeof(float));
    memset(twelve.observations, 0, DRONE_RACE_OBS_SIZE * sizeof(float));
    compute_one_observation(&five, 0);
    compute_one_observation(&twelve, 0);

    CHECK(memcmp(
            five.observations,
            twelve.observations,
            DRONE_RACE_VISUAL_MASK_SIZE * sizeof(float)) == 0,
        "mask geometry must be byte-identical for a shared visible fixture");
    CHECK(memcmp(
            five.observations + DRONE_RACE_VISUAL_BODY_RATE_OFFSET,
            twelve.observations + DRONE_RACE_VISUAL_BODY_RATE_OFFSET,
            (DRONE_RACE_VISUAL_LEGAL_OBS_SIZE
                - DRONE_RACE_VISUAL_BODY_RATE_OFFSET) * sizeof(float)) == 0,
        "22-value legal sensor/history tail must not expose total gate count");
    CHECK(five.observations[DRONE_RACE_VISUAL_PRIVILEGED_OFFSET + 32]
            != twelve.observations[DRONE_RACE_VISUAL_PRIVILEGED_OFFSET + 32],
        "training-only native progress may still depend on episode length");

    free_visual_env(&five);
    free_visual_env(&twelve);
    return 0;
}

static int test_asynchronous_camera_and_action_history(void) {
    DroneRace env = make_visual_env();
    float reset_mask[DRONE_RACE_VISUAL_MASK_SIZE];
    memcpy(reset_mask, env.observations, sizeof(reset_mask));
    env.actions[0] = 0.25f;
    env.actions[1] = -0.50f;
    env.actions[2] = 0.75f;
    env.actions[3] = -1.0f;

    c_step(&env);
    CHECK(env.observations[DRONE_RACE_VISUAL_NEW_FRAME_OFFSET] == 0.0f,
        "64 Hz policy step should hold a 30 Hz camera frame");
    CHECK(memcmp(reset_mask, env.observations, sizeof(reset_mask)) == 0,
        "held camera observation must be bitwise unchanged");
    for (int action = 0; action < 4; action++) {
        CHECK(fabsf(
                env.observations[DRONE_RACE_VISUAL_ACTION_HISTORY_OFFSET + action]
                    - env.actions[action]) < 1e-7f,
            "action-history slot zero must contain the executed action");
        CHECK(env.observations[
                DRONE_RACE_VISUAL_ACTION_HISTORY_OFFSET + 4 + action] == 0.0f,
            "older action history must start at zero");
    }
    c_step(&env);
    CHECK(env.observations[DRONE_RACE_VISUAL_NEW_FRAME_OFFSET] == 0.0f,
        "second 64 Hz step should still hold the 30 Hz frame");
    c_step(&env);
    CHECK(env.observations[DRONE_RACE_VISUAL_NEW_FRAME_OFFSET] == 1.0f,
        "fractional cadence accumulator must eventually publish a new frame");
    free_visual_env(&env);
    return 0;
}

static int test_privileged_intervention_cannot_change_legal_slice(void) {
    DroneRace env = make_visual_env();
    env.visual_camera_hz = 0.0f;
    compute_one_observation(&env, 0);
    float legal_before[DRONE_RACE_VISUAL_LEGAL_OBS_SIZE];
    memcpy(legal_before, env.observations, sizeof(legal_before));
    float info_mass_before = env.observations[
        DRONE_RACE_VISUAL_PRIVILEGED_OFFSET + 26];

    env.agents[0].drone.params.mass *= 1.5f;
    compute_one_observation(&env, 0);
    CHECK(memcmp(legal_before, env.observations, sizeof(legal_before)) == 0,
        "training-only mass intervention must not alter legal actor inputs");
    CHECK(env.observations[DRONE_RACE_VISUAL_PRIVILEGED_OFFSET + 26]
            != info_mass_before,
        "privileged decoder target must expose the dynamics intervention");
    free_visual_env(&env);
    return 0;
}

static int test_ctbr_privileged_targets_follow_effective_plant(void) {
    DroneRace env = make_visual_env();
    env.interface_mode = DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT;
    env.visual_camera_hz = 0.0f;
    env.sitl_hover_thrust = 0.31f;
    env.sitl_vertical_accel_per_thrust = 30.0f;
    env.sitl_max_thrust = 0.46f;
    env.sitl_rate_lag_tau_s = 0.40f;
    env.sitl_gravity_m_s2 = 9.2f;
    env.sitl_linear_drag = 0.20f;
    compute_one_observation(&env, 0);
    int dynamics = DRONE_RACE_VISUAL_PRIVILEGED_OFFSET + 26;
    CHECK(fabsf(env.observations[dynamics]
            - visual_log_ratio(0.31f, 0.27f)) < 1e-7f,
        "CTBR decoder must expose effective hover thrust");
    CHECK(fabsf(env.observations[dynamics + 1]
            - visual_log_ratio(30.0f, 32.81f)) < 1e-7f,
        "CTBR decoder must expose effective vertical acceleration");
    CHECK(fabsf(env.observations[dynamics + 2]
            - visual_log_ratio(0.46f, 0.42f)) < 1e-7f,
        "CTBR decoder must expose effective maximum thrust");
    CHECK(fabsf(env.observations[dynamics + 3]
            - visual_log_ratio(0.40f, 0.52f)) < 1e-7f,
        "CTBR decoder must expose effective rate lag");
    CHECK(fabsf(env.observations[dynamics + 4]
            - visual_log_ratio(9.2f, 8.69f)) < 1e-7f,
        "CTBR decoder must expose effective gravity");
    CHECK(fabsf(env.observations[dynamics + 5] - tanhf(0.20f)) < 1e-7f,
        "CTBR decoder must expose effective linear drag");
    free_visual_env(&env);
    return 0;
}

static int test_ctbr_zero_action_is_hover_centered(void) {
    DroneRace env = make_visual_env();
    env.interface_mode = DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT;
    float initial_z = env.agents[0].drone.state.pos.z;
    for (int step = 0; step < 128; step++) {
        c_step(&env);
    }
    CHECK(!env.agents[0].crash,
        "zero CTBR action must not crash during the hover probe");
    CHECK(fabsf(env.agents[0].drone.state.pos.z - initial_z) < 0.10f,
        "zero CTBR action must remain centered on hover");
    CHECK(fabsf(env.agents[0].drone.state.omega.x) < 1e-7f
            && fabsf(env.agents[0].drone.state.omega.y) < 1e-7f
            && fabsf(env.agents[0].drone.state.omega.z) < 1e-7f,
        "zero CTBR action must not introduce angular motion");
    free_visual_env(&env);
    return 0;
}

static int test_gate_local_curriculum_is_random_and_actor_legal(void) {
    DroneRace env = make_visual_env();
    env.interface_mode = DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT;
    env.gate_local_start_curriculum = 1;
    env.gate_local_start_probability = 1.0f;
    env.gate_local_start_offset_min = 2.0f;
    env.gate_local_start_offset_max = 4.0f;
    env.reset_position_noise_xy = 0.0f;
    env.reset_position_noise_z = 0.0f;

    float initial_reset_count = env.log.reset_count;
    float initial_gate_local_reset_count = env.log.gate_local_reset_count;
    int sampled[2] = {0};
    for (int episode = 0; episode < 128; episode++) {
        reset_agent(&env, &env.agents[0]);
        DroneRaceAgent* agent = &env.agents[0];
        CHECK(agent->gate_local_start_sampled == 1,
            "unit local-start probability must mark every reset local");
        CHECK(agent->current_gate >= 0 && agent->current_gate < env.num_gates,
            "local curriculum must sample a valid active gate");
        sampled[agent->current_gate] += 1;
        Target* gate = agent_gate(&env, agent, agent->current_gate);
        float behind = dot3(
            sub3(gate->pos, agent->drone.state.pos), gate->normal);
        CHECK(behind >= 2.0f && behind <= 4.0f,
            "local reset must stay inside its preregistered gate offset");
        CHECK(agent->drone.state.vel.x == 0.0f
                && agent->drone.state.vel.y == 0.0f
                && agent->drone.state.vel.z == 0.0f,
            "local discovery reset must begin with neutral velocity");
        CHECK(agent->drone.state.quat.w == 1.0f
                && agent->drone.state.quat.x == 0.0f
                && agent->drone.state.quat.y == 0.0f
                && agent->drone.state.quat.z == 0.0f,
            "local discovery reset must begin with neutral attitude");
        compute_one_observation(&env, 0);
        CHECK(DRONE_RACE_VISUAL_PRIVILEGED_OFFSET
                == DRONE_RACE_VISUAL_LEGAL_OBS_SIZE,
            "gate phase must remain beyond the legal actor slice");
    }
    CHECK(sampled[0] > 0 && sampled[1] > 0,
        "local curriculum must cover more than one active gate");
    CHECK(env.log.reset_count == initial_reset_count + 128.0f,
        "reset diagnostic must count every local curriculum draw");
    CHECK(env.log.gate_local_reset_count
            == initial_gate_local_reset_count + 128.0f,
        "local reset diagnostic must count every sampled local reset");

    env.gate_local_start_probability = 0.0f;
    reset_agent(&env, &env.agents[0]);
    CHECK(env.agents[0].gate_local_start_sampled == 0
            && env.agents[0].current_gate == 0,
        "zero local-start probability must preserve the full-course reset");
    CHECK(env.log.reset_count == initial_reset_count + 129.0f
            && env.log.gate_local_reset_count
                == initial_gate_local_reset_count + 128.0f,
        "reset diagnostics must distinguish full-course from local resets");
    free_visual_env(&env);
    return 0;
}

static int test_terminal_visual_observation_precedes_reset_noop(void) {
    DroneRace env = make_visual_env();
    DroneRaceAgent* agent = &env.agents[0];
    agent->valid_run = 0;
    c_step(&env);
    CHECK(env.terminals[0] == 1.0f && agent->pending_reset == 1,
        "visual environment must retain the true terminal state for one step");
    CHECK(agent->step_count == 1,
        "terminal visual observation must be computed before reset");

    c_step(&env);
    CHECK(env.terminals[0] == 0.0f && env.rewards[0] == 0.0f,
        "step after a terminal must be a reset-only no-op");
    CHECK(agent->pending_reset == 0 && agent->step_count == 0,
        "reset-only no-op must expose a fresh episode without advancing it");
    CHECK(env.observations[DRONE_RACE_VISUAL_NEW_FRAME_OFFSET] == 1.0f,
        "reset-only no-op must expose a fresh camera observation");
    free_visual_env(&env);
    return 0;
}

static int test_centered_ordered_gate_reward(void) {
    DroneRace env = make_visual_env();
    env.w_progress = 0.0f;
    env.w_gate = 0.0f;
    env.w_ordered_gate = 30.0f;
    env.w_finish = 0.0f;
    env.w_time = 0.0f;
    DroneRaceAgent* agent = &env.agents[0];
    Target* gate = agent_gate(&env, agent, 0);
    agent->drone.state.pos = sub3(gate->pos, scalmul3(gate->normal, 0.01f));
    agent->drone.state.vel = scalmul3(gate->normal, 1.0f);
    agent->drone.prev_pos = agent->drone.state.pos;
    compute_one_observation(&env, 0);

    c_step(&env);
    CHECK(agent->current_gate == 1,
        "centered forward crossing must advance the ordered gate");
    CHECK(env.rewards[0] > 29.9f && env.rewards[0] <= 30.0f,
        "centered crossing must receive one SkyDreamer-style gate reward");
    free_visual_env(&env);
    return 0;
}

static int test_skydreamer_body_rate_penalty(void) {
    DroneRace env = {0};
    env.dt = 1.0f / 64.0f;
    env.w_body_rate = 1.0f;
    State state = {0};
    state.omega = (Vec3){1.0f, -2.0f, 3.0f};
    float expected = env.dt * 5.0e-6f * expm1f(6.0f);
    CHECK(fabsf(paper_body_rate_penalty(&env, &state) - expected) < 1e-12f,
        "body-rate cost must match the frequency-normalized paper equation");
    state.omega = (Vec3){20.0f, 0.0f, 0.0f};
    expected = env.dt * 5.0e-6f * expm1f(17.0f);
    CHECK(fabsf(paper_body_rate_penalty(&env, &state) - expected) < 1e-6f,
        "body-rate cost must cap the L1 rate at 17 rad/s");
    return 0;
}

int main(void) {
    if (test_visual_schema_and_mask()) return 1;
    if (test_legal_visual_observation_is_count_agnostic()) return 1;
    if (test_asynchronous_camera_and_action_history()) return 1;
    if (test_privileged_intervention_cannot_change_legal_slice()) return 1;
    if (test_ctbr_privileged_targets_follow_effective_plant()) return 1;
    if (test_ctbr_zero_action_is_hover_centered()) return 1;
    if (test_gate_local_curriculum_is_random_and_actor_legal()) return 1;
    if (test_terminal_visual_observation_precedes_reset_noop()) return 1;
    if (test_centered_ordered_gate_reward()) return 1;
    if (test_skydreamer_body_rate_penalty()) return 1;
    printf("drone_race vision native regressions ok\n");
    return 0;
}
