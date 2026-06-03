#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DRONE_RACE_BINDING
#include "ocean/drone_race/drone_race.c"

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

static DroneRace make_test_env_with_interface(int interface_mode) {
    DroneRace env = {0};
    env.num_agents = 1;
    env.rng = 1;
    env.dt = 1.0f / 120.0f;
    env.num_gates = 2;
    env.gate_spacing = 5.0f;
    env.gate_radius = 1.0f;
    env.gate_altitude = 1.0f;
    env.gate_lateral_amplitude = 0.0f;
    env.start_offset = 2.0f;
    env.crash_height = -10.0f;
    env.pos_bound = 100.0f;
    env.plane_cross_tolerance = 1e-3f;
    env.miss_tolerance = 0.5f;
    env.direction_min = 0.05f;
    env.strict_missed_gate = 1;
    env.max_steps = 57600;
    env.time_limit_seconds = 480.0f;
    env.w_progress = 20.0f;
    env.w_gate = 3.0f;
    env.w_finish = 30.0f;
    env.w_time = 1.0f;
    env.w_ctrl = 0.01f;
    env.invalid_penalty = 40.0f;
    env.interface_mode = interface_mode;
    env.observations = (float*)calloc(DRONE_RACE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(DRONE_RACE_NUM_ATNS, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    init(&env);
    c_reset(&env);
    return env;
}

static DroneRace make_test_env(void) {
    return make_test_env_with_interface(DRONE_RACE_INTERFACE_NATIVE_MOTOR);
}

static void free_test_env(DroneRace* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

static int test_gate_crossing_geometry(void) {
    Target gate = {
        .pos = {0.0f, 0.0f, 1.0f},
        .normal = {1.0f, 0.0f, 0.0f},
        .radius = 1.0f,
    };
    float radial = 0.0f;

    Vec3 prev_ok = {-0.1f, 0.0f, 1.0f};
    Vec3 curr_ok = {0.1f, 0.0f, 1.0f};
    CHECK(gate_crossing(prev_ok, curr_ok, &gate, 1e-3f, 0.05f, &radial),
        "in-order crossing through gate should pass");
    CHECK(radial < 1e-4f, "center crossing radial distance should be near zero");

    Vec3 prev_wrong_dir = {0.1f, 0.0f, 1.0f};
    Vec3 curr_wrong_dir = {-0.1f, 0.0f, 1.0f};
    CHECK(!gate_crossing(prev_wrong_dir, curr_wrong_dir, &gate, 1e-3f, 0.05f, NULL),
        "reverse crossing should fail");

    Vec3 prev_miss = {-0.1f, 1.5f, 1.0f};
    Vec3 curr_miss = {0.1f, 1.5f, 1.0f};
    CHECK(!gate_crossing(prev_miss, curr_miss, &gate, 1e-3f, 0.05f, &radial),
        "radial miss should fail");
    CHECK(radial > 1.0f, "miss radial distance should exceed gate radius");

    return 0;
}

static int test_motor_control_semantics(void) {
    DroneRace env = make_test_env();
    DroneRaceAgent* agent = &env.agents[0];
    float start_x = agent->drone.state.pos.x;
    float start_rpm = agent->drone.state.rpms[0];

    for (int i = 0; i < 16; i++) {
        c_step(&env);
    }

    CHECK(fabsf(agent->drone.state.pos.x - start_x) < 0.01f,
        "zero action should not directly command forward x velocity");
    CHECK(fabsf(agent->drone.state.rpms[0] - start_rpm) < 1.0f,
        "zero action should keep motor RPM near hover baseline");

    env.actions[0] = 1.0f;
    env.actions[1] = -1.0f;
    env.actions[2] = 1.0f;
    env.actions[3] = -1.0f;
    for (int i = 0; i < 8; i++) {
        c_step(&env);
    }

    float omega = norm3(agent->drone.state.omega);
    CHECK(omega > 0.01f, "asymmetric motor commands should change angular velocity");

    free_test_env(&env);
    return 0;
}

static int test_timeout_matches_qualifier_timing(void) {
    DroneRace env = make_test_env();
    env.max_steps = 4;
    env.time_limit_seconds = 4.0f / 120.0f;

    for (int i = 0; i < 4; i++) {
        c_step(&env);
    }

    CHECK(env.terminals[0] == 1.0f, "agent should terminate at max_steps/time limit");
    CHECK(env.log.n == 1.0f, "timeout should log one episode");
    CHECK(env.log.timeout == 1.0f, "timeout counter should increment");
    CHECK(fabsf(env.log.episode_length - 4.0f) < 1e-4f, "episode length should match max_steps");

    free_test_env(&env);
    return 0;
}

static int test_observation_bounds(void) {
    DroneRace env = make_test_env();
    for (int i = 0; i < DRONE_RACE_OBS_SIZE; i++) {
        CHECK(env.observations[i] >= -1.0001f && env.observations[i] <= 1.0001f,
            "drone_race observations should stay normalized for Q8 source quantization");
    }
    free_test_env(&env);
    return 0;
}

static int test_ts002_observation_contract(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW);
    float* obs = env.observations;

    CHECK(env.interface_mode == DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW,
        "TS-002 interface mode should survive init defaults");
    CHECK(DRONE_RACE_OBS_SIZE == 23, "TS-002 contract should keep the 23-float policy input size");
    CHECK(obs[10] == 1.0f, "first gate should be visible from reset in the synthetic camera contract");
    CHECK(obs[11] > 0.15f && obs[11] < 0.25f,
        "visible gate forward pose should encode the reset standoff, not raw world position");
    CHECK(fabsf(obs[12]) < 0.2f, "visible gate lateral pose should be near centered at reset");
    CHECK(obs[16] > 0.5f && obs[16] <= 1.0f,
        "visible gate apparent size should be normalized and positive");
    CHECK(obs[18] == 0.0f, "elapsed-time observation should start at zero");
    for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
        CHECK(obs[19 + k] == 0.0f, "last-action observation should reset to zero");
    }

    env.actions[0] = 0.25f;
    env.actions[1] = -0.50f;
    env.actions[2] = 0.75f;
    env.actions[3] = -1.0f;
    c_step(&env);
    obs = env.observations;
    CHECK(obs[18] > 0.0f, "elapsed-time observation should advance after a TS-002 step");
    CHECK(fabsf(obs[19] - 0.25f) < 1e-5f, "last forward command should be observable as controller state");
    CHECK(fabsf(obs[20] + 0.50f) < 1e-5f, "last lateral command should be observable as controller state");
    CHECK(fabsf(obs[21] - 0.75f) < 1e-5f, "last vertical command should be observable as controller state");
    CHECK(fabsf(obs[22] + 1.0f) < 1e-5f, "last yaw-rate command should be observable as controller state");

    free_test_env(&env);
    return 0;
}

static int test_ts002_velocity_yaw_action_contract(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW);
    DroneRaceAgent* agent = &env.agents[0];
    float hover_action[DRONE_RACE_NUM_ATNS] = {0};
    float forward_action[DRONE_RACE_NUM_ATNS] = {1.0f, 0.0f, 0.0f, 0.0f};
    float hover_motor[DRONE_RACE_NUM_ATNS] = {0};
    float forward_motor[DRONE_RACE_NUM_ATNS] = {0};

    policy_to_motor_actions(&env, agent, hover_action, hover_motor);
    policy_to_motor_actions(&env, agent, forward_action, forward_motor);

    for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
        CHECK(fabsf(hover_motor[k]) < 1e-5f,
            "zero TS-002 velocity/yaw command should map to centered hover motor action");
        CHECK(forward_motor[k] >= -1.0f && forward_motor[k] <= 1.0f,
            "TS-002 velocity/yaw controller should clamp mapped motor actions");
    }
    CHECK(fabsf(forward_motor[0] - forward_motor[1]) > 1e-5f
            || fabsf(forward_motor[2] - forward_motor[3]) > 1e-5f
            || fabsf(forward_motor[0] - forward_motor[2]) > 1e-5f,
        "forward velocity setpoint should map to a motor mix, not a raw motor command passthrough");

    free_test_env(&env);
    return 0;
}

static int test_floor_risk_diagnostics(void) {
    DroneRace env = make_test_env();
    env.crash_height = 0.0f;
    DroneRaceAgent* agent = &env.agents[0];
    agent->drone.state.pos.z = 0.001f;
    agent->drone.state.vel.z = -4.0f;

    c_step(&env);

    CHECK(env.log.n == 1.0f, "floor crash should log one episode");
    CHECK(env.log.crash_low == 1.0f, "floor crash should increment low-crash counter");
    CHECK(env.log.crash_low_floor_margin_pre > 0.0f,
        "low-crash diagnostic should preserve pre-step floor margin");
    CHECK(env.log.crash_low_ttf_pre > 0.0f && env.log.crash_low_ttf_pre < 0.01f,
        "low-crash diagnostic should report short pre-step time to floor");
    CHECK(env.log.crash_low_stop_margin_pre < 0.0f,
        "low-crash diagnostic should report negative stopping margin");
    CHECK(env.log.floor_impact_risk == 1.0f,
        "episode should be flagged as a floor impact risk");
    CHECK(env.log.floor_stop_violation == 1.0f,
        "episode should be flagged as a stopping-distance violation");

    free_test_env(&env);
    return 0;
}

int main(void) {
    if (test_gate_crossing_geometry()) return 1;
    if (test_motor_control_semantics()) return 1;
    if (test_timeout_matches_qualifier_timing()) return 1;
    if (test_observation_bounds()) return 1;
    if (test_ts002_observation_contract()) return 1;
    if (test_ts002_velocity_yaw_action_contract()) return 1;
    if (test_floor_risk_diagnostics()) return 1;
    printf("drone_race native regressions ok\n");
    return 0;
}
