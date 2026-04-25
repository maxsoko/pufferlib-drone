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

static DroneRace make_test_env(void) {
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
    env.observations = (float*)calloc(DRONE_RACE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(DRONE_RACE_NUM_ATNS, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    init(&env);
    c_reset(&env);
    return env;
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

int main(void) {
    if (test_gate_crossing_geometry()) return 1;
    if (test_motor_control_semantics()) return 1;
    if (test_timeout_matches_qualifier_timing()) return 1;
    if (test_observation_bounds()) return 1;
    printf("drone_race native regressions ok\n");
    return 0;
}
