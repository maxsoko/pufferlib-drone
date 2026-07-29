#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DRONE_RACE_BINDING
#include "ocean/drone_race/drone_race.c"
#include "src/vecenv.h"

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
    env.gate_radius_randomize_gate_index = -1;
    env.gate_altitude = 1.0f;
    env.gate_lateral_amplitude = 0.0f;
    env.start_offset = 2.0f;
    env.reset_position_noise_xy = 0.1f;
    env.reset_position_noise_z = 0.05f;
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
    env.sitl_horizontal_accel_scale = 1.0f;
    env.sitl_braking_accel_scale = 1.0f;
    env.sitl_lateral_accel_scale = 1.0f;
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

static int test_vecenv_dict_grows_without_heap_overflow(void) {
    Dict* dict = create_dict(1);
    dict_set(dict, "first", 1.0);
    dict_set(dict, "second", 2.0);
    dict_set(dict, "third", 3.0);
    CHECK(dict->size == 3 && dict->capacity >= 3,
        "vector log dictionary should grow beyond its initial capacity");
    dict_set(dict, "second", 4.0);
    CHECK(dict->size == 3 && dict_get(dict, "second")->value == 4.0,
        "updating a vector log dictionary key should not grow its size");
    free(dict->items);
    free(dict);
    return 0;
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

    Vec3 prev_slow = {-0.0005f, 0.0f, 1.0f};
    Vec3 curr_slow = {0.0005f, 0.0f, 1.0f};
    CHECK(gate_crossing(prev_slow, curr_slow, &gate, 1e-3f, 0.05f, &radial),
        "slow directed crossing must not be lost inside the plane tolerance band");

    Vec3 prev_wrong_dir = {0.1f, 0.0f, 1.0f};
    Vec3 curr_wrong_dir = {-0.1f, 0.0f, 1.0f};
    CHECK(!gate_crossing(prev_wrong_dir, curr_wrong_dir, &gate, 1e-3f, 0.05f, NULL),
        "reverse crossing should fail");

    Vec3 prev_miss = {-0.1f, 1.5f, 1.0f};
    Vec3 curr_miss = {0.1f, 1.5f, 1.0f};
    Vec3 radial_vector = {0.0f, 0.0f, 0.0f};
    CHECK(!gate_crossing_details(
            prev_miss, curr_miss, &gate, 1e-3f, 0.05f,
            &radial, &radial_vector),
        "radial miss should fail");
    CHECK(radial > 1.0f, "miss radial distance should exceed gate radius");
    CHECK(fabsf(radial_vector.y - 1.5f) < 1e-5f,
        "crossing diagnostics should preserve signed right error");
    CHECK(fabsf(radial_vector.z) < 1e-5f,
        "crossing diagnostics should preserve signed vertical error");

    return 0;
}

static int test_terminal_crossing_is_bucketed_by_gate(void) {
    DroneRace env = {0};
    DroneRaceAgent agent = {0};
    agent.terminal_crossing_sampled = 1;
    agent.terminal_crossing_gate_index = 1;
    agent.terminal_crossing_radial = 1.25f;
    agent.terminal_crossing_right = -1.0f;
    agent.terminal_crossing_vertical = 0.75f;

    add_log(&env, &agent, false);

    CHECK(env.log.terminal_gate_sampled[1] == 1.0f,
        "terminal crossing should increment its active-gate bucket");
    CHECK(fabsf(env.log.terminal_gate_radial[1] - 1.25f) < 1e-6f,
        "terminal gate bucket should preserve radial error");
    CHECK(fabsf(env.log.terminal_gate_right[1] + 1.0f) < 1e-6f,
        "terminal gate bucket should preserve signed right error");
    CHECK(fabsf(env.log.terminal_gate_vertical[1] - 0.75f) < 1e-6f,
        "terminal gate bucket should preserve signed vertical error");
    CHECK(env.log.terminal_gate_sampled[0] == 0.0f,
        "terminal crossing should not contaminate other gate buckets");
    return 0;
}

static int test_ordered_crossing_and_envelope_diagnostics_are_log_only(void) {
    DroneRace env = {0};
    DroneRaceAgent agent = {0};
    agent.ordered_gate_sampled[2] = 1;
    agent.ordered_gate_radial[2] = 0.25f;
    agent.ordered_gate_right[2] = -0.20f;
    agent.ordered_gate_vertical[2] = 0.15f;
    agent.crossing_margin_violation = 1;
    agent.action_envelope_violation = 1;
    agent.wire_rate_envelope_violation = 1;
    agent.thrust_envelope_violation = 1;

    add_log(&env, &agent, false);

    CHECK(env.log.ordered_gate_sampled[2] == 1.0f,
        "ordered crossing diagnostics should retain their gate bucket");
    CHECK(fabsf(env.log.ordered_gate_radial[2] - 0.25f) < 1e-6f
            && fabsf(env.log.ordered_gate_right[2] + 0.20f) < 1e-6f
            && fabsf(env.log.ordered_gate_vertical[2] - 0.15f) < 1e-6f,
        "ordered crossing diagnostics should preserve signed geometry");
    CHECK(env.log.ordered_gate_sampled[1] == 0.0f,
        "ordered crossing diagnostics should not contaminate another gate");
    CHECK(env.log.crossing_margin_violation == 1.0f
            && env.log.action_envelope_violation == 1.0f
            && env.log.wire_rate_envelope_violation == 1.0f
            && env.log.thrust_envelope_violation == 1.0f,
        "episode-level oracle diagnostics should aggregate without control effects");
    return 0;
}

static int test_per_gate_curriculum_radius(void) {
    DroneRace env = make_test_env();
    env.gate_radius = 0.75f;
    env.custom_gate_radius[0] = 0.0f;
    env.custom_gate_radius[1] = 2.0f;
    build_course(&env);

    CHECK(fabsf(env.gates[0].radius - 0.75f) < 1e-6f,
        "zero per-gate radius should inherit the global aperture");
    CHECK(fabsf(env.gates[1].radius - 2.0f) < 1e-6f,
        "a per-gate radius should override only its selected aperture");

    free_test_env(&env);
    return 0;
}

static int test_course_geometry_scale_preserves_x_and_interpolates_yz(void) {
    DroneRace env = make_test_env();
    env.use_custom_start = 1;
    env.custom_start_pos[1] = 2.0f;
    env.custom_start_pos[2] = -1.0f;
    env.use_custom_gate_layout = 1;
    env.custom_gate_pos[0][0] = 20.0f;
    env.custom_gate_pos[0][1] = 10.0f;
    env.custom_gate_pos[0][2] = -9.0f;
    env.course_geometry_scale = 0.25f;
    build_course(&env);

    CHECK(fabsf(env.gates[0].pos.x - 20.0f) < 1e-6f,
        "geometry curriculum must preserve official longitudinal spacing");
    CHECK(fabsf(env.gates[0].pos.y - 4.0f) < 1e-6f,
        "geometry curriculum should interpolate lateral position from the start");
    CHECK(fabsf(env.gates[0].pos.z + 3.0f) < 1e-6f,
        "geometry curriculum should interpolate vertical position from the start");

    env.course_geometry_scale = 0.0f;
    build_course(&env);
    CHECK(fabsf(env.gates[0].pos.y - 2.0f) < 1e-6f
            && fabsf(env.gates[0].pos.z + 1.0f) < 1e-6f,
        "zero geometry scale should produce a straight level course at start height");

    free_test_env(&env);
    return 0;
}

static int test_course_geometry_scale_randomization_is_per_agent_state(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.use_custom_start = 1;
    env.custom_start_pos[1] = 2.0f;
    env.custom_start_pos[2] = -1.0f;
    env.use_custom_gate_layout = 1;
    env.custom_gate_pos[0][0] = 20.0f;
    env.custom_gate_pos[0][1] = 10.0f;
    env.custom_gate_pos[0][2] = -9.0f;
    env.custom_gate_pos[1][0] = 40.0f;
    env.custom_gate_pos[1][1] = -6.0f;
    env.custom_gate_pos[1][2] = -13.0f;
    env.course_geometry_scale = 1.0f;
    build_course(&env);

    Vec3 nominal = env.gates[0].pos;
    env.course_geometry_scale_randomize = 1;
    env.course_geometry_scale_min = 0.5f;
    env.course_geometry_scale_max = 0.5f;
    c_reset(&env);

    DroneRaceAgent* agent = &env.agents[0];
    CHECK(fabsf(agent->randomized_course_geometry_scale - 0.5f) < 1e-6f,
        "each agent should retain its episode geometry sample");
    CHECK(fabsf(agent->randomized_gates[0].pos.x - 20.0f) < 1e-6f,
        "per-agent geometry sampling must preserve longitudinal timing");
    CHECK(fabsf(agent->randomized_gates[0].pos.y - 6.0f) < 1e-6f,
        "per-agent geometry sampling should interpolate lateral displacement");
    CHECK(fabsf(agent->randomized_gates[0].pos.z + 5.0f) < 1e-6f,
        "per-agent geometry sampling should interpolate vertical displacement");
    CHECK(fabsf(env.gates[0].pos.y - nominal.y) < 1e-6f
            && fabsf(env.gates[0].pos.z - nominal.z) < 1e-6f,
        "per-agent geometry sampling must not mutate exact course geometry");
    CHECK(DRONE_RACE_OBS_SIZE == 32,
        "geometry curricula must not add privileged policy observations");

    free_test_env(&env);
    return 0;
}

static int test_gate_radius_randomization_is_shared_per_episode(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.gate_radius = 0.75f;
    build_course(&env);
    env.gate_radius_randomize = 1;
    env.gate_radius_min = 2.8f;
    env.gate_radius_max = 2.8f;
    c_reset(&env);

    DroneRaceAgent* agent = &env.agents[0];
    for (int gate = 0; gate < env.num_gates; gate++) {
        CHECK(fabsf(agent->randomized_gates[gate].radius - 2.8f) < 1e-6f,
            "aperture smoothing should use one shared radius for every episode gate");
        CHECK(fabsf(env.gates[gate].radius - 0.75f) < 1e-6f,
            "aperture smoothing must not mutate the fixed evaluation course");
    }
    CHECK(DRONE_RACE_OBS_SIZE == 32,
        "aperture smoothing must not expose the sampled radius to the policy");

    free_test_env(&env);
    return 0;
}

static int test_gate_radius_randomization_can_target_one_gate(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.gate_radius = 0.95f;
    env.custom_gate_radius[0] = 0.75f;
    env.custom_gate_radius[1] = 0.875f;
    build_course(&env);
    env.gate_radius_randomize = 1;
    env.gate_radius_randomize_gate_index = 1;
    env.gate_radius_min = 0.75f;
    env.gate_radius_max = 0.75f;
    c_reset(&env);

    DroneRaceAgent* agent = &env.agents[0];
    CHECK(fabsf(agent->randomized_gates[0].radius - 0.75f) < 1e-6f,
        "targeted aperture smoothing must preserve every non-target gate");
    CHECK(fabsf(agent->randomized_gates[1].radius - 0.75f) < 1e-6f,
        "targeted aperture smoothing should replace only the selected gate");
    CHECK(fabsf(env.gates[1].radius - 0.875f) < 1e-6f,
        "targeted aperture smoothing must not mutate exact evaluation geometry");

    free_test_env(&env);
    return 0;
}

static int test_gate_radius_profile_mix_preserves_exact_anchor_and_target(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.gate_radius = 0.95f;
    env.custom_gate_radius[0] = 0.75f;
    env.custom_gate_radius[1] = 0.875f;
    build_course(&env);
    env.gate_radius_profile_mix = 1;
    env.gate_radius_profile_mix_target = 0.75f;

    env.gate_radius_profile_mix_probability = 0.0f;
    reset_agent(&env, &env.agents[0]);
    CHECK(fabsf(env.agents[0].randomized_gates[0].radius - 0.75f) < 1e-6f
            && fabsf(env.agents[0].randomized_gates[1].radius - 0.875f) < 1e-6f,
        "aperture anchor episode must retain exact per-gate radii");

    env.gate_radius_profile_mix_probability = 1.0f;
    reset_agent(&env, &env.agents[0]);
    for (int gate = 0; gate < env.num_gates; gate++) {
        CHECK(fabsf(env.agents[0].randomized_gates[gate].radius - 0.75f) < 1e-6f,
            "aperture target episode must set every gate to the exact target");
    }
    CHECK(fabsf(env.gates[1].radius - 0.875f) < 1e-6f,
        "aperture profile mix must not mutate the fixed anchor course");

    free_test_env(&env);
    return 0;
}

static int test_gate_radius_profile_mix_preserves_vehicle_rng(void) {
    DroneRace reference = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace mixed = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    reference.rng = 3385u;
    mixed.rng = 3385u;
    mixed.gate_radius_profile_mix = 1;
    mixed.gate_radius_profile_mix_probability = 1.0f;
    mixed.gate_radius_profile_mix_target = 0.75f;
    reset_agent(&reference, &reference.agents[0]);
    reset_agent(&mixed, &mixed.agents[0]);

    CHECK(fabsf(reference.agents[0].drone.state.pos.x
            - mixed.agents[0].drone.state.pos.x) < 1e-6f
            && fabsf(reference.agents[0].drone.state.pos.y
                - mixed.agents[0].drone.state.pos.y) < 1e-6f
            && fabsf(reference.agents[0].drone.state.pos.z
                - mixed.agents[0].drone.state.pos.z) < 1e-6f,
        "aperture profile sampling must preserve paired vehicle position RNG");
    CHECK(fabsf(reference.agents[0].drone.state.quat.w
            - mixed.agents[0].drone.state.quat.w) < 1e-6f
            && fabsf(reference.agents[0].drone.state.quat.x
                - mixed.agents[0].drone.state.quat.x) < 1e-6f
            && fabsf(reference.agents[0].drone.state.quat.y
                - mixed.agents[0].drone.state.quat.y) < 1e-6f
            && fabsf(reference.agents[0].drone.state.quat.z
                - mixed.agents[0].drone.state.quat.z) < 1e-6f,
        "aperture profile sampling must preserve paired vehicle attitude RNG");

    free_test_env(&reference);
    free_test_env(&mixed);
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

static int test_strict_near_miss_terminates_at_gate_plane(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    Target* gate = &env.gates[0];
    agent->drone.state.pos = (Vec3){
        gate->pos.x - 0.01f,
        gate->pos.y + gate->radius + 0.10f,
        gate->pos.z,
    };
    agent->drone.state.vel = (Vec3){2.0f, 0.0f, 0.0f};
    agent->drone.prev_pos = agent->drone.state.pos;

    c_step(&env);

    CHECK(env.terminals[0] == 1.0f,
        "strict out-of-aperture plane crossing should terminate immediately");
    CHECK(env.log.n == 1.0f && env.log.missed_gate == 1.0f,
        "strict near miss should be logged as a missed gate");
    CHECK(env.log.gates_passed == 0.0f,
        "strict near miss must not receive gate credit");

    free_test_env(&env);
    return 0;
}

static int test_gate2_terminal_crossing_diagnostics_are_conditioned(void) {
    DroneRace env = make_test_env();
    DroneRaceAgent* agent = &env.agents[0];
    record_terminal_crossing(
        agent, 2, 3.5f, (Vec3){0.0f, 2.0f, -2.8722813f});
    add_log(&env, agent, false);

    CHECK(env.log.gate2_terminal_crossing_sampled == 1.0f,
        "gate-2 crossing diagnostics should retain their conditioning count");
    CHECK(fabsf(env.log.gate2_terminal_crossing_radial - 3.5f) < 1e-6f,
        "gate-2 crossing diagnostics should retain radial error");
    CHECK(fabsf(env.log.gate2_terminal_crossing_right - 2.0f) < 1e-6f,
        "gate-2 crossing diagnostics should retain signed right error");
    CHECK(fabsf(env.log.gate2_terminal_crossing_vertical + 2.8722813f) < 1e-6f,
        "gate-2 crossing diagnostics should retain signed vertical error");

    agent->gate2_crossing_sampled = 1;
    agent->gate2_crossing_radial = 0.5f;
    agent->gate2_crossing_right = -0.3f;
    agent->gate2_crossing_vertical = 0.4f;
    agent->gate2_exit_vx = 20.0f;
    agent->gate2_exit_vy = 3.0f;
    agent->gate2_exit_vz = -5.0f;
    add_log(&env, agent, false);

    CHECK(env.log.gate2_crossing_sampled == 1.0f,
        "successful gate-2 diagnostics should retain their conditioning count");
    CHECK(fabsf(env.log.gate2_crossing_radial - 0.5f) < 1e-6f,
        "successful gate-2 diagnostics should retain radial error");
    CHECK(fabsf(env.log.gate2_crossing_right + 0.3f) < 1e-6f,
        "successful gate-2 diagnostics should retain signed right error");
    CHECK(fabsf(env.log.gate2_crossing_vertical - 0.4f) < 1e-6f,
        "successful gate-2 diagnostics should retain signed vertical error");
    CHECK(fabsf(env.log.gate2_exit_vx - 20.0f) < 1e-6f
            && fabsf(env.log.gate2_exit_vy - 3.0f) < 1e-6f
            && fabsf(env.log.gate2_exit_vz + 5.0f) < 1e-6f,
        "successful gate-2 diagnostics should retain exit velocity");
    CHECK(env.log.gate2_downstream_failure_sampled == 1.0f
            && env.log.gate2_completion_sampled == 0.0f,
        "gate-2 transitions should be partitioned by eventual completion");
    CHECK(fabsf(env.log.gate2_downstream_failure_crossing_radial - 0.5f) < 1e-6f
            && fabsf(env.log.gate2_downstream_failure_exit_vy - 3.0f) < 1e-6f,
        "downstream failures should retain their gate-2 transition state");

    add_log(&env, agent, true);
    CHECK(env.log.gate2_completion_sampled == 1.0f,
        "course completions should retain their gate-2 transition count");
    CHECK(fabsf(env.log.gate2_completion_crossing_radial - 0.5f) < 1e-6f
            && fabsf(env.log.gate2_completion_exit_vy - 3.0f) < 1e-6f,
        "course completions should retain their gate-2 transition state");

    free_test_env(&env);
    return 0;
}

static int test_gate1_action_diagnostics_capture_early_window(void) {
    DroneRace env = make_test_env();
    DroneRaceAgent* agent = &env.agents[0];
    agent->current_gate = 1;
    agent->gate1_action_steps = 59.0f;
    agent->gate1_early_action_steps = 59.0f;
    env.actions[0] = 0.2f;
    env.actions[1] = -0.3f;
    env.actions[2] = 0.4f;
    env.actions[3] = -0.5f;

    c_step(&env);
    CHECK(agent->gate1_action_steps == 60.0f,
        "gate-1 diagnostics should count every active-phase action");
    CHECK(agent->gate1_early_action_steps == 60.0f,
        "gate-1 early diagnostics should include the first 60 actions");
    CHECK(fabsf(agent->gate1_early_action_roll + 0.3f) < 1e-6f,
        "gate-1 early diagnostics should accumulate action components");

    c_step(&env);
    CHECK(agent->gate1_action_steps == 61.0f,
        "gate-1 full diagnostics should continue after the early window");
    CHECK(agent->gate1_early_action_steps == 60.0f,
        "gate-1 early diagnostics should stop after 60 actions");
    CHECK(fabsf(agent->gate1_action_roll + 0.6f) < 1e-6f,
        "gate-1 full diagnostics should accumulate after the early window");
    CHECK(fabsf(agent->gate1_early_action_roll + 0.3f) < 1e-6f,
        "gate-1 early action sum should remain frozen after its window");

    add_log(&env, agent, false);
    CHECK(env.log.gate1_action_steps == 61.0f
            && env.log.gate1_early_action_steps == 60.0f,
        "gate-1 action diagnostics should be included in episode logs");
    free_test_env(&env);
    return 0;
}

static int test_stage_transition_reproduces_official_entry_speed(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.sitl_gate_transition_min_forward_speed = 7.0f;
    DroneRaceAgent* agent = &env.agents[0];
    Target* gate = &env.gates[0];
    agent->drone.state.pos = (Vec3){
        gate->pos.x - 0.01f,
        gate->pos.y,
        gate->pos.z,
    };
    agent->drone.state.vel = (Vec3){2.0f, 0.0f, 0.0f};
    agent->drone.prev_pos = agent->drone.state.pos;

    c_step(&env);

    CHECK(env.terminals[0] == 0.0f && agent->current_gate == 1,
        "first gate should transition into the second-gate curriculum");
    CHECK(agent->drone.state.vel.x >= 6.999f,
        "configured transition should reproduce official minimum entry speed");

    free_test_env(&env);
    return 0;
}

static int test_transition_speed_ramp_is_capped_and_opt_in(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.dt = 0.1f;
    env.sitl_gate_transition_min_forward_speed = 6.0f;
    env.sitl_gate_transition_max_accel_m_s2 = 2.0f;
    agent->current_gate = 1;
    agent->drone.state.vel = (Vec3){3.8f, 0.0f, 0.0f};

    apply_transition_speed_ramp(&env, agent);
    CHECK(fabsf(agent->drone.state.vel.x - 4.0f) < 1e-6f,
        "transition ramp should cap one-step forward acceleration");
    for (int step = 0; step < 20; step++) {
        apply_transition_speed_ramp(&env, agent);
    }
    CHECK(fabsf(agent->drone.state.vel.x - 6.0f) < 1e-6f,
        "transition ramp should converge to the configured speed floor");

    agent->current_gate = 0;
    agent->drone.state.vel.x = 3.8f;
    apply_transition_speed_ramp(&env, agent);
    CHECK(fabsf(agent->drone.state.vel.x - 3.8f) < 1e-6f,
        "transition ramp should not accelerate before the first gate");

    env.sitl_gate_transition_max_accel_m_s2 = 0.0f;
    agent->current_gate = 1;
    apply_transition_speed_ramp(&env, agent);
    CHECK(fabsf(agent->drone.state.vel.x - 3.8f) < 1e-6f,
        "zero acceleration should preserve legacy instantaneous-floor mode");

    env.num_gates = 3;
    env.sitl_gate_transition_from_gate_index = 2;
    agent->current_gate = 1;
    CHECK(!transition_speed_floor_active(&env, agent),
        "floor activation index 2 should leave the Gate-0 to Gate-1 leg unchanged");
    agent->current_gate = 2;
    CHECK(transition_speed_floor_active(&env, agent),
        "floor activation index 2 should enable the Gate-1 to Gate-2 leg");
    free_test_env(&env);
    return 0;
}

static int test_transition_speed_can_preserve_outgoing_direction(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    agent->current_gate = 1;
    agent->drone.state.vel = (Vec3){3.0f, -1.0f, -2.0f};

    raise_transition_forward_speed(&env, agent, 6.0f);
    CHECK(fabsf(agent->drone.state.vel.x - 6.0f) < 1e-6f
            && fabsf(agent->drone.state.vel.y + 1.0f) < 1e-6f
            && fabsf(agent->drone.state.vel.z + 2.0f) < 1e-6f,
        "legacy transition floor should change only gate-normal velocity");

    env.sitl_gate_transition_preserve_velocity_direction = 1;
    agent->drone.state.vel = (Vec3){3.0f, -1.0f, -2.0f};
    raise_transition_forward_speed(&env, agent, 6.0f);
    CHECK(fabsf(agent->drone.state.vel.x - 6.0f) < 1e-6f
            && fabsf(agent->drone.state.vel.y + 2.0f) < 1e-6f
            && fabsf(agent->drone.state.vel.z + 4.0f) < 1e-6f,
        "direction-preserving floor should scale the complete outgoing velocity");

    free_test_env(&env);
    return 0;
}

static int test_transition_speed_impulse_can_wait_for_gate_association(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.sitl_gate_transition_min_forward_speed = 6.0f;
    agent->current_gate = 1;
    agent->drone.state.vel = (Vec3){3.0f, -1.0f, -2.0f};
    agent->transition_speed_floor_delay_steps_remaining = 1;

    apply_pending_transition_speed_floor(&env, agent);
    CHECK(agent->transition_speed_floor_delay_steps_remaining == 0,
        "one-step floor delay should expire on the next control step");
    CHECK(fabsf(agent->drone.state.vel.x - 6.0f) < 1e-6f
            && fabsf(agent->drone.state.vel.y + 1.0f) < 1e-6f
            && fabsf(agent->drone.state.vel.z + 2.0f) < 1e-6f,
        "expired delay should apply the unchanged legacy floor impulse");

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
    CHECK(env.log.closest_gate_range > 0.0f,
        "terminal diagnostics should retain closest gate range");

    free_test_env(&env);
    return 0;
}

static int test_evaluation_episode_offset_advances_reset_sequence_once(void) {
    DroneRace shifted = make_test_env();
    DroneRace reference = make_test_env();
    shifted.rng = 17;
    reference.rng = 17;
    shifted.evaluation_episode_offset = 2;
    shifted.evaluation_episode_offset_applied = 0;

    c_reset(&shifted);
    for (int reset = 0; reset < 3; reset++) {
        reset_agent(&reference, &reference.agents[0]);
    }

    CHECK(shifted.rng == reference.rng,
        "evaluation offset should consume the requested reset sequence");
    CHECK(fabsf(shifted.agents[0].drone.state.pos.x
            - reference.agents[0].drone.state.pos.x) < 1e-6f,
        "evaluation offset should select the expected reset sample");
    CHECK(fabsf(shifted.agents[0].drone.state.pos.y
            - reference.agents[0].drone.state.pos.y) < 1e-6f,
        "evaluation offset should preserve the expected lateral sample");

    unsigned int rng_after_first_reset = shifted.rng;
    c_reset(&shifted);
    reset_agent(&reference, &reference.agents[0]);
    CHECK(shifted.rng == reference.rng && shifted.rng != rng_after_first_reset,
        "evaluation offset must not be reapplied on a later reset");

    free_test_env(&shifted);
    free_test_env(&reference);
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

static int test_gate_position_randomization_is_per_agent_state(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    Vec3 nominal = env.gates[0].pos;
    env.gate_position_domain_randomize = 1;
    env.gate_position_domain_randomize_probability = 1.0f;
    env.gate_position_randomize_from_index = 0;
    env.gate_position_jitter_y = 2.0f;

    c_reset(&env);
    Vec3 randomized = env.agents[0].randomized_gates[0].pos;

    CHECK(fabsf(env.gates[0].pos.y - nominal.y) < 1e-6f,
        "course randomization must not mutate nominal gate coordinates");
    CHECK(randomized.y >= nominal.y - 2.0f && randomized.y <= nominal.y + 2.0f,
        "per-agent randomized gate must remain inside configured bounds");
    CHECK(fabsf(randomized.y - nominal.y) > 1e-6f,
        "enabled gate jitter should produce per-agent course state");
    CHECK(env.observations[10] == 1.0f,
        "randomized visible gate should still use the official camera contract");

    free_test_env(&env);
    return 0;
}

static int test_gate_position_randomization_can_mix_nominal_anchor_episodes(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    Vec3 nominal = env.gates[0].pos;
    env.gate_position_domain_randomize = 1;
    env.gate_position_randomize_from_index = 0;
    env.gate_position_jitter_y = 2.0f;

    env.gate_position_domain_randomize_probability = 0.0f;
    env.gate_position_randomized_radius = 1.5f;
    c_reset(&env);
    CHECK(fabsf(env.agents[0].randomized_gates[0].pos.y - nominal.y) < 1e-6f,
        "zero position-randomization probability should produce an exact nominal anchor");
    CHECK(fabsf(env.agents[0].randomized_gates[0].radius - env.gates[0].radius) < 1e-6f,
        "nominal anchor episodes must preserve the exact configured aperture");

    env.gate_position_domain_randomize_probability = 1.0f;
    c_reset(&env);
    CHECK(fabsf(env.agents[0].randomized_gates[0].pos.y - nominal.y) > 1e-6f,
        "unit position-randomization probability should preserve legacy jitter behavior");
    CHECK(fabsf(env.agents[0].randomized_gates[0].radius - 1.5f) < 1e-6f,
        "randomized episodes should apply the configured teaching aperture");

    int nominal_episodes = 0;
    int randomized_episodes = 0;
    env.gate_position_domain_randomize_probability = 0.5f;
    for (int episode = 0; episode < 128; episode++) {
        reset_agent(&env, &env.agents[0]);
        float delta = fabsf(
            env.agents[0].randomized_gates[0].pos.y - nominal.y);
        nominal_episodes += delta < 1e-6f;
        randomized_episodes += delta >= 1e-6f;
    }
    CHECK(nominal_episodes > 0 && randomized_episodes > 0,
        "half-probability anchor mixture should sample nominal and randomized courses");

    free_test_env(&env);
    return 0;
}

static int test_gate_position_randomization_preserves_start_rng_stream(void) {
    DroneRace nominal = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace randomized = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    nominal.rng = 3385;
    randomized.rng = 3385;
    randomized.gate_position_domain_randomize = 1;
    randomized.gate_position_domain_randomize_probability = 1.0f;
    randomized.gate_position_randomize_from_index = 1;
    randomized.gate_position_jitter_x = 2.0f;
    randomized.gate_position_jitter_y = 6.0f;
    randomized.gate_position_jitter_z = 3.0f;

    c_reset(&nominal);
    c_reset(&randomized);
    State* nominal_state = &nominal.agents[0].drone.state;
    State* randomized_state = &randomized.agents[0].drone.state;
    CHECK(fabsf(nominal_state->pos.x - randomized_state->pos.x) < 1e-7f
            && fabsf(nominal_state->pos.y - randomized_state->pos.y) < 1e-7f
            && fabsf(nominal_state->pos.z - randomized_state->pos.z) < 1e-7f,
        "late-gate position randomization must not shift the initial position RNG stream");
    CHECK(fabsf(nominal_state->quat.w - randomized_state->quat.w) < 1e-7f
            && fabsf(nominal_state->quat.x - randomized_state->quat.x) < 1e-7f
            && fabsf(nominal_state->quat.y - randomized_state->quat.y) < 1e-7f
            && fabsf(nominal_state->quat.z - randomized_state->quat.z) < 1e-7f,
        "late-gate position randomization must not shift the initial attitude RNG stream");

    free_test_env(&nominal);
    free_test_env(&randomized);
    return 0;
}

static int test_ts002_observation_contract(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW);
    float* obs = env.observations;

    CHECK(env.interface_mode == DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW,
        "TS-002 interface mode should survive init defaults");
    CHECK(DRONE_RACE_OBS_SIZE == 32,
        "full-policy contract should append observable phase-gated adapter inputs");
    CHECK(obs[0] == 0.0f && obs[1] == 0.0f && obs[2] == 0.0f,
        "camera gate-motion rates should initialize to zero at reset");
    CHECK(obs[10] == 1.0f, "first gate should be visible from reset in the synthetic camera contract");
    CHECK(obs[11] > 0.15f && obs[11] < 0.25f,
        "visible gate forward pose should encode the reset standoff, not raw world position");
    CHECK(fabsf(obs[12]) < 0.2f, "visible gate lateral pose should be near centered at reset");
    CHECK(obs[16] > 0.5f && obs[16] <= 1.0f,
        "visible gate apparent size should be normalized and positive");
    CHECK(fabsf(obs[17] - (1.0f - 0.5f * (fabsf(obs[14]) + fabsf(obs[15])))) < 1e-6f,
        "native gate centering quality must match the official observable contract");
    CHECK(obs[18] == 0.0f, "elapsed-time observation should start at zero");
    for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
        CHECK(obs[19 + k] == 0.0f, "last-action observation should reset to zero");
    }
    for (int i = 23; i < DRONE_RACE_OBS_SIZE; i++) {
        CHECK(obs[i] == 0.0f,
            "disabled race status should append neutral adapter values");
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

static int test_ts002_gate_track_holds_across_detector_dropout(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    float initial_forward = env.observations[11];
    float initial_right = env.observations[12];
    CHECK(env.observations[10] == 1.0f,
        "gate must be associated before testing detector dropout");

    env.sitl_gate_obs_dropout_prob = 1.0f;
    c_step(&env);

    CHECK(env.observations[10] == 1.0f,
        "an unchanged official gate identity should hold across a missing frame");
    CHECK(fabsf(env.observations[11] - initial_forward) < 0.05f,
        "held gate range should remain near the last associated camera pose");
    CHECK(fabsf(env.observations[12] - initial_right) < 0.05f,
        "held gate lateral pose should remain near the last association");

    env.sitl_gate_obs_dropout_prob = 0.0f;
    env.sitl_gate_obs_dropout_range_m = 10.0f;
    env.sitl_gate_obs_dropout_from_index = 0;
    c_step(&env);
    CHECK(env.observations[10] == 1.0f,
        "close-range detector loss should hold the last official gate association");
    CHECK(fabsf(env.observations[11] - initial_forward) < 0.05f,
        "close-range hold should not leak the native gate position");

    free_test_env(&env);
    return 0;
}

static int test_ts002_gate_track_predicts_across_detector_dropout_when_enabled(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    CHECK(env.observations[10] == 1.0f,
        "gate must be associated before testing predictive dropout");
    float initial_forward = env.observations[11];
    env.sitl_gate_motion_predict_dropout = 1;
    env.sitl_gate_obs_dropout_prob = 1.0f;
    env.dt = 0.5f;
    env.agents[0].gate_motion_rate_world.x = -2.0f;

    compute_one_observation(&env, 0);

    CHECK(env.observations[10] == 1.0f,
        "predictive dropout must retain the official gate identity");
    CHECK(env.observations[11] < initial_forward - 0.05f,
        "predictive dropout must advance the held range with observable rate");

    free_test_env(&env);
    return 0;
}

static int test_ts002_gate_track_integrates_roll_acceleration_when_enabled(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    CHECK(env.observations[10] == 1.0f,
        "gate must be associated before testing control-aware prediction");
    env.sitl_gate_motion_predict_dropout = 1;
    env.sitl_gate_motion_control_accel_gain = 4.295f;
    env.sitl_gate_obs_sample_interval_steps = 2;
    env.agents[0].step_count = 1;
    env.dt = 0.5f;
    env.agents[0].gate_motion_rate_world = (Vec3){0.0f, 0.0f, 0.0f};
    float initial_world_y = env.agents[0].gate_motion_raw_world.y;
    env.agents[0].drone.state.quat = euler_to_quat(
        (EulerAngles){-0.4f, 0.0f, 0.0f});

    compute_one_observation(&env, 0);

    CHECK(env.agents[0].gate_motion_rate_world.y < -0.8f,
        "negative observed roll must drive predicted gate rate toward world left");
    CHECK(env.agents[0].gate_motion_raw_world.y < initial_world_y,
        "control-aware dropout must advance lateral pose with integrated acceleration");

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

static int test_sitl_attitude_plant_closes_absolute_setpoint(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    float pitch_action[DRONE_RACE_NUM_ATNS] = {1.0f, 0.0f, 0.0f, 0.0f};

    agent->drone.state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    agent->drone.state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    EulerAngles att0 = quat_to_euler(agent->drone.state.quat);

    for (int i = 0; i < 120; i++) {
        memcpy(env.actions, pitch_action, sizeof(pitch_action));
        c_step(&env);
    }

    EulerAngles att1 = quat_to_euler(agent->drone.state.quat);
    float pitch_delta = att1.pitch - att0.pitch;
    CHECK(pitch_delta > 0.25f,
        "positive policy pitch should reach a positive absolute attitude setpoint");
    CHECK(fabsf(att1.pitch - env.sitl_max_pitch_rad) < 0.15f,
        "native mode 2 should close the same absolute attitude target as the live runner");
    CHECK(fabsf(agent->drone.state.omega.y) < 0.25f,
        "closed-loop attitude rate should settle instead of integrating forever");

    free_test_env(&env);
    return 0;
}

static int test_sitl_attitude_hover_thrust(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    float hover_cmd[DRONE_RACE_NUM_ATNS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float z0 = agent->drone.state.pos.z;

    for (int i = 0; i < 240; i++) {
        memcpy(env.actions, hover_cmd, sizeof(hover_cmd));
        c_step(&env);
    }

    CHECK(fabsf(agent->drone.state.pos.z - z0) < 0.25f,
        "hover thrust action should keep altitude near equilibrium");
    CHECK(fabsf(agent->drone.state.vel.z) < 0.5f,
        "hover thrust action should keep vertical velocity bounded");

    free_test_env(&env);
    return 0;
}

static int test_sitl_attitude_forward_sign_matches_official_sim(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    float forward_cmd[DRONE_RACE_NUM_ATNS] = {-0.50f, 0.0f, 0.0f, 0.0f};
    float x0 = agent->drone.state.pos.x;

    for (int i = 0; i < 240; i++) {
        memcpy(env.actions, forward_cmd, sizeof(forward_cmd));
        c_step(&env);
    }

    CHECK(agent->drone.state.pos.x > x0 + 0.5f,
        "negative pitch command must accelerate forward, matching the official simulator");

    free_test_env(&env);
    return 0;
}

static int test_sitl_attitude_roll_sign_matches_official_sim(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    float right_cmd[DRONE_RACE_NUM_ATNS] = {0.0f, 0.50f, 0.0f, 0.0f};
    float y0 = agent->drone.state.pos.y;

    for (int i = 0; i < 240; i++) {
        memcpy(env.actions, right_cmd, sizeof(right_cmd));
        c_step(&env);
    }

    CHECK(agent->drone.state.pos.y < y0 - 0.5f,
        "positive roll command must accelerate toward negative Y like the official simulator");

    free_test_env(&env);
    return 0;
}

static int test_sitl_stage_lock_holds_hover_heading_and_lateral(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    float exploratory_cmd[DRONE_RACE_NUM_ATNS] = {0.0f, 1.0f, 1.0f, 1.0f};
    float z0 = agent->drone.state.pos.z;
    float y0 = agent->drone.state.pos.y;

    env.sitl_lock_hover_thrust = 1;
    env.sitl_lock_roll = 1;
    env.sitl_lock_yaw = 1;
    for (int i = 0; i < 240; i++) {
        memcpy(env.actions, exploratory_cmd, sizeof(exploratory_cmd));
        c_step(&env);
    }

    EulerAngles attitude = quat_to_euler(agent->drone.state.quat);
    CHECK(fabsf(agent->drone.state.pos.z - z0) < 0.25f,
        "Stage B thrust lock should hold the calibrated hover point");
    CHECK(fabsf(attitude.yaw) < 0.05f,
        "Stage B yaw lock should hold the official start heading");
    CHECK(fabsf(agent->drone.state.pos.y - y0) < 0.05f,
        "Stage A roll lock should prevent lateral exploration drift");

    free_test_env(&env);
    return 0;
}

static int test_sitl_hover_lock_compensates_forward_tilt(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    float forward_cmd[DRONE_RACE_NUM_ATNS] = {-0.75f, 0.0f, 1.0f, 0.0f};
    float z0 = agent->drone.state.pos.z;

    env.sitl_lock_hover_thrust = 1;
    for (int i = 0; i < 1200; i++) {
        move_drone_sitl_plant(&env, agent, forward_cmd);
    }

    CHECK(fabsf(agent->drone.state.pos.z - z0) < 0.35f,
        "hover lock should compensate vertical lift lost to forward tilt");

    free_test_env(&env);
    return 0;
}

static int test_sitl_teacher_reward_prefers_forward_contract(void) {
    DroneRace preferred = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 200.0f;
    idle.w_action_teacher = 200.0f;
    preferred.teacher_pitch_action = -0.5f;
    idle.teacher_pitch_action = -0.5f;
    preferred.actions[0] = -0.5f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(preferred.rewards[0] > idle.rewards[0] + 0.2f,
        "Stage A teacher reward should favor the official forward pitch action");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_reward_prefers_descent_to_lower_gate(void) {
    DroneRace preferred = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 100.0f;
    idle.w_action_teacher = 100.0f;
    preferred.teacher_pitch_action = 0.0f;
    idle.teacher_pitch_action = 0.0f;
    preferred.teacher_roll_per_m = 0.0f;
    idle.teacher_roll_per_m = 0.0f;
    preferred.teacher_thrust_per_m = 0.1f;
    idle.teacher_thrust_per_m = 0.1f;
    preferred.gates[0].pos.z = preferred.agents[0].drone.state.pos.z - 2.0f;
    idle.gates[0].pos.z = idle.agents[0].drone.state.pos.z - 2.0f;
    preferred.actions[2] = -0.2f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(preferred.rewards[0] > idle.rewards[0] + 0.02f,
        "teacher reward should favor reduced thrust when the visible gate is below");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_reward_supports_thrust_bias(void) {
    DroneRace preferred = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 100.0f;
    idle.w_action_teacher = 100.0f;
    preferred.teacher_pitch_weight = 0.0f;
    idle.teacher_pitch_weight = 0.0f;
    preferred.teacher_roll_weight = 0.0f;
    idle.teacher_roll_weight = 0.0f;
    preferred.teacher_thrust_bias = -0.2f;
    idle.teacher_thrust_bias = -0.2f;
    preferred.actions[2] = -0.2f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(preferred.rewards[0] > idle.rewards[0] + 0.02f,
        "teacher reward should support a gate-local hover-centered thrust bias");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_reward_can_arrest_observed_descent_rate(void) {
    DroneRace preferred = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 100.0f;
    idle.w_action_teacher = 100.0f;
    preferred.w_ctrl = 0.0f;
    idle.w_ctrl = 0.0f;
    preferred.teacher_pitch_weight = 0.0f;
    idle.teacher_pitch_weight = 0.0f;
    preferred.teacher_roll_weight = 0.0f;
    idle.teacher_roll_weight = 0.0f;
    preferred.teacher_thrust_weight = 1.0f;
    idle.teacher_thrust_weight = 1.0f;
    preferred.teacher_thrust_rate_per_m_s = 0.25f;
    idle.teacher_thrust_rate_per_m_s = 0.25f;
    preferred.agents[0].gate_motion_valid = 1;
    idle.agents[0].gate_motion_valid = 1;
    preferred.agents[0].gate_motion_gate_index = 0;
    idle.agents[0].gate_motion_gate_index = 0;
    preferred.agents[0].gate_motion_rate_world.z = 2.0f;
    idle.agents[0].gate_motion_rate_world.z = 2.0f;
    preferred.actions[2] = 0.5f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(preferred.rewards[0] > idle.rewards[0] + 0.2f,
        "teacher reward should favor thrust that arrests observed descent rate");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_reward_supports_roll_bias(void) {
    DroneRace preferred = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 100.0f;
    idle.w_action_teacher = 100.0f;
    preferred.teacher_pitch_weight = 0.0f;
    idle.teacher_pitch_weight = 0.0f;
    preferred.teacher_roll_weight = 1.0f;
    idle.teacher_roll_weight = 1.0f;
    preferred.teacher_thrust_weight = 0.0f;
    idle.teacher_thrust_weight = 0.0f;
    preferred.teacher_roll_bias = 0.2f;
    idle.teacher_roll_bias = 0.2f;
    preferred.teacher_roll_per_m = 0.0f;
    idle.teacher_roll_per_m = 0.0f;
    preferred.actions[1] = 0.2f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(preferred.rewards[0] > idle.rewards[0] + 0.02f,
        "teacher reward should support a gate-local roll bias");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_can_shape_only_lateral_after_gate_handoff(void) {
    DroneRace preferred = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 100.0f;
    idle.w_action_teacher = 100.0f;
    preferred.teacher_from_gate_index = 1;
    idle.teacher_from_gate_index = 1;
    preferred.teacher_pitch_weight = 0.0f;
    idle.teacher_pitch_weight = 0.0f;
    preferred.teacher_roll_weight = 1.0f;
    idle.teacher_roll_weight = 1.0f;
    preferred.teacher_roll_until_gate_index = 2;
    idle.teacher_roll_until_gate_index = 2;
    preferred.teacher_thrust_weight = 0.0f;
    idle.teacher_thrust_weight = 0.0f;
    preferred.teacher_roll_per_m = 0.2f;
    idle.teacher_roll_per_m = 0.2f;
    preferred.agents[0].current_gate = 1;
    idle.agents[0].current_gate = 1;
    preferred.gates[1].pos.y = -1.0f;
    idle.gates[1].pos.y = -1.0f;
    preferred.actions[1] = 0.2f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(preferred.rewards[0] > idle.rewards[0] + 0.02f,
        "lateral-only teacher should prefer steering toward a later gate");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_roll_can_stop_before_next_gate(void) {
    DroneRace preferred = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace idle = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    preferred.w_action_teacher = 100.0f;
    idle.w_action_teacher = 100.0f;
    preferred.w_ctrl = 0.0f;
    idle.w_ctrl = 0.0f;
    preferred.teacher_pitch_weight = 0.0f;
    idle.teacher_pitch_weight = 0.0f;
    preferred.teacher_roll_weight = 1.0f;
    idle.teacher_roll_weight = 1.0f;
    preferred.teacher_thrust_weight = 0.0f;
    idle.teacher_thrust_weight = 0.0f;
    preferred.teacher_roll_from_gate_index = 1;
    idle.teacher_roll_from_gate_index = 1;
    preferred.teacher_roll_until_gate_index = 1;
    idle.teacher_roll_until_gate_index = 1;
    preferred.agents[0].current_gate = 1;
    idle.agents[0].current_gate = 1;
    preferred.gates[1].pos.y = -1.0f;
    idle.gates[1].pos.y = -1.0f;
    preferred.actions[1] = 0.2f;

    c_step(&preferred);
    c_step(&idle);

    CHECK(fabsf(preferred.rewards[0] - idle.rewards[0]) < 0.01f,
        "roll teacher should stop at its exclusive gate bound");

    free_test_env(&preferred);
    free_test_env(&idle);
    return 0;
}

static int test_sitl_teacher_intervention_blends_only_active_channels(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.teacher_action_blend = 1.0f;
    env.teacher_pitch_from_gate_index = 2;
    env.teacher_roll_from_gate_index = 0;
    env.teacher_roll_until_gate_index = 2;
    env.teacher_thrust_from_gate_index = 0;
    env.teacher_roll_bias = 0.4f;
    env.teacher_roll_per_m = 0.0f;
    env.teacher_thrust_bias = -0.3f;
    env.teacher_thrust_per_m = 0.0f;
    env.actions[0] = 0.1f;
    env.actions[1] = -0.2f;
    env.actions[2] = 0.2f;
    env.actions[3] = -0.4f;

    c_step(&env);

    CHECK(fabsf(env.observations[19] - 0.1f) < 1e-5f,
        "inactive pitch intervention must preserve the raw policy command");
    CHECK(fabsf(env.observations[20] - 0.4f) < 1e-5f,
        "active roll intervention should execute the teacher command");
    CHECK(fabsf(env.observations[21] + 0.3f) < 1e-5f,
        "active thrust intervention should execute the teacher command");
    CHECK(fabsf(env.observations[22] + 0.4f) < 1e-5f,
        "yaw must remain owned by the raw recurrent policy");

    free_test_env(&env);
    return 0;
}

static int test_sitl_teacher_yaw_intervention_is_explicitly_opt_in(void) {
    DroneRace raw = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace assisted = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    raw.teacher_action_blend = 1.0f;
    assisted.teacher_action_blend = 1.0f;
    assisted.teacher_yaw_control = 1;
    assisted.teacher_yaw_action = 0.15f;
    raw.actions[3] = -0.4f;
    assisted.actions[3] = -0.4f;

    c_step(&raw);
    c_step(&assisted);

    CHECK(fabsf(raw.observations[22] + 0.4f) < 1e-5f,
        "disabled yaw teacher must preserve PufferLib yaw ownership");
    CHECK(fabsf(assisted.observations[22] - 0.15f) < 1e-5f,
        "enabled training-only yaw teacher should execute its target");

    free_test_env(&raw);
    free_test_env(&assisted);
    return 0;
}

static int test_sitl_teacher_intervention_can_wait_for_training_step(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.teacher_action_blend = 1.0f;
    env.teacher_action_blend_from_step = 2;
    env.teacher_pitch_from_gate_index = 2;
    env.teacher_roll_from_gate_index = 0;
    env.teacher_roll_until_gate_index = 2;
    env.teacher_thrust_from_gate_index = 2;
    env.teacher_roll_bias = 0.4f;
    env.teacher_roll_per_m = 0.0f;
    env.actions[1] = -0.2f;

    c_step(&env);
    CHECK(fabsf(env.observations[20] + 0.2f) < 1e-5f,
        "delayed teacher must preserve the first raw Puffer action");
    c_step(&env);
    CHECK(fabsf(env.observations[20] + 0.2f) < 1e-5f,
        "delayed teacher must preserve raw Puffer actions before its step");
    c_step(&env);
    CHECK(fabsf(env.observations[20] - 0.4f) < 1e-5f,
        "training teacher should activate exactly at the configured step");

    free_test_env(&env);
    return 0;
}

static int test_sitl_teacher_intervention_zero_preserves_policy_dynamics(void) {
    DroneRace reference = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace shaped = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    shaped.teacher_action_blend = 0.0f;
    shaped.teacher_roll_bias = 1.0f;
    shaped.teacher_thrust_bias = -1.0f;
    for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
        reference.actions[k] = 0.1f * (float)(k + 1);
        shaped.actions[k] = reference.actions[k];
    }

    c_step(&reference);
    c_step(&shaped);

    CHECK(fabsf(reference.agents[0].drone.state.pos.x
            - shaped.agents[0].drone.state.pos.x) < 1e-7f
            && fabsf(reference.agents[0].drone.state.pos.y
                - shaped.agents[0].drone.state.pos.y) < 1e-7f
            && fabsf(reference.agents[0].drone.state.pos.z
                - shaped.agents[0].drone.state.pos.z) < 1e-7f,
        "zero teacher intervention must preserve historical plant dynamics");
    for (int index = 19; index <= 22; index++) {
        CHECK(fabsf(reference.observations[index]
                - shaped.observations[index]) < 1e-7f,
            "zero teacher intervention must preserve last-action observations");
    }

    free_test_env(&reference);
    free_test_env(&shaped);
    return 0;
}

static int test_sitl_teacher_pitch_speed_intervention_brakes_and_accelerates(void) {
    DroneRace fast = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace slow = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    fast.teacher_action_blend = 1.0f;
    slow.teacher_action_blend = 1.0f;
    fast.teacher_pitch_speed_control = 1;
    slow.teacher_pitch_speed_control = 1;
    fast.teacher_pitch_speed_target_m_s = 2.0f;
    slow.teacher_pitch_speed_target_m_s = 2.0f;
    fast.teacher_pitch_speed_gain = 1.0f;
    slow.teacher_pitch_speed_gain = 1.0f;
    fast.teacher_pitch_speed_scale = 1.0f;
    slow.teacher_pitch_speed_scale = 1.0f;
    fast.teacher_roll_from_gate_index = 2;
    slow.teacher_roll_from_gate_index = 2;
    fast.teacher_thrust_from_gate_index = 2;
    slow.teacher_thrust_from_gate_index = 2;
    fast.agents[0].drone.state.vel.x = 3.0f;
    slow.agents[0].drone.state.vel.x = 1.0f;

    c_step(&fast);
    c_step(&slow);

    CHECK(fabsf(fast.observations[19] - 0.60f) < 1e-5f,
        "speed teacher should command positive pitch to brake excess speed");
    CHECK(fabsf(slow.observations[19] + 0.60f) < 1e-5f,
        "speed teacher should command negative pitch below target speed");

    free_test_env(&fast);
    free_test_env(&slow);
    return 0;
}

static int test_sitl_teacher_world_vertical_ignores_bank_coupling(void) {
    DroneRace body = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace world = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    body.teacher_thrust_per_m = 1.0f;
    world.teacher_thrust_per_m = 1.0f;
    world.teacher_thrust_world_frame = 1;
    body.gates[0].pos = (Vec3){5.0f, 2.0f, 1.0f};
    world.gates[0].pos = body.gates[0].pos;
    body.agents[0].drone.state.quat = euler_to_quat(
        (EulerAngles){.roll = 0.75f, .pitch = 0.0f, .yaw = 0.0f});
    world.agents[0].drone.state.quat = body.agents[0].drone.state.quat;

    ActionTeacherTarget body_target = action_teacher_target(&body, &body.agents[0], 0);
    ActionTeacherTarget world_target = action_teacher_target(&world, &world.agents[0], 0);

    CHECK(fabsf(world_target.thrust - 1.0f) < 1e-6f,
        "world-vertical teacher should target the world-up gate error");
    CHECK(fabsf(body_target.thrust - world_target.thrust) > 0.2f,
        "banked body-frame altitude error should remain a distinct legacy option");

    free_test_env(&body);
    free_test_env(&world);
    return 0;
}

static int test_sitl_teacher_true_velocity_damping_is_default_off(void) {
    DroneRace legacy = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRace damped = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    legacy.teacher_roll_per_m = 0.0f;
    damped.teacher_roll_per_m = 0.0f;
    legacy.teacher_roll_rate_per_m_s = 0.5f;
    damped.teacher_roll_rate_per_m_s = 0.5f;
    damped.teacher_true_velocity_damping = 1;
    legacy.agents[0].drone.state.vel = (Vec3){0.0f, 1.0f, 0.0f};
    damped.agents[0].drone.state.vel = legacy.agents[0].drone.state.vel;
    legacy.gates[0].vel = (Vec3){0.0f, 0.0f, 0.0f};
    damped.gates[0].vel = legacy.gates[0].vel;

    ActionTeacherTarget legacy_target = action_teacher_target(
        &legacy, &legacy.agents[0], 0);
    ActionTeacherTarget damped_target = action_teacher_target(
        &damped, &damped.agents[0], 0);

    CHECK(fabsf(legacy_target.roll) < 1e-7f,
        "default teacher must preserve the historical absent-rate behavior");
    CHECK(fabsf(damped_target.roll - 0.5f) < 1e-6f,
        "privileged velocity damping should oppose true rightward velocity");

    free_test_env(&legacy);
    free_test_env(&damped);
    return 0;
}

static int test_sitl_attitude_observation_contract(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    float* obs = env.observations;

    CHECK(env.interface_mode == DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT,
        "attitude-setpoint interface mode should survive init defaults");
    CHECK(obs[10] == 1.0f, "first gate should be visible from reset in the attitude contract");

    env.actions[0] = 0.10f;
    env.actions[1] = -0.20f;
    env.actions[2] = 0.30f;
    env.actions[3] = 0.40f;
    c_step(&env);
    obs = env.observations;
    CHECK(fabsf(obs[19] - 0.10f) < 1e-5f, "last pitch command should be observable");
    CHECK(fabsf(obs[20] + 0.20f) < 1e-5f, "last roll command should be observable");
    CHECK(fabsf(obs[21] - 0.30f) < 1e-5f, "last thrust command should be observable");
    CHECK(fabsf(obs[22] - 0.40f) < 1e-5f, "last yaw command should be observable");

    free_test_env(&env);
    return 0;
}

static int test_sitl_observation_can_expose_gate_phase(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.observable_gate_index = 1;
    env.num_gates = 4;
    build_course(&env);
    env.agents[0].current_gate = 1;
    env.agents[0].last_action[3] = -0.75f;
    compute_observations(&env);

    CHECK(fabsf(env.observations[22] + 0.75f) < 1e-6f,
        "phase observation must preserve the previous yaw command");
    CHECK(fabsf(env.observations[23] - 1.0f) < 1e-6f,
        "phase observation should append the active gate-one indicator");
    CHECK(env.observations[24] == 0.0f && env.observations[25] == 0.0f,
        "inactive gate indicators should remain zero");
    CHECK(env.observations[26] == 0.0f && env.observations[27] == 0.0f,
        "gate-two adapter features should stay inactive at gate one");

    env.agents[0].current_gate = 2;
    env.agents[0].drone.state.pos = sub3(env.gates[2].pos, (Vec3){2.0f, 0.5f, -0.25f});
    env.agents[0].gate_motion_valid = 0;
    compute_observations(&env);
    CHECK(env.observations[23] == 0.0f && env.observations[24] == 1.0f
            && env.observations[25] == 0.0f,
        "gate-two status should select only the gate-two indicator");
    CHECK(fabsf(env.observations[26] - env.observations[12]) < 1e-6f
            && fabsf(env.observations[27] - env.observations[1]) < 1e-6f
            && fabsf(env.observations[28] - env.observations[13]) < 1e-6f
            && fabsf(env.observations[29] - env.observations[2]) < 1e-6f,
        "gate-two adapter should copy official-observable lateral/vertical pose and rates");
    CHECK(env.observations[30] == 0.0f && env.observations[31] == 0.0f,
        "gate-three adapter features should stay inactive at gate two");

    env.agents[0].current_gate = 3;
    env.agents[0].drone.state.pos = sub3(env.gates[3].pos, (Vec3){2.0f, -0.4f, 0.3f});
    env.agents[0].gate_motion_valid = 0;
    compute_observations(&env);
    CHECK(env.observations[23] == 0.0f && env.observations[24] == 0.0f
            && env.observations[25] == 1.0f,
        "gate-three status should select only the gate-three indicator");
    CHECK(fabsf(env.observations[30] - env.observations[12]) < 1e-6f
            && fabsf(env.observations[31] - env.observations[1]) < 1e-6f,
        "gate-three adapter should copy official-observable lateral pose and rate");
    CHECK(env.observations[26] == 0.0f && env.observations[27] == 0.0f
            && env.observations[28] == 0.0f && env.observations[29] == 0.0f,
        "gate-two adapter features should stay inactive at gate three");

    free_test_env(&env);
    return 0;
}

static int test_sitl_gate_phase_can_keep_full_course_denominator(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.observable_gate_index = 1;
    env.observable_gate_index_denominator = 3.0f;
    env.agents[0].current_gate = 1;
    compute_observations(&env);

    CHECK(fabsf(env.observations[23] - 1.0f) < 1e-6f,
        "two-gate curriculum should preserve the full-course gate-one identity");

    env.num_gates = 6;
    env.observable_gate_progress = 1;
    env.observable_gate_index_denominator = 6.0f;
    env.agents[0].current_gate = 5;
    env.agents[0].last_action[3] = -0.25f;
    compute_observations(&env);
    CHECK(fabsf(env.observations[22] + 0.25f) < 1e-6f,
        "six-gate progress adapter must preserve previous yaw");
    CHECK(fabsf(env.observations[23] - 5.0f / 6.0f) < 1e-6f,
        "six-gate progress adapter should normalize the active gate index");
    for (int index = 24; index < DRONE_RACE_OBS_SIZE; index++) {
        CHECK(env.observations[index] == 0.0f,
            "six-gate progress adapter reserved values must stay zero");
    }

    env.observable_gate_phase_onehot = 1;
    for (int gate = 0; gate < 6; gate++) {
        env.agents[0].current_gate = gate;
        compute_observations(&env);
        CHECK(fabsf(env.observations[23] - (float)gate / 6.0f) < 1e-6f,
            "six-gate one-hot adapter must preserve normalized progress");
        for (int phase = 0; phase < 6; phase++) {
            CHECK(env.observations[24 + phase] == (gate == phase ? 1.0f : 0.0f),
                "six-gate one-hot adapter must select exactly the active gate");
        }
        CHECK(env.observations[30] == 0.0f && env.observations[31] == 0.0f,
            "six-gate one-hot adapter reserved values must stay zero");
    }

    free_test_env(&env);
    return 0;
}

static int test_gate_exit_reward_targets_next_segment_lateral_velocity(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.gates[0].pos = (Vec3){20.0f, 0.0f, 0.0f};
    env.gates[1].pos = (Vec3){40.0f, 2.0f, 0.0f};
    agent->drone.state.vel = (Vec3){10.0f, 1.0f, 0.0f};

    float aligned = gate_exit_lateral_velocity_error_sq(&env, agent, 0);
    agent->drone.state.vel.y = 0.0f;
    float straight = gate_exit_lateral_velocity_error_sq(&env, agent, 0);

    CHECK(aligned < 1e-6f,
        "gate-exit reward should accept velocity aligned with the next course segment");
    CHECK(fabsf(straight - 1.0f) < 1e-6f,
        "gate-exit reward should penalize a straight exit before a lateral turn");

    free_test_env(&env);
    return 0;
}

static int test_gate_exit_reward_targets_feasible_3d_velocity(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.gate_exit_target_forward_speed = 10.0f;
    env.gates[0].pos = (Vec3){20.0f, 0.0f, 0.0f};
    env.gates[1].pos = (Vec3){40.0f, 2.0f, -4.0f};
    agent->drone.state.vel = (Vec3){10.0f, 1.0f, -2.0f};

    float aligned = gate_exit_velocity_error_sq(&env, agent, 0);
    agent->drone.state.vel = (Vec3){20.0f, 0.0f, 0.0f};
    float sprint = gate_exit_velocity_error_sq(&env, agent, 0);

    CHECK(aligned < 1e-6f,
        "3D gate-exit reward should accept the feasible next-segment velocity");
    CHECK(fabsf(sprint - 105.0f) < 1e-5f,
        "3D gate-exit reward should penalize terminal sprint and missing yz velocity");

    free_test_env(&env);
    return 0;
}

static int test_gate_exit_reward_skips_hidden_randomized_next_gate(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.num_gates = 6;
    env.gate_position_randomize_from_index = 4;
    env.gate_exit_reward_skip_randomized_next_gate = 1;
    agent->gate_positions_randomized = 1;

    CHECK(gate_exit_reward_active(&env, agent, 2),
        "exit shaping should remain active before the hidden randomized suffix");
    CHECK(!gate_exit_reward_active(&env, agent, 3),
        "exit shaping must not target randomized gate 4 before it is observed");
    CHECK(!gate_exit_reward_active(&env, agent, 4),
        "exit shaping must not target randomized gate 5 before it is observed");

    agent->gate_positions_randomized = 0;
    CHECK(gate_exit_reward_active(&env, agent, 3),
        "nominal anchor episodes should retain the fixed-course exit shaping");
    env.gate_exit_reward_skip_randomized_next_gate = 0;
    agent->gate_positions_randomized = 1;
    CHECK(gate_exit_reward_active(&env, agent, 3),
        "the compatibility default should preserve legacy reward behavior");

    free_test_env(&env);
    return 0;
}

static int test_forward_speed_excess_penalty_has_no_low_speed_floor(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.target_forward_speed = 10.0f;
    agent->current_gate = 0;
    agent->drone.state.vel = (Vec3){20.0f, 3.0f, -2.0f};
    CHECK(fabsf(forward_speed_excess_error_sq(&env, agent) - 100.0f) < 1e-6f,
        "speed curriculum should penalize squared forward excess only");
    agent->drone.state.vel = (Vec3){8.0f, 20.0f, -20.0f};
    CHECK(forward_speed_excess_error_sq(&env, agent) == 0.0f,
        "speed curriculum must not penalize feasible forward speed or lateral motion");

    free_test_env(&env);
    return 0;
}

static int test_transition_speed_curriculum_samples_per_episode(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.sitl_gate_transition_min_forward_speed = 4.0f;
    env.sitl_gate_transition_min_forward_speed_randomize = 1;
    env.sitl_gate_transition_min_forward_speed_min = 4.125f;
    env.sitl_gate_transition_min_forward_speed_max = 4.1875f;

    reset_agent(&env, agent);
    CHECK(agent->randomized_sitl_gate_transition_min_forward_speed >= 4.125f,
        "randomized transition speed should respect its lower bound");
    CHECK(agent->randomized_sitl_gate_transition_min_forward_speed <= 4.1875f,
        "randomized transition speed should respect its upper bound");

    env.sitl_gate_transition_min_forward_speed_min = 5.0f;
    env.sitl_gate_transition_min_forward_speed_max = 5.0f;
    reset_agent(&env, agent);
    CHECK(agent->randomized_sitl_gate_transition_min_forward_speed == 5.0f,
        "degenerate transition-speed range should be deterministic");

    env.sitl_gate_transition_min_forward_speed_randomize = 0;
    reset_agent(&env, agent);
    CHECK(agent->randomized_sitl_gate_transition_min_forward_speed == 4.0f,
        "disabled transition-speed randomization should use the fixed floor");

    free_test_env(&env);
    return 0;
}


static int test_cross_track_reward_can_start_at_late_gate(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.w_cross_track = 10.0f;
    env.cross_track_from_gate_index = 1;
    env.gates[0].pos = (Vec3){20.0f, 2.0f, 0.0f};
    env.gates[1].pos = (Vec3){40.0f, 2.0f, 0.0f};

    agent->current_gate = 0;
    CHECK(cross_track_penalty(&env, agent, (Vec3){10.0f, 0.0f, 0.0f}) == 0.0f,
        "late-stage cross-track reward must preserve earlier course segments");
    agent->current_gate = 1;
    CHECK(cross_track_penalty(&env, agent, (Vec3){30.0f, 0.0f, 0.0f}) > 0.0f,
        "late-stage cross-track reward should activate at its configured gate");

    env.gates[1].pos = (Vec3){40.0f, 0.0f, 0.0f};
    env.cross_track_vertical_weight = 4.0f;
    float lateral = cross_track_penalty(
        &env, agent, (Vec3){30.0f, 1.0f, 0.0f});
    float vertical = cross_track_penalty(
        &env, agent, (Vec3){30.0f, 0.0f, 1.0f});
    CHECK(vertical > 3.9f * lateral,
        "vertical cross-track weight should provide dense altitude priority");

    free_test_env(&env);
    return 0;
}

static int test_gate_crossing_error_reward_targets_configured_gate(void) {
    DroneRace env = make_test_env();
    env.num_gates = 4;
    env.w_gate_crossing_error = 0.5f;
    env.gate_crossing_error_from_gate_index = 3;

    CHECK(gate_crossing_error_penalty(&env, 2, 2.0f) == 0.0f,
        "crossing-error reward should preserve gates before its configured start");
    CHECK(fabsf(gate_crossing_error_penalty(&env, 3, 2.0f) - 2.0f) < 1e-6f,
        "crossing-error reward should penalize squared radial error");
    CHECK(gate_crossing_error_penalty(&env, 3, 0.0f) == 0.0f,
        "a centered crossing should have no crossing-error penalty");

    free_test_env(&env);
    return 0;
}

static int test_gate_camera_alignment_reward_is_centered_and_phase_local(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.w_gate_camera_alignment = 10.0f;
    env.gate_camera_alignment_from_gate_index = 1;
    env.gates[1].pos = (Vec3){10.0f, 0.0f, 0.0f};
    agent->drone.state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    agent->drone.state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};

    agent->current_gate = 0;
    CHECK(gate_camera_alignment_error_sq(&env, agent) == 0.0f,
        "camera alignment reward should preserve earlier gates");
    agent->current_gate = 1;
    CHECK(gate_camera_alignment_error_sq(&env, agent) < 1e-6f,
        "a centered level gate should have zero camera alignment error");
    env.gates[1].pos.y = 10.0f;
    CHECK(gate_camera_alignment_error_sq(&env, agent) > 0.9f,
        "an off-axis gate should incur camera alignment error");

    free_test_env(&env);
    return 0;
}

static int test_late_invalid_penalty_preserves_prefix(void) {
    DroneRace env = make_test_env();
    env.invalid_penalty = 600.0f;
    env.late_invalid_penalty = 40.0f;
    env.late_invalid_penalty_from_gate_index = 2;

    CHECK(invalid_run_penalty(&env, 0) == 600.0f,
        "late invalid reward must preserve the gate-0 penalty");
    CHECK(invalid_run_penalty(&env, 1) == 600.0f,
        "late invalid reward must preserve the solved prefix");
    CHECK(invalid_run_penalty(&env, 2) == 40.0f,
        "late invalid reward should unsaturate the configured gate");
    CHECK(invalid_run_penalty(&env, 3) == 40.0f,
        "late invalid reward should apply to downstream gates");

    free_test_env(&env);
    return 0;
}

static int test_course_spline_teacher_is_reward_only(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.gates[0].pos = (Vec3){0.0f, 0.0f, 0.0f};
    env.gates[1].pos = (Vec3){5.0f, 2.0f, -2.0f};
    agent->current_gate = 0;
    agent->drone.state.pos = (Vec3){-1.0f, 0.0f, 0.0f};
    agent->drone.state.vel = (Vec3){8.0f, 0.0f, 0.0f};
    compute_observations(&env);
    float observations_before[DRONE_RACE_OBS_SIZE];
    memcpy(observations_before, env.observations, sizeof(observations_before));

    env.teacher_course_spline = 1;
    build_course_teacher_profile(&env);
    CHECK(env.teacher_course_spline_valid == 1,
        "course-aware action reward should build a valid monotonic profile");
    for (int i = 0; i < DRONE_RACE_OBS_SIZE; i++) {
        CHECK(fabsf(env.observations[i] - observations_before[i]) < 1e-6f,
            "training-only course profile must not change the policy observation");
    }

    float teacher_roll = 0.0f;
    float teacher_thrust = 0.0f;
    course_spline_teacher_action(
        &env, agent, &teacher_roll, &teacher_thrust);
    CHECK(isfinite(teacher_roll) && teacher_roll >= -1.0f && teacher_roll <= 1.0f,
        "course teacher roll target should be finite and normalized");
    CHECK(isfinite(teacher_thrust)
            && teacher_thrust >= -1.0f && teacher_thrust <= 1.0f,
        "course teacher thrust target should be finite and normalized");
    CHECK(fabsf(teacher_roll) > 0.01f || fabsf(teacher_thrust) > 0.01f,
        "course teacher should provide a nontrivial target before a turning segment");

    free_test_env(&env);
    return 0;
}

static int test_minimum_jerk_segment_reference_is_transition_continuous(void) {
    float value;
    float first;
    float second;
    minimum_jerk_basis(0.0f, &value, &first, &second);
    CHECK(value == 0.0f && first == 0.0f && second == 0.0f,
        "minimum-jerk basis should start at rest with zero acceleration");
    minimum_jerk_basis(0.5f, &value, &first, &second);
    CHECK(fabsf(value - 0.5f) < 1e-6f,
        "minimum-jerk midpoint should be centered");
    CHECK(fabsf(first - 1.875f) < 1e-6f && fabsf(second) < 1e-6f,
        "minimum-jerk midpoint derivatives should match the analytic basis");
    minimum_jerk_basis(1.0f, &value, &first, &second);
    CHECK(value == 1.0f && first == 0.0f && second == 0.0f,
        "minimum-jerk basis should end at rest with zero acceleration");

    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.num_gates = 3;
    env.gates[0].pos = (Vec3){0.0f, 0.0f, 0.0f};
    env.gates[1].pos = (Vec3){10.0f, 4.0f, 2.0f};
    env.gates[2].pos = (Vec3){20.0f, -3.0f, -1.0f};
    agent->drone.state.pos = env.gates[1].pos;
    agent->drone.state.vel = (Vec3){2.5f, 0.0f, 0.0f};
    agent->drone.state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    float before_roll = 0.0f;
    float before_thrust = 0.0f;
    float after_roll = 0.0f;
    float after_thrust = 0.0f;
    agent->current_gate = 1;
    course_segment_minimum_jerk_teacher_action(
        &env, agent, &before_roll, &before_thrust);
    agent->current_gate = 2;
    course_segment_minimum_jerk_teacher_action(
        &env, agent, &after_roll, &after_thrust);
    CHECK(fabsf(before_roll - after_roll) < 1e-6f,
        "segment transition should preserve the lateral action reference");
    CHECK(fabsf(before_thrust - after_thrust) < 1e-6f,
        "segment transition should preserve the vertical action reference");

    free_test_env(&env);
    return 0;
}

static int test_alignment_governor_brakes_and_centers_with_default_off(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    DroneRaceAgent* agent = &env.agents[0];
    env.num_gates = 1;
    env.gates[0].pos = (Vec3){10.0f, 0.0f, 0.0f};
    env.teacher_pitch_action = 0.25f;
    env.teacher_pitch_speed_target_m_s = 2.0f;
    env.teacher_from_gate_index = 0;
    env.teacher_roll_from_gate_index = 0;
    env.teacher_roll_until_gate_index = 1;
    env.teacher_thrust_from_gate_index = 0;
    env.sitl_linear_drag = 0.14f;
    env.sitl_gravity_m_s2 = 8.69f;
    env.sitl_lateral_accel_scale = 0.107491f;
    env.sitl_max_roll_rad = 0.50f;
    env.sitl_vertical_accel_per_thrust = 32.81f;
    env.sitl_hover_thrust = 0.27f;
    env.sitl_min_thrust = 0.18f;
    env.sitl_max_thrust = 0.42f;
    agent->current_gate = 0;
    agent->drone.state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    agent->drone.state.vel = (Vec3){2.0f, 0.0f, 0.0f};
    agent->drone.state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};

    ActionTeacherTarget historical = action_teacher_target(&env, agent, 0);
    CHECK(fabsf(historical.pitch - 0.25f) < 1e-6f,
        "default-off alignment governor should preserve the historical pitch target");

    env.teacher_alignment_governor = 1;
    ActionTeacherTarget centered = action_teacher_target(&env, agent, 0);
    CHECK(fabsf(centered.pitch) < 1e-6f && fabsf(centered.roll) < 1e-6f,
        "centered gate at approach speed should be a pitch/roll equilibrium");
    CHECK(isfinite(centered.thrust)
            && centered.thrust >= -1.0f && centered.thrust <= 1.0f,
        "alignment governor thrust should be finite and bounded");

    env.gates[0].pos.y = 4.0f;
    ActionTeacherTarget offset = action_teacher_target(&env, agent, 0);
    CHECK(fabsf(offset.pitch - 0.80f) < 1e-6f,
        "large radial error should command the preregistered braking limit");
    CHECK(offset.roll < 0.0f,
        "a gate to world right should command negative roll");
    CHECK(isfinite(offset.thrust)
            && offset.thrust >= -1.0f && offset.thrust <= 1.0f,
        "bank-compensated alignment thrust should remain bounded");

    free_test_env(&env);
    return 0;
}

static int test_segment_curriculum_starts_at_requested_gate_and_velocity(void) {
    DroneRace env = make_test_env_with_interface(DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.start_gate_index = 1;
    env.start_elapsed_time = 4.5f;
    env.use_custom_start = 1;
    env.custom_start_vel[0] = 10.0f;
    env.custom_start_vel[1] = 1.5f;
    env.custom_start_vel[2] = -2.0f;
    env.custom_start_quat[0] = 0.9950042f;
    env.custom_start_quat[2] = 0.0998334f;
    env.custom_start_omega[1] = 0.25f;
    reset_agent(&env, &env.agents[0]);

    CHECK(env.agents[0].current_gate == 1,
        "segment curriculum should begin at the configured active gate");
    CHECK(env.agents[0].target.pos.x == env.gates[1].pos.x,
        "segment curriculum target should match its configured active gate");
    CHECK(fabsf(env.agents[0].elapsed_time - 4.5f) < 1e-6f,
        "segment curriculum should preserve the measured race-time offset");
    CHECK(fabsf(env.agents[0].drone.state.vel.x - 10.0f) < 1e-6f,
        "segment curriculum should preserve configured forward velocity");
    CHECK(fabsf(env.agents[0].drone.state.vel.y - 1.5f) < 1e-6f,
        "segment curriculum should preserve configured lateral velocity");
    CHECK(fabsf(env.agents[0].drone.state.vel.z + 2.0f) < 1e-6f,
        "segment curriculum should preserve configured vertical velocity");
    CHECK(fabsf(env.agents[0].drone.state.quat.y - 0.0998334f) < 1e-5f,
        "segment curriculum should preserve configured attitude");
    CHECK(fabsf(env.agents[0].drone.state.omega.y - 0.25f) < 1e-6f,
        "segment curriculum should preserve configured body rate");

    free_test_env(&env);
    return 0;
}

static int test_measured_segment_start_can_disable_legacy_position_noise(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.use_custom_start = 1;
    env.start_gate_index = 1;
    env.custom_start_pos[0] = 3.0f;
    env.custom_start_pos[1] = -2.0f;
    env.custom_start_pos[2] = 1.0f;
    env.reset_position_noise_xy = 0.0f;
    env.reset_position_noise_z = 0.0f;

    for (int episode = 0; episode < 8; episode++) {
        reset_agent(&env, &env.agents[0]);
        Vec3 pos = env.agents[0].drone.state.pos;
        CHECK(pos.x == 3.0f && pos.y == -2.0f && pos.z == 1.0f,
            "zero legacy noise should reproduce the measured position exactly");
    }

    free_test_env(&env);
    return 0;
}

static int test_segment_start_distribution_is_bounded_and_non_degenerate(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.use_custom_start = 1;
    env.start_gate_index = 1;
    env.start_elapsed_time = 10.7f;
    env.start_elapsed_time_jitter = 0.5f;
    env.custom_start_pos[0] = 64.0f;
    env.custom_start_pos[1] = 0.0f;
    env.custom_start_pos[2] = -7.0f;
    env.custom_start_pos_jitter[0] = 6.0f;
    env.custom_start_pos_jitter[1] = 4.0f;
    env.custom_start_pos_jitter[2] = 3.0f;
    env.custom_start_vel[0] = 9.0f;
    env.custom_start_vel[1] = 0.0f;
    env.custom_start_vel[2] = -4.0f;
    env.custom_start_vel_jitter[0] = 4.0f;
    env.custom_start_vel_jitter[1] = 4.0f;
    env.custom_start_vel_jitter[2] = 3.0f;
    env.custom_start_quat[0] = 1.0f;
    env.custom_start_attitude_jitter[0] = 0.20f;
    env.custom_start_attitude_jitter[1] = 0.08f;
    env.custom_start_attitude_jitter[2] = 0.12f;
    env.custom_start_omega[0] = -0.2f;
    env.custom_start_omega_jitter[0] = 0.5f;

    float min_vx = 999.0f;
    float max_vx = -999.0f;
    float min_roll = 999.0f;
    float max_roll = -999.0f;
    for (int episode = 0; episode < 256; episode++) {
        reset_agent(&env, &env.agents[0]);
        State state = env.agents[0].drone.state;
        EulerAngles attitude = quat_to_euler(state.quat);
        CHECK(env.agents[0].current_gate == 1,
            "distributed segment reset should retain the requested gate");
        CHECK(env.agents[0].elapsed_time >= 10.2f
                && env.agents[0].elapsed_time <= 11.2f,
            "distributed segment elapsed time should stay inside its bounds");
        // The common reset path adds its historical 0.10/0.10/0.05 m spawn
        // perturbation after the empirical position distribution.
        CHECK(state.pos.x >= 57.9f && state.pos.x <= 70.1f,
            "distributed segment X should stay inside jitter plus spawn noise");
        CHECK(state.pos.y >= -4.1f && state.pos.y <= 4.1f,
            "distributed segment Y should stay inside jitter plus spawn noise");
        CHECK(state.pos.z >= -10.05f && state.pos.z <= -3.95f,
            "distributed segment Z should stay inside jitter plus spawn noise");
        CHECK(state.vel.x >= 5.0f && state.vel.x <= 13.0f,
            "distributed segment forward velocity should stay bounded");
        CHECK(state.vel.y >= -4.0f && state.vel.y <= 4.0f,
            "distributed segment lateral velocity should stay bounded");
        CHECK(state.vel.z >= -7.0f && state.vel.z <= -1.0f,
            "distributed segment vertical velocity should stay bounded");
        CHECK(fabsf(attitude.roll) <= 0.20001f
                && fabsf(attitude.pitch) <= 0.08001f
                && fabsf(attitude.yaw) <= 0.12001f,
            "distributed segment attitude should stay inside Euler bounds");
        CHECK(state.omega.x >= -0.7f && state.omega.x <= 0.3f,
            "distributed segment body rate should stay bounded");
        min_vx = fminf(min_vx, state.vel.x);
        max_vx = fmaxf(max_vx, state.vel.x);
        min_roll = fminf(min_roll, attitude.roll);
        max_roll = fmaxf(max_roll, attitude.roll);
    }
    CHECK(max_vx - min_vx > 6.0f,
        "distributed segment reset should sample a velocity range");
    CHECK(max_roll - min_roll > 0.30f,
        "distributed segment reset should sample an attitude range");

    free_test_env(&env);
    return 0;
}

static int test_mixed_curriculum_preserves_full_and_segment_starts(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT);
    env.use_custom_start = 1;
    env.start_gate_index = 1;
    env.start_elapsed_time = 4.5f;
    env.custom_start_pos[0] = 3.0f;
    env.custom_start_vel[0] = 10.0f;
    env.custom_start_quat[0] = 1.0f;
    env.mixed_start_curriculum = 1;

    env.segment_start_probability = 0.0f;
    reset_agent(&env, &env.agents[0]);
    CHECK(env.agents[0].current_gate == 0,
        "zero segment probability should reset at the full-course first gate");
    CHECK(env.agents[0].elapsed_time == 0.0f,
        "full-course member of a mixed curriculum should start race time at zero");
    CHECK(env.agents[0].drone.state.vel.x == 0.0f,
        "full-course member should not inherit segment entry velocity");

    env.segment_start_probability = 1.0f;
    reset_agent(&env, &env.agents[0]);
    CHECK(env.agents[0].current_gate == 1,
        "unit segment probability should reset at the measured segment gate");
    CHECK(fabsf(env.agents[0].elapsed_time - 4.5f) < 1e-6f,
        "segment member should retain its measured elapsed-time offset");
    CHECK(fabsf(env.agents[0].drone.state.vel.x - 10.0f) < 1e-6f,
        "segment member should retain measured entry velocity");

    int full_starts = 0;
    int segment_starts = 0;
    env.segment_start_probability = 0.5f;
    for (int episode = 0; episode < 128; episode++) {
        reset_agent(&env, &env.agents[0]);
        full_starts += env.agents[0].current_gate == 0;
        segment_starts += env.agents[0].current_gate == 1;
    }
    CHECK(full_starts > 0 && segment_starts > 0,
        "parallel mixed curriculum should sample both start-state families");

    free_test_env(&env);
    return 0;
}

static int test_vq1_telemetry_velocity_plant_tracks_local_command(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY);
    env.max_cmd_forward = 12.0f;
    env.max_cmd_lateral = 8.0f;
    env.max_cmd_vertical = 8.0f;
    env.telemetry_velocity_response_tau_s = 0.25f;
    env.telemetry_velocity_max_accel_m_s2 = 8.0f;
    DroneRaceAgent* agent = &env.agents[0];
    agent->drone.state.pos = (Vec3){-5.0f, 0.0f, 1.0f};
    agent->drone.state.vel = (Vec3){0.0f, 0.0f, 0.0f};
    float action[4] = {0.5f, -0.25f, 0.25f, 0.0f};

    for (int step = 0; step < 120; step++) {
        move_drone_vq1_telemetry_velocity(&env, agent, action);
    }

    CHECK(agent->drone.state.vel.x > 5.7f
            && agent->drone.state.vel.x <= 6.0f,
        "telemetry velocity plant should converge to normalized forward command");
    CHECK(agent->drone.state.vel.y < -1.9f
            && agent->drone.state.vel.y >= -2.0f,
        "telemetry velocity plant should converge to local lateral command");
    CHECK(agent->drone.state.vel.z < -1.9f
            && agent->drone.state.vel.z >= -2.0f,
        "positive down command should produce negative native-world Z velocity");

    free_test_env(&env);
    return 0;
}

static int test_vq1_telemetry_observation_is_exact_local_course_state(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY);
    env.observable_gate_index_denominator = 6.0f;
    env.observable_gate_phase_onehot = 1;
    DroneRaceAgent* agent = &env.agents[0];
    env.gates[0].pos = (Vec3){20.0f, 2.0f, -3.0f};
    agent->current_gate = 0;
    agent->drone.state.pos = (Vec3){5.0f, -1.0f, -1.0f};
    agent->drone.state.vel = (Vec3){4.0f, -2.0f, -3.0f};
    compute_one_observation(&env, 0);

    CHECK(fabsf(env.observations[0] - tanhf(-0.8f)) < 1e-6f,
        "telemetry observation should expose exact local forward gate rate");
    CHECK(fabsf(env.observations[1] - tanhf(2.0f / 3.0f)) < 1e-6f,
        "telemetry observation should expose exact local right gate rate");
    CHECK(fabsf(env.observations[2] - tanhf(-1.0f)) < 1e-6f,
        "telemetry observation should expose exact local down gate rate");
    CHECK(env.observations[6] == 1.0f && env.observations[10] == 1.0f,
        "telemetry observation should use fixed course attitude and a present gate");
    CHECK(fabsf(env.observations[11] - tanhf(1.5f)) < 1e-6f,
        "telemetry observation should encode exact gate-forward displacement");
    CHECK(fabsf(env.observations[12] - tanhf(0.6f)) < 1e-6f,
        "telemetry observation should encode exact gate-right displacement");
    CHECK(fabsf(env.observations[13] - tanhf(0.4f)) < 1e-6f,
        "telemetry observation should encode exact gate-down displacement");
    CHECK(env.observations[24] == 1.0f,
        "telemetry observation should expose the active six-gate phase");

    free_test_env(&env);
    return 0;
}

static int test_vq1_telemetry_teacher_targets_active_gate(void) {
    DroneRace env = make_test_env_with_interface(
        DRONE_RACE_INTERFACE_VQ1_TELEMETRY_VELOCITY);
    env.max_cmd_forward = 12.0f;
    env.max_cmd_lateral = 8.0f;
    env.max_cmd_vertical = 8.0f;
    env.telemetry_velocity_teacher_speed_m_s = 8.0f;
    DroneRaceAgent* agent = &env.agents[0];
    env.gates[0].pos = (Vec3){20.0f, 2.0f, -4.0f};
    agent->current_gate = 0;
    agent->drone.state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    float teacher[DRONE_RACE_NUM_ATNS] = {0};

    CHECK(telemetry_velocity_teacher_action(&env, agent, teacher),
        "telemetry teacher should be valid for the active gate");
    CHECK(teacher[0] > 0.0f && teacher[1] > 0.0f && teacher[2] > 0.0f,
        "telemetry teacher should point forward, right, and down toward the gate");
    CHECK(teacher[0] <= 1.0f && teacher[1] <= 1.0f && teacher[2] <= 1.0f,
        "telemetry teacher actions should remain normalized");
    CHECK(teacher[3] == 0.0f,
        "local-course telemetry teacher should not require yaw control");

    free_test_env(&env);
    return 0;
}

int main(void) {
    if (test_vecenv_dict_grows_without_heap_overflow()) return 1;
    if (test_gate_crossing_geometry()) return 1;
    if (test_terminal_crossing_is_bucketed_by_gate()) return 1;
    if (test_ordered_crossing_and_envelope_diagnostics_are_log_only()) return 1;
    if (test_per_gate_curriculum_radius()) return 1;
    if (test_course_geometry_scale_preserves_x_and_interpolates_yz()) return 1;
    if (test_course_geometry_scale_randomization_is_per_agent_state()) return 1;
    if (test_gate_radius_randomization_is_shared_per_episode()) return 1;
    if (test_gate_radius_randomization_can_target_one_gate()) return 1;
    if (test_gate_radius_profile_mix_preserves_exact_anchor_and_target()) return 1;
    if (test_gate_radius_profile_mix_preserves_vehicle_rng()) return 1;
    if (test_motor_control_semantics()) return 1;
    if (test_strict_near_miss_terminates_at_gate_plane()) return 1;
    if (test_gate2_terminal_crossing_diagnostics_are_conditioned()) return 1;
    if (test_gate1_action_diagnostics_capture_early_window()) return 1;
    if (test_stage_transition_reproduces_official_entry_speed()) return 1;
    if (test_transition_speed_ramp_is_capped_and_opt_in()) return 1;
    if (test_transition_speed_can_preserve_outgoing_direction()) return 1;
    if (test_transition_speed_impulse_can_wait_for_gate_association()) return 1;
    if (test_timeout_matches_qualifier_timing()) return 1;
    if (test_evaluation_episode_offset_advances_reset_sequence_once()) return 1;
    if (test_observation_bounds()) return 1;
    if (test_gate_position_randomization_is_per_agent_state()) return 1;
    if (test_gate_position_randomization_can_mix_nominal_anchor_episodes()) return 1;
    if (test_gate_position_randomization_preserves_start_rng_stream()) return 1;
    if (test_ts002_observation_contract()) return 1;
    if (test_ts002_gate_track_holds_across_detector_dropout()) return 1;
    if (test_ts002_gate_track_predicts_across_detector_dropout_when_enabled()) return 1;
    if (test_ts002_gate_track_integrates_roll_acceleration_when_enabled()) return 1;
    if (test_ts002_velocity_yaw_action_contract()) return 1;
    if (test_sitl_attitude_plant_closes_absolute_setpoint()) return 1;
    if (test_sitl_attitude_hover_thrust()) return 1;
    if (test_sitl_attitude_forward_sign_matches_official_sim()) return 1;
    if (test_sitl_attitude_roll_sign_matches_official_sim()) return 1;
    if (test_sitl_stage_lock_holds_hover_heading_and_lateral()) return 1;
    if (test_sitl_hover_lock_compensates_forward_tilt()) return 1;
    if (test_sitl_teacher_reward_prefers_forward_contract()) return 1;
    if (test_sitl_teacher_reward_prefers_descent_to_lower_gate()) return 1;
    if (test_sitl_teacher_reward_supports_thrust_bias()) return 1;
    if (test_sitl_teacher_reward_can_arrest_observed_descent_rate()) return 1;
    if (test_sitl_teacher_reward_supports_roll_bias()) return 1;
    if (test_sitl_teacher_can_shape_only_lateral_after_gate_handoff()) return 1;
    if (test_sitl_teacher_roll_can_stop_before_next_gate()) return 1;
    if (test_sitl_teacher_intervention_blends_only_active_channels()) return 1;
    if (test_sitl_teacher_yaw_intervention_is_explicitly_opt_in()) return 1;
    if (test_sitl_teacher_intervention_can_wait_for_training_step()) return 1;
    if (test_sitl_teacher_intervention_zero_preserves_policy_dynamics()) return 1;
    if (test_sitl_teacher_pitch_speed_intervention_brakes_and_accelerates()) return 1;
    if (test_sitl_teacher_world_vertical_ignores_bank_coupling()) return 1;
    if (test_sitl_teacher_true_velocity_damping_is_default_off()) return 1;
    if (test_sitl_attitude_observation_contract()) return 1;
    if (test_sitl_observation_can_expose_gate_phase()) return 1;
    if (test_sitl_gate_phase_can_keep_full_course_denominator()) return 1;
    if (test_gate_exit_reward_targets_next_segment_lateral_velocity()) return 1;
    if (test_gate_exit_reward_targets_feasible_3d_velocity()) return 1;
    if (test_gate_exit_reward_skips_hidden_randomized_next_gate()) return 1;
    if (test_forward_speed_excess_penalty_has_no_low_speed_floor()) return 1;
    if (test_transition_speed_curriculum_samples_per_episode()) return 1;
    if (test_cross_track_reward_can_start_at_late_gate()) return 1;
    if (test_gate_crossing_error_reward_targets_configured_gate()) return 1;
    if (test_gate_camera_alignment_reward_is_centered_and_phase_local()) return 1;
    if (test_late_invalid_penalty_preserves_prefix()) return 1;
    if (test_course_spline_teacher_is_reward_only()) return 1;
    if (test_minimum_jerk_segment_reference_is_transition_continuous()) return 1;
    if (test_alignment_governor_brakes_and_centers_with_default_off()) return 1;
    if (test_segment_curriculum_starts_at_requested_gate_and_velocity()) return 1;
    if (test_measured_segment_start_can_disable_legacy_position_noise()) return 1;
    if (test_segment_start_distribution_is_bounded_and_non_degenerate()) return 1;
    if (test_mixed_curriculum_preserves_full_and_segment_starts()) return 1;
    if (test_vq1_telemetry_velocity_plant_tracks_local_command()) return 1;
    if (test_vq1_telemetry_observation_is_exact_local_course_state()) return 1;
    if (test_vq1_telemetry_teacher_targets_active_gate()) return 1;
    if (test_floor_risk_diagnostics()) return 1;
    printf("drone_race native regressions ok\n");
    return 0;
}
