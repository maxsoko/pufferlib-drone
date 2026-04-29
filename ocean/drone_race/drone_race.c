#include "drone_race.h"
#include <stdio.h>
#include <time.h>

static inline float race_dist3(Vec3 a, Vec3 b) {
    return norm3(sub3(a, b));
}

static void build_course(DroneRace* env) {
    if (env->num_gates < 1) env->num_gates = 1;
    if (env->num_gates > DRONE_RACE_MAX_GATES) env->num_gates = DRONE_RACE_MAX_GATES;

    for (int i = 0; i < env->num_gates; i++) {
        float y = (i % 2 == 0) ? env->gate_lateral_amplitude : -env->gate_lateral_amplitude;
        env->gates[i] = (Target){
            .pos = {i * env->gate_spacing, y, env->gate_altitude},
            .vel = {0.0f, 0.0f, 0.0f},
            .orientation = {1.0f, 0.0f, 0.0f, 0.0f},
            .normal = {1.0f, 0.0f, 0.0f},
            .radius = env->gate_radius,
        };
        env->between_gate_distance[i] = 0.0f;
    }

    for (int i = 0; i < env->num_gates - 1; i++) {
        env->between_gate_distance[i] = race_dist3(env->gates[i + 1].pos, env->gates[i].pos);
    }
}

static float remaining_distance(DroneRace* env, DroneRaceAgent* agent) {
    if (agent->current_gate >= env->num_gates) {
        return 0.0f;
    }

    float rem = race_dist3(agent->drone.state.pos, env->gates[agent->current_gate].pos);
    for (int i = agent->current_gate; i < env->num_gates - 1; i++) {
        rem += env->between_gate_distance[i];
    }
    return rem;
}

static bool gate_crossing(
        Vec3 prev_pos,
        Vec3 pos,
        Target* gate,
        float plane_tol,
        float direction_min,
        float* radial_out) {
    Vec3 move = sub3(pos, prev_pos);
    float move_norm = norm3(move);
    if (move_norm < 1e-8f) return false;

    Vec3 move_dir = scalmul3(move, 1.0f / move_norm);
    Vec3 prev_rel = sub3(prev_pos, gate->pos);
    Vec3 curr_rel = sub3(pos, gate->pos);
    float d_prev = dot3(prev_rel, gate->normal);
    float d_curr = dot3(curr_rel, gate->normal);

    if (!(d_prev <= -plane_tol && d_curr >= plane_tol)) return false;
    float denom = d_prev - d_curr;
    if (fabsf(denom) < 1e-8f) return false;

    float t = clampf(d_prev / denom, 0.0f, 1.0f);
    Vec3 hit = add3(prev_pos, scalmul3(move, t));
    Vec3 radial = sub3(hit, gate->pos);
    float normal_part = dot3(radial, gate->normal);
    radial = sub3(radial, scalmul3(gate->normal, normal_part));

    float radial_dist = norm3(radial);
    if (radial_out != NULL) *radial_out = radial_dist;
    return radial_dist <= gate->radius && dot3(move_dir, gate->normal) > direction_min;
}

static void set_agent_target(DroneRace* env, DroneRaceAgent* agent) {
    int gate_idx = agent->current_gate < env->num_gates ? agent->current_gate : env->num_gates - 1;
    agent->target = env->gates[gate_idx];
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

static void reset_agent(DroneRace* env, DroneRaceAgent* agent) {
    memset(agent, 0, sizeof(*agent));
    agent->valid_run = 1;
    agent->drone.target = &agent->target;
    init_drone(&agent->drone, &env->rng, 0.05f);

    Target* first_gate = &env->gates[0];
    agent->drone.state.pos = sub3(first_gate->pos, scalmul3(first_gate->normal, env->start_offset));
    agent->drone.state.pos.x += rndf(-0.1f, 0.1f, &env->rng);
    agent->drone.state.pos.y += rndf(-0.1f, 0.1f, &env->rng);
    agent->drone.state.pos.z += rndf(-0.05f, 0.05f, &env->rng);
    agent->drone.prev_pos = agent->drone.state.pos;
    set_agent_target(env, agent);
}

static void add_log(DroneRace* env, DroneRaceAgent* agent, bool success) {
    env->log.perf += success ? 1.0f : 0.0f;
    env->log.score += (float)agent->current_gate;
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += (float)agent->step_count;
    env->log.valid_run_rate += agent->valid_run ? 1.0f : 0.0f;
    env->log.success_rate += success ? 1.0f : 0.0f;
    env->log.gates_passed += (float)agent->current_gate;
    env->log.completion_time += success ? agent->elapsed_time : 0.0f;
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
    env->log.out_of_order += agent->out_of_order ? 1.0f : 0.0f;
    env->log.missed_gate += agent->missed_gate ? 1.0f : 0.0f;
    env->log.timeout += agent->timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static void compute_one_observation(DroneRace* env, int i) {
    DroneRaceAgent* agent = &env->agents[i];
    float* obs = &env->observations[i * DRONE_RACE_OBS_SIZE];
    compute_drone_observations(&agent->drone, obs);
    for (int j = 0; j < DRONE_RACE_OBS_SIZE; j++) {
        obs[j] = clampf(obs[j], -1.0f, 1.0f);
    }
}

static void compute_observations(DroneRace* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_one_observation(env, i);
    }
}

void init(DroneRace* env) {
    build_course(env);
    env->agents = (DroneRaceAgent*)calloc(env->num_agents, sizeof(DroneRaceAgent));
    memset(&env->log, 0, sizeof(Log));
}

void c_reset(DroneRace* env) {
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, &env->agents[i]);
    }
    compute_observations(env);
}

void c_step(DroneRace* env) {
    for (int i = 0; i < env->num_agents; i++) {
        DroneRaceAgent* agent = &env->agents[i];
        float* action = &env->actions[i * DRONE_RACE_NUM_ATNS];

        agent->drone.prev_pos = agent->drone.state.pos;
        float prev_remaining = remaining_distance(env, agent);
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            agent->last_action[k] = action[k];
        }

        move_drone_for_dt(&agent->drone, action, env->dt);
        agent->elapsed_time += env->dt;
        agent->step_count += 1;

        bool success = false;
        float radial = 0.0f;
        if (agent->current_gate < env->num_gates && gate_crossing(
                agent->drone.prev_pos, agent->drone.state.pos,
                &env->gates[agent->current_gate],
                env->plane_cross_tolerance,
                env->direction_min, &radial)) {
            agent->current_gate += 1;
            set_agent_target(env, agent);
            if (agent->current_gate >= env->num_gates) {
                success = true;
            }
        } else if (agent->current_gate < env->num_gates && env->strict_missed_gate) {
            Target miss_gate = env->gates[agent->current_gate];
            miss_gate.radius += env->miss_tolerance;
            float miss_radial = 0.0f;
            bool crossed_plane = gate_crossing(
                agent->drone.prev_pos, agent->drone.state.pos,
                &miss_gate,
                env->plane_cross_tolerance,
                env->direction_min,
                &miss_radial);
            if (!crossed_plane && miss_radial > env->gate_radius + env->miss_tolerance) {
                agent->missed_gate = 1;
                agent->valid_run = 0;
            }
        }

        for (int gate = agent->current_gate + 1; gate < env->num_gates; gate++) {
            if (gate_crossing(agent->drone.prev_pos, agent->drone.state.pos, &env->gates[gate],
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

        Vec3 pos = agent->drone.state.pos;
        bool crash_xy = fabsf(pos.x) > env->pos_bound || fabsf(pos.y) > env->pos_bound;
        bool crash_low = pos.z < env->crash_height;
        bool crash_high = pos.z > env->pos_bound;
        if (crash_xy || crash_low || crash_high) {
            agent->crash = 1;
            agent->crash_xy = crash_xy ? 1 : 0;
            agent->crash_low = crash_low ? 1 : 0;
            agent->crash_high = crash_high ? 1 : 0;
            agent->valid_run = 0;
        }
        if (agent->step_count >= env->max_steps || agent->elapsed_time >= env->time_limit_seconds) {
            agent->timeout = 1;
        }

        float ctrl = 0.0f;
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) ctrl += action[k] * action[k];
        float reward = env->w_progress * progress_delta - env->w_time * env->dt - env->w_ctrl * ctrl;
        float altitude_deficit = fmaxf(env->safety_altitude - pos.z, 0.0f);
        if (env->w_altitude_floor > 0.0f && altitude_deficit > 0.0f) {
            reward -= env->w_altitude_floor * altitude_deficit * altitude_deficit;
        }
        if (env->w_descent_floor > 0.0f && altitude_deficit > 0.0f && agent->drone.state.vel.z < 0.0f) {
            reward -= env->w_descent_floor * -agent->drone.state.vel.z;
        }
        if (agent->current_gate > 0 && progress_delta > 0.0f) reward += env->w_gate * progress_delta;
        if (success) reward += env->w_finish;
        if (!agent->valid_run) reward -= env->invalid_penalty;

        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool done = success || !agent->valid_run || agent->timeout;
        env->terminals[i] = done ? 1.0f : 0.0f;
        if (done) {
            add_log(env, agent, success && agent->valid_run);
            reset_agent(env, agent);
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
