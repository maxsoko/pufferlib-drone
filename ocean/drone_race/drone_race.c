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
static const float TS002_CAMERA_UPTILT_RAD = 0.3490658504f;
static const float TS002_CAMERA_HALF_FOV_RAD = 0.7853981634f;
static const float TS002_GATE_INNER_WIDTH_M = 1.5f;

typedef struct {
    float roll;
    float pitch;
    float yaw;
} EulerAngles;

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

static void apply_interface_defaults(DroneRace* env) {
    if (env->interface_mode != DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW) {
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
    agent->min_floor_ttf = FLOOR_TTF_SENTINEL;
    agent->min_floor_stop_margin = FLOOR_TTF_SENTINEL;
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
    env->log.out_of_order += agent->out_of_order ? 1.0f : 0.0f;
    env->log.missed_gate += agent->missed_gate ? 1.0f : 0.0f;
    env->log.timeout += agent->timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static void compute_one_observation(DroneRace* env, int i) {
    DroneRaceAgent* agent = &env->agents[i];
    float* obs = &env->observations[i * DRONE_RACE_OBS_SIZE];
    if (env->interface_mode == DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW) {
        memset(obs, 0, DRONE_RACE_OBS_SIZE * sizeof(float));
        Quat q = agent->drone.state.quat;
        Quat q_inv = quat_inverse(q);
        Vec3 vel_body = quat_rotate(q_inv, agent->drone.state.vel);

        int idx = 0;
        obs[idx++] = vel_body.x / 10.0f;
        obs[idx++] = vel_body.y / 10.0f;
        obs[idx++] = -vel_body.z / 10.0f;
        obs[idx++] = agent->drone.state.omega.x / agent->drone.params.max_omega;
        obs[idx++] = agent->drone.state.omega.y / agent->drone.params.max_omega;
        obs[idx++] = agent->drone.state.omega.z / agent->drone.params.max_omega;
        obs[idx++] = q.w;
        obs[idx++] = q.x;
        obs[idx++] = q.y;
        obs[idx++] = q.z;

        if (agent->current_gate < env->num_gates) {
            Vec3 rel_world = sub3(env->gates[agent->current_gate].pos, agent->drone.state.pos);
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
    } else {
        compute_drone_observations(&agent->drone, obs);
    }
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
    apply_interface_defaults(env);
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
        float* policy_action = &env->actions[i * DRONE_RACE_NUM_ATNS];
        float motor_action[DRONE_RACE_NUM_ATNS] = {0};

        agent->drone.prev_pos = agent->drone.state.pos;
        float prev_remaining = remaining_distance(env, agent);
        for (int k = 0; k < DRONE_RACE_NUM_ATNS; k++) {
            agent->last_action[k] = policy_action[k];
        }
        FloorRisk pre_floor_risk = compute_floor_risk(env, agent);
        update_floor_risk_stats(agent, pre_floor_risk);

        policy_to_motor_actions(env, agent, policy_action, motor_action);
        move_drone_for_dt(&agent->drone, motor_action, env->dt);
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
