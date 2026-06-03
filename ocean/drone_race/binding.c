#define DRONE_RACE_BINDING
#include "drone_race.c"

#define OBS_SIZE DRONE_RACE_OBS_SIZE
#define NUM_ATNS DRONE_RACE_NUM_ATNS
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

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
    env->gate_spacing = get_float(kwargs, "gate_spacing", 5.0f);
    env->gate_radius = get_float(kwargs, "gate_radius", 1.0f);
    env->gate_altitude = get_float(kwargs, "gate_altitude", 1.0f);
    env->gate_lateral_amplitude = get_float(kwargs, "gate_lateral_amplitude", 1.5f);
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
    env->w_finish = get_float(kwargs, "w_finish", 30.0f);
    env->w_time = get_float(kwargs, "w_time", 1.0f);
    env->w_ctrl = get_float(kwargs, "w_ctrl", 0.01f);
    env->w_altitude_floor = get_float(kwargs, "w_altitude_floor", 0.0f);
    env->w_descent_floor = get_float(kwargs, "w_descent_floor", 0.0f);
    env->invalid_penalty = get_float(kwargs, "invalid_penalty", 40.0f);
    env->interface_mode = get_int(kwargs, "interface_mode", DRONE_RACE_INTERFACE_NATIVE_MOTOR);
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
    dict_set(out, "completion_time", log->completion_time);
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
    dict_set(out, "out_of_order", log->out_of_order);
    dict_set(out, "missed_gate", log->missed_gate);
    dict_set(out, "timeout", log->timeout);
}
