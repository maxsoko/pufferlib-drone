#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define Log DroneBaseLog
#include "../drone/dronelib.h"
#undef Log

#define DRONE_RACE_OBS_SIZE 23
#define DRONE_RACE_NUM_ATNS 4
#define DRONE_RACE_MAX_GATES 16
#define DRONE_RACE_INTERFACE_NATIVE_MOTOR 0
#define DRONE_RACE_INTERFACE_TS002_VELOCITY_YAW 1

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float valid_run_rate;
    float success_rate;
    float gates_passed;
    float completion_time;
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
    float out_of_order;
    float missed_gate;
    float timeout;
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
    float progress;
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
    int step_count;
    int current_gate;
    int valid_run;
    int floor_impact_risk;
    int floor_stop_violation;
    int floor_risk_sampled;
    int out_of_order;
    int missed_gate;
    int crash;
    int crash_low;
    int crash_high;
    int crash_xy;
    int timeout;
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
    float gate_spacing;
    float gate_radius;
    float gate_altitude;
    float gate_lateral_amplitude;
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
    float w_finish;
    float w_time;
    float w_ctrl;
    float w_altitude_floor;
    float w_descent_floor;
    float invalid_penalty;
    int interface_mode;
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

    Target gates[DRONE_RACE_MAX_GATES];
    float between_gate_distance[DRONE_RACE_MAX_GATES];
    DroneRaceAgent* agents;
} DroneRace;

void init(DroneRace* env);
void c_reset(DroneRace* env);
void c_step(DroneRace* env);
void c_render(DroneRace* env);
void c_close(DroneRace* env);
