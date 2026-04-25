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
    int step_count;
    int current_gate;
    int valid_run;
    int out_of_order;
    int missed_gate;
    int crash;
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
    float w_progress;
    float w_gate;
    float w_finish;
    float w_time;
    float w_ctrl;
    float invalid_penalty;

    Target gates[DRONE_RACE_MAX_GATES];
    float between_gate_distance[DRONE_RACE_MAX_GATES];
    DroneRaceAgent* agents;
} DroneRace;

void init(DroneRace* env);
void c_reset(DroneRace* env);
void c_step(DroneRace* env);
void c_render(DroneRace* env);
void c_close(DroneRace* env);
