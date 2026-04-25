#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ocean/drone/drone.h"
#include "src/puffernet_q8.h"

void c_close_client(Client* client) {
    (void)client;
}

static void configure_env(DroneEnv* env, int num_agents, unsigned int seed) {
    env->num_agents = num_agents;
    env->rng = seed;
    env->max_rings = 10;
    env->task = HOVER;
    env->alpha_dist = 0.782192f;
    env->alpha_hover = 0.071445f;
    env->alpha_shaping = 3.9754f;
    env->alpha_omega = 0.00135588f;
    env->hover_target_dist = 5.0f;
    env->hover_dist = 0.1f;
    env->hover_omega = 0.1f;
    env->hover_vel = 0.1f;
    env->observations = (float*)calloc((size_t)num_agents * 23, sizeof(float));
    env->actions = (float*)calloc((size_t)num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc((size_t)num_agents, sizeof(float));
    env->terminals = (float*)calloc((size_t)num_agents, sizeof(float));
    init(env);
    c_reset(env);
}

static void free_env(DroneEnv* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : "resources/drone/drone_weights.bin";
    int steps = argc > 2 ? atoi(argv[2]) : 1024;
    int num_agents = argc > 3 ? atoi(argv[3]) : 64;
    int hidden_size = argc > 4 ? atoi(argv[4]) : 128;
    int num_layers = argc > 5 ? atoi(argv[5]) : 3;
    int logit_sizes[4] = {1, 1, 1, 1};

    Weights* weights = load_weights(weights_path);
    if (weights == NULL) {
        fprintf(stderr, "failed to load weights: %s\n", weights_path);
        return 1;
    }

    DroneEnv fp32_env = {0};
    DroneEnv q8_env = {0};
    configure_env(&fp32_env, num_agents, 123);
    configure_env(&q8_env, num_agents, 123);

    PufferNet* fp32 = make_puffernet(weights, num_agents, 23, hidden_size, num_layers, logit_sizes, 4);
    Q8PufferNet* q8 = make_q8_puffernet_from_float(fp32);

    float mean_abs_action_diff = 0.0f;
    float max_abs_action_diff = 0.0f;
    float mean_abs_reward_diff = 0.0f;
    int action_count = 0;
    int reward_count = 0;

    for (int step = 0; step < steps; step++) {
        forward_puffernet(fp32, fp32_env.observations, fp32_env.actions);
        forward_q8_puffernet(q8, q8_env.observations, q8_env.actions);

        for (int i = 0; i < num_agents * 4; i++) {
            float d = fabsf(fp32_env.actions[i] - q8_env.actions[i]);
            mean_abs_action_diff += d;
            if (d > max_abs_action_diff) max_abs_action_diff = d;
            action_count += 1;
        }

        c_step(&fp32_env);
        c_step(&q8_env);

        for (int i = 0; i < num_agents; i++) {
            mean_abs_reward_diff += fabsf(fp32_env.rewards[i] - q8_env.rewards[i]);
            reward_count += 1;
        }
    }

    mean_abs_action_diff /= fmaxf((float)action_count, 1.0f);
    mean_abs_reward_diff /= fmaxf((float)reward_count, 1.0f);

    printf("weights=%s steps=%d agents=%d hidden=%d layers=%d\n",
        weights_path, steps, num_agents, hidden_size, num_layers);
    printf("mean_abs_action_diff=%.6f max_abs_action_diff=%.6f mean_abs_reward_diff=%.6f\n",
        mean_abs_action_diff, max_abs_action_diff, mean_abs_reward_diff);
    printf("fp32_log_n=%.0f fp32_score=%.6f fp32_perf=%.6f fp32_oob=%.0f\n",
        fp32_env.log.n, fp32_env.log.score, fp32_env.log.perf, fp32_env.log.oob);
    printf("q8_log_n=%.0f q8_score=%.6f q8_perf=%.6f q8_oob=%.0f\n",
        q8_env.log.n, q8_env.log.score, q8_env.log.perf, q8_env.log.oob);

    free_q8_puffernet(q8);
    free_puffernet(fp32);
    free(weights);
    free_env(&fp32_env);
    free_env(&q8_env);
    return 0;
}
