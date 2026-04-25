#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "src/puffernet_q8.h"

typedef struct {
    Weights weights;
    float* data;
    int capacity;
} WeightBuilder;

static int align8(int idx) {
    return (idx + 7) & ~7;
}

static WeightBuilder make_weight_builder(int capacity) {
    WeightBuilder builder = {0};
    builder.data = (float*)calloc((size_t)capacity, sizeof(float));
    builder.capacity = capacity;
    builder.weights.data = builder.data;
    builder.weights.size = capacity;
    builder.weights.idx = 0;
    return builder;
}

static void append_aligned(WeightBuilder* builder, int count, float scale, unsigned int* rng) {
    int idx = builder->weights.idx;
    if (idx + count >= builder->capacity) {
        fprintf(stderr, "weight builder overflow\n");
        exit(1);
    }
    for (int i = 0; i < count; i++) {
        float u = ((float)rand_r(rng) / (float)RAND_MAX) * 2.0f - 1.0f;
        builder->data[idx + i] = u * scale;
    }
    builder->weights.idx = align8(idx + count);
}

static WeightBuilder make_puffernet_weights(
        int input_dim, int hidden_dim, int num_layers, int num_actions) {
    int decoder_dim = num_actions + 1;
    int approximate = align8(hidden_dim * input_dim)
        + align8(decoder_dim * hidden_dim)
        + align8(num_actions)
        + num_layers * align8(3 * hidden_dim * hidden_dim)
        + 64;
    WeightBuilder builder = make_weight_builder(approximate);
    unsigned int rng = 7;
    append_aligned(&builder, hidden_dim * input_dim, 0.15f, &rng);
    append_aligned(&builder, decoder_dim * hidden_dim, 0.10f, &rng);
    append_aligned(&builder, num_actions, 0.01f, &rng);
    for (int l = 0; l < num_layers; l++) {
        append_aligned(&builder, 3 * hidden_dim * hidden_dim, 0.05f, &rng);
    }
    builder.weights.idx = 0;
    return builder;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    int batch = argc > 1 ? atoi(argv[1]) : 64;
    int input_dim = argc > 2 ? atoi(argv[2]) : 25;
    int hidden_dim = argc > 3 ? atoi(argv[3]) : 128;
    int num_layers = argc > 4 ? atoi(argv[4]) : 3;
    int iters = argc > 5 ? atoi(argv[5]) : 1000;
    int num_actions = 4;
    int logit_sizes[4] = {1, 1, 1, 1};

    WeightBuilder builder = make_puffernet_weights(input_dim, hidden_dim, num_layers, num_actions);
    PufferNet* fp32 = make_puffernet(&builder.weights, batch, input_dim, hidden_dim, num_layers, logit_sizes, num_actions);
    Q8PufferNet* q8 = make_q8_puffernet_from_float(fp32);

    float* obs = (float*)calloc((size_t)batch * input_dim, sizeof(float));
    float* actions_fp32 = (float*)calloc((size_t)batch * num_actions, sizeof(float));
    float* actions_q8 = (float*)calloc((size_t)batch * num_actions, sizeof(float));
    for (int i = 0; i < batch * input_dim; i++) {
        obs[i] = sinf((float)i * 0.013f);
    }

    double t0 = now_seconds();
    for (int i = 0; i < iters; i++) {
        forward_puffernet(fp32, obs, actions_fp32);
    }
    double fp32_seconds = now_seconds() - t0;

    t0 = now_seconds();
    for (int i = 0; i < iters; i++) {
        forward_q8_puffernet(q8, obs, actions_q8);
    }
    double q8_seconds = now_seconds() - t0;

    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    int action_count = batch * num_actions;
    for (int i = 0; i < action_count; i++) {
        float d = fabsf(actions_fp32[i] - actions_q8[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        mean_abs_diff += d;
    }
    mean_abs_diff /= (float)action_count;

    printf("batch=%d input_dim=%d hidden_dim=%d layers=%d iters=%d\n",
        batch, input_dim, hidden_dim, num_layers, iters);
    printf("fp32_total_ms=%.3f fp32_us_per_iter=%.3f\n",
        fp32_seconds * 1000.0, fp32_seconds * 1e6 / iters);
    printf("q8_total_ms=%.3f q8_us_per_iter=%.3f\n",
        q8_seconds * 1000.0, q8_seconds * 1e6 / iters);
    printf("mean_abs_action_diff=%.6f max_abs_action_diff=%.6f\n",
        mean_abs_diff, max_abs_diff);

    free(actions_fp32);
    free(actions_q8);
    free(obs);
    free_q8_puffernet(q8);
    free_puffernet(fp32);
    free(builder.data);
    return 0;
}
