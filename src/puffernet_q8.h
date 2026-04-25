#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "puffernet.h"

// Mixed-precision Q8 PufferNet runtime.
//
// This intentionally keeps nonlinear MinGRU math and recurrent state in float
// for the first edge-inference milestone. Linear layers use int8 weights,
// dynamic int8 activations, int32 accumulators, and per-output-channel scales.
// That gives us a deterministic low-bandwidth MAC path while preserving enough
// control fidelity to measure closed-loop drift before tightening state/gates.

typedef struct Q8Linear Q8Linear;
struct Q8Linear {
    int8_t* weights;
    int8_t* qinput;
    float* weight_scales;
    float* input_scales;
    float* output;
    int batch_size;
    int input_dim;
    int output_dim;
};

static inline int8_t q8_quantize_unit(float x) {
    x = fmaxf(-1.0f, fminf(1.0f, x));
    int v = (int)lrintf(x * 127.0f);
    if (v < -127) v = -127;
    if (v > 127) v = 127;
    return (int8_t)v;
}

static inline Q8Linear* make_q8linear_from_float(
        float* weights, int batch_size, int input_dim, int output_dim) {
    size_t weight_count = (size_t)input_dim * output_dim;
    size_t buffer_size = weight_count * sizeof(int8_t)
        + (size_t)batch_size * input_dim * sizeof(int8_t)
        + (size_t)output_dim * sizeof(float)
        + (size_t)batch_size * sizeof(float)
        + (size_t)batch_size * output_dim * sizeof(float);
    Q8Linear* layer = (Q8Linear*)calloc(1, sizeof(Q8Linear) + buffer_size);
    char* buffer = (char*)(layer + 1);
    layer->weights = (int8_t*)buffer;
    buffer += weight_count * sizeof(int8_t);
    layer->qinput = (int8_t*)buffer;
    buffer += (size_t)batch_size * input_dim * sizeof(int8_t);
    layer->weight_scales = (float*)buffer;
    buffer += (size_t)output_dim * sizeof(float);
    layer->input_scales = (float*)buffer;
    buffer += (size_t)batch_size * sizeof(float);
    layer->output = (float*)buffer;
    layer->batch_size = batch_size;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;

    for (int o = 0; o < output_dim; o++) {
        float max_abs = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            float w = weights[o * input_dim + i];
            max_abs = fmaxf(max_abs, fabsf(w));
        }
        float scale = max_abs > 1e-8f ? max_abs / 127.0f : 1.0f / 127.0f;
        layer->weight_scales[o] = scale;
        for (int i = 0; i < input_dim; i++) {
            float normalized = weights[o * input_dim + i] / (127.0f * scale);
            layer->weights[o * input_dim + i] = q8_quantize_unit(normalized);
        }
    }

    return layer;
}

static inline void q8linear(Q8Linear* layer, float* input) {
    for (int b = 0; b < layer->batch_size; b++) {
        float max_abs = 0.0f;
        for (int i = 0; i < layer->input_dim; i++) {
            max_abs = fmaxf(max_abs, fabsf(input[b * layer->input_dim + i]));
        }
        float scale = max_abs > 1e-8f ? max_abs / 127.0f : 1.0f / 127.0f;
        layer->input_scales[b] = scale;
        for (int i = 0; i < layer->input_dim; i++) {
            float normalized = input[b * layer->input_dim + i] / (127.0f * scale);
            layer->qinput[b * layer->input_dim + i] = q8_quantize_unit(normalized);
        }
    }

    for (int b = 0; b < layer->batch_size; b++) {
        for (int o = 0; o < layer->output_dim; o++) {
            int32_t acc = 0;
            for (int i = 0; i < layer->input_dim; i++) {
                acc += (int32_t)layer->qinput[b * layer->input_dim + i]
                    * (int32_t)layer->weights[o * layer->input_dim + i];
            }
            layer->output[b * layer->output_dim + o] =
                (float)acc * layer->input_scales[b] * layer->weight_scales[o];
        }
    }
}

typedef struct Q8MinGRU Q8MinGRU;
struct Q8MinGRU {
    float* state;
    float* output;
    Q8Linear** proj;
    int batch_size;
    int hidden_size;
    int num_layers;
};

static inline Q8MinGRU* make_q8mingru_from_float(MinGRU* src) {
    Q8MinGRU* layer = (Q8MinGRU*)calloc(1, sizeof(Q8MinGRU));
    int B = src->batch_size;
    int H = src->hidden_size;
    int L = src->num_layers;
    layer->state = (float*)calloc((size_t)L * B * H, sizeof(float));
    layer->output = (float*)calloc((size_t)B * H, sizeof(float));
    layer->proj = (Q8Linear**)calloc((size_t)L, sizeof(Q8Linear*));
    layer->batch_size = B;
    layer->hidden_size = H;
    layer->num_layers = L;
    for (int l = 0; l < L; l++) {
        layer->proj[l] = make_q8linear_from_float(src->proj[l]->weights, B, H, 3 * H);
    }
    return layer;
}

static inline void q8mingru(Q8MinGRU* layer, float* input) {
    int B = layer->batch_size;
    int H = layer->hidden_size;
    float* x = input;
    for (int l = 0; l < layer->num_layers; l++) {
        float* state_l = layer->state + (size_t)l * B * H;
        q8linear(layer->proj[l], x);
        float* combined = layer->proj[l]->output;
        for (int b = 0; b < B; b++) {
            float* cb = combined + (size_t)b * 3 * H;
            float* sb = state_l + (size_t)b * H;
            float* xb = x + (size_t)b * H;
            float* ob = layer->output + (size_t)b * H;
            for (int h = 0; h < H; h++) {
                float hidden = cb[h];
                float gate = cb[H + h];
                float hw = cb[2 * H + h];
                float s = sb[h];
                float gate_s = _sigmoid(gate);
                float h_tilde = hidden >= 0.0f ? hidden + 0.5f : _sigmoid(hidden);
                float mingru_out = s + gate_s * (h_tilde - s);
                float hw_s = _sigmoid(hw);
                ob[h] = hw_s * mingru_out + (1.0f - hw_s) * xb[h];
                sb[h] = mingru_out;
            }
        }
        x = layer->output;
    }
}

static inline void free_q8mingru(Q8MinGRU* layer) {
    for (int l = 0; l < layer->num_layers; l++) {
        free(layer->proj[l]);
    }
    free(layer->state);
    free(layer->output);
    free(layer->proj);
    free(layer);
}

typedef struct Q8PufferNet Q8PufferNet;
struct Q8PufferNet {
    int num_agents;
    Q8Linear* encoder;
    Q8MinGRU* mingru;
    Q8Linear* decoder;
    int is_continuous;
    int num_actions;
};

static inline Q8PufferNet* make_q8_puffernet_from_float(PufferNet* src) {
    Q8PufferNet* net = (Q8PufferNet*)calloc(1, sizeof(Q8PufferNet));
    net->num_agents = src->num_agents;
    net->encoder = make_q8linear_from_float(
        src->encoder->weights, src->encoder->batch_size,
        src->encoder->input_dim, src->encoder->output_dim);
    net->mingru = make_q8mingru_from_float(src->mingru);
    net->decoder = make_q8linear_from_float(
        src->decoder->weights, src->decoder->batch_size,
        src->decoder->input_dim, src->decoder->output_dim);
    net->is_continuous = src->is_continuous;
    net->num_actions = src->num_actions;
    return net;
}

static inline void forward_q8_puffernet(Q8PufferNet* net, float* observations, float* actions) {
    q8linear(net->encoder, observations);
    q8mingru(net->mingru, net->encoder->output);
    q8linear(net->decoder, net->mingru->output);
    if (net->is_continuous) {
        _gaussian_mean(net->decoder->output, actions, net->num_agents, net->num_actions);
    } else {
        // Discrete Q8 policies are not needed for drone control yet.
        memset(actions, 0, (size_t)net->num_agents * net->num_actions * sizeof(float));
    }
}

static inline void free_q8_puffernet(Q8PufferNet* net) {
    free(net->encoder);
    free_q8mingru(net->mingru);
    free(net->decoder);
    free(net);
}
