#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/puffernet.h"

#define OBS 32
#define ACTIONS 4
#define WIDTH (OBS + ACTIONS + 1)

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s CHECKPOINT.bin DATASET.bin\n", argv[0]);
        return 2;
    }
    FILE* data = fopen(argv[2], "rb");
    if (data == NULL) return 2;
    Weights* weights = load_weights(argv[1]);
    if (weights == NULL) return 2;
    int logit_sizes[ACTIONS] = {1, 1, 1, 1};
    PufferNet* net = make_puffernet(weights, 1, OBS, 128, 3, logit_sizes, ACTIONS);
    float record[WIDTH];
    float predicted[ACTIONS];
    double squared[ACTIONS] = {0};
    double phase_squared[4][ACTIONS] = {{0}};
    long phase_rows[4] = {0};
    long rows = 0;
    while (fread(record, sizeof(record), 1, data) == 1) {
        if (record[WIDTH - 1] > 0.5f) {
            memset(net->mingru->state, 0,
                (size_t)net->mingru->num_layers * net->mingru->batch_size
                    * net->mingru->hidden_size * sizeof(float));
        }
        forward_puffernet(net, record, predicted);
        if (rows == 0) {
            printf("first predicted=(%.6f,%.6f,%.6f,%.6f) target=(%.6f,%.6f,%.6f,%.6f)\n",
                predicted[0], predicted[1], predicted[2], predicted[3],
                record[OBS], record[OBS + 1], record[OBS + 2], record[OBS + 3]);
        }
        for (int k = 0; k < ACTIONS; k++) {
            double error = (double)predicted[k] - record[OBS + k];
            squared[k] += error * error;
            int phase = record[25] > 0.5f ? 3
                : record[24] > 0.5f ? 2
                : record[23] > 0.5f ? 1 : 0;
            phase_squared[phase][k] += error * error;
        }
        int phase = record[25] > 0.5f ? 3
            : record[24] > 0.5f ? 2
            : record[23] > 0.5f ? 1 : 0;
        phase_rows[phase]++;
        rows++;
    }
    fclose(data);
    printf("rows=%ld rmse=(%.6f,%.6f,%.6f,%.6f)\n", rows,
        sqrt(squared[0] / rows), sqrt(squared[1] / rows),
        sqrt(squared[2] / rows), sqrt(squared[3] / rows));
    for (int phase = 0; phase < 4; phase++) {
        if (phase_rows[phase] == 0) continue;
        printf("phase=%d rows=%ld rmse=(%.6f,%.6f,%.6f,%.6f)\n",
            phase, phase_rows[phase],
            sqrt(phase_squared[phase][0] / phase_rows[phase]),
            sqrt(phase_squared[phase][1] / phase_rows[phase]),
            sqrt(phase_squared[phase][2] / phase_rows[phase]),
            sqrt(phase_squared[phase][3] / phase_rows[phase]));
    }
    free_puffernet(net);
    free(weights);
    return 0;
}
