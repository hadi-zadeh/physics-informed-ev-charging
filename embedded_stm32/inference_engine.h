#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <stdint.h>

#define SEQ_LEN 24
#define INPUT_DIM 3
#define LSTM_HIDDEN 64
#define FUSION_DIM 66
#define FC1_OUT 512
#define FC2_OUT 256
#define FC3_OUT 128
#define FC4_OUT 1

typedef struct {
    const float *w_ih;    // [256, 3]
    const float *w_hh;    // [256, 64]
    const float *b_ih;    // [256]
    const float *b_hh;    // [256]
    const float *w_fc1;   // [512, 66]
    const float *b_fc1;   // [512]
    const float *w_fc2;   // [256, 512]
    const float *b_fc2;   // [256]
    const float *w_fc3;   // [128, 256]
    const float *b_fc3;   // [128]
    const float *w_fc4;   // [1, 128]
    const float *b_fc4;   // [1]
} ModelWeights;

void model_init_weights(ModelWeights *weights, const float *raw_binary);
float model_predict(const ModelWeights *weights, const float *x_74);

// Deterministic Safety Post-Processing (SPP)
typedef struct {
    float pos_clip;
    float neg_clip;
    float clip_total;
    int interventions;
} SPPMetrics;

void spp_project(float a_raw, float cur_soc, int step_idx, int dwell_len,
                 float cur_pv, float cur_load,
                 float *a_spp, float *soc_next, SPPMetrics *met);

#endif // INFERENCE_ENGINE_H
