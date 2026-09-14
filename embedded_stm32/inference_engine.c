#include "inference_engine.h"
#include <math.h>

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float hardtanh(float x, float min_v, float max_v) {
    if (x < min_v) return min_v;
    if (x > max_v) return max_v;
    return x;
}

void model_init_weights(ModelWeights *weights, const float *raw) {
    int idx = 0;
    weights->w_ih = &raw[idx]; idx += 256 * 3;
    weights->w_hh = &raw[idx]; idx += 256 * 64;
    weights->b_ih = &raw[idx]; idx += 256;
    weights->b_hh = &raw[idx]; idx += 256;
    weights->w_fc1 = &raw[idx]; idx += 512 * 66;
    weights->b_fc1 = &raw[idx]; idx += 512;
    weights->w_fc2 = &raw[idx]; idx += 256 * 512;
    weights->b_fc2 = &raw[idx]; idx += 256;
    weights->w_fc3 = &raw[idx]; idx += 128 * 256;
    weights->b_fc3 = &raw[idx]; idx += 128;
    weights->w_fc4 = &raw[idx]; idx += 1 * 128;
    weights->b_fc4 = &raw[idx]; idx += 1;
}

float model_predict(const ModelWeights *weights, const float *x_74) {
    // Hidden and cell state buffers
    float h[LSTM_HIDDEN] = {0.0f};
    float c[LSTM_HIDDEN] = {0.0f};
    float gates[256];

    // Series inputs: p=x[0:24], pv=x[24:48], l=x[48:72]
    for (int t = 0; t < SEQ_LEN; t++) {
        float x_t[3] = { x_74[t], x_74[24 + t], x_74[48 + t] };

        // gates = W_ih * x_t + b_ih + W_hh * h + b_hh
        for (int i = 0; i < 256; i++) {
            float sum = weights->b_ih[i] + weights->b_hh[i];
            sum += weights->w_ih[i * 3 + 0] * x_t[0];
            sum += weights->w_ih[i * 3 + 1] * x_t[1];
            sum += weights->w_ih[i * 3 + 2] * x_t[2];

            const float *w_hh_row = &weights->w_hh[i * 64];
            for (int j = 0; j < 64; j++) {
                sum += w_hh_row[j] * h[j];
            }
            gates[i] = sum;
        }

        // Apply activations
        for (int j = 0; j < LSTM_HIDDEN; j++) {
            float in_gate  = sigmoid(gates[0   + j]);
            float f_gate   = sigmoid(gates[64  + j]);
            float g_gate   = tanhf  (gates[128 + j]);
            float out_gate = sigmoid(gates[192 + j]);

            c[j] = f_gate * c[j] + in_gate * g_gate;
            h[j] = out_gate * tanhf(c[j]);
        }
    }

    // Fusion layer: [h (64), t_norm (1), soc_norm (1)]
    float fusion[FUSION_DIM];
    for (int j = 0; j < 64; j++) {
        fusion[j] = h[j];
    }
    fusion[64] = x_74[72]; // time
    fusion[65] = x_74[73]; // soc

    // FC1: 66 -> 512
    static float z1[FC1_OUT];
    for (int i = 0; i < FC1_OUT; i++) {
        float sum = weights->b_fc1[i];
        const float *w_row = &weights->w_fc1[i * FUSION_DIM];
        for (int j = 0; j < FUSION_DIM; j++) {
            sum += w_row[j] * fusion[j];
        }
        z1[i] = relu(sum);
    }

    // FC2: 512 -> 256
    static float z2[FC2_OUT];
    for (int i = 0; i < FC2_OUT; i++) {
        float sum = weights->b_fc2[i];
        const float *w_row = &weights->w_fc2[i * FC1_OUT];
        for (int j = 0; j < FC1_OUT; j++) {
            sum += w_row[j] * z1[j];
        }
        z2[i] = relu(sum);
    }

    // FC3: 256 -> 128
    static float z3[FC3_OUT];
    for (int i = 0; i < FC3_OUT; i++) {
        float sum = weights->b_fc3[i];
        const float *w_row = &weights->w_fc3[i * FC2_OUT];
        for (int j = 0; j < FC2_OUT; j++) {
            sum += w_row[j] * z2[j];
        }
        z3[i] = relu(sum);
    }

    // FC4: 128 -> 1
    float a_out = weights->b_fc4[0];
    for (int j = 0; j < FC3_OUT; j++) {
        a_out += weights->w_fc4[j] * z3[j];
    }

    return hardtanh(a_out, -7.0f, 7.0f);
}

void spp_project(float a_raw, float cur_soc, int step_idx, int dwell_len,
                 float cur_pv, float cur_load,
                 float *a_spp, float *soc_next, SPPMetrics *met) {
    const float soc_min = 4.0f;
    const float soc_max = 40.0f;
    const float eta_ch = 0.98f;
    const float eta_dis = 0.98f;
    const float e_max = 7.0f;
    const float E_max_trans = 100.0f;

    float a = a_raw;
    float pos_clip = 0.0f, neg_clip = 0.0f;
    int interv = 0;

    // Step 1: SoC boundary clamping
    if (a >= 0.0f) {
        float max_ch = (soc_max - cur_soc) / eta_ch;
        if (a > max_ch) {
            pos_clip += (a - max_ch);
            a = max_ch;
            interv++;
        }
    } else {
        float max_dis = (soc_min - cur_soc) * eta_dis;
        if (a < max_dis) {
            neg_clip += (max_dis - a);
            a = max_dis;
            interv++;
        }
    }

    // Step 2: Departure reachability
    float soc_tentative = (a >= 0.0f) ? (cur_soc + eta_ch * a) : (cur_soc + a / eta_dis);
    if (step_idx == dwell_len - 1) {
        if ((soc_max - soc_tentative) / eta_ch > 1e-4f) {
            a = (soc_max - cur_soc) / eta_ch;
            interv++;
        }
    } else {
        int rem = dwell_len - step_idx - 1;
        if ((soc_max - soc_tentative) / eta_ch > e_max * rem) {
            a = (soc_max - cur_soc) / eta_ch - e_max * rem;
            interv++;
        }
    }

    // Step 3: Feeder capacity limit
    float net_e = a + cur_load - cur_pv;
    if (net_e > E_max_trans) {
        a = E_max_trans + cur_pv - cur_load;
        interv++;
    } else if (net_e < -E_max_trans) {
        a = -E_max_trans + cur_pv - cur_load;
        interv++;
    }

    // Compute final next SoC
    *soc_next = (a >= 0.0f) ? (cur_soc + eta_ch * a) : (cur_soc + a / eta_dis);
    *a_spp = a;

    met->pos_clip = pos_clip;
    met->neg_clip = neg_clip;
    met->clip_total = pos_clip + neg_clip;
    met->interventions = interv;
}
