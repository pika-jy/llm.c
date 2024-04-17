/*
This is the C++ version of the Mixtral-8x7B model.
- it runs on CPU.
- it is for inference currently.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#ifdef OMP
#include <omp.h>
#endif

using namespace std;

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void embedding_forward() {

}

void rmsnorm_forward() {

}

void matmul_forward() {

}

void attention_forward() {

}

void sparse_moe_forward() {

}

void silu_forward() {

}

void residual_forward() {

}

void softmax_forward() {

}

// ----------------------------------------------------------------------------
// Mixtral-8x7B model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
class ParameterTensors {
    vector<vector<float>> wte; // (V, C)
    vector<vector<float>> wpe; // (maxT, C)
    vector<vector<float>> ln1w; // (L, C)
    vector<vector<float>> ln1b; // (L, C)
    vector<vector<vector<float>>> qkvw; // (L, 3*C, C)
    vector<vector<float>> qkvb; // (L, 3*C)
    vector<vector<vector<float>>> attnprojw; // (L, C, C)
    vector<vector<float>> attnprojb; // (L, C)
    vector<vector<float>> ln2w; // (L, C)
    vector<vector<float>> ln2b; // (L, C)
    vector<vector<float>> fcup1w; // (L, 3.5*C, C)
    vector<vector<float>> fcup1b; // (L, 3.5*C)
    vector<vector<float>> fcup2w; // (L, 3.5*C, C)
    vector<vector<float>> fcup2b; // (L, 3.5*C)
    vector<vector<float>> fcdownw; // (L, C, 3.5*C)
    vector<vector<float>> fcdownb; // (L, C)
};

#define NUM_ACTIVATION_TENSORS 23
class ActivationTensors {
    vector<vector<vector<float>>> embedded; // (B, T, C)
    vector<vector<vector<vector<float>>>> ln1; // (L, B, T, C)
    vector<vector<vector<float>>> ln1; // (L, B, T)
    vector<vector<vector<float>>> ln1_mean; // (L, B, T)
    vector<vector<vector<float>>> ln1_rstd; // (L, B, T)
    vector<vector<vector<vector<float>>>> qkv; // (L, B, T, 3*C)
    vector<vector<vector<vector<float>>>> atty; // (L, B, T, C)
    vector<vector<vector<vector<vector<float>>>>> preatt; // (L, B, NH, T, T)
    vector<vector<vector<vector<vector<float>>>>> att; // (L, B, NH, T, T)
    vector<vector<vector<vector<float>>>> attproj; // (L, B, T, C)
    vector<vector<vector<vector<float>>>> residual2; // (L, B, T, C)
    vector<vector<vector<vector<float>>>> ln2; // (L, B, T, C)
    vector<vector<vector<float>>> ln2_mean; // (L, B, T)
    vector<vector<vector<float>>> ln2_rstd; // (L, B, T)
    vector<vector<vector<vector<float>>>> fcup1h; // (L, B, T, 3.5*C)
    vector<vector<vector<vector<float>>>> fcup1h_silu; // (L, B, T, 3.5*C)
    vector<vector<vector<vector<float>>>> fcup2h; // (L, B, T, 3.5*C)
    vector<vector<vector<vector<float>>>> fcdownh; // (L, B, T, C)
    vector<vector<vector<vector<float>>>> residual3; // (L, B, T, C)
    vector<vector<vector<float>>> lnf; // (B, T, C)
    vector<vector<float>> lnf_mean; // (B, T)
    vector<vector<float>> lnf_rstd; // (B, T)
    vector<vector<vector<float>>> logits; // (B, T, V)
    vector<vector<vector<float>>> probs; // (B, T, V)
};

class MixtralConfig {
    int max_seq_len; // max sequence length, e.g. 
};