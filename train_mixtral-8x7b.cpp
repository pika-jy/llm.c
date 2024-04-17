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

} MixtralParameters;