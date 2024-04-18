/*
This is the C++ version of the Mixtral-8x7B model.
- it runs on CPU.
- it is for inference currently.
*/

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#ifdef OMP
#include <omp.h>
#endif

using namespace std;

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

namespace llm {
    class Module {
    public:
        virtual void forward() = 0;
        virtual void backward() = 0;
    };

    class WTE: public Module {
    public:
        void forward() {}
    };

    class WPERoPE: public Module {
    public:
        void forward() {}
    };

    class RMSNorm: public Module {
    public:
        void forward() {}
    };

    class Matmul: public Module {
    public:
        void forward() {}
    };

    class Attention: public Module {
    public:
        void forward() {}
    };

    class SparseMoE: public Module {
    public:
        void forward() {}
    };

    class SiLU: public Module {
    public:
        void forward() {}
    };

    class Residual: public Module {
    public:
        void forward() {}
    };

    class Softmax: public Module {
    public:
        void forward() {}
    };
}

// ----------------------------------------------------------------------------
// Mixtral-8x7B model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
class ParameterTensors {
public:
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

#define NUM_ACTIVATION_TENSORS 24
class ActivationTensors {
public:
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
// {
//   "architectures": [
//     "MixtralForCausalLM"
//   ],
//   "attention_dropout": 0.0,
//   "bos_token_id": 1,
//   "eos_token_id": 2,
//   "hidden_act": "silu",
//   "hidden_size": 4096,
//   "initializer_range": 0.02,
//   "intermediate_size": 14336,
//   "max_position_embeddings": 32768,
//   "model_type": "mixtral",
//   "num_attention_heads": 32,
//   "num_experts_per_tok": 2,
//   "num_hidden_layers": 32,
//   "num_key_value_heads": 8,
//   "num_local_experts": 8,
//   "output_router_logits": false,
//   "rms_norm_eps": 1e-05,
//   "rope_theta": 1000000.0,
//   "router_aux_loss_coef": 0.02,
//   "sliding_window": null,
//   "tie_word_embeddings": false,
//   "torch_dtype": "bfloat16",
//   "transformers_version": "4.36.0.dev0",
//   "use_cache": true,
//   "vocab_size": 32000
// }
public:
    int max_seq_len; // max sequence length, e.g. 32768
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 32
    int num_q_heads; // number of q heads in the attention, e.g. 32
    int num_kvheads; // number of kvheads in the attention, e.g. 8
    int channels; // number of channels, e.g. 4096
};

class Mixtral {
public:
    MixtralConfig config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    shared_ptr<ParameterTensors> params_ptr;
    size_t num_parameters;
    // the activations of the model, and their sizes
    ActivationTensors activs;
    size_t activ_sizes[NUM_ACTIVATION_TENSORS];
    shared_ptr<ActivationTensors> activs_ptr;
    size_t num_activations;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    vector<vector<int>> inputs; // the input tokens for the current forward pass
};

void mixtral_build_from_checkpoint(shared_ptr<Mixtral>& model_ptr, const string checkpoint_path) {
    // read in model from a checkpoint file

}

void mixtral_forward(shared_ptr<Mixtral>& model_ptr, const vector<vector<int>>& inputs, int B, int T) {

}

void mixtral_free() {

}

#ifndef TESTING

// ----------------------------------------------------------------------------
// sampler

// the Mixtral-8x7B end-of-text token id
#define Mixtral_EOT 50256

// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)
class Tokenizer {
public:
    void print() {}
    void init(const string& filename) {}
    void decode() {}

    uint32_t vocab_size;
    vector<string> token_table;
    bool init_ok;
};

// ----------------------------------------------------------------------------
// main inference
int main() {
    // build the GPT-2 model from a checkpoint
    shared_ptr<Mixtral> model_ptr = make_shared<Mixtral>();
    // model url: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    mixtral_build_from_checkpoint(model_ptr, "mistralai/Mixtral-8x7B-v0.1");

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer.init("gpt2_tokenizer.bin");

    // inference

    // free

    return 0;
}
#endif