/*
This is the C++ version of the Mixtral-8x7B model.
- it runs on CPU.
- it is for inference currently.
*/

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
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

    template <typename T>
    class WTE: public Module {
    public:
        void forward(vector<vector<vector<T>>>& output,
                     const vector<vector<T>>& input,
                     const vector<vector<T>>& wte,
                     int B, int S, int C) {
            /*
            output: (B, S, C)
            input: (B, S)
            wte: (V, C)
            */
            #pragma omp parallel collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < S; ++t) {
                    const auto& input_bt = input[b][t];
                    const auto& wte_t = wte[t];
                    auto& output_bt = output[b][t];

                    for (int c = 0; c < C; ++c) {
                        output_bt[c] = input_bt[c] + wte_t[c];
                    }
                }
            }
        }

        void backward() {}
    };

    template <typename T>
    class WPERoPE: public Module {
    public:
        /*
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        */
        void WPERoPE(int HD) {
            /*
            HD: head dim
            */
            this->inv_freq.resize(HD / 2);
            T n = 0;
            generate(this->inv_freq.begin(), this->inv_freq.end(), [&n](){
                n += 2;
                return 1.0f / powf(base, n / HD);
            });

            this->set_cos_sin_cache(this->max_seq_len);
        }

        void set_cos_sin_cache(int S) {
            vector<T> position(S);
            itoa(position.begin(), position.end(), 0);
            vector<vector<T>> freqs(S, vector<T>(this->inv_freq.size(), 0.f));
            this->cos_cached.resize(S);
            this->sin_cached.resize(S);

            for (int t = 0; t < S; ++t) {
                const auto& inv_freq_t = this->inv_freq[t];
                auto& freqs_t = freqs[t];
                auto& cos_cached_t = this->cos_cached[t];
                auto& sin_cached_t = this->sin_cached[t];

                for (int c = 0; c < this->inv_freq.size(); ++c) {
                    auto& freq = freqs_t[c];
                    freq = static_cast<T>(t) * this->inv_freq[c];
                    this->cos_cached_t[c] = cos(freq);
                    this->sin_cached_t[c] = sin(freq);
                }
            }
        }

        void forward(vecotr<vector<vector<vector<T>>>>& output,
                     const vecotr<vector<vector<vector<T>>>>& input,
                     int B, int S, int C, int NH) {
            /*
            RoPE paper: https://arxiv.org/pdf/2104.09864.pdf
            Reference: https://github.com/lucidrains/rotary-embedding-torch
            output: (B, NH, S, HD)
            input: (B, NH, S, HD)
            */
            int HD = C / NH; // head dim
            assert(HD % 2 == 0), string("head dim needs to be divisible by 2");

            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < S; ++t) {
                    const auto& input_bt = input[b][t];
                    auto& output_bt = output[b][t];

                    // embed = x * cos + rotate_half(x) * sin
                    // x * cos
                    for (int c = 0; i < HD; ++i) {
                        output_bt[c] = input_bt[c] * this->cos_cached.at(c);
                    }
                    // rotate_half(x) * sin
                    int offset = HD / 2;
                    for (int c = 0; i < offset; ++i) {
                        output_bt[c] -= input_bt[c + offset] * this->sin_cached.at(c);
                    }
                    for (int c = offset; i < HD; ++i) {
                        output_bt[c] += input_bt[c - offset] * this->sin_cached.at(c);
                    }
                }
            }
        }

        void backward() {}

    private:
        static constexpr T base{10000.0f};
        int max_seq_len{32768};
        vector<T> inv_freq;
        vector<T> cos_cached;
        vector<T> sin_cached;
    };

    template <typename T>
    class RMSNorm: public Module {
    public:
        /*
        llama:
        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
        */
        void forward(vector<vecotr<vector<T>>>& output,
                     const vector<vector<vector<T>>>& input,
                     const vector<T>& rstd,
                     const vector<T>& weight,
                     int B, int S, int C) {
            /*
            LayerNorm paper: https://arxiv.org/pdf/1607.06450.pdf
            RMSNorm paper: https://arxiv.org/pdf/1910.07467.pdf
            RMSNorm is a simplification of the LayerNorm, removes mean (re-center)
            output: (B, S, C) activations
            input: (B, S, C) activations
            rstd are (B, S) buffers, to be used later in backward pass
            at each position (b, t) of the input, the C-dimensional vector
            of activations gets normalized, then scaled and shifted
            */
            float eps = 1e-6f;
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < S; ++t) {
                    const auto& input_bt = input[b][t];
                    auto& output_bt = output[b][t];

                    // variance
                    T varval = 0.0f;
                    for_each(input_bt.begin(), input_bt.end(), [&](const auto& v){
                        varval += powf(v, 2);
                    });
                    varval /= C;

                    // rstd (reciprocal standard deviation)
                    T rstd = 1.0f / sqrtf(varval + eps);

                    // normalize, scale
                    for (int c = 0; c < C; ++c) {
                        auto& v = output_bt[b][t][c];
                        v *= rstd; // normalize
                        v *= weight[c]; // scale
                    }
                }
            }
        }

        void backward() {}
    };

    template <typename T>
    class Matmul: public Module {
    public:
        void forward(vector<vector<vector<T>>>& output
                     const vector<vector<vector<T>>>& input,
                     vector<vector<T>>& weight,
                     vector<T>& bias,
                     int B, int S, int C, int OC) {
            /*
            output: (B, S, OC)
            input: (B, S, C)
            weight: (OC, C)
            bias: (OC)
            */
            #pragma omp parallel for collpase(2)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < S, ++t) {
                    const auto& input_bt = input[b][t];
                    auto& output_bt = output[b][t];

                    for (int oc = 0; oc < OC; ++oc) {
                        auto& v = output_bt[b][t][oc];
                        v = inner_product(input_bt.cbegin(), input_bt.cend(), weight.cbegin(), 0.f);
                        v += bias[oc];
                    }
                }
            }
        }

        void backward() {}
    };

    template <typename T>
    class Attention: public Module {
    public:
        void forward(vector<vecotr<vector<T>>>& output,
                     const vector<vecotr<vector<T>>>& input,
                     const vector<vecotr<T>>& mask,
                     vector<vecotr<vector<vector<T>>>>& preatt,
                     vector<vecotr<vector<vector<T>>>>& att,
                     int B, int S, int C, int NH) {
            /*
            input is (B, S, 3C) holding the query, key, value (Q, K, V) vectors
            preatt, att are (B, NH, S, S). NH = number of heads, S = sequence length
            that holds the pre-attention and post-attention scores (used in backward)
            output is (B, S, C)
            attention is the only layer that mixes information across time
            every other operation is applied at every (b,t) position independently
            (and of course, no layer mixes information across batch)
            */
            int HD = C / NH; // head dim
            T norm_factor = 1.0 / sqrtf(HD);

            #pragma omp parallel for collapse(3)
            for (int b = 0; b < B; ++b) {
                auto& preatt_b = preatt[b];
                auto& att_b = att[b];

                for (int t1 = 0; t1 < S; ++t1) {
                    // Todo: optimize memory usage
                    const auto& input_bt = input[b][t1];
                    auto input_bt_start = input_bt.cbegin();
                    auto input_bt_end = input_bt.end();
                    const auto&& query_bt = vector<T>(input_bt_start, input_bt_start + C);
                    const auto&& key_bt = vector<T>(input_bt_start + C, input_bt_start + 2 * C);
                    const auto&& value_bt = vector<T>(input_bt_start + 2 * C, input_bt_end);
                    auto& output_bt = output[b][t1];
                    
                    for (int m = 0; m < NH; ++m) {
                        const auto&& query_btm = vector<T>(query_bt.cbegin() + m * HD, query_bt.cbegin() + (m + 1) * HD);
                        const auto&& key_btm = vector<T>(key_bt.cbegin() + m * HD, key_bt.cbegin() + (m + 1) * HD);
                        const auto&& value_btm = vector<T>(value_bt.cbegin() + m * HD, value_bt.cbegin() + (m + 1) * HD);
                        const auto& mask_t = mask[t1];
                        auto& preatt_bmt1 = preatt_b[m][t1];
                        auto& att_bmt1 = att_b[m][t1];

                        // Q @ K_T
                        for (int t2 = 0; t2 < S; ++t2) {
                            auto val = inner_product(query_btm.cbegin(), query_btm.cend(), key_btm.cbegin(), 0.f);
                            val *= norm_factor; // scale
                            preatt_bmt1[t2] = val;
                        }

                        // Apply causal mask
                        for (int t2 = 0; t2 < S; ++t2) {
                            preatt_bmt1[t2] += mask_t[t2];
                        }

                        // Safe softmax (subtract maxval)
                        auto maxval = max(preatt_bmt1.cbegin(), preatt_bmt1.cend());
                        for (int t2 = 0; t2 < S; ++t2) {
                            preatt_bmt1[t2] = expf(logits_bt[t2] - maxval);
                        }
                        auto expsum = accumulate(preatt_bmt1); // sum as the denomiator
                        for_each(preatt_bmt1.begin(), preatt_bmt1.end(), [&](auto& v){ // divide by sum
                            v /= expsum;
                        });

                        // Score @ V

                    }
                }
            }
        }

        void backward() {}
    };

    class SparseMoE: public Module {
    public:
        void forward() {}
    };

    class SiLU: public Module {
    public:
        void forward() {

        }

        void backward() {}
    };

    template <typename T>
    class Residual: public Module {
    public:
        void forward(vector<vecotr<vector<T>>>& output,
                     const vector<vecotr<vector<T>>>& input1,
                     const vector<vecotr<vector<T>>>& input2,
                     int B, int S, int C) {
            for (int i = 0; i < N; i++) {
                out[i] = inp1[i] + inp2[i];
            }
             /*
            output: (B, S, C)
            input1: (B, S, C)
            input2: (B, S, C)
            */
            #pragma omp parallel for collpase(2)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < S, ++t) {
                    const auto& input1_bt = input1[b][t];
                    const auto& input2_bt = input2[b][t];
                    auto& output_bt = output[b][t];

                    for (int c = 0; c < C; ++c) {
                        output_bt[c] = input1_bt[c] + input2_bt[c];
                    }
                }
            }
        }

        void backward() {}
    };

    template <typename T>
    class Softmax: public Module {
    public:
        void forward(vector<vector<vector<T>>>& probs,
                     const vector<vector<vector<T>>>& logits,
                     int B, int S, int V) {
            /*
            output: probs, (B, S, V) of the probabilities (sums to 1.0 in each b,t position)
            input: logits, (B, S, V) of the unnormalized log probabilities
            */
            // softmax(logits) -> probs
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < S, ++t) {
                    const auto& logits_bt = logits[b][t];
                    auto& probs_bt = probs[b][t];

                    // safe softmax (subtract maxval)
                    auto maxval = max(logits_bt.cbegin(), logits_bt.cend());
                    for (int i = 0; i < V; ++i) {
                        probs_bt[i] = expf(logits_bt[i] - maxval);
                    }

                    // sum as the denomiator
                    auto expsum = accumulate(probs_bt);

                    // divide by sum
                    for_each(probs_bt.begin(), probs_bt.end(), [&](auto& v){
                        v /= expsum;
                    });
                }
            }
        }

        void backward() {}
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
int sample_mult(const vector<float>& probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// the Mixtral-8x7B end-of-text token id
#define Mixtral_EOT 50256

// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)
class Tokenizer {
public:
    void print(string) {}
    void init(const string& filename) {}
    string decode() {}

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

    // build the inputs
    int B = 32;
    int T = 128;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer.init("mixtral_tokenizer.bin");

    vector<vector<int>> inputs;
    vector<vector<int>> gen_tokens(B, vector<int>(T, Mixtral_EOT));
    constexpr int genT = 128;

    // inference
    for (const auto& tokens : inputs) {
        std::cout << "generating:\n---\n";
        for (int t = 1; t < genT; t++) {
            // Todo: add KV Cache for inference to remove re-calculations
            mixtral_forward(model_ptr, gen_tokens, B, T);
            auto probs = model_ptr->activs_ptr->probs[0][t - 1];
            float coin = 0; // random_f32(&rng_state);
            auto next_token = sample_mult(probs, model_ptr->config.vocab_size, coin);
            gen_tokens[0][t] = next_token;
            if (tokenizer.init_ok) {
                auto token_str = tokenizer.decode();
                tokenizer.print(token_str);
            } else {
                std::cout << "token id: " << next_token << "\n";
            }
            fflush(stdout);
        }
        std::cout << "\n---\n";
    }

    // free

    return 0;
}
#endif