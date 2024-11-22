#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <vector>

// ============ GPT2 tensor manifest ============

// derived from ggml gpt2 reference implementation
struct gpt2_layer {
  struct ggml_tensor *layer_norm_1_weights;
  struct ggml_tensor *layer_norm_1_bias;

  struct ggml_tensor *layer_norm_2_weights;
  struct ggml_tensor *layer_norm_2_bias;

  // attention
  struct ggml_tensor *c_attention_attention_weights;
  struct ggml_tensor *c_attention_attention_bias;

  struct ggml_tensor *c_attention_projection_weights;
  struct ggml_tensor *c_attention_projection_bias;

  // mlp
  struct ggml_tensor *c_multi_layer_perceptron_fully_connected_weights;
  struct ggml_tensor *c_multi_layer_perceptron_fully_connected_bias;

  struct ggml_tensor *c_multi_layer_perceptron_projection_weights;
  struct ggml_tensor *c_multi_layer_perceptron_projection_bias;
};

struct autoregressive_model {

  struct ggml_tensor *embedding;

  std::map<std::string, struct ggml_tensor *> tensors;

  struct ggml_tensor *text_embedding_weights;
  struct ggml_tensor *text_position_embedding_weights;

  struct ggml_tensor *mel_embedding_weights;
  struct ggml_tensor *mel_position_embedding_weights;

  struct ggml_tensor *final_layer_norm_weights;
  struct ggml_tensor *final_layer_norm_bias;

  struct ggml_tensor *language_model_head_layer_norm_weights;
  struct ggml_tensor *language_model_head_layer_norm_bias;

  struct ggml_tensor *language_model_head_linear_weights;
  struct ggml_tensor *language_model_head_linear_bias;

  struct ggml_tensor *memory_key;
  struct ggml_tensor *memory_value;

  std::vector<gpt2_layer> layers;

  struct ggml_context *ctx;

  ggml_backend_buffer_t buffer_w;

  ggml_backend_t backend = NULL;
};

// ===================================================

// ============ Diffusion Tensor Manifest ============

struct latent_conditioner_attention_block {

  struct ggml_tensor *norm_weight;
  struct ggml_tensor *norm_bias;

  struct ggml_tensor *qkv_weight;
  struct ggml_tensor *qkv_bias;

  struct ggml_tensor *projection_out_weight;
  struct ggml_tensor *projection_out_bias;

  struct ggml_tensor
      *relative_position_embeddings_relative_attention_bias_weight;
};

struct residual_block {

  struct ggml_tensor *in_layers_0_weight;

  struct ggml_tensor *in_layers_0_bias;

  struct ggml_tensor *in_layers_2_weight;

  struct ggml_tensor *in_layers_2_bias;

  struct ggml_tensor *emb_layers_1_weight;

  struct ggml_tensor *emb_layers_1_bias;

  struct ggml_tensor *out_layers_0_weight;

  struct ggml_tensor *out_layers_0_bias;

  struct ggml_tensor *out_layers_3_weight;

  struct ggml_tensor *out_layers_3_bias;
};

struct diffusion_layer {

  struct ggml_tensor *resblock_in_layers_0_weight;

  struct ggml_tensor *resblock_in_layers_0_bias;

  struct ggml_tensor *resblock_in_layers_2_weight;

  struct ggml_tensor *resblock_in_layers_2_bias;

  struct ggml_tensor *resblock_emb_layers_1_weight;

  struct ggml_tensor *resblock_emb_layers_1_bias;

  struct ggml_tensor *resblock_out_layers_0_weight;

  struct ggml_tensor *resblock_out_layers_0_bias;

  struct ggml_tensor *resblock_out_layers_3_weight;

  struct ggml_tensor *resblock_out_layers_3_bias;

  struct ggml_tensor *attn_norm_weight;

  struct ggml_tensor *attn_norm_bias;

  struct ggml_tensor *attn_qkv_weight;

  struct ggml_tensor *attn_qkv_bias;

  struct ggml_tensor *attn_proj_out_weight;

  struct ggml_tensor *attn_proj_out_bias;

  struct ggml_tensor
      *attn_relative_pos_embeddings_relative_attention_bias_weight;
};

struct diffusion_model {

  struct ggml_tensor *diffusion_conditioning_latent;

  struct ggml_tensor *latent_conditioner_convolution_weight;

  struct ggml_tensor *latent_conditioner_convolution_bias;

  std::vector<latent_conditioner_attention_block>
      latent_conditioner_attention_blocks;

  struct ggml_tensor *code_norm_weight;

  struct ggml_tensor *code_norm_bias;

  struct ggml_tensor *time_emb_linear_0_weight;

  struct ggml_tensor *time_emb_linear_0_bias;

  struct ggml_tensor *time_emb_linear_1_weight;

  struct ggml_tensor *time_emb_layer_norm_1_bias;

  std::vector<diffusion_layer>
      timestep_conditioning_integrator_diffusion_layers;

  struct ggml_tensor *inp_block_weight;

  struct ggml_tensor *inp_block_bias;

  struct ggml_tensor *integrating_conv_weight;

  struct ggml_tensor *integrating_conv_bias;

  std::vector<diffusion_layer> main_diffusion_layers;

  std::vector<residual_block> main_residual_blocks;

  struct ggml_tensor *out_group_norm_weight;
  struct ggml_tensor *out_group_norm_bias;

  struct ggml_tensor *out_convolution_weight;
  struct ggml_tensor *out_convolution_bias;

  struct ggml_tensor *unconditioned_embedding;

  std::map<std::string, struct ggml_tensor *> tensors;

  struct ggml_context *ctx;

  ggml_backend_buffer_t buffer_w;

  ggml_backend_t backend = NULL;
};

// ===================================================

// ============= Vocoder Tensor Manifest =============

struct residual_conv_block {
  struct ggml_tensor *residual_convs_1_bias;
  struct ggml_tensor *residual_convs_1_weight;

  struct ggml_tensor *residual_convs_3_bias;
  struct ggml_tensor *residual_convs_3_weight;
};

struct conv_block {
  struct ggml_tensor *conv_block_1_bias;
  struct ggml_tensor *conv_block_1_weight;
};

struct vocoder_residual_block {

  struct ggml_tensor *kernel_predictor_input_convolution_weight;
  struct ggml_tensor *kernel_predictor_input_convolution_bias;

  std::vector<residual_conv_block> kernel_predictor_residual_conv_blocks;

  struct ggml_tensor *kernel_predictor_kernel_convolution_weight;
  struct ggml_tensor *kernel_predictor_kernel_convolution_bias;

  struct ggml_tensor *kernel_predictor_bias_convolution_weight;
  struct ggml_tensor *kernel_predictor_bias_convolution_bias;

  struct ggml_tensor *convolution_t_pre_weight;
  struct ggml_tensor *convolution_t_pre_bias;

  std::vector<conv_block> conv_blocks;
};

// model tether
struct vocoder_model {

  struct ggml_tensor *convolution_pre_weight;
  struct ggml_tensor *convolution_pre_bias;

  std::vector<vocoder_residual_block> residual_stack;

  struct ggml_tensor *convolution_post_weight;
  struct ggml_tensor *convolution_post_bias;

  std::map<std::string, struct ggml_tensor *> tensors;

  struct ggml_context *ctx;

  ggml_backend_buffer_t buffer_w;

  ggml_backend_t backend = NULL;
};

// ===================================================

// ================ Helper structs ===================

using trimmed_latents_vector = std::vector<std::vector<float>>;
using sequence_vector = std::vector<std::vector<int>>;
// ===================================================

bool autoregressive_model_load(const std::string &fname, autoregressive_model &model);
bool diffusion_model_load(const std::string &fname, diffusion_model &model);
bool vocoder_model_load(const std::string &fname, vocoder_model &model);

std::pair<trimmed_latents_vector, sequence_vector> autoregressive(
  std::vector<gpt_vocab::id> tokens,
  std::string voice_path,
  int batch_size);

std::vector<float> diffusion(diffusion_model& dfsn_model, std::vector<float> trimmed_latents);
std::vector<float> vocoder(std::vector<float> mel);

void writeWav(const char *filename, const std::vector<float> &data, int sampleRate);
