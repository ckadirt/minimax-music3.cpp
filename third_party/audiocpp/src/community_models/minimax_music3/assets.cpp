#include "engine/community_models/minimax_music3/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/filesystem.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

std::vector<int64_t> i64_array(const engine::io::json::Value & value) {
    std::vector<int64_t> out;
    for (const auto & item : value.as_array()) {
        out.push_back(item.as_i64());
    }
    return out;
}

engine::io::json::Value parse_json_file(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("missing MiniMax Music 3 config file: " + path.string());
    }
    return engine::io::json::parse_file(path);
}

MiniMaxMusic3QwenConfig parse_qwen_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "language_model.json");
    MiniMaxMusic3QwenConfig out;
    out.vocab_size = root.require("vocab_size").as_i64();
    out.hidden_size = root.require("hidden_size").as_i64();
    out.intermediate_size = root.require("intermediate_size").as_i64();
    out.layers = root.require("num_hidden_layers").as_i64();
    out.attention_heads = root.require("num_attention_heads").as_i64();
    out.kv_heads = root.require("num_key_value_heads").as_i64();
    out.head_dim = root.require("head_dim").as_i64();
    out.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    out.rms_norm_eps = root.require("rms_norm_eps").as_f32();
    if (const auto * rope = root.find("rope_parameters")) {
        out.rope_theta = rope->require("rope_theta").as_f32();
    }
    return out;
}

MiniMaxMusic3DepthConfig parse_depth_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "rvq_depth_decoder.json");
    MiniMaxMusic3DepthConfig out;
    out.audio_vocab_size = root.require("audio_vocab_size").as_i64();
    out.hidden_size = root.require("hidden_size").as_i64();
    out.intermediate_size = root.require("intermediate_size").as_i64();
    out.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    out.attention_heads = root.require("num_attention_heads").as_i64();
    out.codebooks = root.require("num_codebooks").as_i64();
    out.layers = root.require("num_layers").as_i64();
    return out;
}

MiniMaxMusic3ConditionConfig parse_condition_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "condition_encoder.json");
    MiniMaxMusic3ConditionConfig out;
    out.condition_hidden_dim = root.require("condition_hidden_dim").as_i64();
    out.condition_layers = root.require("num_condition_layers").as_i64();
    out.out_dim = root.require("out_dim").as_i64();
    out.input_sample_rate = root.require("input_sampling_rate").as_i64();
    out.input_hop_length = root.require("input_hop_length").as_i64();
    out.output_sample_rate = root.require("output_sampling_rate").as_i64();
    out.output_hop_length = root.require("output_hop_length").as_i64();
    return out;
}

MiniMaxMusic3FlowConfig parse_flow_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "transformer.json");
    MiniMaxMusic3FlowConfig out;
    out.in_channels = root.require("in_channels").as_i64();
    out.condition_dim = root.require("condition_dim").as_i64();
    out.layers = root.require("num_layers").as_i64();
    out.attention_heads = root.require("num_attention_heads").as_i64();
    out.head_dim = root.require("attention_head_dim").as_i64();
    out.ff_inner_dim = root.require("ff_inner_dim").as_i64();
    out.rotary_dim = root.require("rotary_dim").as_i64();
    out.fourier_embedding_dim = root.require("fourier_embedding_dim").as_i64();
    return out;
}

MiniMaxMusic3VocoderConfig parse_vocoder_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "vocoder.json");
    MiniMaxMusic3VocoderConfig out;
    out.latent_channels = root.require("latent_channels").as_i64();
    out.decoder_input_dim = root.require("decoder_input_dim").as_i64();
    out.decoder_hidden_dim = root.require("decoder_hidden_dim").as_i64();
    out.sample_rate = static_cast<int>(root.require("sampling_rate").as_i64());
    out.upsample_ratios = i64_array(root.require("upsampling_ratios"));
    return out;
}

std::shared_ptr<const assets::TensorSource> open_model_root_gguf(
    const std::filesystem::path & model_root,
    const std::filesystem::path & relative_path) {
    const auto path = model_root / relative_path;
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("missing MiniMax Music 3 component GGUF: " + path.string());
    }
    return assets::open_tensor_source(path);
}

void validate_pinned_config(const MiniMaxMusic3Config & config) {
    const auto & qwen = config.qwen;
    if (qwen.vocab_size != 200000 || qwen.hidden_size != 4096 || qwen.intermediate_size != 12288 ||
        qwen.layers != 36 || qwen.attention_heads != 32 || qwen.kv_heads != 8 || qwen.head_dim != 128 ||
        qwen.max_position_embeddings != 10240 || qwen.rope_theta != 1000000.0F) {
        throw std::runtime_error("MiniMax Music 3 language_model config does not match the pinned architecture");
    }
    const auto & depth = config.depth;
    if (depth.audio_vocab_size != 1024 || depth.hidden_size != 4096 || depth.intermediate_size != 6144 ||
        depth.max_position_embeddings != 16 || depth.attention_heads != 16 || depth.codebooks != 8 ||
        depth.layers != 4) {
        throw std::runtime_error("MiniMax Music 3 RVQ config does not match the pinned architecture");
    }
    const auto & condition = config.condition;
    if (condition.condition_hidden_dim != 4096 || condition.condition_layers != 8 || condition.out_dim != 2048 ||
        condition.input_sample_rate != 24000 || condition.input_hop_length != 960 ||
        condition.output_sample_rate != 44100 || condition.output_hop_length != 512) {
        throw std::runtime_error("MiniMax Music 3 condition config does not match the pinned architecture");
    }
    const auto & flow = config.flow;
    if (flow.in_channels != 128 || flow.condition_dim != 2048 || flow.layers != 36 ||
        flow.attention_heads != 32 || flow.head_dim != 64 || flow.ff_inner_dim != 8192 ||
        flow.rotary_dim != 32 || flow.fourier_embedding_dim != 256) {
        throw std::runtime_error("MiniMax Music 3 DiT config does not match the pinned architecture");
    }
    const auto & vocoder = config.vocoder;
    if (vocoder.latent_channels != 128 || vocoder.decoder_input_dim != 1024 ||
        vocoder.decoder_hidden_dim != 1536 || vocoder.sample_rate != 44100 ||
        vocoder.upsample_ratios != std::vector<int64_t>({8, 8, 4, 2})) {
        throw std::runtime_error("MiniMax Music 3 vocoder config does not match the pinned architecture");
    }
}

}  // namespace

void validate_minimax_music3_anchors(const MiniMaxMusic3Assets & assets) {
    assets.language_model_weights->require_metadata("model.embed_tokens.weight");
    assets.language_model_weights->require_metadata("model.layers.0.self_attn.q_proj.weight");
    assets.language_model_weights->require_metadata("model.norm.weight");
    assets.depth_decoder_weights->require_metadata("audio_embeddings.weight");
    assets.depth_decoder_weights->require_metadata("layers.0.attn.to_q.weight");
    assets.condition_encoder_weights->require_metadata("proj.weight");
    assets.transformer_weights->require_metadata("preprocess_conv.weight");
    assets.transformer_weights->require_metadata("transformer_blocks.0.attn.to_q.weight");
    assets.vocoder_weights->require_metadata("dec_in_proj.weight");
    assets.vocoder_weights->require_metadata("conv_in.weight_v");
}

std::shared_ptr<const MiniMaxMusic3Assets> load_minimax_music3_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MiniMaxMusic3Assets>();
    assets->model_root = assets::prepare_model_directory(model_path).model_root;
    assets->tokenizer_json_path = assets->model_root / "tokenizer" / "tokenizer.json";
    assets->tokenizer_config_path = assets->model_root / "tokenizer" / "tokenizer_config.json";
    if (!engine::io::is_existing_file(assets->tokenizer_json_path)) {
        throw std::runtime_error("missing MiniMax Music 3 tokenizer file: " + assets->tokenizer_json_path.string());
    }
    if (!engine::io::is_existing_file(assets->tokenizer_config_path)) {
        throw std::runtime_error(
            "missing MiniMax Music 3 tokenizer config file: " + assets->tokenizer_config_path.string());
    }
    assets->config.qwen = parse_qwen_config(assets->model_root);
    assets->config.depth = parse_depth_config(assets->model_root);
    assets->config.condition = parse_condition_config(assets->model_root);
    assets->config.flow = parse_flow_config(assets->model_root);
    assets->config.vocoder = parse_vocoder_config(assets->model_root);
    validate_pinned_config(assets->config);
    assets->language_model_weights = open_model_root_gguf(assets->model_root, "lm-Q4_K_M.gguf");
    assets->depth_decoder_weights = open_model_root_gguf(assets->model_root, "rvq-Q4_K_M.gguf");
    assets->condition_encoder_weights = open_model_root_gguf(assets->model_root, "condition-F32.gguf");
    assets->transformer_weights = open_model_root_gguf(assets->model_root, "dit-Q4_K_M.gguf");
    assets->vocoder_weights = open_model_root_gguf(assets->model_root, "vae-F16.gguf");
    validate_minimax_music3_anchors(*assets);
    return assets;
}

std::shared_ptr<const MiniMaxMusic3Assets> load_minimax_music3_component_assets(
    const std::filesystem::path & language_model,
    const std::filesystem::path & rvq_depth_decoder,
    const std::filesystem::path & condition_encoder,
    const std::filesystem::path & flow_transformer,
    const std::filesystem::path & vocoder) {
    const std::vector<std::filesystem::path> paths{
        language_model, rvq_depth_decoder, condition_encoder, flow_transformer, vocoder};
    for (const auto & path : paths) {
        if (!engine::io::is_existing_file(path) || path.extension() != ".gguf") {
            throw std::runtime_error("MiniMax Music 3 Cantor component is not an existing GGUF: " + path.string());
        }
    }
    auto result = std::make_shared<MiniMaxMusic3Assets>();
    result->model_root = assets::prepare_model_directory(language_model).model_root;
    result->tokenizer_json_path = result->model_root / "tokenizer" / "tokenizer.json";
    result->tokenizer_config_path = result->model_root / "tokenizer" / "tokenizer_config.json";
    if (!engine::io::is_existing_file(result->tokenizer_json_path) ||
        !engine::io::is_existing_file(result->tokenizer_config_path)) {
        throw std::runtime_error("MiniMax Music 3 LM GGUF does not provide the tokenizer sidecars");
    }
    result->config.qwen = parse_qwen_config(result->model_root);
    result->config.depth = parse_depth_config(result->model_root);
    result->config.condition = parse_condition_config(result->model_root);
    result->config.flow = parse_flow_config(result->model_root);
    result->config.vocoder = parse_vocoder_config(result->model_root);
    validate_pinned_config(result->config);
    result->language_model_weights = assets::open_tensor_source(language_model);
    result->depth_decoder_weights = assets::open_tensor_source(rvq_depth_decoder);
    result->condition_encoder_weights = assets::open_tensor_source(condition_encoder);
    result->transformer_weights = assets::open_tensor_source(flow_transformer);
    result->vocoder_weights = assets::open_tensor_source(vocoder);
    validate_minimax_music3_anchors(*result);
    return result;
}

assets::TensorStorageType conv_safe_storage_type(
    const assets::TensorSource & source,
    const std::string & tensor_prefix,
    assets::TensorStorageType requested) {
    if (requested == assets::TensorStorageType::BF16) {
        return assets::TensorStorageType::F32;
    }
    if (requested == assets::TensorStorageType::Native &&
        assets::ggml_type_for_tensor_dtype(source.require_metadata(tensor_prefix + ".weight").dtype) ==
            GGML_TYPE_BF16) {
        return assets::TensorStorageType::F32;
    }
    return requested;
}

}  // namespace engine::models::minimax_music3
