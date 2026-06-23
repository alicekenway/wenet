#include "wenet_bridge/wenet_runtime_bridge.h"

#include <algorithm>
#include <exception>

#include "decoder/asr_decoder.h"
#include "decoder/onnx_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "package/model_package_validator.h"
#include "post_processor/post_processor.h"
#include "wenet_bridge/wenet_shared.h"

namespace asr_sdk::internal {
namespace {

bool SamePath(const std::filesystem::path& a, const std::filesystem::path& b) {
  std::error_code ec;
  return std::filesystem::equivalent(a, b, ec);
}

Status ValidateFlatRuntimePackage(const ModelPackage& package) {
  if (!SamePath(package.runtime_dir, package.root)) {
    return Status::FailedPrecondition(
        "this WeNet C API bridge expects a flat runtime package; set "
        "onnx_dir to '.' or symlink encoder.onnx/ctc.onnx/decoder.onnx "
        "into model_dir");
  }
  if (!SamePath(package.units_txt.parent_path(), package.root)) {
    return Status::FailedPrecondition(
        "units.txt must be in model_dir for the WeNet C API bridge");
  }
  if (package.has_wfst &&
      (!SamePath(package.tlg_fst.parent_path(), package.root) ||
       !SamePath(package.words_txt.parent_path(), package.root))) {
    return Status::FailedPrecondition(
        "TLG.fst and words.txt must be in model_dir for the WeNet C API bridge");
  }
  return Status::Ok();
}

StatusOr<std::shared_ptr<wenet_types::Shared>> LoadSharedResources(
    const EngineConfig& config, const ModelPackage& package) {
  auto shared = std::make_shared<wenet_types::Shared>();
  shared->feature_config = std::make_shared<wenet::FeaturePipelineConfig>(
      80, package.sample_rate);
  shared->resource = std::make_shared<wenet::DecodeResource>();

  try {
    wenet::OnnxAsrModel::InitEngineThreads(
        std::max(1, config.num_threads));
    auto model = std::make_shared<wenet::OnnxAsrModel>();
    model->Read(package.root.string());
    shared->resource->model = model;

    shared->resource->unit_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(package.units_txt.string()));
    if (shared->resource->unit_table == nullptr) {
      return Status::Internal("failed to load units.txt: " +
                              package.units_txt.string());
    }

    if (package.has_wfst) {
      shared->resource->fst = std::shared_ptr<fst::VectorFst<fst::StdArc>>(
          fst::VectorFst<fst::StdArc>::Read(package.tlg_fst.string()));
      if (shared->resource->fst == nullptr) {
        return Status::Internal("failed to load TLG.fst: " +
                                package.tlg_fst.string());
      }
      shared->resource->symbol_table = std::shared_ptr<fst::SymbolTable>(
          fst::SymbolTable::ReadText(package.words_txt.string()));
      if (shared->resource->symbol_table == nullptr) {
        return Status::Internal("failed to load words.txt: " +
                                package.words_txt.string());
      }
    } else {
      shared->resource->symbol_table = shared->resource->unit_table;
    }

    wenet::PostProcessOptions post_opts;
    post_opts.language_type =
        package.language == "en" ? wenet::kIndoEuropean
                                 : wenet::kMandarinEnglish;
    shared->resource->post_processor =
        std::make_shared<wenet::PostProcessor>(post_opts);

    shared->decode_options = std::make_shared<wenet::DecodeOptions>();
    shared->decode_options->chunk_size = package.chunk_size;
    shared->decode_options->num_left_chunks = package.num_left_chunks;
    shared->decode_options->ctc_weight = 1.0;
    shared->decode_options->rescoring_weight = 0.0;
    shared->decode_options->reverse_weight = 0.0;
    shared->decode_options->ctc_wfst_search_opts.max_active = 7000;
    shared->decode_options->ctc_wfst_search_opts.min_active = 200;
    shared->decode_options->ctc_wfst_search_opts.beam = 16.0;
    shared->decode_options->ctc_wfst_search_opts.lattice_beam = 10.0;
    shared->decode_options->ctc_wfst_search_opts.acoustic_scale = 1.0;
    shared->decode_options->ctc_wfst_search_opts.blank = 0;
    shared->decode_options->ctc_wfst_search_opts.blank_skip_thresh = 1.0;
    shared->decode_options->ctc_wfst_search_opts.blank_scale = 1.0;
    shared->decode_options->ctc_wfst_search_opts.length_penalty = 0.0;
    shared->decode_options->ctc_wfst_search_opts.nbest = package.nbest;
    shared->decode_options->ctc_prefix_search_opts.first_beam_size =
        package.nbest;
    shared->decode_options->ctc_prefix_search_opts.second_beam_size =
        package.nbest;
    shared->decode_options->ctc_prefix_search_opts.blank = 0;
    shared->decode_options->ctc_endpoint_config.blank = 0;
    shared->decode_options->ctc_endpoint_config.blank_scale = 1.0;
  } catch (const std::exception& e) {
    return Status::Internal(std::string("failed to load WeNet runtime: ") +
                            e.what());
  }
  return shared;
}

}  // namespace

WenetRuntimeBridge::WenetRuntimeBridge(EngineConfig config,
                                       ModelPackage package)
    : config_(std::move(config)), package_(std::move(package)) {}

StatusOr<std::unique_ptr<WenetRuntimeBridge>> WenetRuntimeBridge::Create(
    const EngineConfig& config, ModelPackage package) {
  Status status = ValidateModelPackage(package);
  if (!status.ok()) {
    return status;
  }
  status = ValidateFlatRuntimePackage(package);
  if (!status.ok()) {
    return status;
  }
  EngineConfig resolved_config = config;
  resolved_config.model_dir = package.root.string();
  resolved_config.sample_rate = package.sample_rate;
  resolved_config.chunk_size = package.chunk_size;
  resolved_config.nbest = package.nbest;
  resolved_config.enable_continuous_decoding =
      package.enable_continuous_decoding;
  resolved_config.enable_timestamps = package.enable_timestamps;
  resolved_config.language = package.language;
  resolved_config.num_left_chunks = package.num_left_chunks;
  auto shared_or = LoadSharedResources(resolved_config, package);
  if (!shared_or.ok()) {
    return shared_or.status();
  }
  auto bridge = std::unique_ptr<WenetRuntimeBridge>(
      new WenetRuntimeBridge(std::move(resolved_config), std::move(package)));
  bridge->shared_ = std::move(shared_or).value();
  return bridge;
}

StatusOr<std::unique_ptr<WenetStreamAdapter>>
WenetRuntimeBridge::CreateStream() {
  auto adapter = std::make_unique<WenetStreamAdapter>(shared_, config_);
  Status status = adapter->Init();
  if (!status.ok()) {
    return status;
  }
  return adapter;
}

}  // namespace asr_sdk::internal
