#include "package/model_package_validator.h"

#include "utils/file_utils.h"

namespace asr_sdk::internal {
namespace {

void AddFileLine(const char* label, const std::filesystem::path& path,
                 ModelPackageReport* report) {
  const bool ok = FileExists(path);
  report->lines.push_back(std::string(label) + ": " +
                          (ok ? "ok " : "missing ") + path.string());
  report->ok = report->ok && ok;
}

Status RequireFile(const std::filesystem::path& path, const char* label) {
  if (!FileExists(path)) {
    return Status::NotFound(std::string(label) + " not found: " +
                            path.string());
  }
  return Status::Ok();
}

}  // namespace

Status ValidateModelPackage(const ModelPackage& package) {
  if (!DirectoryExists(package.root)) {
    return Status::NotFound("model root not found: " + package.root.string());
  }
  if (!DirectoryExists(package.runtime_dir)) {
    return Status::NotFound("ONNX runtime directory not found: " +
                            package.runtime_dir.string());
  }
  for (const auto& item :
       {std::pair{package.encoder_onnx, "encoder.onnx"},
        std::pair{package.ctc_onnx, "ctc.onnx"},
        std::pair{package.decoder_onnx, "decoder.onnx"},
        std::pair{package.units_txt, "units.txt"}}) {
    Status status = RequireFile(item.first, item.second);
    if (!status.ok()) {
      return status;
    }
  }
  if (package.has_wfst || FileExists(package.tlg_fst)) {
    Status status = RequireFile(package.tlg_fst, "TLG.fst");
    if (!status.ok()) {
      return status;
    }
    status = RequireFile(package.words_txt, "words.txt");
    if (!status.ok()) {
      return status;
    }
  }
  if (package.sample_rate != 16000) {
    return Status::InvalidArgument(
        "only 16 kHz PCM is supported by this WeNet runtime package");
  }
  if (package.chunk_size == 0) {
    return Status::InvalidArgument("chunk_size must not be 0");
  }
  return Status::Ok();
}

ModelPackageReport InspectModelPackage(const ModelPackage& package) {
  ModelPackageReport report;
  report.ok = true;
  report.lines.push_back("model_dir: " + package.root.string());
  report.lines.push_back("manifest: " +
                         std::string(package.has_manifest ? "yes " : "no ") +
                         package.manifest.string());
  report.lines.push_back("runtime_dir: " + package.runtime_dir.string());
  report.lines.push_back("sample_rate: " + std::to_string(package.sample_rate));
  report.lines.push_back("chunk_size: " + std::to_string(package.chunk_size));
  report.lines.push_back("num_left_chunks: " +
                         std::to_string(package.num_left_chunks));
  report.lines.push_back("nbest: " + std::to_string(package.nbest));
  AddFileLine("encoder.onnx", package.encoder_onnx, &report);
  AddFileLine("ctc.onnx", package.ctc_onnx, &report);
  AddFileLine("decoder.onnx", package.decoder_onnx, &report);
  AddFileLine("units.txt", package.units_txt, &report);
  if (FileExists(package.tlg_fst)) {
    AddFileLine("TLG.fst", package.tlg_fst, &report);
    AddFileLine("words.txt", package.words_txt, &report);
  } else {
    report.lines.push_back("TLG.fst: not configured");
  }
  const Status status = ValidateModelPackage(package);
  if (!status.ok()) {
    report.ok = false;
    report.lines.push_back("status: " + status.ToString());
  } else {
    report.lines.push_back("status: OK");
  }
  return report;
}

}  // namespace asr_sdk::internal
