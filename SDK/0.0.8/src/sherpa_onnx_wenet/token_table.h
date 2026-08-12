#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_TOKEN_TABLE_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_TOKEN_TABLE_H_

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace asr_sdk::internal::sherpa_onnx_wenet {

class TokenTable {
 public:
  explicit TokenTable(const std::filesystem::path& path);

  int Size() const { return static_cast<int>(id_to_token_.size()); }
  int ModelVocabSize() const { return model_vocab_size_; }
  int BlankId() const { return blank_id_; }
  bool Contains(const std::string& token) const;
  int Id(const std::string& token) const;
  const std::string& Token(int id) const;
  std::string DecodeIds(const std::vector<int>& ids) const;

 private:
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  int model_vocab_size_ = 0;
  int blank_id_ = 0;
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_TOKEN_TABLE_H_
