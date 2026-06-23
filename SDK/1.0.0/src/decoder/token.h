#ifndef WENET_SDK_SRC_DECODER_TOKEN_H_
#define WENET_SDK_SRC_DECODER_TOKEN_H_

#include <memory>
#include <vector>

namespace wenet_sdk::internal {

struct Token {
  int label = -1;
  int frame = -1;
  float score = 0.0f;
  std::shared_ptr<Token> prev;
};

std::vector<int> TraceLabels(const std::shared_ptr<Token>& token);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_TOKEN_H_
