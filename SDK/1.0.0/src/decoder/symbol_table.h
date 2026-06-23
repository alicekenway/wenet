#ifndef WENET_SDK_SRC_DECODER_SYMBOL_TABLE_H_
#define WENET_SDK_SRC_DECODER_SYMBOL_TABLE_H_

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/status.h"

namespace wenet_sdk::internal {

class SymbolTable {
 public:
  Status Load(const std::filesystem::path& path);

  const std::string& Symbol(int id) const;
  int Id(const std::string& symbol) const;
  bool Contains(int id) const;
  int Size() const { return static_cast<int>(id_to_symbol_.size()); }
  bool empty() const { return id_to_symbol_.empty(); }

 private:
  std::vector<std::string> id_to_symbol_;
  std::unordered_map<std::string, int> symbol_to_id_;
};

std::string Trim(std::string text);
bool ParseInteger(const std::string& text, int* value);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_SYMBOL_TABLE_H_
