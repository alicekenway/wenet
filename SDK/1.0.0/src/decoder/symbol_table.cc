#include "decoder/symbol_table.h"

#include <fstream>
#include <sstream>

namespace wenet_sdk::internal {
namespace {

const std::string& UnknownSymbol() {
  static const std::string kUnknown = "<unk>";
  return kUnknown;
}

void PutSymbol(std::vector<std::string>* id_to_symbol,
               std::unordered_map<std::string, int>* symbol_to_id, int id,
               const std::string& symbol) {
  if (id < 0 || symbol.empty()) {
    return;
  }
  if (static_cast<int>(id_to_symbol->size()) <= id) {
    id_to_symbol->resize(static_cast<size_t>(id + 1));
  }
  (*id_to_symbol)[static_cast<size_t>(id)] = symbol;
  (*symbol_to_id)[symbol] = id;
}

}  // namespace

std::string Trim(std::string text) {
  const char* ws = " \t\r\n";
  const auto begin = text.find_first_not_of(ws);
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(ws);
  return text.substr(begin, end - begin + 1);
}

bool ParseInteger(const std::string& text, int* value) {
  if (text.empty()) {
    return false;
  }
  size_t pos = 0;
  try {
    int parsed = std::stoi(text, &pos);
    if (pos != text.size()) {
      return false;
    }
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

Status SymbolTable::Load(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    return Status::NotFound("failed to open symbol table: " + path.string());
  }

  id_to_symbol_.clear();
  symbol_to_id_.clear();

  std::string line;
  int implicit_id = 0;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::vector<std::string> cols;
    std::string col;
    while (iss >> col) {
      cols.push_back(col);
    }
    if (cols.empty()) {
      continue;
    }

    int id = implicit_id;
    std::string symbol = cols[0];
    if (cols.size() >= 2) {
      int first = 0;
      int second = 0;
      if (ParseInteger(cols[0], &first)) {
        id = first;
        symbol = cols[1];
      } else if (ParseInteger(cols.back(), &second)) {
        id = second;
        symbol = cols[0];
      }
    }

    if (id < 0) {
      return Status::InvalidArgument("negative id in " + path.string() +
                                     " at line " + std::to_string(line_no));
    }
    PutSymbol(&id_to_symbol_, &symbol_to_id_, id, symbol);
    implicit_id = std::max(implicit_id + 1, id + 1);
  }

  if (id_to_symbol_.empty()) {
    return Status::InvalidArgument("symbol table is empty: " + path.string());
  }
  return Status::OK();
}

const std::string& SymbolTable::Symbol(int id) const {
  if (id < 0 || id >= static_cast<int>(id_to_symbol_.size()) ||
      id_to_symbol_[static_cast<size_t>(id)].empty()) {
    return UnknownSymbol();
  }
  return id_to_symbol_[static_cast<size_t>(id)];
}

int SymbolTable::Id(const std::string& symbol) const {
  const auto it = symbol_to_id_.find(symbol);
  return it == symbol_to_id_.end() ? -1 : it->second;
}

bool SymbolTable::Contains(int id) const {
  return id >= 0 && id < static_cast<int>(id_to_symbol_.size()) &&
         !id_to_symbol_[static_cast<size_t>(id)].empty();
}

}  // namespace wenet_sdk::internal
