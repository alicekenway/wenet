#include <cassert>

#include "decoder/symbol_table.h"

int main() {
  wenet_sdk::internal::SymbolTable table;
  auto status = table.Load("model_example/tokens.txt");
  assert(status.ok());
  assert(table.Size() == 4);
  assert(table.Id("HELLO") == 1);
  assert(table.Symbol(2) == "WORLD");
  assert(table.Symbol(100) == "<unk>");
  return 0;
}
