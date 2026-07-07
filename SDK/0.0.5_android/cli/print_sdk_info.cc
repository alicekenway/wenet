#include <iostream>

#include "asr_sdk/version.h"

int main() {
  std::cout << asr_sdk::BuildInfoJson() << "\n";
  return 0;
}
