#include <iostream>

#include "asr_sdk/version.h"

// Print SDK version and compilation metadata as JSON.

int main() {
  std::cout << asr_sdk::BuildInfoJson() << "\n";
  return 0;
}
