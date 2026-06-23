#include <cassert>

#include "postprocess/endpoint.h"

int main() {
  wenet_sdk::internal::Endpoint endpoint({100, 1000, 10});
  endpoint.AdvanceFrames(5, true);
  assert(!endpoint.IsEndpoint(false));
  endpoint.AdvanceFrames(5, true);
  assert(endpoint.IsEndpoint(false));
  endpoint.Reset();
  assert(endpoint.IsEndpoint(true));
  return 0;
}
