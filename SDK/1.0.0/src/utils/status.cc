#include "utils/status.h"

namespace wenet_sdk::internal {
namespace {
static_assert(static_cast<int>(StatusCode::kOk) == 0,
              "StatusCode must stay ABI-aligned with C API statuses");
}  // namespace
}  // namespace wenet_sdk::internal
