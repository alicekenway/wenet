#ifndef ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_VALIDATOR_H_
#define ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_VALIDATOR_H_

#include <string>
#include <vector>

#include "package/model_package.h"

namespace asr_sdk::internal {

struct ModelPackageReport {
  bool ok = false;
  std::vector<std::string> lines;
};

Status ValidateModelPackage(const ModelPackage& package);
ModelPackageReport InspectModelPackage(const ModelPackage& package);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_VALIDATOR_H_
