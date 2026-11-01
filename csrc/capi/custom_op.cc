// custom_op.cc -- custom-operator loading for the C ABI layer (P16).
// Implements pd_load_op_meta_info (paddle_c.h Group 13).
//
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, HEAD 2bfeb115b0):
//   paddle/fluid/framework/custom_operator.h:316
//                                                  LoadOpMetaInfoAndRegisterOp
//   paddle/fluid/framework/custom_operator.cc:1279-1315
//                                                  RegisterOperatorWithMetaInfoMap
//     -- registers BOTH the PIR dialect op (CustomOpDialect::RegisterCustomOp
//        via custom_dialect, op_dialect.cc:1180, name = "custom_op." + op,
//        custom_operator_utils.h:29) so the op participates in pir program
//        assembly, AND the legacy fluid OpInfo table (RegisterOperatorWithMetaInfo,
//        custom_operator.cc:962) which owns the executable kernel
//        (RegisterOperatorKernel -> OperatorWithKernel, custom_operator.cc:898-961)
//   paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:1180-1229
//                                                  CustomOpDialect::RegisterCustomOp
//   paddle/fluid/pir/transforms/pd_op_to_kernel_pass.cc:3744-3751
//                                                  IsCustomOp -> HandleForCustomOp
//   paddle/phi/api/ext/op_meta_info.h:1164         PD_BUILD_OP macro (user side)
//
// The PIR path is what makes a loaded custom op usable from our Lua trace /
// program-assembly machinery (P15): once registered as "custom_op.<name>",
// a builder can look it up via IrContext::GetRegisteredOpInfo and run it
// through run_program_ad_func exactly like a generated op.

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

#include "capi_impl.h"
#include "paddle_c.h"

extern "C" {

int pd_load_op_meta_info(const char* dso_path, char* names, int cap) {
  PD_GUARD_BEGIN;
  if (dso_path == nullptr || dso_path[0] == '\0') {
    pd_set_error(PD_INVALID_ARG, "pd_load_op_meta_info: empty dso_path");
    return -1;
  }
  if (cap > 0 && names == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_load_op_meta_info: names null with cap>0");
    return -1;
  }
  const auto& diff =
      paddle::framework::LoadOpMetaInfoAndRegisterOp(std::string(dso_path));
  size_t written = 0;
  int total = 0;
  for (const auto& pair : diff) {
    ++total;
    if (written + pair.first.size() + 1 <= static_cast<size_t>(cap)) {
      std::memcpy(names + written, pair.first.c_str(),
                  pair.first.size() + 1);
      written += pair.first.size() + 1;
    }
  }
  return total;
  PD_GUARD_END(-1);
}

}  // extern "C"