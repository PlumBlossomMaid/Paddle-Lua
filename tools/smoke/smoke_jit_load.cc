// tools/smoke/smoke_jit_load.cc
// T-M0-17: verify that jit::Layer Load() can load a jit.save export and run
// forward() in a WITH_PYTHON=OFF + ON_INFER=OFF build.
//
// Question under test: can the Lua frontend load a model saved by Python's
// paddle.jit.save (the standard interop path: Python trains + Lua infers)?
// The .pdmodel/.pdiparams/.meta files are binary program + params + attrs.
//
// References:
//   paddle/fluid/jit/serializer.h:78               -> jit::Load(path, place)
//   paddle/fluid/jit/serializer.cc:176             -> Load -> Deserializer()(path, place)
//   paddle/fluid/jit/serializer.cc:37-98           -> read .pdmodel + .pdiparams + .meta
//   paddle/fluid/jit/serializer_utils.h:29-32      -> suffixes .pdmodel/.pdiparams/.meta
//   paddle/fluid/jit/layer.h:41                    -> Layer::forward(const vector<Tensor>&)
//   paddle/common/flags.cc:1888                    -> FLAGS_enable_pir_api default true
//   test/cpp/jit/layer_test.cc:86-126              -> reference assertions (fare attribute
//                                                     fbias=1.4, attrs dict, forward=0.02194316,
//                                                     Function("infer")=1.41562390)
//   test/cpp/jit/CMakeLists.txt:6                  -> sample export tarball URL (downloaded)
//
// The sample export was produced in non-PIR mode; PIR mode expects .json
// program files, so this smoke sets FLAGS_enable_pir_api=false (matching the
// upstream layer_test.cc guard which skips tests under PIR).
//
// Build: same 483-lib baseline as tools/smoke/smoke.cc (m0-report.md section 2),
//   must add /LIBPATH:E:\code\Paddle\build-nopy (libonednn/libcblas at root),
//   /FORCE:MULTIPLE, /bigobj, and the 35 DEFINES (see smoke_strided_slice.cc).
// Run with a sample export at <cwd>/multi_program_load/export.pdmodel etc.

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_utils.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/pir/include/core/builtin_attribute.h"

// ---------------------------------------------------------------------------
// Upstream link gap shim (WITH_PYTHON=OFF only):
//   libfluid_jit's PirFunctionInfo ctor (function_schema.cc:112/115) calls
//   paddle::GetFeedTargetNames/GetFetchTargetNames unconditionally, but those
//   free functions live in paddle/fluid/pybind/pir_utils.cc:20/36 -- the
//   pybind directory, which is NOT built under WITH_PYTHON=OFF (verified:
//   build-nopy has no pybind .lib output).  The implementations themselves are
//   pure PIR (DataOp/FeedOp/FetchOp/ShadowOutputOp + StrAttribute) with zero
//   Python dependency, so we provide equivalents here for the smoke EXE.
//   Lua bindings that link libfluid_jit will need the same shim (or an
//   upstream patch moving these two free functions out of the pybind dir).
//   See process/m0-report.md T-M0-17.
namespace paddle {
std::vector<std::string> GetFeedTargetNames(pir::Program* prog) {
  std::vector<std::string> feed_target;
  for (auto& op : *(prog->block())) {
    if (op.isa<paddle::dialect::DataOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      feed_target.push_back(name);
      continue;
    } else if (op.isa<paddle::dialect::FeedOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      feed_target.push_back(name);
      continue;
    }
  }
  return feed_target;
}

std::vector<std::string> GetFetchTargetNames(pir::Program* prog) {
  std::vector<std::string> fetch_target;
  for (auto& op : *(prog->block())) {
    if (op.isa<paddle::dialect::FetchOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      fetch_target.push_back(name);
      continue;
    } else if (op.isa<pir::ShadowOutputOp>()) {
      auto name = op.attribute<pir::StrAttribute>("output_name").AsString();
      fetch_target.push_back(name);
      continue;
    }
  }
  return fetch_target;
}
}  // namespace paddle
// ---------------------------------------------------------------------------

// Forwards/backwards of the sample net are elementwise_add -> matmul_v2 -> relu
// -> reduce_mean plus feed/fetch/scale/transfer_layout wrappers.
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(matmul_v2);
USE_OP_ITSELF(relu);
USE_OP_ITSELF(reduce_mean);
USE_OP_ITSELF(feed);
USE_OP_ITSELF(fetch);
USE_OP_ITSELF(scale);
USE_OP_ITSELF(transfer_layout);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);

COMMON_DECLARE_bool(enable_pir_api);
COMMON_DECLARE_string(jit_engine_type);
COMMON_DECLARE_bool(use_mkldnn);

namespace paddle {
namespace jit {
using DenseTensor = phi::DenseTensor;

std::vector<paddle::Tensor> PrepareInputs(const phi::Place& place) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(place);
  phi::DenseTensor t;
  t.Resize(common::make_ddim({2, 4}));
  t.mutable_data<float>(place);
  phi::funcs::set_constant(dev_ctx, &t, static_cast<float>(2.));
  return utils::ToTensors({t});
}
}  // namespace jit
}  // namespace paddle

int RunMain();

int main() {
  try {
    return RunMain();
  } catch (const std::exception& e) {
    std::cerr << "SMOKE EXCEPTION: " << e.what() << std::endl;
    return 2;
  }
}

int RunMain() {
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  // The downloaded sample export is non-PIR; PIR mode would look for .json
  // program files instead of .pdmodel (serializer.cc:41/83).
  FLAGS_enable_pir_api = false;
  // Default jit_engine_type is "Predictor" (flags.cc:1702) which runs the
  // analysis-predictor pipeline; that path hits an upstream oneDNN fc bug
  // (onednn_context.cc:347 "OneDnnContext does not have the input W").
  // We target InterpreterCore as our static-graph engine (D16), so use the
  // "New" engine here (serializer.cc:73-76 -> InterpreterEngine).
  FLAGS_jit_engine_type = "New";
  // fc 的 onednn kernel 会无条件查询输入名 W(onednn_context.cc:347),旧 sample
  // 在 New executor 下无此映射 → 显式禁用 oneDNN 排除该触发器。
  FLAGS_use_mkldnn = false;

  std::string path = "./multi_program_load/export";
  auto layer = paddle::jit::Load(path, phi::CPUPlace());

  // Attr round-trip: .meta file (Property) must have been deserialized.
  float fbias = layer.Attribute<float>("fbias");
  std::string fstr = layer.Attribute<paddle::framework::String>("fstr");
  std::vector<int> ints = layer.Attribute<std::vector<int>>("ints");

  // forward() through the loaded program.
  auto inputs = paddle::jit::PrepareInputs(phi::CPUPlace());
  auto outs = layer.forward(inputs);
  float out0 = outs[0].data<float>()[0];

  // Named sub-function "infer" (from export.infer.pdmodel).
  auto func = layer.Function("infer");
  bool fn_valid = func.IsValid();
  float inf0 = 0.0f;
  if (fn_valid) {
    auto inf_outs = func(inputs);
    inf0 = inf_outs[0].data<float>()[0];
  }

  bool ok = (std::abs(fbias - 1.4f) < 1e-6f) && (fstr == "save str property") &&
            (ints[0] == 10) && (ints[1] == 20) &&
            (std::abs(out0 - 0.02194316f) < 1e-6f) && fn_valid &&
            (std::abs(inf0 - 1.41562390f) < 1e-6f);

  if (ok) {
    std::cout << "SMOKE PASS: jit::Load ran forward()=" << out0
              << " infer()=" << inf0 << " fbias=" << fbias << " fstr=" << fstr
              << " ints=[" << ints[0] << "," << ints[1] << "]" << std::endl;
    return 0;
  }
  std::cout << "SMOKE FAIL: fbias=" << fbias << " fstr=" << fstr
            << " ints=[" << (ints.empty() ? -1 : ints[0]) << ","
            << (ints.size() < 2 ? -1 : ints[1]) << "] out0=" << out0
            << " fn_valid=" << fn_valid << " inf0=" << inf0 << std::endl;
  return 1;
}