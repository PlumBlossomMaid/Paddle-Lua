// tools/smoke/smoke_jit_tiny.cc -- isolate the 0xC0000005: is it the C++
// Load+forward pipeline (product+engine+flags) or the sol binding layer?
// Mirrors smoke_jit_load.cc (T-M0-17) but against OUR tiny non-PIR export
// and WITHOUT the Lua layer: same FLAGS, same jit::Load, same forward.
//
// Build: same 483-lib baseline as smoke_jit_load.cc (m0-report.md sec.2),
//   /LIBPATH:E:\code\Paddle\build-nopy, /FORCE:MULTIPLE, /bigobj, 35 DEFINES.
//   The GetFeedTarget/GetFetchTarget shim IS needed (PirFunctionInfo ctor
//   calls them unconditionally, function_schema.cc:112/115 -- same as
//   smoke_jit_load.cc). Run: smoke_jit_tiny.exe
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/pir/include/core/builtin_attribute.h"

// The tiny net is elementwise_add -> bias-broadcast add -> relu (old-IR
// dy2static leaves feed/fetch/scale/transfer_layout wrappers). New executor
// resolves ops by name at runtime, so the EXE must link the op/kernel
// registrations in or they silently go missing (m0-report.md sec.4.3) --
// mirrors smoke_jit_load.cc.
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(relu);
USE_OP_ITSELF(feed);
USE_OP_ITSELF(fetch);
USE_OP_ITSELF(scale);
USE_OP_ITSELF(transfer_layout);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);

// ---------------------------------------------------------------------------
// Upstream link gap shim (WITH_PYTHON=OFF only) -- same source as the
// T-M0-17 smoke (smoke_jit_load.cc): libfluid_jit's PirFunctionInfo ctor
// (function_schema.cc:112/115) calls paddle::GetFeedTargetNames /
// GetFetchTargetNames unconditionally, but those free functions live in
// paddle/fluid/pybind/pir_utils.cc:20/36 -- the pybind directory, NOT built
// under WITH_PYTHON=OFF. Pure PIR, zero Python dependency. See
// process/m0-report.md T-M0-17.
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

COMMON_DECLARE_bool(enable_pir_api);
COMMON_DECLARE_string(jit_engine_type);
COMMON_DECLARE_bool(use_mkldnn);

int main() {
  try {
    paddle::framework::InitMemoryMethod();
    paddle::framework::InitDevices();

    FLAGS_enable_pir_api = false;   // non-PIR export (serializer.cc:48)
    FLAGS_jit_engine_type = "New";  // InterpreterEngine (D16 / serializer.cc:73)
    FLAGS_use_mkldnn = false;       // avoid onednn input-name mapping bug (Q-24)

    std::cout << "flags: pir=" << FLAGS_enable_pir_api
              << " engine=" << FLAGS_jit_engine_type
              << " mkldnn=" << FLAGS_use_mkldnn << std::endl;

    auto layer = paddle::jit::Load("tools/smoke/data/jit_export/tiny",
                                   phi::CPUPlace());
    std::cout << "LOADED" << std::endl;

    // input x = [2,4] all 1.0; net: y = relu(x + x + b) with b=[0.5]
    paddle::Tensor x = paddle::experimental::full(
        {2, 4}, paddle::experimental::Scalar(1.0f), phi::DataType::FLOAT32,
        phi::CPUPlace());
    std::vector<paddle::Tensor> outs = layer.forward({x});
    std::cout << "FORWARD ok n=" << outs.size() << std::endl;
    const float* p = outs[0].data<float>();
    std::cout << "out[0]=" << p[0] << std::endl;

    float expect = 2.0f + 0.5f;  // add(x,x)->2.0, +b(0.5)->2.5, relu(2.5)=2.5
    if (std::abs(p[0] - expect) < 1e-6f) {
      std::cout << "SMOKE_JIT_TINY PASS" << std::endl;
      return 0;
    }
    std::cout << "SMOKE_JIT_TINY FAIL expect=" << expect << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "SMOKE_JIT_TINY EXCEPTION: " << e.what() << std::endl;
    return 2;
  }
}