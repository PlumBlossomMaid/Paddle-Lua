// tools/smoke/smoke_static_trace.cc
// P15 (static-graph trace, plan A: inference-first) polish probe:
// build a minimal pir::Program BY HAND (no pybind, no AST) and run it
// through egr::to_static::run_program_ad_func in inference mode
// (is_test=true + empty backward_program, partial_program.py:427 pattern).
//
// Program under test:
//   (%0) = pd_op.data("x", shape=[2,2], FLOAT32, CPUPlace)
//   (%1) = pd_op.scale(%0, scale=2.0, bias=0.0)
// Ground truth: input full({2,2},3.0) -> output 6.0 everywhere.
// Pass condition: out == 6.0 (max_err < 1e-6). This proves that a
// hand-assembled Program is consumed correctly by InterpreterCore,
// which is the load-bearing fact for the trace recorder.
//
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, HEAD 2bfeb115b0):
//   paddle/pir/include/core/program.h:44          -> Program(IrContext*)
//   paddle/pir/src/core/program.cc:55-57          -> ctor builds ModuleOp
//   paddle/pir/include/core/builder.h:139-141     -> Builder::Build(...,OpInfo)
//   paddle/pir/include/core/ir_context.h:124      -> GetRegisteredOpInfo
//   paddle/fluid/pir/dialect/operator/ir/pd_op.h (generated, build-nopy):
//     :5050  DataOp::Build(builder,arg,name,shape,dtype,place)
//     :17306 ScaleOp::Build(builder,arg,x_,scale,bias,bias_after_scale)
//   paddle/fluid/pybind/pir.cc:2606-2610          -> GetOrRegisterDialect
//   paddle/fluid/eager/to_static/run_program_func.h:23-28
//                                                 -> run_program_ad_func
//   paddle/fluid/eager/to_static/run_program_impl.cc:448-500
//                                                 -> prog_attrs keys
//   python/paddle/jit/dy2static/partial_program.py:427
//                                                 -> empty backward for eval
//
// Build: add_executable smoke_static_trace in the top-level CMakeLists
// (same LibPaddle baseline as paddle_core.dll; standalone EXE needs
// /FORCE:MULTIPLE per m0-report.md sec.4.1).
// Run: smoke_static_trace.exe

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/to_static/run_program_func.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"

// Operator dialect declaration: OperatorDialect (pd_op.h does NOT declare
// it; it lives in op_dialect.h:31). DataOp/ScaleOp typed wrappers come from
// the generated pd_op.h in the build tree (build-nopy); the include path
// for it comes from LibPaddle_INCLUDE_DIRS, same as csrc/capi/jit.cc.
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

int main() {
  // Same init contract as tools/smoke/smoke.cc (m0-report.md sec.4.2):
  // InitMemoryMethod before any phi allocation.
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  pir::IrContext* ctx = pir::IrContext::Instance();
  // Register the pd_op dialect in this process (pybind does this on import;
  // pir.cc:2606-2610). Without it GetRegisteredOpInfo("pd_op.scale") fails.
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  // ---- hand-build the program ----------------------------------------
  auto program = std::make_shared<pir::Program>(ctx);
  pir::Builder builder(ctx, program->block());

  // (%0) = pd_op.data("x", [2,2], FLOAT32, CPUPlace)
  paddle::dialect::DataOp data_op = builder.Build<paddle::dialect::DataOp>(
      "x", std::vector<int64_t>{2, 2}, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::Value x = data_op.out();

  // (%1) = pd_op.scale(%0, scale=2.0, bias=0.0, bias_after_scale=true)
  paddle::dialect::ScaleOp scale_op = builder.Build<paddle::dialect::ScaleOp>(
      x, 2.0f, 0.0f, true);
  pir::Value y = scale_op.out();

  // Mark the output: builtin.shadow_output writes the value into a named
  // scope variable so RunProgramImpl's ShareTensorsFromScopeWithName can
  // find it by fo_names (run_program_impl.cc:225). Mirrors Python's
  // append_shadow_outputs (pybind/pir.cc:2247-2265). ShadowOutputOp::Build
  // takes (builder, argument, value, name) per builtin_op.h:109-114.
  builder.Build<pir::ShadowOutputOp>(y, "out");

  std::cout << "program built: " << program->num_ops() << " ops" << std::endl;

  // ---- prog_attrs / cuda_graph_attrs ----------------------------------
  paddle::framework::AttributeMap prog_attrs;
  prog_attrs["program_id"] = static_cast<int64_t>(program->id());
  prog_attrs["is_test"] = true;  // inference path (partial_program.py:427)
  prog_attrs["in_sot_mode"] = false;
  prog_attrs["fx_names"] = std::vector<std::string>{"x"};
  prog_attrs["fp_names"] = std::vector<std::string>{};
  prog_attrs["fo_names"] = std::vector<std::string>{"out"};
  prog_attrs["no_need_buffers_names"] = std::vector<std::string>{};
  prog_attrs["fo_values"] = std::vector<pir::Value>{y};
  prog_attrs["forward_program"] = program;
  // Inference: empty backward program mirrors Python's eval path
  // (partial_program.py:427 _empty_backward_program_for_eval).
  prog_attrs["backward_program"] = std::make_shared<pir::Program>(ctx);

  paddle::framework::AttributeMap cuda_graph_attrs;
  cuda_graph_attrs["cuda_graph_state"] = static_cast<int64_t>(0);
  cuda_graph_attrs["cuda_graph_dispatch_key"] = static_cast<int64_t>(0);

  // Single outer scope (run_program_impl.cc:466-473).
  auto scope = std::make_shared<paddle::framework::Scope>();
  std::vector<paddle::framework::Scope*> step_scope{scope.get()};

  // Input: x = full({2,2}, 3.0) -> expected output = 6.0 everywhere.
  paddle::Tensor x_in = paddle::experimental::full(
      {2, 2},
      paddle::experimental::Scalar(3.0),
      phi::DataType::FLOAT32,
      phi::CPUPlace());

  std::vector<paddle::Tensor> out;
  try {
    out = egr::to_static::run_program_ad_func(
        {x_in}, {}, step_scope, prog_attrs, cuda_graph_attrs);
  } catch (const std::exception& e) {
    std::cout << "SMOKE FAIL: run_program_ad_func threw: "
              << e.what() << std::endl;
    return 1;
  }

  if (out.size() != 1) {
    std::cout << "SMOKE FAIL: expected 1 output, got " << out.size()
              << std::endl;
    return 1;
  }
  auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(out[0].impl());
  if (!dense) {
    std::cout << "SMOKE FAIL: output is not a dense tensor" << std::endl;
    return 1;
  }
  const float* data = dense->data<float>();
  float max_err = 0.0f;
  for (int i = 0; i < 4; ++i) {
    max_err = std::max(max_err, std::abs(data[i] - 6.0f));
  }

  if (max_err < 1e-6f) {
    std::cout << "SMOKE PASS: hand-built Program ran via InterpreterCore, "
                 "out == 6.0, max_err="
              << max_err << std::endl;
    return 0;
  }
  std::cout << "SMOKE FAIL: out != 6.0, first=" << data[0]
            << " max_err=" << max_err << std::endl;
  return 1;
}
