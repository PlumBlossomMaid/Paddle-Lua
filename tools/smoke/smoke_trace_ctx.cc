// tools/smoke/smoke_trace_ctx.cc
// P15-A step 2: validate the C ABI trace context (trace.cc) end-to-end.
// Flow mirrors what the generated op wrappers' trace branch will do once
// emit_capi emits it: run the real eager op, then build the SAME op into
// the recorded pir::Program via the typed <Op>Op::Build wrapper, then
// replay through run_program_ad_func (inference) and compare to eager.
//
// Ops under test: pd_op.scale on a [2,2] tensor (scale=2 -> out==6.0).
//
// Paddle symbols (all read from $PADDLE_ROOT, HEAD 2bfeb115b0):
//   csrc/capi_gen/ops_s.cc:27                  pd_op_scale (eager entry)
//   build-nopy/.../ir/pd_op.h                 ScaleOp typed wrapper
//   build-nopy/.../ir/pd_op4.cc:4386           raw AttributeMap overload
//                                              (scale -> FullOp input)
//   paddle/fluid/pir/dialect/operator/ir/op_attribute.h:60
//                                              ScalarAttribute::get
//   paddle/fluid/eager/to_static/run_program_func.h:23-28
//                                              run_program_ad_func
//
// Build: add_executable smoke_trace_ctx (top-level CMakeLists).
// Run: needs build-nopy/paddle/{common,phi} on PATH.

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

// Generated operator dialect: ScaleOp typed wrapper.
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
// pd dialect attribute types (ScalarAttribute).
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

#include "capi_impl.h"        // pd_tensor_t
#include "ops_gen.h"          // pd_op_scale (generated eager entry)
#include "trace_ctx.h"        // trace recorder helpers + PdTraceCtx

int main() {
  // Same init contract as tools/smoke/smoke.cc (m0-report.md sec.4.2).
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  // x = full({2,2}, 3.0)
  paddle::Tensor x = paddle::experimental::full(
      {2, 2}, paddle::experimental::Scalar(3.0),
      phi::DataType::FLOAT32, phi::CPUPlace());
  pd_tensor x_handle =
      reinterpret_cast<pd_tensor>(new pd_tensor_t{x});

  // Eager reference: out = scale(x, 2.0) == 6.0 (real generated op).
  pd_tensor out_handle = pd_op_scale(x_handle, 2.0, 0.0, 1);

  // ---- trace: record the SAME op into a pir::Program -----------------
  pd_trace t = pd_trace_begin();
  if (pd_trace_input(t, x_handle, "x") != PD_OK) {
    std::cout << "SMOKE FAIL: pd_trace_input: " << pd_last_error()
              << std::endl;
    return 1;
  }
  pir::Value vx = pd_trace_value_or_feed(x_handle);
  // Mirror the future generated trace branch: typed Build with the op's
  // attribute types (scale/bias are ScalarAttribute; pd_op4.cc:4386).
  pir::AttributeMap attrs = {
      {"scale", paddle::dialect::ScalarAttribute::get(
                    pd_trace_ir(), phi::Scalar(2.0))},
      {"bias", paddle::dialect::ScalarAttribute::get(
                   pd_trace_ir(), phi::Scalar(0.0))},
      {"bias_after_scale", pir::BoolAttribute::get(pd_trace_ir(), true)},
  };
  auto op = pd_trace_builder()->Build<paddle::dialect::ScaleOp>(vx, attrs);
  pd_trace_map_output(out_handle, op->result(0));
  if (pd_trace_output(t, out_handle, "out") != PD_OK) {
    std::cout << "SMOKE FAIL: pd_trace_output: " << pd_last_error()
              << std::endl;
    return 1;
  }

  // ---- replay through run_program_ad_func (inference) -----------------
  pd_tensor outs[1] = {nullptr};
  int n = pd_trace_run(t, outs, 1);
  if (n != 1 || outs[0] == nullptr) {
    std::cout << "SMOKE FAIL: pd_trace_run n=" << n << " err="
              << pd_last_error() << std::endl;
    return 1;
  }

  // Compare traced-run output to eager reference (both should be 6.0).
  const paddle::Tensor& eager = reinterpret_cast<pd_tensor_t*>(out_handle)->value;
  const paddle::Tensor& traced = reinterpret_cast<pd_tensor_t*>(outs[0])->value;
  auto de = std::dynamic_pointer_cast<phi::DenseTensor>(eager.impl());
  auto dt = std::dynamic_pointer_cast<phi::DenseTensor>(traced.impl());
  const float* pe = de->data<float>();
  const float* pt = dt->data<float>();
  float max_err = 0.0f;
  for (int i = 0; i < 4; ++i) {
    max_err = std::max(max_err, std::abs(pt[i] - pe[i]));
    max_err = std::max(max_err, std::abs(pt[i] - 6.0f));
  }

  pd_trace_free(t);
  delete reinterpret_cast<pd_tensor_t*>(x_handle);
  delete reinterpret_cast<pd_tensor_t*>(out_handle);
  delete reinterpret_cast<pd_tensor_t*>(outs[0]);

  if (n == 1 && max_err < 1e-6f) {
    std::cout << "SMOKE PASS: trace ctx recorded scale, run == eager, "
                 "out == 6.0, max_err=" << max_err << std::endl;
    return 0;
  }
  std::cout << "SMOKE FAIL: max_err=" << max_err << std::endl;
  return 1;
}
