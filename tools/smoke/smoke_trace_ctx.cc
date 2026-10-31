// tools/smoke/smoke_trace_ctx.cc
// P15-A step 2: validate the C ABI trace context (trace.cc) + the
// GENERATED trace branches (emit_capi.lua) end-to-end, eager-record.
// The generated pd_op_* wrappers execute eagerly AND record themselves
// into the active trace via the typed <Op>Op::Build overload selected by
// tools/gen/pd_op_builds.lua. This smoke chains matmul -> add -> relu on
// real tensors, marks the chain output, replays the recorded
// pir::Program through run_program_ad_func (inference, empty backward),
// and requires the traced output to be bitwise-identical to eager.
//
// Deterministic values: x = ones[2,3], w = ones[3,4], b = full(-1)[4]
//   -> matmul = 3s[2,4] -> add = 2s[2,4] -> relu = 2s[2,4].
//
// Paddle symbols (all read from $PADDLE_ROOT, HEAD 2bfeb115b0):
//   csrc/capi_gen/ops_gen.h:319/411   pd_op_matmul / pd_op_relu
//   build-nopy/.../ir/pd_op.h         MatmulOp:43601 / AddOp:41817 /
//                                     ReluOp:16353 typed wrappers
//   paddle/fluid/eager/to_static/run_program_func.h:23-28
//                                     run_program_ad_func
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

#include "capi_impl.h"        // pd_tensor_t
#include "ops_gen.h"          // generated eager entries (with trace branch)
#include "trace_ctx.h"        // trace recorder helpers + PdTraceCtx

static pd_tensor full_handle(std::initializer_list<int64_t> shape,
                             double v) {
  paddle::Tensor t = paddle::experimental::full(
      phi::IntArray(std::vector<int64_t>(shape)),
      paddle::experimental::Scalar(v), phi::DataType::FLOAT32,
      phi::CPUPlace());
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{t});
}

static float max_abs_diff(pd_tensor a, pd_tensor b, int64_t n) {
  const paddle::Tensor& ta = reinterpret_cast<pd_tensor_t*>(a)->value;
  const paddle::Tensor& tb = reinterpret_cast<pd_tensor_t*>(b)->value;
  auto da = std::dynamic_pointer_cast<phi::DenseTensor>(ta.impl());
  auto db = std::dynamic_pointer_cast<phi::DenseTensor>(tb.impl());
  const float* pa = da->data<float>();
  const float* pb = db->data<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}

int main() {
  // Same init contract as tools/smoke/smoke.cc (m0-report.md sec.4.2).
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  pd_tensor x = full_handle({2, 3}, 1.0);
  pd_tensor w = full_handle({3, 4}, 1.0);
  pd_tensor b = full_handle({4}, -1.0);

  // ---- eager reference (no trace active: branches stay off) ----------
  pd_tensor h_e = pd_op_matmul(x, w, 0, 0);
  pd_tensor z_e = pd_op_add(h_e, b);
  pd_tensor out_e = pd_op_relu(z_e);

  // ---- trace: same calls; generated wrappers record themselves -------
  pd_trace t = pd_trace_begin();
  if (pd_trace_input(t, x, "x") != PD_OK) {
    std::cout << "SMOKE FAIL: pd_trace_input: " << pd_last_error()
              << std::endl;
    return 1;
  }
  pd_tensor h_t = pd_op_matmul(x, w, 0, 0);  // w -> auto feed_0
  pd_tensor z_t = pd_op_add(h_t, b);         // b -> auto feed_1
  pd_tensor out_t = pd_op_relu(z_t);
  if (pd_trace_output(t, out_t, "out") != PD_OK) {
    std::cout << "SMOKE FAIL: pd_trace_output: " << pd_last_error()
              << std::endl;
    return 1;
  }

  // ---- replay through run_program_ad_func (inference) ----------------
  pd_tensor outs[1] = {nullptr};
  int n = pd_trace_run(t, outs, 1);
  if (n != 1 || outs[0] == nullptr) {
    std::cout << "SMOKE FAIL: pd_trace_run n=" << n << " err="
              << pd_last_error() << std::endl;
    return 1;
  }

  // Traced-run == eager-chain (bitwise) == traced-eager == 2.0.
  float err_run = max_abs_diff(outs[0], out_e, 8);
  float err_rec = max_abs_diff(out_t, out_e, 8);
  const paddle::Tensor& tv = reinterpret_cast<pd_tensor_t*>(outs[0])->value;
  auto dt = std::dynamic_pointer_cast<phi::DenseTensor>(tv.impl());
  const float* pt = dt->data<float>();
  float err_val = 0.0f;
  for (int i = 0; i < 8; ++i) err_val = std::max(err_val, std::abs(pt[i] - 2.0f));

  pd_trace_free(t);
  pd_tensor xs[7] = {x, w, b, h_e, z_e, out_e, out_t};
  for (auto h : xs) delete reinterpret_cast<pd_tensor_t*>(h);
  delete reinterpret_cast<pd_tensor_t*>(h_t);
  delete reinterpret_cast<pd_tensor_t*>(z_t);
  delete reinterpret_cast<pd_tensor_t*>(outs[0]);

  if (err_run < 1e-6f && err_rec < 1e-6f && err_val < 1e-6f) {
    std::cout << "SMOKE PASS: generated trace branches recorded "
                 "matmul->add->relu, run == eager bitwise, out == 2.0 "
                 "(err_run=" << err_run << " err_rec=" << err_rec
              << " err_val=" << err_val << ")" << std::endl;
    return 0;
  }
  std::cout << "SMOKE FAIL: err_run=" << err_run << " err_rec=" << err_rec
            << " err_val=" << err_val << std::endl;
  return 1;
}
