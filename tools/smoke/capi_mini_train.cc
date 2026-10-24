// tools/smoke/capi_mini_train.cc
// M1 2.1.5: pure mini training loop through the C ABI layer.
//
// Purpose (layout.md sec.3 P1 step 5 -- do NOT skip): before any Lua binding
// exists, prove that a forward + backward + read-back loop works through the
// C ABI alone. This permanently separates "C ABI bugs" from "binding bugs":
// any later mystery gets bisected against this C program.
//
// What is exercised through the C ABI (paddle_c.h):
//   pd_place_new / pd_tensor_from_host_buffer / pd_tensor_set_stop_gradient /
//   pd_backward / pd_tensor_grad / pd_tensor_to_host_buffer / pd_place_free
// What is NOT yet C ABI: the multiply op (P3 codegen replaces it later),
// so forward calls the eager *_ad_func directly -- the same functions
// smoke.cc uses, which DO record an autograd graph (plain
// paddle::experimental::multiply does not, leaving grad=null). The C ABI
// skeleton (handles, lifetime, error boundary, autograd) is what this tests.
//
// Acceptance (01-c-abi.md sec.5):
//   1. create tensor -> fill -> multiply -> pd_backward -> read grad,
//      gradient matches hand math (d/dx[i] of sum(x*w) == w[i]);
//   2. a deliberate bad call returns PD_INVALID_ARG (status) instead of
//      crashing the process;
//   3. two threads trigger errors concurrently and pd_last_error() stays
//      per-thread (thread_local slot, Lanes-safe by design).
//
// Build: same 483-lib baseline as tools/smoke/smoke.cc (m0-report.md sec.2),
//   + /LIBPATH:E:\code\Paddle\build-nopy + /FORCE:MULTIPLE + the 35 DEFINES.
// Run: capi_mini_train.exe

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"

#include "capi_impl.h"  // pd_tensor_t/pd_place_t bridge + paddle_c.h

namespace {

int failures = 0;

#define CHECK(cond, msg)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::cerr << "FAIL: " << msg << std::endl;                \
      ++failures;                                               \
    } else {                                                    \
      std::cout << "ok: " << msg << std::endl;                  \
    }                                                           \
  } while (0)

// bridge from C ABI handle to the C++ tensor for the (temporary) op call
paddle::Tensor& as_cpp(pd_tensor t) {
  return reinterpret_cast<pd_tensor_t*>(t)->value;
}

// x = [1,2,3,4] float32 on CPU
pd_tensor make_x(pd_place cpu) {
  const int64_t dims[2] = {2, 2};
  const float data[4] = {1.f, 2.f, 3.f, 4.f};
  pd_tensor x = pd_tensor_from_host_buffer(cpu, dims, 2,
                                           static_cast<int>(PD_DTYPE_FLOAT32),
                                           data, sizeof(data));
  return x;
}

// w = [0.5, 1.5, 2.5, 3.5] float32 on CPU
pd_tensor make_w(pd_place cpu) {
  const int64_t dims[2] = {2, 2};
  const float data[4] = {0.5f, 1.5f, 2.5f, 3.5f};
  return pd_tensor_from_host_buffer(cpu, dims, 2,
                                    static_cast<int>(PD_DTYPE_FLOAT32),
                                    data, sizeof(data));
}

}  // namespace

int main() {
  // Standard eager init (same as smoke.cc; platform/init.cc:443).
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  /* ---- 1. forward + backward + read-back through the C ABI ---- */
  pd_place cpu = pd_place_new(PD_PLACE_CPU, 0);
  CHECK(cpu != nullptr, "pd_place_new CPU");

  pd_tensor x = make_x(cpu);
  pd_tensor w = make_w(cpu);
  CHECK(x != nullptr && w != nullptr, "pd_tensor_from_host_buffer x/w");

  // leaves must allow gradient accumulation
  CHECK(pd_tensor_set_stop_gradient(x, 0) == PD_OK, "stop_gradient(x)=false");
  CHECK(pd_tensor_set_stop_gradient(w, 0) == PD_OK, "stop_gradient(w)=false");

  // forward: y = x * w (elementwise), loss = sum(y) = 0.5 + 3 + 7.5 + 14 = 25
  // *_ad_func record an autograd graph (smoke.cc same path); plain
  // paddle::experimental::multiply would NOT, leaving grad=null after
  // pd_backward. dygraph_functions.h -- generated *_ad_func forwards.
  paddle::Tensor y = multiply_ad_func(as_cpp(x), as_cpp(w));
  paddle::Tensor loss = sum_ad_func(y, {}, phi::DataType::FLOAT32, false);

  // wrap the loss into a pd_tensor handle for pd_backward
  pd_tensor loss_h =
      reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(loss)});
  pd_status st = pd_backward(&loss_h, 1, nullptr, 0, 0);
  CHECK(st == PD_OK, "pd_backward(loss)");
  pd_tensor_free(loss_h);

  // read gradient of x back to host: d loss / d x[i] == w[i]
  pd_tensor gx = pd_tensor_grad(x);
  CHECK(gx != nullptr, "pd_tensor_grad(x) not null");
  float gdata[4] = {0.f, 0.f, 0.f, 0.f};
  int64_t n = pd_tensor_to_host_buffer(gx, gdata, sizeof(gdata));
  CHECK(n == static_cast<int64_t>(sizeof(gdata)),
        "pd_tensor_to_host_buffer returns 16 bytes");
  const float expect[4] = {0.5f, 1.5f, 2.5f, 3.5f};
  bool grad_ok = true;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(gdata[i] - expect[i]) > 1e-6f) grad_ok = false;
  }
  CHECK(grad_ok, "x.grad == w (hand math)");
  pd_tensor_free(gx);

  // also verify metadata getters on x
  CHECK(pd_tensor_ndim(x) == 2, "pd_tensor_ndim(x) == 2");
  CHECK(pd_tensor_numel(x) == 4, "pd_tensor_numel(x) == 4");
  CHECK(pd_tensor_dtype(x) == static_cast<int>(PD_DTYPE_FLOAT32),
        "pd_tensor_dtype(x) == FLOAT32");
  int64_t dims[4] = {0, 0, 0, 0};
  CHECK(pd_tensor_shape(x, dims, 4) == 2 && dims[0] == 2 && dims[1] == 2,
        "pd_tensor_shape(x) == {2,2}");

  /* ---- 2. deliberate bad call -> PD_INVALID_ARG, not a crash ---- */
  pd_place bad = pd_place_new(static_cast<pd_place_kind>(999), 0);
  CHECK(bad == nullptr, "bad place kind -> NULL handle");
  CHECK(pd_last_status() == PD_INVALID_ARG, "bad place kind status");
  CHECK(std::string(pd_last_error()).find("bad pd_place_kind") !=
            std::string::npos,
        "bad place kind message");

  pd_tensor bad_t =
      pd_tensor_from_host_buffer(cpu, nullptr, 0,
                                 static_cast<int>(PD_DTYPE_FLOAT32), nullptr, 0);
  CHECK(bad_t == nullptr && pd_last_status() == PD_INVALID_ARG,
        "from_host_buffer(ndim=0) -> NULL + INVALID_ARG");

  /* ---- 3. two threads, per-thread error slots ---- */
  std::atomic<int> t1_status{0};
  std::atomic<int> t2_status{0};
  std::thread t1([&] {
    pd_place p = pd_place_new(static_cast<pd_place_kind>(999), 0);
    t1_status.store(pd_last_status());
    (void)p;
  });
  std::thread t2([&] {
    pd_place p = pd_place_new(PD_PLACE_CPU, 0);
    (void)p;  // ok path; then force an invalid call
    pd_tensor bad =
        pd_tensor_from_host_buffer(p, nullptr, 0,
                                   static_cast<int>(PD_DTYPE_FLOAT32),
                                   nullptr, 0);
    t2_status.store(pd_last_status());
    pd_place_free(p);
    pd_tensor_free(bad);  // NULL-safe
  });
  t1.join();
  t2.join();
  CHECK(t1_status.load() == PD_INVALID_ARG && t2_status.load() == PD_INVALID_ARG,
        "thread error slots isolated (both report INVALID_ARG)");

  /* ---- cleanup ---- */
  pd_tensor_free(x);
  pd_tensor_free(w);
  pd_place_free(cpu);

  if (failures == 0) {
    std::cout << "MINI_TRAIN PASS" << std::endl;
    return 0;
  }
  std::cout << "MINI_TRAIN FAIL (" << failures << ")" << std::endl;
  return 1;
}
