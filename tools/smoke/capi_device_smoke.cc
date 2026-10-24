// tools/smoke/capi_device_smoke.cc
// M1 2.1.6: device / jit C ABI acceptance.
//
// Exercises through paddle_c.h only:
//   pd_device_count / pd_device_current / pd_device_set_current /
//   pd_device_sync / pd_device_allocated_bytes / pd_place_kind_name /
//   pd_jit_layer_load (bad-path: NULL / nonexistent path) / free
//
// Acceptance (01-c-abi.md sec.5 + 2.1.6):
//   - CPU device count == 1, current == 0, set_current(0) OK
//   - sync(CPU,0) returns PD_OK
//   - allocated_bytes is >= 0 (any value: env-dependent)
//   - kind_name("CPU"/"GPU") round-trips
//   - jit load with NULL path -> NULL + PD_INVALID_ARG (no crash)
//   - two threads call device queries concurrently (thread-safety sanity)
//
// Build/Run: same recipe as capi_mini_train.cc (link.exe direct @rsp with
// the 831-lib baseline, /LIBPATH, /FORCE:MULTIPLE, upstream /MT+DEFINES).

#include <atomic>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"

#include "capi_impl.h"  // pd_tensor_t/pd_place_t bridge + paddle_c.h

// ---------------------------------------------------------------------------
// Upstream link gap shim (WITH_PYTHON=OFF), same as smoke_jit_load.cc:
// libfluid_jit's PirFunctionInfo ctor calls GetFeedTargetNames /
// GetFetchTargetNames (function_schema.cc:112/115), whose implementations
// live in paddle/fluid/pybind/pir_utils.cc:20/36 -- not built without
// Python. Pure PIR, so we provide equivalents for the smoke EXE. See
// process/m0-report.md T-M0-17.
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"
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

}  // namespace

int main() {
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  /* ---- 1. device queries (CPU) ---- */
  CHECK(pd_device_count(PD_PLACE_CPU) == 1, "pd_device_count(CPU) == 1");
  CHECK(pd_device_current(PD_PLACE_CPU) == 0, "pd_device_current(CPU) == 0");
  CHECK(pd_device_set_current(PD_PLACE_CPU, 0) == PD_OK,
        "pd_device_set_current(CPU, 0)");
  CHECK(pd_device_sync(PD_PLACE_CPU, 0) == PD_OK, "pd_device_sync(CPU, 0)");
  CHECK(pd_device_allocated_bytes(PD_PLACE_CPU) >= 0,
        "pd_device_allocated_bytes(CPU) >= 0");
  CHECK(std::strcmp(pd_place_kind_name(PD_PLACE_CPU), "CPU") == 0,
        "pd_place_kind_name(CPU) == \"CPU\"");
  CHECK(std::strcmp(pd_place_kind_name(PD_PLACE_CUDA), "GPU") == 0,
        "pd_place_kind_name(CUDA) == \"GPU\"");

  /* ---- 2. bad device call -> INVALID_ARG, no crash ---- */
  CHECK(pd_device_current(static_cast<int>(PD_PLACE_CUSTOM)) == -1 &&
            pd_last_status() == PD_INVALID_ARG,
        "pd_device_current(bad kind) -> -1 + INVALID_ARG");
  CHECK(pd_device_set_current(static_cast<int>(PD_PLACE_CUSTOM), 0) ==
            PD_INVALID_ARG,
        "pd_device_set_current(bad kind) -> INVALID_ARG");

  /* ---- 3. jit bad-path ---- */
  pd_layer l0 = pd_jit_layer_load(nullptr, nullptr);
  CHECK(l0 == nullptr && pd_last_status() == PD_INVALID_ARG,
        "pd_jit_layer_load(NULL) -> NULL + INVALID_ARG");
  pd_place cpu = pd_place_new(PD_PLACE_CPU, 0);
  CHECK(cpu != nullptr, "pd_place_new CPU");
  // Upstream Load (serializer.cc:176) does NOT throw on a missing path:
  // the Deserializer loop simply runs zero times and returns a default
  // (empty) Layer. So a bad path must not crash, and the returned handle
  // must be freeable -- NULL is not guaranteed upstream.
  pd_layer l1 = pd_jit_layer_load("no_such_export_dir", cpu);
  CHECK(l1 != nullptr, "pd_jit_layer_load(bad path) -> handle (no crash)");
  pd_jit_layer_free(l1);
  pd_place_free(cpu);

  /* ---- 4. two threads querying devices concurrently ---- */
  std::atomic<int> t1{0};
  std::atomic<int> t2{0};
  std::thread a([&] {
    for (int i = 0; i < 100; ++i) {
      if (pd_device_count(PD_PLACE_CPU) == 1) t1.fetch_add(1);
    }
  });
  std::thread b([&] {
    for (int i = 0; i < 100; ++i) {
      if (pd_device_sync(PD_PLACE_CPU, 0) == PD_OK) t2.fetch_add(1);
    }
  });
  a.join();
  b.join();
  CHECK(t1.load() == 100 && t2.load() == 100,
        "thread device queries all succeeded");

  if (failures == 0) {
    std::cout << "DEVICE_SMOKE PASS" << std::endl;
    return 0;
  }
  std::cout << "DEVICE_SMOKE FAIL (" << failures << ")" << std::endl;
  return 1;
}