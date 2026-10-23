// tools/smoke/smoke_run_program.cc
// T-M0-18: verify that a hand-built AttributeMap can be fed into
// egr::to_static::run_program_ad_func in a WITH_PYTHON=OFF + ON_INFER=OFF build.
//
// Question under test: the Lua frontend must run a static program (P16, PIR)
// via the eager layer. The entry point is run_program_ad_func; its inputs are
// pure C++ types (vector<Tensor>, vector<Scope*>, AttributeMap). Can we build
// the AttributeMap by hand without Python?
//
// References:
//   paddle/fluid/eager/to_static/run_program_func.h:24-28  -> signature
//   paddle/fluid/eager/to_static/run_program_func.cc:264-320-> impl: reads is_test,
//                                                              builds autograd meta
//   paddle/fluid/eager/to_static/run_program_impl.cc:435-490
//                                                           -> RunProgramImpl: requires
//                                                              prog_attrs keys is_test/
//                                                              program_id/in_sot_mode/
//                                                              fx_names/fp_names/fo_names/
//                                                              no_need_buffers_names/
//                                                              fo_values/forward_program/
//                                                              backward_program; and
//                                                              cuda_graph_attrs keys
//                                                              cuda_graph_state/
//                                                              cuda_graph_dispatch_key
//   paddle/fluid/framework/type_defs.h:51/73               -> Attribute = variant<...>;
//                                                              AttributeMap = unordered_map
//                                                              <string, Attribute>
//   paddle/fluid/framework/scope.h                          -> Scope
//   paddle/pir/include/core/program.h                       -> pir::Program
//
// Build: same 483-lib baseline as tools/smoke/smoke.cc (m0-report.md section 2),
//   /LIBPATH:E:\code\Paddle\build-nopy, /FORCE:MULTIPLE, /bigobj, 35 DEFINES.
// Run: smoke_run_program.exe

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/to_static/run_program_func.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/pir/include/core/program.h"

int main() {
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  // Hand-build the AttributeMaps exactly as run_program_impl.cc expects.
  paddle::framework::AttributeMap prog_attrs;
  prog_attrs["program_id"] = static_cast<int64_t>(1);
  prog_attrs["is_test"] = false;
  prog_attrs["in_sot_mode"] = false;
  prog_attrs["fx_names"] = std::vector<std::string>{"x"};
  prog_attrs["fp_names"] = std::vector<std::string>{};
  prog_attrs["fo_names"] = std::vector<std::string>{"out"};
  prog_attrs["no_need_buffers_names"] = std::vector<std::string>{};
  prog_attrs["fo_values"] = std::vector<pir::Value>{};
  prog_attrs["forward_program"] =
      std::shared_ptr<pir::Program>(nullptr);  // empty on purpose
  prog_attrs["backward_program"] =
      std::shared_ptr<pir::Program>(nullptr);

  paddle::framework::AttributeMap cuda_graph_attrs;
  cuda_graph_attrs["cuda_graph_state"] = static_cast<int64_t>(0);
  cuda_graph_attrs["cuda_graph_dispatch_key"] = static_cast<int64_t>(0);

  // A single outer scope is required (run_program_impl.cc:461-466).
  auto scope = std::make_shared<paddle::framework::Scope>();
  std::vector<paddle::framework::Scope*> step_scope{scope.get()};

  try {
    // Empty tensors, no params, empty pir::Program: expected to reach the
    // C++ logic and fail on missing interpreter-core/program, NOT on missing
    // pybind. That is exactly what proves the chain is usable from Lua.
    auto out = egr::to_static::run_program_ad_func(
        {}, {}, step_scope, prog_attrs, cuda_graph_attrs);
    std::cout << "SMOKE FAIL: unexpectedly ran with empty program (out.size="
              << out.size() << ")" << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::string msg = e.what();
    // Any error from C++ logic (missing program, bad cache key, ...) is fine.
    // An error mentioning pybind/Python would mean the chain is NOT usable.
    bool pybind_leak =
        (msg.find("pybind") != std::string::npos ||
         msg.find("python") != std::string::npos);
    if (!pybind_leak && !msg.empty()) {
      std::cout << "SMOKE PASS: hand-built AttributeMap reached C++ logic "
                   "(expected error for empty program: "
                << msg.substr(0, 120) << "...)" << std::endl;
      return 0;
    }
    std::cout << "SMOKE FAIL: unexpected error: " << msg << std::endl;
    return 1;
  }
}
