// error.cc -- C ABI status code -> Lua error conversion
// Stage: P2 / M1 (WORKPLAN 2.2.1). First file of csrc/sol (layout.md sec.3).
//
// Design: docs/migration/plan/modules/02-binding.md sec.3.3.
// The C ABI reports failures as pd_status + thread-local message. Lua must
// see a thrown error() instead of a nil return. Two layers:
//   - pd::check() (sol_common.hpp): throw sol::error after a pd_* call
//     whose status is not PD_OK; sol2 turns it into a Lua error at the
//     call boundary.
//   - raise_last_error() (here): longjmp a Lua error directly, for call
//     sites that must raise without C++ exceptions in flight.
//
// Every exported Lua function in this layer goes through one of the two;
// neither lets a pd_status silently become nil.

#include "sol_common.hpp"

namespace pd {

int raise_last_error(lua_State* L) {
  const char* msg = pd_last_error();
  return luaL_error(L, "paddle: %s", msg ? msg : "unknown error");
}

}  // namespace pd