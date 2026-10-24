// sol_common.hpp -- shared internal header for csrc/sol (NOT part of the
// public ABI). Each .cc in this directory includes it; version differences
// between Lua 5.1/5.2/5.3/5.4/LuaJIT are handled by sol2 (02-binding.md
// sec.3.1) -- our code must never contain #if LUA_VERSION.
//
// Design: docs/migration/plan/modules/02-binding.md sec.3.3.
// The C ABI returns pd_status; Lua must see error() instead. pd::check()
// throws a sol::error carrying the C ABI's thread-local message, which
// sol2 converts into a lua_error at the call boundary.

#ifndef PADDLE_LUA_SOL_COMMON_H
#define PADDLE_LUA_SOL_COMMON_H

extern "C" {
#include "paddle_c.h"
}

#include <sol/sol.hpp>

#include <stdexcept>
#include <string>

namespace pd {

/* Throw sol::error with the C ABI's last error message. [[noreturn]]. */
inline void throw_status() {
  const char* msg = pd_last_error();
  throw sol::error(std::string("paddle: ") + (msg ? msg : "unknown error"));
}

/* Call after any pd_* call that can fail (02-binding.md sec.3.3). */
inline void check() {
  if (pd_last_status() != PD_OK) {
    throw_status();
  }
}

}  // namespace pd

#endif  // PADDLE_LUA_SOL_COMMON_H