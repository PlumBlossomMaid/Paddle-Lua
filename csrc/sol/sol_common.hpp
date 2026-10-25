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

/* One Lua-visible Tensor (02-binding.md sec.3.2). Ownership is
 * transfer-on-create, release on GC. The struct lives HERE, not in
 * tensor_binding.cc, because the generated gen_ops_*.cc must construct
 * and unwrap it (AGENTS sec.8.3) -- both TUs include sol_common.hpp.
 *
 * Copy shares storage via pd_tensor_retain (new shell, same data);
 * required by the __lanesclone slot and by sol2 value returns. */
struct LuaTensor {
  pd_tensor h;                        // opaque C ABI handle

  explicit LuaTensor(pd_tensor handle) : h(handle) {}
  ~LuaTensor() { pd_tensor_free(h); }

  LuaTensor(const LuaTensor& other) : h(pd_tensor_retain(other.h)) {}
  LuaTensor(LuaTensor&& other) noexcept : h(other.h) { other.h = nullptr; }
  LuaTensor& operator=(const LuaTensor&) = delete;
};

/* One Lua-visible Place (dtype-device.md sec.2.2: place is a value, but
 * at the paddle_core layer it is a RAII shell over the opaque C ABI
 * handle -- pd_place_new transfers ownership, ~LuaPlace frees it.
 * Not copyable: pd_place has no retain (paddle_c.h ownership rule), so
 * an op lambda borrows `const LuaPlace&` and the factory returns by
 * move. pd_place_free(NULL) is `delete nullptr` (tensor.cc:173) -- a
 * moved-from shell frees safely. */
struct LuaPlace {
  pd_place h;

  explicit LuaPlace(pd_place handle) : h(handle) {}
  ~LuaPlace() { pd_place_free(h); }

  LuaPlace(const LuaPlace&) = delete;
  LuaPlace& operator=(const LuaPlace&) = delete;
  LuaPlace(LuaPlace&& other) noexcept : h(other.h) { other.h = nullptr; }
};

/* Register the LuaTensor usertype + tensor factory into the module table
 * that sits at stack index `module_index` (defined in tensor_binding.cc,
 * 2.2.2). Called at the end of luaopen_paddle_core. */
void register_tensor(lua_State* L, int module_index);

/* Register core.slice_spec (lua spec -> 0-based string, D14) into a sol
 * table (slice.cc, 2.2.4). Pure string transform for P5 consumption. */
void register_slice(sol::table& mod);

/* Register the PD_DTYPE_* / PD_PLACE_* integer constants, the LuaPlace
 * usertype and the pd_place_new factory (enums_place.cc, 2.2.5). The
 * generated _ops wrappers reference these constants for attr defaults
 * (emit_lua DATA_TYPE_DEFAULTS), so this MUST run before any op call. */
void register_enums_place(lua_State* L, int module_index);

/* Register every pd_op_* C ABI function as a sol2 lambda on the module
 * table (generated: gen_ops_*.cc / gen_ops_all.cc, 2.3.4 closing item
 * 3). Runs last in luaopen_paddle_core, after the usertypes it needs. */
void register_ops(lua_State* L, int module_index);

}  // namespace pd

#endif  // PADDLE_LUA_SOL_COMMON_H