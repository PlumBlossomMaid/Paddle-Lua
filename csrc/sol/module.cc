// module.cc -- luaopen_paddle_core: module entry + version/device surface
// Stage: P2 / M1 (WORKPLAN 2.2.1). Second file of csrc/sol (layout.md
// sec.3: "module.cc  luaopen_paddle_core").
//
// Design: docs/migration/plan/modules/02-binding.md.
//   - one source tree, five binaries (5.1/5.2/5.3/5.4/LuaJIT); no
//     #if LUA_VERSION anywhere (sec.4 acceptance, compat.hpp isolates
//     differences if they ever appear);
//   - the Lua module is named paddle_core; the public Lua API lives in
//     lua/paddle/ (a different layer) and consumes this C++ surface;
//   - every pd_* call goes through pd::check() (sol_common.hpp) so a C
//     ABI failure becomes a Lua error, never a silent nil.
//
// The tensor usertype arrives in tensor_binding.cc (2.2.2). This TU only
// proves luaopen works and the error path is wired (2.2.1 acceptance).

#include "sol_common.hpp"

namespace {

const char* kModuleName = "paddle_core";
const char* kModuleVersion = "0.1.0";

/* Version + shout test for the 2.2.1 milestone: require-able, and a bad
 * call on the C ABI side turns into a readable Lua error. */
int l_version(lua_State* L) {
  lua_pushstring(L, kModuleVersion);
  return 1;
}

/* Device count probe: exercises pd_device_count + pd::check() so the
 * error path is proven from Lua on day one (bad kind -> lua_error). */
int l_device_count(lua_State* L) {
  int kind = static_cast<int>(luaL_checkinteger(L, 1));
  int n = pd_device_count(kind);   // 01-c-abi.md group 5
  pd::check();                     // non-PD_OK status -> Lua error
  lua_pushinteger(L, n);
  return 1;
}

/* Place-kind name string, pure table lookup, never errors. */
int l_place_kind_name(lua_State* L) {
  int kind = static_cast<int>(luaL_checkinteger(L, 1));
  lua_pushstring(L, pd_place_kind_name(kind));
  return 1;
}

const luaL_Reg kModuleFns[] = {
    {"version", l_version},
    {"device_count", l_device_count},
    {"place_kind_name", l_place_kind_name},
    {nullptr, nullptr},
};

}  // namespace

extern "C" int luaopen_paddle_core(lua_State* L) {
  // Plain Lua C API: build the module table by hand (tiny), no reason to
  // drag sol::state in for three functions. Later TUs bring the usertypes.
  // NB: NOT luaL_setfuncs -- that is Lua 5.2+; 5.1 has no such API and the
  // milestone requires all five Lua versions to require() this module.
  lua_newtable(L);
  for (const luaL_Reg* fn = kModuleFns; fn->name != nullptr; ++fn) {
    lua_pushcfunction(L, fn->func);
    lua_setfield(L, -2, fn->name);
  }
  return 1;
}