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

/* Device memory allocation stats (device.cc pd_device_allocated_bytes).
 * Kind defaults to CPU. Current allocated bytes or -1 when that stat is
 * unavailable on this build. Backs paddle.scope's entry/exit water level
 * (api/scope.md sec.2.1, 14-gc.md sec.3 / sec.5 acceptance). */
int l_memory_allocated(lua_State* L) {
  int kind = static_cast<int>(luaL_optinteger(L, 1, PD_PLACE_CPU));
  int64_t bytes = pd_device_allocated_bytes(kind);
  pd::check();                     // bad kind -> Lua error
  lua_pushinteger(L, static_cast<lua_Integer>(bytes));
  return 1;
}

/* Eager runtime init (idempotent, device.cc pd_init_once). MUST be called
 * before any tensor{} / op touches the phi kernel factory. Only the C ABI
 * layer links Paddle (02-binding.md sec.2); the Lua embedder never does. */
int l_init(lua_State* L) {
  pd_status st = pd_init_once();
  pd::check();                     // non-PD_OK -> Lua error
  lua_pushboolean(L, st == PD_OK);
  return 1;
}

/* Place-kind name string, pure table lookup, never errors. */
int l_place_kind_name(lua_State* L) {
  int kind = static_cast<int>(luaL_checkinteger(L, 1));
  lua_pushstring(L, pd_place_kind_name(kind));
  return 1;
}

/* ---- 3.3 GC heap tracking (14-gc.md sec.3.1, layers 1-3) ----------
 * pd_gc_account / pd_gc_set_enabled / pd_gc_enabled / pd_gc_set_softmax /
 * pd_gc_set_handler / pd_gc_tracked_bytes live in gc.cc (a faithful port
 * of Torch7 THGeneral.c:134-238). Default ON per D6; the switch here is
 * what paddle.set_heap_tracking(false) reaches (14-gc.md 坑④). */

/* Enable/disable heap tracking (D6: defaults ON). Disabling resets the
 * global ledger (gc.cc pd_gc_set_enabled). */
int l_set_heap_tracking(lua_State* L) {
  int on = lua_toboolean(L, 1) ? 1 : 0;
  pd_gc_set_enabled(on);
  return 0;
}

/* Current tracking state (0/1). */
int l_heap_tracking(lua_State* L) {
  lua_pushboolean(L, pd_gc_enabled() != 0);
  return 1;
}

/* Sum of live tensor bytes across all lanes (diagnostic; smoke asserts
 * on it). */
int l_tracked_bytes(lua_State* L) {
  lua_pushinteger(L, static_cast<lua_Integer>(pd_gc_tracked_bytes()));
  return 1;
}

/* Adjust the thread-local softmax cap (default 300MB). A tuning knob for
 * small machines (earlier = cheaper collects) and the smoke sets 1MB to
 * drive the softmax path deterministically. */
int l_set_heap_softmax(lua_State* L) {
  int64_t bytes = static_cast<int64_t>(luaL_checkinteger(L, 1));
  pd_gc_set_softmax(bytes);
  return 0;
}

const luaL_Reg kModuleFns[] = {
    {"version", l_version},
    {"init", l_init},
    {"device_count", l_device_count},
    {"memory_allocated", l_memory_allocated},
    {"place_kind_name", l_place_kind_name},
    {"set_heap_tracking", l_set_heap_tracking},
    {"heap_tracking", l_heap_tracking},
    {"tracked_bytes", l_tracked_bytes},
    {"set_heap_softmax", l_set_heap_softmax},
    {nullptr, nullptr},
};

/* 3.3: bridge the C-side softmax trigger to a full Lua GC on THIS
 * thread's state. LUA_GCCOLLECT (2) exists on every targeted Lua version;
 * LUA_GCSTEP is 5.2+ only, and 14-gc.md's LUA_GCSTEP sketch is adjusted
 * here for the 5.1 surface (C5). Each lane registers its OWN lua_State at
 * luaopen time (the handler slot in gc.cc is thread-local), so the
 * collect always runs on the lane that crossed its softmax -- never
 * another lane's state (14-gc.md sec.4 坑②). extern "C" so the function
 * pointer has C language linkage, matching the C ABI's parameter type
 * (language linkage is part of the function type in C++). */
extern "C" static void gc_softmax_bridge(void* data) {
  lua_State* L = static_cast<lua_State*>(data);
  lua_gc(L, LUA_GCCOLLECT, 0);
}

}  // namespace

// The DLL entry point MUST be dllexport: MSVC does not export symbols from
// a DLL unless marked __declspec(dllexport) (or listed in a .def). A plain
// `extern "C"` gives C linkage but NO export -- require() then loads the
// module and fails GetProcAddress("luaopen_paddle_core") with
// ERROR_PROC_NOT_FOUND/127 (caught 2026-10-25; the 2.2.3 sol smoke never
// hit this because it embedded Lua statically instead of loading a DLL).
// Non-Windows linkers export all symbols by default, so the guard is
// Windows-only.
#if defined(_WIN32)
#define PD_LUAOPEN_EXPORT __declspec(dllexport)
#else
#define PD_LUAOPEN_EXPORT
#endif

extern "C" PD_LUAOPEN_EXPORT int luaopen_paddle_core(lua_State* L) {
  // Plain Lua C API: build the module table by hand (tiny), no reason to
  // drag sol::state in for three functions. Later TUs bring the usertypes.
  // NB: NOT luaL_setfuncs -- that is Lua 5.2+; 5.1 has no such API and the
  // milestone requires all five Lua versions to require() this module.
  lua_newtable(L);
  for (const luaL_Reg* fn = kModuleFns; fn->name != nullptr; ++fn) {
    lua_pushcfunction(L, fn->func);
    lua_setfield(L, -2, fn->name);
  }
  // tensor_binding.cc: LuaTensor usertype + tensor{} factory (2.2.2).
  // NB: the module table is NOT at index 1 -- require() pushes the module
  // name onto the stack before calling luaopen, so index 1 is the string
  // "paddle_core". The table we just built sits on top: pass -1 (relative
  // to the current top), not an absolute index.
  pd::register_tensor(L, -1);
  // enums_place.cc: PD_DTYPE_* / PD_PLACE_* constants + pd_place_new
  // factory + LuaPlace usertype (2.2.5). MUST precede register_ops: the
  // generated op lambdas take `const LuaPlace&` and the _ops wrappers
  // read the constants for attr defaults.
  pd::register_enums_place(L, -1);
  // gen_ops_all.cc: every pd_op_* C ABI function as a sol2 lambda
  // (2.3.4 closing item 3 -- generated, AGENTS sec.8.3).
  pd::register_ops(L, -1);
  // jit_binding.cc: LuaLayer usertype + core.jit_load (2.8). Runs last:
  // load needs the op registry (InterpreterCore) already in place.
  pd::register_jit(L, -1);
  // 3.3 GC: bind THIS thread's lua_State to the C-side softmax trigger.
  // Each lane registers its own state here (handler slot is thread-local
  // in gc.cc), so a collect always runs on the lane that crossed its own
  // softmax -- never another lane's state (14-gc.md sec.4 坑②). The
  // bridge is a plain C function; e.g. no C++ exceptions can cross.
  pd_gc_set_handler(&gc_softmax_bridge, L);
  return 1;
}