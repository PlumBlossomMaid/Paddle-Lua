// tools/smoke/sol_smoke.cc
// M1 2.2.1: luaopen milestone -- require "paddle_core" works and a C ABI
// failure becomes a Lua error.
//
// The local Lua 5.1.5 install is x86 while paddle_core.dll is x64, so this
// smoke embeds Lua 5.1 from vcpkg (x64 lua51.lib/dll) and loads the module
// through require -- the exact path Lua users will take.
//
// Acceptance (02-binding.md sec.5 + layout.md P2 step 3):
//   - require("paddle_core") returns a table (luaopen ran)
//   - core.version() == "0.1.0"
//   - core.device_count(0) == 1            (PD_PLACE_CPU)
//   - core.device_count(999) raises a Lua error (bad kind -> INVALID_ARG
//     -> pd::check -> lua_error), not a crash
//   - core.place_kind_name(1) == "GPU"
//
// Build: same /MT + /std:c++17 recipe, link vcpkg lua51.lib + paddle_core
// deps (@capi_libs.rsp /LIBPATH /FORCE:MULTIPLE). Run with lua51.dll and
// Paddle runtime DLLs on PATH.

#include <cstdio>
#include <cstring>
#include <string>

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

namespace {

int failures = 0;

#define CHECK(cond, msg)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::fprintf(stderr, "FAIL: %s\n", msg);                  \
      ++failures;                                               \
    } else {                                                    \
      std::printf("ok: %s\n", msg);                             \
    }                                                           \
  } while (0)

// Run a Lua snippet; returns true if it completed without error.
bool run_ok(lua_State* L, const char* code) {
  if (luaL_loadstring(L, code) != 0) {
    std::fprintf(stderr, "load error: %s\n", lua_tostring(L, -1));
    lua_pop(L, 1);
    return false;
  }
  if (lua_pcall(L, 0, 0, 0) != 0) {
    std::fprintf(stderr, "run error: %s\n", lua_tostring(L, -1));
    lua_pop(L, 1);
    return false;
  }
  return true;
}

}  // namespace

int main() {
  lua_State* L = luaL_newstate();
  if (L == nullptr) {
    std::fprintf(stderr, "FAIL: luaL_newstate\n");
    return 1;
  }
  luaL_openlibs(L);

  // Point require at our .dll. package.cpath default on Windows is
  // ".\?.dll;..." -- but be explicit so we do not depend on CWD.
  if (!run_ok(L,
              "package.cpath = [[.qwen\\tmp\\?.dll;]] .. package.cpath\n"
              "local ok, core = pcall(require, 'paddle_core')\n"
              "if not ok then error('require failed: ' .. tostring(core)) end\n"
              "paddle_core = core")) {
    lua_close(L);
    return 1;
  }

  // Eager runtime init through the C ABI (pd_init_once), NOT by statically
  // linking Paddle into this EXE: paddle_core.dll already links the static
  // libs, and a second copy would re-register every gflags flag
  // (flags_native.cc:309) -- the exact failure this smoke used to hit.
  CHECK(run_ok(L,
               "assert(paddle_core.init() == true, 'paddle_core.init()')"),
        "paddle_core.init() idempotent eager init");

  CHECK(run_ok(L,
               "assert(type(paddle_core) == 'table', 'module is a table')\n"
               "assert(paddle_core.version() == '0.1.0', 'version')\n"
               "assert(paddle_core.device_count(0) == 1, 'cpu count')\n"
               "assert(paddle_core.place_kind_name(1) == 'GPU', 'gpu name')"),
        "require + version + device_count + place_kind_name");

  // Bad kind: C ABI returns PD_INVALID_ARG -> pd::check -> Lua error.
  CHECK(run_ok(L,
               "local ok, err = pcall(paddle_core.device_count, 999)\n"
               "assert(not ok, 'bad kind must raise')\n"
               "assert(type(err) == 'string' and #err > 0, 'err is string')"),
        "device_count(999) raises a Lua error (not a crash)");

  // 2.2.2: LuaTensor usertype -- factory, shape/dtype/ndim/numel.
  CHECK(run_ok(L,
               "local t = paddle_core.tensor{2, 3}\n"
               "print('DBG ndim=', t:ndim(), 'numel=', t:numel())\n"
               "print('DBG shape=', t:shape())\n"
               "assert(paddle_core.tensor ~= nil, 'tensor factory exists')\n"
               "assert(t:ndim() == 2, 'ndim == 2')\n"
               "assert(t:numel() == 6, 'numel == 6')\n"
               "assert(t:shape()[1] == 2 and t:shape()[2] == 3, 'shape {2,3}')\n"
               "assert(t:is_contiguous() == 1, 'contiguous')\n"
               "assert(t:dtype() == 10, 'dtype == FLOAT32 (pd_dtype 10)')"),
        "tensor{2,3}: ndim/numel/shape/is_contiguous/dtype");

  // Stale-error regression (PD_GUARD_BEGIN now clears the slot): a success
  // after a failure must NOT inherit the previous error.
  CHECK(run_ok(L,
               "pcall(paddle_core.device_count, 999)  -- leaves INVALID_ARG\n"
               "local t = paddle_core.tensor{1}\n"
               "assert(t:numel() == 1, 'stale error must be cleared')"),
        "stale error slot cleared by successful call");

  lua_close(L);

  if (failures == 0) {
    std::printf("SOL_SMOKE PASS\n");
    return 0;
  }
  std::printf("SOL_SMOKE FAIL (%d)\n", failures);
  return 1;
}