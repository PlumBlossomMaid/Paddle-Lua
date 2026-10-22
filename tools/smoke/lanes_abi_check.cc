// tools/smoke/lanes_abi_check.cc
// M0 #5 acceptance: verify Lanes v3.17.2 (pure C) links into a C++ translation
// unit with the same compiler/CRT as the binding layer and the Lua 5.1 we ship.
//
// Key fact verified by source: lanes.h declares C functions (luaopen_lanes_core
// / luaopen_lanes_embedded) WITHOUT an extern "C" guard, so a C++ consumer must
// wrap the include itself.  deep.h exports luaG_newdeepuserdata / luaG_todeep
// the same way.  v3.17.2 is plain C src (10 .c, 0 .cpp, PROJECT(lanes C)), so
// there is no C++ vtable ABI to match (that concern only applies to 4.x).

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "lanes.h"
}  // extern "C"

#include <cstdio>

int main() {
  // Minimal smoke: create a fresh 5.1 state, load the Lanes C core through the
  // public C entry point, confirm the module table exists, then tear down.
  lua_State* L = luaL_newstate();
  if (!L) {
    std::fprintf(stderr, "FAIL: luaL_newstate\n");
    return 1;
  }
  luaL_openlibs(L);

  // luaopen_lanes_core(L) returns 1 module table on the stack (per lanes.c
  // luaL_register-style init); call it through Lua so we exercise the C ABI
  // across the state boundary, not just the linker.
  lua_pushcfunction(L, luaopen_lanes_core);
  lua_call(L, 0, 1);
  if (!lua_istable(L, -1)) {
    std::fprintf(stderr, "FAIL: luaopen_lanes_core did not return a table\n");
    lua_close(L);
    return 1;
  }
  lua_setglobal(L, "lanes_core");

  lua_getglobal(L, "lanes_core");
  int ok = lua_istable(L, -1) ? 1 : 0;
  lua_pop(L, 1);
  lua_close(L);

  if (!ok) {
    std::fprintf(stderr, "FAIL: lanes_core global is not a table\n");
    return 1;
  }
  std::printf("LANES ABI OK: v3.17.2 C core linked from C++ TU under MSVC\n");
  return 0;
}