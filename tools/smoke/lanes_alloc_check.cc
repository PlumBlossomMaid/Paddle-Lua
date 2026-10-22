// tools/smoke/lanes_alloc_check.cc
// M0 #10 acceptance: heap tracking (a counting lua_Alloc on the master state)
// vs Lanes' allocator machinery.
//
// Question (research/dataloader.md §9.5(e), 13-lanes.md 坑④):
//   Torch7-style heap tracking counts allocation bytes and triggers lua_gc on
//   threshold. Lanes creates one lua_State per lane. Do the two conflict, and
//   which allocator does a lane state actually get?
//
// Source facts pinned by this smoke (Lanes v3.17.2):
//   lanes.lua:47   default allocator = (jit) and "protected" or nil
//   lanes.lua:48   default internal_allocator = (jit) and "libc" or "allocator"
//   tools.c:194-232  "protected" wraps the CURRENT master allocator
//                    (lua_getallocf) in a mutex and lua_setallocf's it back.
//   state.c:253-255  #if LUAJIT_FLAVOR()==64 -> lane states are created with
//                    luaL_newstate() ONLY; a custom allocator NEVER reaches
//                    lane states under LuaJIT.
//   state.c:257-271  PUC Lua -> lane state reuses the master's allocator
//                    (U->protected_allocator.definition).
//
// Method: install counting_alloc on the master, load the real lanes.lua,
// configure{ on_state_create = probe }, launch a lane that allocates ~1.2MB,
// and assert:
//   (1) on_state_create fired >= 1 time         (hook usable per lane)
//   (2) the lane states' own allocator (probed via lua_getallocf inside the
//       hook, which runs ON the new lane state) is NOT counting_alloc under
//       LuaJIT   -> heap-tracking via master allocator does NOT see lane
//       allocations; our tracking must stay a tensor-bytes counter, which is
//       per-thread and independent of any Lua allocator.
//   (3) the master-side counting allocator still works after configure
//       (protected wraps-and-forwards, does not discard our allocator).
//   (4) the lane runs and returns the expected value.
//
// Build (dev-only, mirrors lanes_abi_check): compile the 10 Lanes .c files
// against the same LuaJIT headers as this TU, link lua51.lib, same CRT.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "compat.h"   // provides luaL_requiref for Lua 5.1 (Lanes compat.c)
#include "lanes.h"
}  // extern "C"

static std::atomic<long long> g_alloc_calls{0};
static std::atomic<long long> g_alloc_bytes{0};
static std::atomic<int> g_on_state_create{0};
static std::atomic<int> g_lane_inherited_alloc{0};

static lua_Alloc g_orig_alloc = NULL;   // the allocator present at install time
static void* g_orig_ud = NULL;

// Heap-tracking probe. MUST chain to the pre-install allocator instead of
// calling realloc/free directly: vcpkg LuaJIT ships as lua51.dll with its own
// CRT heap; freeing a pointer it produced with this exe's (static) CRT heap
// corrupts the heap (0xC0000374). This is exactly how Lanes' "protected"
// allocator wraps the master allocator (tools.c:173-190 protected_lua_Alloc).
static void* counting_alloc(void* ud, void* ptr, size_t osize, size_t nsize) {
  (void)ud; (void)osize;
  g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
  void* p = g_orig_alloc(g_orig_ud, ptr, osize, nsize);
  if (p && nsize > 0 && ptr == NULL) {   // pure grow: count live bytes once
    g_alloc_bytes.fetch_add((long long)nsize, std::memory_order_relaxed);
  }
  return p;
}

// Runs as the on_state_create hook. Lanes calls it ON the freshly created
// lane state, so lua_getallocf here answers "what allocator did my lane get?".
static int on_state_create_probe(lua_State* L) {
  g_on_state_create.fetch_add(1, std::memory_order_relaxed);
  void* ud = NULL;
  lua_Alloc f = lua_getallocf(L, &ud);
  if (f == counting_alloc) {
    g_lane_inherited_alloc.fetch_add(1, std::memory_order_relaxed);
  } else {
    std::printf("  [lane state #%d] allocator != counting_alloc (fine under LuaJIT)\n",
                (int)g_on_state_create.load());
  }
  return 0;
}

int main(int argc, char** argv) {
  const char* lanes_lua = (argc > 1) ? argv[1]
                                     : "E:\\code\\lanes-3.17\\src\\lanes.lua";
  bool install_alloc = true;
  if (argc > 2 && std::strcmp(argv[2], "noalloc") == 0) install_alloc = false;

  lua_State* L = luaL_newstate();
  if (!L) { std::fprintf(stderr, "FAIL: luaL_newstate\n"); return 1; }
  std::printf("[stage] newstate ok\n"); std::fflush(stdout);
  luaL_openlibs(L);
  if (install_alloc) {
    g_orig_alloc = lua_getallocf(L, &g_orig_ud);  // capture BEFORE replacing
    lua_setallocf(L, counting_alloc, NULL);        // our probe
  }
  std::printf("[stage] openlibs+setallocf ok (install_alloc=%d)\n", install_alloc);
  std::fflush(stdout);

  // Load lanes.core through the official entry (compat.c provides
  // luaL_requiref for 5.1), then the real lanes.lua module.
  luaL_requiref(L, "lanes.core", luaopen_lanes_core, 1);
  lua_pop(L, 1);
  if (luaL_dofile(L, lanes_lua) != 0) {
    std::fprintf(stderr, "FAIL: lanes.lua: %s\n", lua_tostring(L, -1));
    return 1;
  }
  lua_setglobal(L, "lanes");  // lanes.lua returns the module table

  // Give the Lua chunk our C hook. Configure with DEFAULTS (LuaJIT ->
  // allocator="protected", internal_allocator="libc") + the hook.
  lua_pushcfunction(L, on_state_create_probe);
  lua_setglobal(L, "nix_alloc_probe");
  lua_pushstring(L, "nix");
  lua_setglobal(L, "nix_sentinel");

  const char* script =
      "local lanes = require'lanes'\n"
      "lanes.configure{ on_state_create = nix_alloc_probe }\n"
      "local gen = lanes.gen('string', function()\n"
      "  local t = {}\n"
      "  for i = 1, 20000 do t[i] = string.rep('x', 64) end\n"
      "  return #t\n"
      "end)\n"
      "local h = gen()\n"
      "return h:join()\n";
  if (luaL_dostring(L, script) != 0) {
    std::fprintf(stderr, "FAIL: scenario: %s\n", lua_tostring(L, -1));
    return 1;
  }
  long long result = (long long)lua_tointeger(L, -1);
  lua_pop(L, 1);

  long long calls = g_alloc_calls.load();
  long long bytes = g_alloc_bytes.load();
  int created = g_on_state_create.load();
  int inherited = g_lane_inherited_alloc.load();

  lua_getglobal(L, "jit");
  bool is_luajit = lua_istable(L, -1) != 0;   // LuaJIT exposes the "jit" module table
  lua_pop(L, 1);
  std::printf("== LUA: %s (%s)\n", LUA_RELEASE, is_luajit ? "LuaJIT" : "PUC");
  std::printf("lane result            : %lld (expect 20000)\n", result);
  std::printf("on_state_create fired  : %d times\n", created);
  std::printf("lane states inherited counting_alloc: %d\n", inherited);
  std::printf("master allocator calls : %lld, bytes %lld\n", calls, bytes);

  int failures = 0;
  if (result != 20000) { std::printf("FAIL: lane result\n"); ++failures; }
  if (created < 1) { std::printf("FAIL: on_state_create never fired\n"); ++failures; }
  if (inherited != 0) {
    std::printf("FAIL: lane state inherited our allocator; heap tracking via\n"
                "      master allocator would double-count lane allocations\n");
    ++failures;
  }
  if (install_alloc && calls < 1) {
    std::printf("FAIL: counting allocator never called\n");
    ++failures;
  }

  lua_close(L);
  if (failures == 0) {
    std::printf("LANES ALLOC PASS: heap-tracking allocator is master-only; "
                "lane Lua heap is NOT tracked through it (LuaJIT). "
                "Tracking must be tensor-bytes per-thread, not a Lua allocator.\n");
    return 0;
  }
  std::printf("LANES ALLOC FAIL: %d assertion(s)\n", failures);
  return 1;
}