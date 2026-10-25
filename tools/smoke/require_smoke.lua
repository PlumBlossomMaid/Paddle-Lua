-- tools/smoke/require_smoke.lua
-- M1 2.4: require "paddle_core" + require "paddle" smoke (04-packaging).
--
-- The exact route a user takes: LuaJIT (or any 5.1 ABI Lua) `require`s
-- the built C module, then the pure-Lua entry point. Proves the CMake
-- build produces a loadable paddle_core.dll whose luaopen works and whose
-- C ABI surface (init / tensor / slice_spec) answers correctly.
--
-- Build first:  cmake -S . -B build-test
--               cmake --build build-test --target paddle_core -j 6
-- Run (from the repo root):
--   PATH=<paddle build dirs + vcpkg luajit dir>;$PATH  luajit tools/smoke/require_smoke.lua
-- The paddle_core.dll runtime deps (phi.dll common.dll mkldnn.dll
-- libiomp5md.dll lua51.dll) must resolve via PATH or the working
-- directory -- Windows has no RPATH (04-packaging.md sec.4 pit 1).
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
assert(type(core) == "table", "module is a table")
assert(core.version() == "0.1.0", "version() == 0.1.0")
assert(core.init() == true, "init() idempotent")

local t = core.tensor{2, 3}
assert(t:ndim() == 2 and t:numel() == 6, "tensor{2,3} ndim/numel")
local s = t:shape()
assert(s[1] == 2 and s[2] == 3, "shape {2,3}")
assert(t:is_contiguous() == 1, "is_contiguous")
assert(t:dtype() == 10, "dtype FLOAT32")

assert(core.slice_spec("1:3") == "0:2", "slice_spec 1:3")
assert(core.slice_spec("2,1:-1") == "1,0:-1", "slice_spec multi-dim")

local ok2, P = pcall(require, "paddle")
assert(ok2, "require paddle failed: " .. tostring(P))
assert(P.init() == true, "paddle.init passthrough")
assert(P._version == "0.1.0", "paddle._version")

print("REQUIRE_SMOKE PASS")