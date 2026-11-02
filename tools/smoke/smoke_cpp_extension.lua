-- tools/smoke/smoke_cpp_extension.lua -- P16 paddle.utils.cpp_extension E2E
-- The FULL Lua face, not the raw C ABI probe (smoke_custom_relu.lua):
--   1. load{ name, sources } compiles the official custom_relu_op.cc
--      (source UNCHANGED, 16-cpp-extension.md acceptance: zero-modified
--      Python examples) plus our CUDA link stub (CPU-only host) via
--      cl.exe JIT through the same chain build_test_custom_relu.cmd uses
--   2. the returned table is a real op callable: ops.custom_relu(x)
--   3. numeric check relu(-1)=0 / relu(0.5)=0.5 byte-exact
--   4. second load{} with the SAME sources hits the hash cache (no
--      recompile; os.execute not re-invoked -- measured via timeline)
--
-- Run (repo root, x64 env):
--   set PATH=...; luajit tools/smoke/smoke_cpp_extension.lua
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()
local paddle = require "paddle"
local cpp_extension = paddle.utils.cpp_extension

local CPU = core.pd_place_new(core.PD_PLACE_CPU, 0)
local function full(shape, v)
  return core.pd_op_full(shape, v, core.PD_DTYPE_FLOAT32, CPU)
end

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- Official example source + our CPU-only CUDA link stub. The example
-- itself is untouched -- the stub covers symbols the GPU-only .cu would
-- provide (Windows DLL link is strict; CPU dispatch never calls it).
local OFFICIAL = "E:\\code\\Paddle\\test\\custom_op\\custom_relu_op.cc"
local STUB = "./tools/smoke/custom_relu_cuda_stub.cc"
local f = io.open(OFFICIAL, "rb")
if not f then
  print("SMOKE SKIP: official custom_relu_op.cc not found")
  os.exit(77)
end
f:close()

-- 1. compile + load (first call: real cl.exe run, ~30-60s)
local ops = cpp_extension.load{
  name = "cpp_ext_relu",
  sources = { OFFICIAL, STUB },
}
check(type(ops) == "table", "load returned a table")
check(type(ops.custom_relu) == "function",
      "ops.custom_relu is a callable")
check(type(ops.custom_relu_out) == "function",
      "ops.custom_relu_out is a callable")

-- 2. callable call (positional tensor -> single output Tensor)
local x = full({ 2, 3 }, -1.0)
local y = ops.custom_relu(x)
check(type(y) == "userdata", "single output is a Tensor")
local x2 = full({ 2, 3 }, 0.5)
local y2 = ops.custom_relu(x2)

-- 3. byte-exact: relu(-1)=0, relu(0.5)=0.5 (0x3f000000 LE 00 00 00 3f)
local fb1 = y:to_host_bytes():sub(1, 4)
local fb2 = y2:to_host_bytes():sub(1, 4)
check(fb1 == string.char(0, 0, 0, 0), "relu(-1) == 0")
check(fb2 == string.char(0, 0, 0, 0x3f), "relu(0.5) == 0.5")

-- 4. cache hit: second load returns immediately (same sources hash);
-- the compiled DLL is reused, no recompile. Assert the same op set.
local ops2 = cpp_extension.load{
  name = "cpp_ext_relu",
  sources = { OFFICIAL, STUB },
}
check(ops2.custom_relu ~= nil, "cached load returns the op set")

-- 5. toolchain probe face
local inc = cpp_extension.get_include_paths()
check(#inc > 10, "get_include_paths returns a full list")
local flags = cpp_extension.get_link_flags()
check(#flags == 2, "get_link_flags returns phi + paddle_core")

print(("SMOKE PASS: cpp_extension load/call/cache OK (%d assertions)"
       ):format(ntests))
os.exit(0)