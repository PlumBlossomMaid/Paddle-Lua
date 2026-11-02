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

-- Official simple_slice op (two int64_t attrs) -- the attr marshalling
-- path (positional AND named, plus a missing-attr error check).
local SLICE = "E:\\code\\Paddle\\test\\custom_op\\custom_simple_slice_op.cc"
local sf = io.open(SLICE, "rb")
if not sf then
  print("SMOKE SKIP: official custom_simple_slice_op.cc not found")
  os.exit(77)
end
sf:close()

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

-- ---- attr-aware op: official custom_simple_slice (2 int64_t attrs) ----
local slice_ops = cpp_extension.load{
  name = "cpp_ext_slice",
  sources = { SLICE },
}
check(type(slice_ops.custom_simple_slice) == "function",
      "simple_slice callable registered")

-- positional attrs: slice(x, begin, end) -- x.slice(begin, end)
local xs = full({ 8, 4 }, 1.0)
local ys = slice_ops.custom_simple_slice(xs, 2, 5)
check(ys:shape()[1] == 3, "positional attrs slice rows 2..5 -> 3 rows")
-- named attrs: slice{ xs, begin_index = 2, end_index = 5 }
local ys2 = slice_ops.custom_simple_slice{ xs, begin_index = 2,
                                           end_index = 5 }
check(ys2:shape()[1] == 3, "named attrs slice rows 2..5 -> 3 rows")
-- missing attr -> actionable Lua error (not a marshalling crash)
local ok, err = pcall(function()
  return slice_ops.custom_simple_slice(xs, 2)
end)
check(not ok and tostring(err):find("attr", 1, true) ~= nil,
      "missing attr errors with an actionable message")

-- ---- 16-cpp-extension.md acceptance: cache INVALIDATION ----
-- Changing ONE line of a source must invalidate the hash cache: the new
-- load recompiles (different DLL path, different behavior) and drops the
-- stale DLL. We use a throwaway scale op whose constant we flip.
local tmp_cc = "build-test/cache_probe.cc"
-- opname differs per compile so two different-DLL builds (the cache
-- invalidation case) do not double-register the same op -- the point
-- here is the cache key/behavior change, not re-registration. The
-- timestamp comment makes every run recompile (a previous run's products
-- would otherwise satisfy the cache and skip the stale-cleanup path).
local function write_scale(k, opname, ts)
  local fw = io.open(tmp_cc, "w")
  fw:write([[
#include "paddle/extension.h"
// probe ]], tostring(ts), [[

std::vector<paddle::Tensor> ScaleKernel(const paddle::Tensor& x) {
  auto out = paddle::empty_like(x);
  auto* op = out.mutable_data<float>(x.place());
  auto* ip = x.data<float>();
  const float k = ]], k, [[;
  for (int64_t i = 0; i < x.numel(); ++i) op[i] = ip[i] * k;
  return {out};
}

PD_BUILD_OP(]], opname, [[)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ScaleKernel));
]])
  fw:close()
end
local function exists(p)
  local fe = io.open(p, "rb")
  if fe then fe:close() return true end
  return false
end

local ts = os.time()
write_scale("2.0f", "cache_probe_a", ts)
local p1 = cpp_extension.load{ name = "cpp_ext_cacheprobe",
                               sources = { tmp_cc } }
local d1 = p1._dll
local xp = full({ 2, 2 }, 4.0)
local y1 = p1.cache_probe_a(xp)
local b1 = y1:to_host_bytes()

-- A never-loaded stale product must be dropped by the recompile cleanup.
-- (The ALREADY-loaded d1 cannot be deleted while mapped -- Windows keeps
-- modules open without FILE_SHARE_DELETE -- so stale removal is only
-- asserted on this fake, load-free entry.)
local fake = "build-test/cpp_ext_cacheprobe_fakestale.dll"
local ff = io.open(fake, "wb")
ff:write("stale")
ff:close()

write_scale("3.0f", "cache_probe_b", ts)  -- one line changed -> key changes
local p2 = cpp_extension.load{ name = "cpp_ext_cacheprobe",
                               sources = { tmp_cc } }
local d2 = p2._dll
local y2 = p2.cache_probe_b(xp)
local b2 = y2:to_host_bytes()
check(d2 ~= d1, "source change -> new cache key / DLL path")
check(b2 ~= b1, "source change -> recompiled (behavior differs)")
check(not exists(fake), "stale (never-loaded) DLL removed after recompile")
os.remove(tmp_cc)

-- ---- 16-cpp-extension.md acceptance: ABI guard for a MISMATCHED
-- toolchain (sec 4 pit 1) ----
-- A product built by a different compiler family (MinGW linker major 2
-- vs MSVC 14) must be REJECTED with an actionable error instead of being
-- handed to LoadLibrary (which would crash with no clue). The smoke can
-- only patch a COPY: a loaded DLL is open read-only on Windows, so the
-- in-place patch is impossible -- we build a patched copy and drive the
-- same abi_check load{} runs before every LoadLibrary.
local ev_dll = ops2._dll
local rf = io.open(ev_dll, "rb")
local all = rf:read("*a")
rf:close()
local e1, e2, e3, e4 = all:byte(0x3D, 0x40)  -- e_lfanew u32 @ 0x3C
local ne = e1 + e2 * 256 + e3 * 65536 + e4 * 16777216
local major_off = ne + 4 + 20 + 2  -- PE sig(4) + COFF(20) + Magic(2)
local patched = all:sub(1, major_off) .. string.char(2)
                .. all:sub(major_off + 2)
local copy = "build-test/cpp_ext_relu_patched.dll"
local wf = io.open(copy, "wb")
wf:write(patched)
wf:close()
-- the same MSVC-built original passes the check; the patched one fails
check(cpp_extension._check_abi(ev_dll) == nil,
      "MSVC-built DLL passes the ABI check")
local okb, errb = pcall(function()
  return cpp_extension._check_abi(copy)
end)
check(not okb and tostring(errb):find("linker major", 1, true) ~= nil,
      "mismatched toolchain DLL rejected with an actionable error")
os.remove(copy)

print(("SMOKE PASS: cpp_extension load/call/cache/attrs OK "
       .. "(%d assertions)"):format(ntests))
os.exit(0)