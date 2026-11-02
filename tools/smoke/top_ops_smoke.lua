-- tools/smoke/top_ops_smoke.lua -- paddle 顶层算子层 smoke (4.4 前置)
-- Validates lua/paddle/top_ops.lua (api/top.md 契约首批):
--   paddle.zeros/ones/full{shape,...}, paddle.cast{x, "int32"},
--   paddle.equal/greater_than, paddle.sum/mean{axis=...},
--   paddle.abs/sqrt/square/subtract/divide/maximum/minimum
-- The layer wraps the generated _ops wrappers and adds what they lack:
--   dtype string -> pd enum (pd_op_cast("int32") crashes, enum works),
--   IntArray scalar -> table (pd_op_sum(0) crashes, {0} works),
--   argrule table-call contract (api/README.md sec2.1.5).
-- Run: repo root, x64 env (same PATH as the other smokes).
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()
local paddle = require "paddle"

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- f32 LE decode (clone of the other smokes).
local function f32_at(s, idx)
  local b1, b2, b3, b4 = s:byte(idx, idx + 3)
  local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local sign = bits >= 2147483648 and 1 or 0
  local exp = math.floor(bits / 8388608) % 256
  local mant = bits % 8388608
  if exp == 0 then return 0.0 end
  if exp == 255 then return math.huge end
  local v = (1 + mant / 8388608) * 2 ^ (exp - 127)
  return sign == 1 and -v or v
end

-- ---- creation ----
local z = paddle.zeros{ 2, 2 }
check(type(z) == "userdata", "paddle.zeros returns a Tensor")
check(z:shape()[1] == 2 and z:shape()[2] == 2, "zeros shape {2,2}")
check(z:dtype() == core.PD_DTYPE_FLOAT32,
      "zeros default dtype float32 (pd enum)")
local o = paddle.ones{ 3 }
check(o:to_host_bytes() == string.char(0, 0, 0x80, 0x3f)
      .. string.char(0, 0, 0x80, 0x3f) .. string.char(0, 0, 0x80, 0x3f),
      "ones {3} == 1.0 float32")
local f = paddle.full{ { 2 }, 2.5 }
local fb = f:to_host_bytes()
check(math.abs(f32_at(fb, 1) - 2.5) < 1e-6, "full fill_value 2.5")

-- ---- cast: string dtype -> enum ----
local i32 = paddle.cast{ z, dtype = "int32" }
check(i32:dtype() == core.PD_DTYPE_INT32, "cast{x,'int32'} dtype string accepted")
local f64 = paddle.cast{ z, dtype = paddle.float64 }
check(f64:dtype() == core.PD_DTYPE_FLOAT64, "cast accepts paddle.float64 constant")

-- ---- logic ----
local a = paddle.full{ { 2 }, 1.0 }
local b = paddle.full{ { 2 }, 1.0 }
local eq = paddle.equal{ a, b }
check(eq:dtype() == core.PD_DTYPE_BOOL, "equal output dtype bool")
local gt = paddle.greater_than{ b, a }
local gtb = gt:to_host_bytes()
check(gtb == string.char(0) .. string.char(0), "greater_than B>A == false,false")

-- ---- reduction: scalar axis -> table, 1-based ----
-- x = [[1,2,3],[4,5,6]] ; 1-based axis=1 = C++ dim 0 -> reduce ROWS,
-- result [5,7,9] (column sums)
local x = require("paddle_core").from_host_bytes(
  string.char(0, 0, 0x80, 0x3f, 0, 0, 0, 0x40, 0, 0, 0x40, 0x40,
                0, 0, 0x80, 0x40, 0, 0, 0xa0, 0x40, 0, 0, 0xc0, 0x40),
  { 2, 3 }, core.PD_DTYPE_FLOAT32,
  core.pd_place_new(core.PD_PLACE_CPU, 0))
local s = paddle.sum{ x, axis = 1 }
local sb = s:to_host_bytes()
check(s:shape()[1] == 3, "sum axis=1 -> 3 columns")
check(math.abs(f32_at(sb, 1) - 5.0) < 1e-6, "sum col0 == 1+4 == 5")
check(math.abs(f32_at(sb, 5) - 7.0) < 1e-6, "sum col1 == 2+5 == 7")
check(math.abs(f32_at(sb, 9) - 9.0) < 1e-6, "sum col2 == 3+6 == 9")
local sall = paddle.sum{ x }
check(sall:ndim() == 0, "sum full reduce -> 0-d scalar")
check(math.abs(f32_at(sall:to_host_bytes(), 1) - 21.0) < 1e-6,
      "sum full == 21")
local m = paddle.mean{ x, axis = { 1 } }
local mb = m:to_host_bytes()
check(math.abs(f32_at(mb, 1) - 2.5) < 1e-6, "mean axis{1} col0 == 2.5")
check(math.abs(f32_at(mb, 9) - 4.5) < 1e-6, "mean axis{1} col2 == 4.5")

-- ---- unary/binary basics ----
local y = paddle.full{ { 3 }, 4.0 }
local sq = paddle.square{ y }
check(math.abs(f32_at(sq:to_host_bytes(), 1) - 16.0) < 1e-6, "square 4 == 16")
local rt = paddle.sqrt{ y }
check(math.abs(f32_at(rt:to_host_bytes(), 1) - 2.0) < 1e-6, "sqrt 4 == 2")
local sub = paddle.subtract{ y, paddle.full{ { 3 }, 1.0 } }
check(math.abs(f32_at(sub:to_host_bytes(), 1) - 3.0) < 1e-6, "subtract 4-1")
local div = paddle.divide{ y, paddle.full{ { 3 }, 2.0 } }
check(math.abs(f32_at(div:to_host_bytes(), 1) - 2.0) < 1e-6, "divide 4/2")
local mx = paddle.maximum{ y, paddle.full{ { 3 }, 5.0 } }
check(math.abs(f32_at(mx:to_host_bytes(), 1) - 5.0) < 1e-6, "maximum 4,5 -> 5")
local mn = paddle.minimum{ y, paddle.full{ { 3 }, 5.0 } }
check(math.abs(f32_at(mn:to_host_bytes(), 1) - 4.0) < 1e-6, "minimum 4,5 -> 4")
local ab = paddle.abs{ paddle.full{ { 2 }, -3.0 } }
check(math.abs(f32_at(ab:to_host_bytes(), 1) - 3.0) < 1e-6, "abs -3 -> 3")

-- ---- error paths: bad dtype / positional form ----
local ok, err = pcall(function() return paddle.cast{ z, dtype = "bogus" } end)
check(not ok and tostring(err):find("dtype", 1, true) ~= nil,
      "unknown dtype errors")
local ok2 = pcall(function() return paddle.sum(x, 2, nil, true) end)
check(ok2, "positional form sum(x, 2, nil, true) works")

print(("SMOKE PASS: top_ops (paddle.zeros/cast/sum/...) OK "
       .. "(%d assertions)"):format(ntests))
os.exit(0)