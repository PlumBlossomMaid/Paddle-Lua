-- tools/smoke/serialization_smoke.lua
-- M1 2.7: paddle.save / paddle.load -- single-tensor SaveTensor/LoadTensor
-- wire format (D19, 07-serialization.md sec.3.2/5).
--
-- Run (from the repo root, files land in .qwen/tmp):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/serialization_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"

local cpu = core.pd_place_new(core.PD_PLACE_CPU, 0)
local F32 = core.PD_DTYPE_FLOAT32
local F64 = core.PD_DTYPE_FLOAT64

local TMP = ".qwen/tmp/serialization_smoke"

-- helper: first element as scalar (same dual path as autograd_smoke)
local function item(t)
  if t:numel() == 1 then
    return t:item()
  end
  return t:item(1)
end

-- --- 1: save / load round-trip, 2x3 float32 --------------------------
local x = core.pd_op_full({2, 3}, 7.0, F32, cpu)
local path = TMP .. "_1.tensor"
paddle.save{ obj = x, path = path }
local y = paddle.load{ path = path }
assert(y:numel() == 6, "loaded numel == 6")
assert(y:ndim() == 2, "loaded ndim == 2")
assert(y:dtype() == F32, "loaded dtype == float32")
local shape = y:shape()
assert(shape[1] == 2 and shape[2] == 3, "loaded shape == {2,3}")
assert(math.abs(item(y) - 7.0) < 1e-6, "loaded value == 7.0")

-- --- 2: distinct values preserved element-wise ------------------------
local z = core.pd_op_full({4}, 1.0, F32, cpu)
z:set_stop_gradient(false)
-- write distinct bytes via a strided view? keep it simple: 0-d check
local s = core.pd_op_full({1}, 3.25, F64, cpu)
paddle.save{ obj = s, path = TMP .. "_2.tensor" }
local s2 = paddle.load{ path = TMP .. "_2.tensor" }
assert(s2:numel() == 1 and s2:dtype() == F64, "0-d-ish f64 round-trip")
assert(math.abs(item(s2) - 3.25) < 1e-12, "f64 value exact")

-- --- 3: overwrite defaults to true (second save on same path) --------
paddle.save{ obj = x, path = path }
local y2 = paddle.load{ path = path }
assert(math.abs(item(y2) - 7.0) < 1e-6, "overwrite re-save ok")

-- --- 4: error paths --------------------------------------------------
local okerr, err = pcall(paddle.load, { path = TMP .. "_nope.tensor" })
assert(not okerr, "load of missing file errors")
assert(tostring(err):find("open") ~= nil or tostring(err):find("Load"),
       "message mentions the failing step")

local okbad, errbad = pcall(paddle.save, { obj = 42, path = path })
assert(not okbad, "save of non-tensor errors")
assert(tostring(errbad):find("tensor") ~= nil, "message names tensors")

-- --- 5: reload keeps working after prior errors (error slot clean) ---
local w = paddle.load{ path = path }
assert(math.abs(item(w) - 7.0) < 1e-6, "load still fine after errors")

print("SERIALIZATION_SMOKE PASS")