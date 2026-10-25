-- tools/smoke/serialization_smoke.lua
-- M1 2.7: paddle.save / paddle.load -- single-tensor SaveTensor/LoadTensor
-- wire format (D19, 07-serialization.md sec.3.2/5).
--
-- 07-serialization.md sec.5 acceptance demands a RESTARTED-PROCESS read:
-- a save may not depend on anything still in memory. So this smoke runs in
-- two modes, meant to be invoked as SEPARATE processes (see run-two.ps1 or
-- run it manually):
--   --save : builds tensors, saves them, prints their checksums
--   --load : loads the files, verifies checksums + error paths
--
-- Run (from the repo root, files land in .qwen/tmp):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/serialization_smoke.lua --save
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/serialization_smoke.lua --load
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

local mode = arg and arg[1]
assert(mode == "--save" or mode == "--load",
       "usage: serialization_smoke.lua --save|--load")

if mode == "--save" then
  -- --- save process: build, save, print checksums -------------------
  local x = core.pd_op_full({2, 3}, 7.0, F32, cpu)
  paddle.save{ obj = x, path = TMP .. "_a.tensor" }
  print("SAVED a 7.0 f32 2x3")

  local s = core.pd_op_full({1}, 3.25, F64, cpu)
  paddle.save{ obj = s, path = TMP .. "_b.tensor" }
  print("SAVED b 3.25 f64 1x1")

  -- overwrite path: save a again over the same file
  local x2 = core.pd_op_full({2, 3}, 8.5, F32, cpu)
  paddle.save{ obj = x2, path = TMP .. "_a.tensor", overwrite = true }
  print("SAVED a' 8.5 f32 2x3 (overwrite)")
  print("SAVE_PHASE DONE")
else
  -- --- load process: fresh VM, verify round-trip + error paths --------
  local a = paddle.load{ path = TMP .. "_a.tensor" }
  assert(a:numel() == 6 and a:ndim() == 2, "a numel/ndim")
  assert(a:dtype() == F32, "a dtype float32")
  local shp = a:shape()
  assert(shp[1] == 2 and shp[2] == 3, "a shape 2x3")
  assert(math.abs(item(a) - 8.5) < 1e-6, "a value 8.5 (overwrite applied)")

  local b = paddle.load{ path = TMP .. "_b.tensor" }
  assert(b:numel() == 1 and b:dtype() == F64, "b dtype float64")
  assert(math.abs(item(b) - 3.25) < 1e-12, "b value exact")

  -- error paths, then confirm the error slot is clean afterwards
  local okerr, err = pcall(paddle.load, { path = TMP .. "_nope.tensor" })
  assert(not okerr, "load of missing file errors")
  local msg1 = tostring(err)
  assert(msg1:find("open") ~= nil or msg1:find("Load") ~= nil,
         "missing-file message mentions the failing step")

  local okbad, errbad = pcall(paddle.save, { obj = 42, path = TMP .. "_x.tensor" })
  assert(not okbad, "save of non-tensor errors")
  assert(tostring(errbad):find("tensor") ~= nil, "bad-save names tensors")

  local w = paddle.load{ path = TMP .. "_a.tensor" }
  assert(math.abs(item(w) - 8.5) < 1e-6, "load fine after errors")

  print("SERIALIZATION_SMOKE PASS (restarted-process round-trip + errors)")
end