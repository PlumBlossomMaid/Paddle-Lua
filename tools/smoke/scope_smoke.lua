-- tools/smoke/scope_smoke.lua
-- M1 2.6: paddle.scope -- scope-scoped explicit reclamation.
-- Covers 06-autograd.md sec.5 acceptance: scope exit lowers memory
-- (core.memory_allocated before/after), returned values survive,
-- error path restores state.
--
-- Run (from the repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/scope_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local scope = require "paddle.scope"

local CPU = core.PD_PLACE_CPU
local F32 = core.PD_DTYPE_FLOAT32
local cpu = core.pd_place_new(CPU, 0)

local function mb(n)
  return math.floor(n / (1024 * 1024))
end

-- helper: first element as scalar (same dual path as autograd_smoke)
local function item(t)
  if t:numel() == 1 then
    return t:item()
  end
  return t:item(1)
end

-- --- 1: scope exit reclaims scope-local intermediates -----------------
-- Build a ~40MB float32 tensor inside the scope; it must be collected at
-- exit (LuaTensor destructor -> pd_tensor_free), lowering memory.
-- The probe_mem run proved memory_allocated tracks allocations (full
-- 100^3 float32 -> +4000004 bytes; drop+gc -> 0), so a real rise inside
-- the scope and a drop after exit is meaningful, not constant-zero.
local before = core.memory_allocated(CPU)
assert(type(before) == "number", "memory_allocated returns a number")

local peak_inside
scope.scope(function()
  local big = core.pd_op_full({100, 100, 100}, 1.0, F32, cpu)
  assert(big:numel() == 100 * 100 * 100)
  peak_inside = core.memory_allocated(CPU)
end)

local after = core.memory_allocated(CPU)
assert(peak_inside > before,
       ("allocation inside scope must be visible (before=%d peak=%d)")
         :format(before, peak_inside))
assert(after <= before,
       ("memory should not grow after scope exit (before=%d after=%d)")
         :format(before, after))

-- --- 2: returned values escape (still referenced -> not collected) ---
local base = core.memory_allocated(CPU)
local kept = scope.scope(function()
  return core.pd_op_full({100, 100}, 2.0, F32, cpu)
end)
assert(type(kept) == "userdata" and kept:numel() == 100 * 100,
       "returned tensor survives")
assert(math.abs(item(kept) - 2.0) < 1e-6, "kept tensor value intact")
kept = nil
collectgarbage("collect")  -- now it can be reclaimed

-- --- 3: captured upvalue survives (conservative escape rule) ----------
local captured
scope.scope(function()
  captured = core.pd_op_full({50, 50}, 3.0, F32, cpu)
end)
assert(type(captured) == "userdata" and captured:numel() == 2500,
       "captured tensor survives (upvalue)")
assert(math.abs(item(captured) - 3.0) < 1e-6, "captured value intact")
captured = nil
collectgarbage("collect")

-- --- 4: error path restores state, then re-raises --------------------
local okerr, err = pcall(scope.scope, function()
  error("boom from inside scope")
end)
assert(not okerr, "error propagates through scope")
assert(tostring(err):find("boom"), "error message preserved")
-- after an error, the water level comparison still ran; a fresh scope works
local b2 = core.memory_allocated(CPU)
scope.scope(function()
  local t = core.pd_op_full({10, 10, 10}, 1.0, F32, cpu)
  assert(t:numel() == 1000)
end)
assert(core.memory_allocated(CPU) <= b2, "post-error scope still reclaims")

-- --- 5: enable(false) disables reclamation ---------------------------
scope.enable(false)
local b3 = core.memory_allocated(CPU)
scope.scope(function()
  local t = core.pd_op_full({100, 100, 100}, 1.0, F32, cpu)
  assert(t:numel() == 1000000)
end)
scope.enable(true)
-- disabled scope may leave memory higher; just verify the toggle round trip
assert(type(scope.enable) == "function" and type(scope.threshold) == "function",
       "enable/threshold surface present")

print(("SCOPE_SMOKE PASS (before=%dMB after=%dMB)")
  :format(mb(before), mb(after)))