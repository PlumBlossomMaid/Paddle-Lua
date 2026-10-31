-- tools/smoke/jit_trace_smoke.lua -- paddle.jit.trace (P15-A, M3 4.1)
-- Validates the full Lua face over the generated trace branches:
--   1. trace a matmul->add->relu MLP (weight/bias auto-feed as
--      constants, the declared input is replaceable on re-run)
--   2. traced(x) == eager(x) bitwise, both on the example input and on
--      a FRESH input (pd_trace_run2 input substitution)
--   3. Tensor:item() during the recording run ERRORS (never silently
--      freezes a branch, 15-to-static.md sec.4 pit 3) and the failed
--      trace leaves no active trace behind (pcall + close)
--   4. wrong input arity on the traced callable -> clean Lua error
--   5. after finish(), plain eager ops no longer record (trace off)
--
-- Run (repo root, x64 env): set PATH=... ; luajit tools/smoke/jit_trace_smoke.lua

package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()
local paddle = require "paddle"

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

-- matmul -> add -> relu; w/b captured by the closure (auto-feeds)
local w = core.pd_op_full({ 3, 4 }, 1, core.PD_DTYPE_FLOAT32, CPU)
local b = core.pd_op_full({ 4 }, -1, core.PD_DTYPE_FLOAT32, CPU)
local function mlp(x)
  local h = core.pd_op_matmul(x, w, 0, 0)
  local z = core.pd_op_add(h, b)
  return core.pd_op_relu(z)
end

local x1 = full({ 2, 3 }, 1)
local x2 = full({ 2, 3 }, 2)

-- eager references (no trace active yet)
local e1 = mlp(x1)
local e2 = mlp(x2)

-- 1. trace
local traced = paddle.jit.trace(mlp, x1)
check(type(traced) == "function", "trace returns a callable")

-- 2a. replay with the example input: bitwise == eager
local r1 = traced(x1)
check(r1:to_host_bytes() == e1:to_host_bytes(),
      "traced(x1) != eager(x1)")

-- 2b. replay with a FRESH input: input substitution works, still exact
local r2 = traced(x2)
check(r2:to_host_bytes() == e2:to_host_bytes(),
      "traced(x2) != eager(x2) (input substitution broken)")

-- sanity on the deterministic values: x2=2s -> matmul 6s -> add 5s
local v = r2:item(1)
check(math.abs(v - 5.0) < 1e-6, "expected 5.0, got " .. tostring(v))

-- 3. item() during trace must ERROR, and the trace must be cleaned up
check(not core.trace_is_active(), "trace still active after finish()")
local ok, err = pcall(function()
  return paddle.jit.trace(function(x)
    if x:item() > 0 then return x end
    return core.pd_op_relu(x)
  end, x1)
end)
check(not ok, "item() during trace did not error")
check(tostring(err):find("freeze") ~= nil,
      "item-during-trace error should name the freeze hazard, got: "
      .. tostring(err))
check(not core.trace_is_active(),
      "failed trace left the active-trace slot set")

-- 4. wrong arity -> clean Lua error
local ok2, err2 = pcall(traced, x1, x2)
check(not ok2 and tostring(err2):find("expects 1 input") ~= nil,
      "arity error missing, got: " .. tostring(err2))

-- 5. after finish(), eager ops do not record (a second trace is clean)
local traced2 = paddle.jit.trace(mlp, x1)
local r3 = traced2(x2)
check(r3:to_host_bytes() == e2:to_host_bytes(),
      "second trace disagrees with eager")

print(("JIT_TRACE_SMOKE PASS (%d assertions)"):format(ntests))
