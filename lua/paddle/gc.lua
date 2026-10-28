-- lua/paddle/gc.lua -- 3.3 GC 九层机制:层1-3 堆追踪开关 + 诊断
-- Stage: M2 / 3.3 (WORKPLAN 3.3, plan/modules/14-gc.md sec.3.1).
-- Layer 1-3 (account / 1MB batch / x1.4 softmax) live in the C ABI
-- (pd_gc_account, gc.cc -- a faithful port of Torch7 THGeneral.c). This
-- module is the user-facing switch + diagnostics. Default ON per D6
-- (14-gc.md 坑④). The per-thread lua_State bridge (calls collectgarbage
-- when the C-side softmax is exceeded) is registered in luaopen_paddle_core
-- (module.cc), not here -- only there do we have the C linkage + the Lua
-- state at load time.
--
-- NOTE: the bridge uses collectgarbage("collect") (LUA_GCCOLLECT, value 2
-- in every 5.x / LuaJIT). 14-gc.md sec.3.1's sketch uses LUA_GCSTEP,
-- which only exists from Lua 5.2+; our five-version constraint (C5) and
-- the LuaJIT 5.1 API surface forbid it. Full-collect on the softmax path
-- is the safe, portable choice and matches the OOM-callback layer (6-8)
-- which also does a full collect.
--
-- Lua 5.1 subset (C3).

local argrule = require "argrule"
local rule = require "argrule.rule"
local core = require "paddle_core"

local M = {}

-- paddle.set_heap_tracking(bool) -- enable/disable GC heap tracking (D6).
-- Writes through to core.set_heap_tracking, which resets the global
-- ledger on disable (gc.cc pd_gc_set_enabled). Default state is ON.
M.set_heap_tracking = rule {
  { name = "on", type = argrule.boolean, doc = "true=on (default), false=off" },
}(function(on)
  core.set_heap_tracking(on)
end)

-- paddle.heap_tracking() -> bool (current state, 14-gc.md sec.3.1 / sec.5)
-- core.heap_tracking pushes a Lua boolean (module.cc l_heap_tracking), so
-- returning it verbatim -- comparing to 1 would always be false.
M.heap_tracking = function()
  return core.heap_tracking()
end

-- paddle.tracked_bytes() -> number (sum of live tensor bytes across all
-- lanes, gc.cc pd_gc_tracked_bytes). Diagnostic for smoke + tuning.
M.tracked_bytes = function()
  return core.tracked_bytes()
end

return M
