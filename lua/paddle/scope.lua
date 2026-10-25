-- lua/paddle/scope.lua -- paddle.scope: scope-scoped explicit reclamation
-- Stage: P6 / M1 (WORKPLAN 2.6). After autograd.lua (same stage).
--
-- Design: docs/migration/plan/api/scope.md + modules/06-autograd.md sec.3.3.
--   - OUR OWN API (upstream has NO paddle.scope; scope.md header note) --
--     the only user-visible layer of the 9-layer GC scheme (gc.md sec.4
--     layer 6). Pure Lua 5.1 (~40 lines), no C-side epoch list.
--   - Mechanism (scope.md sec.2.1 / sec.6 ③): record the live-tensor
--     water level at entry (core.memory_allocated, C ABI
--     pd_device_allocated_bytes -> HostMemoryStatCurrentValue, device.cc:
--     135-150); on exit, if the scope GREW past the threshold, pressure
--     the Lua GC once so intermediate tensors that lost their last
--     reference are collected NOW -- their LuaTensor destructor runs
--     pd_tensor_free (sol_common.hpp), releasing the memory immediately
--     instead of at the next automatic GC.
--   - Escape is CONSERVATIVE BY CONSTRUCTION (scope.md sec.2.1 / sec.6 ②):
--     returned values and objects captured by external upvalues still
--     have a live reference, so Lua GC cannot collect them. scope is
--     suggested reclamation, never a correctness dependency.
--   - Error path restores the water level before re-raising (scope.md
--     sec.6 ⑤): the growth comparison still runs on the error path, so
--     the pressure decision is not distorted by the exception.

local core = require "paddle_core"

local M = {}

-- Default on (scope.md sec.6 ①: same spirit as D6 heap tracking on by
-- default; the threshold default is tuned after the P6 training
-- benchmark, so keep a conservative 0 = pressure whenever the scope grew).
local enabled = true
local threshold = 0

local function allocated()
  return core.memory_allocated(core.PD_PLACE_CPU)
end

function M.enable(v)
  enabled = v and true or false
end

function M.threshold(bytes)
  threshold = bytes or 0
end

-- paddle.scope(fn) -- run fn, then reclaim scope-local intermediates.
-- Returns fn's results (they survive: still referenced).
function M.scope(fn)
  if type(fn) ~= "function" then
    error("paddle.scope: expected a function", 2)
  end
  if not enabled then
    return fn()
  end
  local base = allocated()
  local ok, a, b, c = pcall(fn)
  -- Pressure even on the error path (scope.md sec.6 ⑤): the growth
  -- comparison restores the water level view before we re-raise.
  local grown = allocated() - base
  if grown > threshold then
    -- Force a full Lua GC cycle: now-unreachable intermediate tensors
    -- are collected and their LuaTensor destructor releases the memory
    -- immediately (scope.md sec.2.1). 5.1 form: collectgarbage("collect").
    collectgarbage("collect")
  end
  if not ok then
    error(a, 0)  -- keep the caller's frame (conventions.md / 06-autograd.md)
  end
  return a, b, c
end

return M