-- lua/paddle/autograd.lua -- paddle.autograd: backward / grad / no_grad
-- Stage: P6 / M1 (WORKPLAN 2.6). Depends on P5 (paddle.Tensor).
--
-- Design: docs/migration/plan/api/autograd.md + modules/06-autograd.md.
-- All exports here are THIN table-call wrappers over paddle_core's
-- module-level autograd bindings (tensor_binding.cc register_autograd_fns)
-- which wrap the C ABI (autograd.cc) which calls eager/backward.h -- one
-- bottom layer shared with Tensor:backward() (api/autograd.md sec.2 note).
--
-- no_grad is FUNCTION-scoped (api/autograd.md sec.6 ①): Lua has no `with`,
-- so the equivalence of `with paddle.no_grad():` is
-- `paddle.autograd.no_grad(function() ... end)`. The pcall-guard restores
-- the previous state even when fn errors (06-autograd.md sec.3.2).

local core = require "paddle_core"

local M = {}

-- ---------------------------------------------------------------------
-- backward{tensors, grad_tensors=nil, retain_graph=false}
-- ---------------------------------------------------------------------
-- api/autograd.md sec.2.1. tensors/grad_tensors are arrays of tensors;
-- a bare tensor (backward{loss}) is wrapped into a one-element list.
-- A failed backward gets a retain-graph hint (sec.6 ④): Paddle releases
-- the graph after the first backward, so a second one errors -- the
-- message should point at retain_graph=true once the error is about a
-- freed graph, not for unrelated failures.
local function as_tensor_list(v)
  if type(v) == "userdata" then
    return { v }
  end
  return v
end

local RETAIN_HINTS = {
  "freed", "already", "released", "graph", "backward",
}

-- core.backward(grad_tensors) must NEVER see a middle nil: sol2 treats
-- nil as the argument-list terminator (measured via _probe_optional,
-- 2026-10-25) -- a nil grad_tensors silently drops retain_graph. Use {}
-- (empty list == no seeds, same semantics as nil upstream).
function M.backward(t)
  local ok, err = pcall(core.backward, as_tensor_list(t.tensors or t[1]),
                        t.grad_tensors or {},
                        t.retain_graph ~= nil and t.retain_graph or false)
  if ok then
    return nil
  end
  local msg = tostring(err)
  local lower = msg:lower()
  local release_suspect = false
  for _, kw in ipairs(RETAIN_HINTS) do
    if lower:find(kw, 1, true) then
      release_suspect = true
      break
    end
  end
  if release_suspect and not (t.retain_graph) then
    error(msg .. " -- did you mean: backward{..., retain_graph = true}?", 2)
  end
  error(msg, 2)
end

-- ---------------------------------------------------------------------
-- grad{outputs, inputs, grad_outputs=nil, retain_graph=nil,
--      create_graph=false, only_inputs=true, allow_unused=false}
--      -> list of gradient tensors
-- ---------------------------------------------------------------------
-- api/autograd.md sec.2.2. Python defaults are kept (sec.6 ③): a silently
-- returned nil hides a disconnected graph, so allow_unused stays false.
-- core.grad(outputs, inputs, grad_outputs, retain_graph, create_graph,
-- only_inputs, allow_unused): every optional slot must be FILLED, never
-- nil -- sol2 treats a nil middle argument as the argument-list
-- terminator (measured via _probe_optional, 2026-10-25) and drops every
-- later param. Defaults mirror Python (base.py:706): retain_graph=false,
-- create_graph=false, only_inputs=true, allow_unused=false.
function M.grad(t)
  return core.grad(as_tensor_list(t.outputs or t[1]),
                   as_tensor_list(t.inputs or t[2]),
                   t.grad_outputs or {},
                   t.retain_graph or false,
                   t.create_graph or false,
                   t.only_inputs == nil and true or t.only_inputs,
                   t.allow_unused or false)
end

-- ---------------------------------------------------------------------
-- no_grad(fn) / enable_grad(fn) -- function-scoped context managers
-- ---------------------------------------------------------------------
local function with_grad(mode, fn)
  local prev = core.has_grad()
  core.set_has_grad(mode)
  local ok, a, b = pcall(fn)
  core.set_has_grad(prev)
  if not ok then
    error(a, 0)   -- 0 keeps the caller's frame (06-autograd.md sec.3.2)
  end
  return a, b
end

function M.no_grad(fn)
  return with_grad(false, fn)
end

function M.enable_grad(fn)
  return with_grad(true, fn)
end

-- ---------------------------------------------------------------------
-- set_grad_enabled(mode) / is_grad_enabled()
-- ---------------------------------------------------------------------
function M.set_grad_enabled(mode)
  core.set_has_grad(mode and true or false)
end

function M.is_grad_enabled()
  return core.has_grad()
end

return M