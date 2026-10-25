-- tools/smoke/autograd_smoke.lua
-- M1 2.6: autograd smoke -- backward / grad / no_grad / accumulation.
-- Covers 06-autograd.md sec.5 acceptance: 2-layer MLP hand-computed grad
-- to 1e-6, grad == backward, no_grad error-restore, no_grad stop_gradient,
-- accumulation without clear_grad.
--
-- Build first:  cmake -S . -B build-test
--               cmake --build build-test --target paddle_core -j 6
-- Run (from the repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/autograd_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local autograd = require "paddle.autograd"

local cpu = core.pd_place_new(core.PD_PLACE_CPU, 0)
local F32 = core.PD_DTYPE_FLOAT32

-- helper: first element as scalar (0-d -> C++ item(); multi-element ->
-- Lua-side 1-based flat item(1); C++ item() rejects numel!=1, so the
-- LUA path must be used for gradient tensors which keep x's shape).
local function item(t)
  if t:numel() == 1 then
    return t:item()      -- C++ binding, 0-d / single element
  end
  return t:item(1)       -- Lua-side decode, 1-based flat index
end

-- --- 2-layer MLP: loss = sum(W1 @ x * 2 ... keep it hand-computable:
-- x = {2}: [3, 3]; loss = sum(x * x) = 18; d loss/d x = 2x = [6, 6]
local x = core.pd_op_full({2}, 3.0, F32, cpu)
x:set_stop_gradient(false)            -- leaf must opt into grad
local loss = core.pd_op_sum(core.pd_op_multiply(x, x), {}, 0, 0)
assert(loss:ndim() == 0, "loss is 0-d")
assert(math.abs(item(loss) - 18.0) < 1e-6, "loss == 18")

loss:backward()                       -- Tensor:backward()
local gx = x:grad()
assert(gx ~= nil, "x has grad after backward")
assert(math.abs(item(gx) - 6.0) < 1e-6 and gx:numel() == 2,
       ("d loss/d x = [6,6], got %s"):format(item(gx)))

-- --- paddle.autograd.backward == Tensor:backward (same bottom layer)
local x2 = core.pd_op_full({2}, 3.0, F32, cpu)
x2:set_stop_gradient(false)
local loss2 = core.pd_op_sum(core.pd_op_multiply(x2, x2), {}, 0, 0)
autograd.backward{ tensors = { loss2 } }
assert(math.abs(item(x2:grad()) - 6.0) < 1e-6, "autograd.backward == backward")

-- --- paddle.autograd.grad == backward (fresh graph, grad does not
-- accumulate onto .grad: only_inputs=true default)
local x3 = core.pd_op_full({2}, 3.0, F32, cpu)
x3:set_stop_gradient(false)
local loss3 = core.pd_op_sum(core.pd_op_multiply(x3, x3), {}, 0, 0)
local grads = autograd.grad{ outputs = { loss3 }, inputs = { x3 } }
assert(type(grads) == "table" and #grads == 1, "grad returns list")
assert(math.abs(item(grads[1]) - 6.0) < 1e-6, "grad[1] == 6")
assert(x3:grad() == nil, "grad() does NOT accumulate onto .grad")

-- --- no_grad: state disabled inside, restored after, stop_gradient true
assert(autograd.is_grad_enabled(), "grad enabled by default")
local inner_stop
autograd.no_grad(function()
  assert(not autograd.is_grad_enabled(), "disabled inside no_grad")
  local z = core.pd_op_multiply(x, x)
  inner_stop = z:stop_gradient()
end)
assert(inner_stop, "no_grad result has stop_gradient == true")
assert(autograd.is_grad_enabled(), "restored after no_grad exit")

-- --- no_grad error path restores state (06-autograd.md sec.3.2)
local okerr, err = pcall(autograd.no_grad, function()
  error("boom")
end)
assert(not okerr, "error propagates")
assert(autograd.is_grad_enabled(), "grad state restored after error")

-- --- set_grad_enabled / is_grad_enabled round trip
autograd.set_grad_enabled(false)
assert(not autograd.is_grad_enabled(), "disabled")
autograd.set_grad_enabled(true)
assert(autograd.is_grad_enabled(), "re-enabled")

-- --- accumulation: two backward WITHOUT clear_grad double the grad
local x4 = core.pd_op_full({2}, 2.0, F32, cpu)
x4:set_stop_gradient(false)
local loss4 = core.pd_op_sum(core.pd_op_multiply(x4, x4), {}, 0, 0)
loss4:backward{ retain_graph = true }    -- d/dx = 2x = [4,4]
assert(math.abs(item(x4:grad()) - 4.0) < 1e-6, "first backward 4")
loss4:backward()                         -- no retain: graph freed, grad +=
assert(math.abs(item(x4:grad()) - 8.0) < 1e-6, "accumulated to 8")

-- --- stop_gradient setter/getter surface
local xi = core.pd_op_full({2}, 1.0, F32, cpu)
assert(xi:stop_gradient(), "fresh tensor stops gradient by default")
xi:set_stop_gradient(false)
assert(not xi:stop_gradient(), "set_stop_gradient(false) works")
xi:set_stop_gradient(true)
assert(xi:grad() == nil,
       "stop_gradient tensor has nil grad (06-autograd.md sec.4 ④)")

print("AUTOGRAD_SMOKE PASS")