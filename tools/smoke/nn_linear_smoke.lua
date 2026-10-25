-- tools/smoke/nn_linear_smoke.lua -- M1 2.9.2: paddle.nn.Linear end to end.
-- Covers 09-nn.md sec.5 acceptance subset for the FIRST real layer:
--   - auto-registration of weight/bias parameters
--   - state_dict keys == "weight" / "bias" (byte-identical to Python)
--   - forward numerics: Paddle Linear is y = xW + b (common.py:121
--     Out = XW + b, weight shape [in, out])
--   - no-bias branch (bias_attr=false) falls back to matmul
--   - parameters are trainable (stop_gradient=false) and backprop
--     flows into them (grad_w == x^T, grad_b == 1)
--
-- Build first:  cmake -S . -B build-test
--               cmake --build build-test --target paddle_core -j 6
-- Run (from the repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/nn_linear_smoke.lua
--   (runtime dirs + PATH gotcha: see tasks.md T-M1-11)
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
assert(paddle.nn ~= nil and paddle.nn.Linear ~= nil,
       "paddle.nn.Linear exported")

local F32 = core.PD_DTYPE_FLOAT32

local function item(t)
  if t:numel() == 1 then return t:item() end
  return t:item(1)
end

-- --- 1: creation + auto-registration -----------------------------------
local lin = paddle.nn.Linear{ 2, 3 }
assert(lin ~= nil, "Linear created")
assert(lin.weight ~= nil and lin.bias ~= nil,
       "weight and bias auto-registered")
local wshape = lin.weight:shape()
assert(wshape[1] == 2 and wshape[2] == 3, "weight shape [2,3]")
assert(lin.bias:shape()[1] == 3, "bias shape [3]")
assert(lin.weight:stop_gradient() == false, "weight is trainable")
assert(lin.bias:stop_gradient() == false, "bias is trainable")

-- --- 2: named_parameters / state_dict keys match Python ----------------
local names = {}
for name in lin:named_parameters() do names[#names + 1] = name end
assert(names[1] == "weight" and names[2] == "bias",
       ("params == weight,bias got %s"):format(table.concat(names, ",")))
local sd = lin:state_dict()
assert(sd["weight"] ~= nil and sd["bias"] ~= nil,
       "state_dict has weight and bias")

-- --- 3: zero-init forward -> y == 0 (default Constant(0) fill) ---------
local x0 = core.pd_op_full({1, 2}, 5.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0))
local y0 = lin(x0)
assert(y0:shape()[1] == 1 and y0:shape()[2] == 3, "output [1,3]")
assert(math.abs(item(y0) - 0.0) < 1e-6,
       "default zero weights -> y == 0 (got " .. item(y0) .. ")")

-- --- 4: no-bias branch (bias_attr=false) -> matmul, real numerics ------
-- NOTE: {2, 3, nil, false} would be a HOLY table -- the array part
-- stops at the first nil, so argrule's table_array_len can never see the
-- trailing false. Use the mixed named form (api/README.md sec.2.1.5).
local lin2 = paddle.nn.Linear{ 2, 3, bias_attr = false }
assert(lin2.bias == nil, "bias_attr=false -> no bias param")
local n2 = 0
for _ in lin2:named_parameters() do n2 = n2 + 1 end
assert(n2 == 1, ("no-bias Linear has 1 param, got %d"):format(n2))
-- W2 = all 2 (2x3); x = [1,1] -> y_j = 1*2 + 1*2 = 4 -> [4,4,4]
local W2 = core.pd_op_full({2, 3}, 2.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0))
lin2:set_state_dict{ weight = W2 }
local x2 = core.pd_op_full({1, 2}, 1.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0))
local y2 = lin2(x2)
assert(y2:shape()[1] == 1 and y2:shape()[2] == 3, "output [1,3]")
assert(math.abs(item(y2) - 4.0) < 1e-6,
       ("no-bias y == [4,4,4], got %s"):format(item(y2)))

-- --- 5: with-bias numerics (y = xW + b via pd_op_linear_v2) ------------
local lin3 = paddle.nn.Linear{ 2, 3 }
lin3:set_state_dict{
  weight = core.pd_op_full({2, 3}, 2.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0)),
  bias   = core.pd_op_full({3}, 1.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0)),
}
local x3 = core.pd_op_full({1, 2}, 1.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0))
local y3 = lin3(x3)
-- y_j = (1*2 + 1*2) + 1 = 5 -> [5,5,5]
assert(math.abs(item(y3) - 5.0) < 1e-6,
       ("with-bias y == [5,5,5], got %s"):format(item(y3)))

-- --- 6: trainable parameters actually backprop -------------------------
local lin4 = paddle.nn.Linear{ 2, 1 }   -- weight [2,1], bias [1]
lin4:set_state_dict{
  weight = core.pd_op_full({2, 1}, 2.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0)),
  bias   = core.pd_op_full({1}, 1.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0)),
}
local x4 = core.pd_op_full({1, 2}, 3.0, F32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0))
local y4 = lin4(x4)
-- y = (3*2 + 3*2) + 1 = 13
assert(math.abs(item(y4) - 13.0) < 1e-6,
       ("forward y = xW+b = 13, got %s"):format(item(y4)))
y4:backward()
local gw = lin4.weight:grad()
local gb = lin4.bias:grad()
assert(gw ~= nil, "weight got grad")
assert(gb ~= nil, "bias got grad")
-- dy/dw_ij = x_i -> for x=[3,3], grad_w = [3,3]; dy/db = 1
assert(math.abs(gw:item(1) - 3.0) < 1e-6, "grad_w[1] == 3")
assert(math.abs(gw:item(2) - 3.0) < 1e-6, "grad_w[2] == 3")
assert(math.abs(gb:item(1) - 1.0) < 1e-6, "grad_b == 1")

print("NN_LINEAR_SMOKE PASS")
