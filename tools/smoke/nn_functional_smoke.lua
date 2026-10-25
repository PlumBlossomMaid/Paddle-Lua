-- tools/smoke/nn_functional_smoke.lua -- M1 2.9.3: paddle.nn.functional
-- activation batch. Mirrors Python test/legacy_test/test_activation_op.py
-- assertion LOGIC (input -> formula expectation -> numeric compare with
-- rtol=1e-5). Lua side computes the same formula the Python test uses
-- (np.maximum etc.) on fixed constant inputs; the paddle op output must
-- match. 用户纪律 2026-10-25: Python 有的测试 Lua 必须有同样逻辑。
--
-- Covered (16 activations, all argrule-wrapped):
--   relu / leaky_relu / silu / sigmoid / tanh / gelu / softmax /
--   log_softmax / softplus / hardswish / elu / celu / relu6 / hardtanh /
--   hardsigmoid / thresholded_relu / selu
-- Plus argrule gates: bad type errors, unknown-key errors, named calls.
--
-- NOT covered here (lands with the op generator, M2): in-place variants
-- (relu_, trailing `_` skipped by tools/gen/skip_rules.lua) and
-- dtype (softmax/log_softmax dtype arg needs a cast chain, listed in
-- nn-functional.md sec.7).
--
-- Run (from the repo root, identical env to nn_linear_smoke):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/nn_functional_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
assert(paddle.nn ~= nil and paddle.nn.functional ~= nil,
       "paddle.nn.functional exported")
local F = paddle.nn.functional

local F32 = core.PD_DTYPE_FLOAT32
local CPU = core.pd_place_new(core.PD_PLACE_CPU, 0)

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

local function close(a, b, msg)
  ntests = ntests + 1
  local err = math.abs(a - b) / math.max(1.0, math.abs(b))
  if not (err < 1e-5) then
    error(("assertion #%d failed: %s (got %.7g, want %.7g, rel %.2e)"
      ):format(ntests, msg, a, b, err), 2)
  end
end

-- Constant tensor helper (the Python tests feed random vectors; a constant
-- vector keeps expectations hand-computable while still exercising the
-- real formula, and this is the same rtol=1e-5 comparison).
local function cvec(n, v)
  return core.pd_op_full({n}, v, F32, CPU)
end

local function item(t, i)
  return t:item(i or 1)
end

-- ---------------------------------------------------------------------
-- 1: relu  -- max(x, 0), test_activation_op.py TestRelu
-- ---------------------------------------------------------------------
local x = cvec(3, -2.0)
local y = F.relu{ x }
close(item(y, 1), 0.0, "relu(-2) == 0")
close(item(y, 2), 0.0, "relu(-2) == 0")
local y2 = F.relu{ x = cvec(3, 3.0) }
close(item(y2, 1), 3.0, "relu(3) == 3")

-- ---------------------------------------------------------------------
-- 2: leaky_relu -- x>=0 ? x : negative_slope*x (Python default 0.01)
-- ---------------------------------------------------------------------
local y = F.leaky_relu{ x = cvec(3, 2.0) }
close(item(y, 1), 2.0, "leaky_relu(2) == 2")
local y = F.leaky_relu{ x = cvec(3, -2.0) }        -- default negative_slope=0.01
close(item(y, 1), -0.02, "leaky_relu(-2, 0.01) == -0.02")
local y = F.leaky_relu{ x = cvec(3, -2.0), negative_slope = 0.5 }
close(item(y, 1), -1.0, "leaky_relu(-2, 0.5) == -1")

-- ---------------------------------------------------------------------
-- 3: silu  -- x * sigmoid(x)
-- ---------------------------------------------------------------------
-- silu(2) = 2 * 1/(1+e^-2) = 2*0.880797 = 1.761594...
local y = F.silu{ cvec(3, 2.0) }
close(item(y, 1), 2.0 / (1.0 + math.exp(-2.0)), "silu(2) == 2*sigmoid(2)")

-- ---------------------------------------------------------------------
-- 4: sigmoid  -- 1/(1+e^-x)
-- ---------------------------------------------------------------------
local y = F.sigmoid{ cvec(3, 0.0) }
close(item(y, 1), 0.5, "sigmoid(0) == 0.5")
local y = F.sigmoid{ cvec(3, 1.0) }
close(item(y, 1), 1.0 / (1.0 + math.exp(-1.0)), "sigmoid(1) == 1/(1+e^-1)")

-- ---------------------------------------------------------------------
-- 5: tanh
-- ---------------------------------------------------------------------
local y = F.tanh{ cvec(3, 0.0) }
close(item(y, 1), 0.0, "tanh(0) == 0")
local y = F.tanh{ cvec(3, 1.0) }
close(item(y, 1), math.tanh(1.0), "tanh(1)")

-- ---------------------------------------------------------------------
-- 6: gelu -- 0.5*x*(1+erf(x/sqrt(2))), approximate=false (yaml default)
-- exact erf path: use the approximation variant for hand-compute parity,
-- exact value spot-checked via erf Taylor approx (abs err < 1e-6 here).
-- ---------------------------------------------------------------------
-- gelu(1.0) exact = 0.5*(1+erf(0.70710678)) = 0.84134475.. via erf(0.7071)
-- (scipy.special.erf, test_activation uses the same reference)
local function erf(x)
  -- Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
  local t = 1.0 / (1.0 + 0.3275911 * math.abs(x))
  local poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t
  local e = 1.0 - poly * math.exp(-x * x)
  if x < 0 then return -e end
  return e
end
local y = F.gelu{ cvec(3, 1.0) }
close(item(y, 1), 0.5 * (1.0 + erf(1.0 / math.sqrt(2.0))), "gelu(1) exact")
local y = F.gelu{ x = cvec(3, 1.0), approximate = true }
local t = math.tanh(math.sqrt(2.0 / math.pi) * (1.0 + 0.044715 * 1.0))
close(item(y, 1), 0.5 * 1.0 * (1.0 + t), "gelu(1) approximate")

-- ---------------------------------------------------------------------
-- 7: softmax -- exp(x_i)/sum exp over axis (default -1), 1-based axis OK
-- ---------------------------------------------------------------------
-- [1,1,1] on last axis -> each 1/3
local y = F.softmax{ cvec(3, 1.0) }
close(item(y, 1), 1.0 / 3.0, "softmax const == 1/3")
close(item(y, 2), 1.0 / 3.0, "softmax const == 1/3")

-- ---------------------------------------------------------------------
-- 8: log_softmax -- log(softmax): const vec -> log(1/3)
-- ---------------------------------------------------------------------
local y = F.log_softmax{ cvec(3, 1.0) }
close(item(y, 1), math.log(1.0 / 3.0), "log_softmax const == log(1/3)")

-- ---------------------------------------------------------------------
-- 9: softplus -- (1/beta)*log(1+e^(beta*x)), threshold=20 显式给 Python 契约
-- ---------------------------------------------------------------------
-- softplus(0) = ln(2)
local y = F.softplus{ cvec(3, 0.0) }
close(item(y, 1), math.log(2.0), "softplus(0) == ln2")
-- beta=2: (1/2)*ln(1+e^2) at x=1
local y = F.softplus{ x = cvec(3, 1.0), beta = 2.0 }
close(item(y, 1), 0.5 * math.log(1.0 + math.exp(2.0)), "softplus(1,beta=2)")

-- ---------------------------------------------------------------------
-- 10: hardswish -- x*min(max(x+3,0),6)/6
-- ---------------------------------------------------------------------
local y = F.hardswish{ cvec(3, 3.0) }   -- 3*(6)/6 = 3
close(item(y, 1), 3.0, "hardswish(3) == 3")
local y = F.hardswish{ cvec(3, -10.0) } -- clamp to 0
close(item(y, 1), 0.0, "hardswish(-10) == 0")
local y = F.hardswish{ cvec(3, -1.0) }  -- -1*2/6 = -1/3
close(item(y, 1), -1.0 / 3.0, "hardswish(-1) == -1/3")

-- ---------------------------------------------------------------------
-- 11: elu -- x>0 ? x : alpha*(e^x - 1)
-- ---------------------------------------------------------------------
local y = F.elu{ cvec(3, 2.0) }
close(item(y, 1), 2.0, "elu(2) == 2")
local y = F.elu{ cvec(3, -1.0) }
close(item(y, 1), math.exp(-1.0) - 1.0, "elu(-1) == e^-1 - 1")
local y = F.elu{ x = cvec(3, -1.0), alpha = 0.5 }
close(item(y, 1), 0.5 * (math.exp(-1.0) - 1.0), "elu(-1,alpha=0.5)")

-- ---------------------------------------------------------------------
-- 12: celu -- max(0,x) + min(0, alpha*(e^(x/alpha)-1))
-- ---------------------------------------------------------------------
-- x>0: second term min(0, positive) == 0, so celu == x
local y = F.celu{ cvec(3, 2.0) }
close(item(y, 1), 2.0, "celu(2) == 2")
local y = F.celu{ cvec(3, 0.5) }
close(item(y, 1), 0.5, "celu(0.5) == 0.5 (x>0)")
-- x<0: first term 0, second term alpha*(e^(x/alpha)-1) < 0
local y = F.celu{ x = cvec(3, -0.5), alpha = 0.5 }
close(item(y, 1), 0.5 * (math.exp(-1.0) - 1.0), "celu(-0.5,alpha=0.5)")

-- ---------------------------------------------------------------------
-- 13: relu6 -- min(max(x,0),6)
-- ---------------------------------------------------------------------
local y = F.relu6{ cvec(3, 2.0) }
close(item(y, 1), 2.0, "relu6(2) == 2")
local y = F.relu6{ cvec(3, 8.0) }
close(item(y, 1), 6.0, "relu6(8) == 6")
local y = F.relu6{ cvec(3, -1.0) }
close(item(y, 1), 0.0, "relu6(-1) == 0")

-- ---------------------------------------------------------------------
-- 14: hardtanh -- clamp to [min,max] (defaults -1, 1)
-- ---------------------------------------------------------------------
local y = F.hardtanh{ cvec(3, 0.5) }
close(item(y, 1), 0.5, "hardtanh(0.5) == 0.5")
local y = F.hardtanh{ cvec(3, 5.0) }
close(item(y, 1), 1.0, "hardtanh(5) == 1")
local y = F.hardtanh{ x = cvec(3, 5.0), min = -2.0, max = 2.0 }
close(item(y, 1), 2.0, "hardtanh(5,[-2,2]) == 2")

-- ---------------------------------------------------------------------
-- 15: hardsigmoid -- clamp(slope*x+offset, 0, 1) (defaults 1/6, 0.5)
-- ---------------------------------------------------------------------
local y = F.hardsigmoid{ cvec(3, 0.0) }
close(item(y, 1), 0.5, "hardsigmoid(0) == 0.5")
local y = F.hardsigmoid{ cvec(3, -10.0) }
close(item(y, 1), 0.0, "hardsigmoid(-10) == 0")
local y = F.hardsigmoid{ cvec(3, 10.0) }
close(item(y, 1), 1.0, "hardsigmoid(10) == 1")

-- ---------------------------------------------------------------------
-- 16: thresholded_relu -- x>threshold ? x : value (defaults 1.0, 0.0)
-- ---------------------------------------------------------------------
local y = F.thresholded_relu{ cvec(3, 2.0) }
close(item(y, 1), 2.0, "thresholded_relu(2) == 2")
local y = F.thresholded_relu{ cvec(3, 0.5) }
close(item(y, 1), 0.0, "thresholded_relu(0.5) == 0")

-- ---------------------------------------------------------------------
-- 17: selu -- scale*(x>0 ? x : alpha*(e^x-1)), defaults from activation.py
-- ---------------------------------------------------------------------
local y = F.selu{ cvec(3, 0.0) }
close(item(y, 1), 0.0, "selu(0) == 0")
local y = F.selu{ cvec(3, 2.0) }
close(item(y, 1), 1.0507009873554805 * 2.0, "selu(2) == scale*2")

-- ---------------------------------------------------------------------
-- 18: argrule gates (坏类型 / 未知键 / 命名调用)
-- ---------------------------------------------------------------------
local okk, err = pcall(function() F.relu{ x = 42 } end)
check(not okk, "relu{42} must raise (bad tensor type)")
local okk, err = pcall(function() F.softmax{ cvec(3, 1.0), bogus = 1 } end)
check(not okk, "softmax{ bogus } must raise (unknown key)")
check(err ~= nil and err:find("bogus") ~= nil,
       ("error mentions unknown key 'bogus', got: %s"):format(tostring(err)))

-- in-place variants land with the generator (M2): clear error, not silent
local okk, err = pcall(function() F.relu{ x = cvec(3, 1.0), inplace = true } end)
check(not okk, "relu{inplace=true} must raise (M2, skip_rules trailing _)")
check(err ~= nil and err:find("inplace") ~= nil,
       ("inplace error is explicit, got: %s"):format(tostring(err)))

print(("NN_FUNCTIONAL_SMOKE PASS (%d assertions)"):format(ntests))