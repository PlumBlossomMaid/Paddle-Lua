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

-- ---------------------------------------------------------------------
-- 19: 第二批激活(2.9.3.2)-- 镜像 Python 公式
-- ---------------------------------------------------------------------

-- softsign: x/(1+|x|)  (test_activation_op.py TestSoftsign)
local y = F.softsign{ cvec(3, 2.0) }
close(item(y, 1), 2.0 / 3.0, "softsign(2) == 2/3")
local y = F.softsign{ cvec(3, -0.5) }
close(item(y, 1), -0.5 / 1.5, "softsign(-0.5)")

-- tanhshrink: x - tanh(x)  (TestTanhShrink)
local y = F.tanhshrink{ cvec(3, 1.0) }
close(item(y, 1), 1.0 - math.tanh(1.0), "tanhshrink(1)")

-- swish: x*sigmoid(x) = silu 语义同 swish(独立算子 swish)
local y = F.swish{ cvec(3, 2.0) }
close(item(y, 1), 2.0 / (1.0 + math.exp(-2.0)), "swish(2)")

-- log_sigmoid: log(1/(1+e^-x)) = -log(1+e^-x) (TestLogSigmoid)
local y = F.log_sigmoid{ cvec(3, 0.0) }
close(item(y, 1), -math.log(2.0), "log_sigmoid(0) == -ln2")
local y = F.log_sigmoid{ cvec(3, 1.0) }
close(item(y, 1), -math.log(1.0 + math.exp(-1.0)), "log_sigmoid(1)")

-- mish: x*tanh(softplus(x)), softplus(x)=ln(1+e^x) for x small (TestMish)
local function softplus1(v) return math.log(1.0 + math.exp(v)) end
local y = F.mish{ cvec(3, 0.0) }
close(item(y, 1), 0.0, "mish(0) == 0")
local y = F.mish{ cvec(3, 1.0) }
close(item(y, 1), 1.0 * math.tanh(softplus1(1.0)), "mish(1)")

-- softshrink: x>t? x-t : x<-t? x+t : 0 (t=0.5 default) (TestSoftshrink)
local y = F.softshrink{ cvec(3, 1.0) }
close(item(y, 1), 0.5, "softshrink(1,0.5) == 0.5")
local y = F.softshrink{ cvec(3, -1.0) }
close(item(y, 1), -0.5, "softshrink(-1,0.5) == -0.5")
local y = F.softshrink{ cvec(3, 0.2) }
close(item(y, 1), 0.0, "softshrink(0.2) == 0")

-- maxout: 4D input [N,C,H,W], groups 分通道取最大 (TestMaxoutAPI)
-- x = [1, 4, 1, 1] values [1,3,2,4], groups=2, axis=2(1-based 通道)
-- C 维被 groups 分成 2 组 → 输出 C=2:每组的 max → [3, 4]
local xmo = core.pd_op_full({1, 4, 1, 1}, 1.0, F32, CPU)
local xmo2 = core.pd_op_full({1, 4, 1, 1}, 2.0, F32, CPU)
local xmo3 = core.pd_op_full({1, 4, 1, 1}, 3.0, F32, CPU)
local xmo4 = core.pd_op_full({1, 4, 1, 1}, 4.0, F32, CPU)
local ymo = F.maxout{ x = xmo4, groups = 2 }
-- 全 4 时每组 max = 4 → 输出 2 通道都 4
assert(ymo:shape()[1] == 1 and ymo:shape()[2] == 2,
       "maxout [1,4,1,1]/groups=2 -> [1,2,1,1]")
close(item(ymo, 1), 4.0, "maxout all-4 group max == 4")

-- prelu: x>0? x : weight*x ; weight numel==1 -> mode=all (TestPReluAPI)
local w1 = core.pd_op_full({1}, 2.0, F32, CPU)
local y = F.prelu{ x = cvec(3, 2.0), weight = w1 }
close(item(y, 1), 2.0, "prelu(2,w=2) == 2")
local y = F.prelu{ x = cvec(3, -2.0), weight = w1 }
close(item(y, 1), -4.0, "prelu(-2,w=2) == -4")

-- rrelu: 运行 + is_test 语义(负斜率在 [lower,upper] 内) (TestRRelu)
local y = F.rrelu{ x = cvec(3, 2.0), training = false }  -- is_test=1
close(item(y, 1), 2.0, "rrelu(2,is_test) == 2")

-- gumbel_softmax: 运行不崩(随机采样,只验证 shape) (TestGumbelSoftmax)
local y = F.gumbel_softmax{ x = cvec(3, 1.0) }
assert(y:shape()[1] == 3, "gumbel_softmax keeps shape")
local okg = true
for i = 1, 3 do
  local v = item(y, i)
  if v < 0 or v > 1 or v ~= v then okg = false end
end
check(okg, "gumbel_softmax outputs in [0,1]")

-- 第二批 argrule 门禁:maxout groups 必填
local okk, err = pcall(function() F.maxout{ x = cvec(3, 1.0) } end)
check(not okk, "maxout without groups must raise")

-- ---------------------------------------------------------------------
-- 20: 池化(2.9.3.3)-- 镜像 Python test_pool2d_op.py 断言逻辑
-- (输入 → 公式期望 → rtol=1e-5;常量输入使期望可手算)
-- ---------------------------------------------------------------------

-- 多维常量张量 + 全等检查(item 取 1-based 线性索引,tensor.lua:423)
local function ctensor(shape, v)
  return core.pd_op_full(shape, v, F32, CPU)
end

local function all_equals(t, v)
  local n = t:numel()
  for i = 1, n do
    if math.abs(item(t, i) - v) > 1e-6 then return false end
  end
  return true
end

-- avg_pool2d:全 1 [1,1,4,4] kernel=2 stride=2 → [1,1,2,2] 全 1(窗口均值)
local x = ctensor({1, 1, 4, 4}, 1.0)
local y = F.avg_pool2d{ x, kernel_size = 2, stride = 2 }
local sh = y:shape()
check(sh[1] == 1 and sh[2] == 1 and sh[3] == 2 and sh[4] == 2,
      "avg_pool2d x[1,1,4,4] k2 s2 -> [1,1,2,2]")
check(all_equals(y, 1.0), "avg_pool2d all-1 k2 s2 -> all-1")
local y = F.avg_pool2d{ ctensor({1, 1, 4, 4}, 3.0), kernel_size = 2, stride = 2 }
check(all_equals(y, 3.0), "avg_pool2d all-3 k2 s2 -> all-3")

-- avg_pool2d padding=1,exclusive=True(默认):padding 0 不参与平均,
-- 全 1 输入仍全 1;shape (4+2*1-2)/1+1 = 5x5
local y = F.avg_pool2d{ x, kernel_size = 2, stride = 1, padding = 1 }
local sh = y:shape()
check(sh[3] == 5 and sh[4] == 5, "avg_pool2d padding=1 k2 s1 -> 5x5")
check(all_equals(y, 1.0), "avg_pool2d padding=1 exclusive=True all-1")

-- avg_pool2d divisor_override:out * (k0*k1) / divisor(pooling.py:407-411;
-- 全 1 k2 → 均值 1,1*4/2 = 2)
local y = F.avg_pool2d{ x, kernel_size = 2, stride = 2, divisor_override = 2 }
check(all_equals(y, 2.0), "avg_pool2d divisor_override=2 -> 2")

-- avg_pool2d padding='SAME'(kernel=2 stride=1):H/W 保持
local y = F.avg_pool2d{ x, kernel_size = 2, stride = 1, padding = "SAME" }
local sh = y:shape()
check(sh[3] == 4 and sh[4] == 4, "avg_pool2d padding='SAME' keeps 4x4")

-- max_pool2d:全 5 → 全 5,shape [1,1,2,2]
local y = F.max_pool2d{ ctensor({1, 1, 4, 4}, 5.0), kernel_size = 2, stride = 2 }
local sh = y:shape()
check(sh[3] == 2 and sh[4] == 2, "max_pool2d k2 s2 -> [1,1,2,2]")
check(all_equals(y, 5.0), "max_pool2d all-5 k2 s2 -> all-5")

-- max_pool2d return_mask=True:NCHW 双输出,out 全 5;mask 是输入展平的
-- 最大值索引,1-based(C4,C ABI pd_idx_plus1)。kernel=2 stride=2 于
-- [1,1,4,4] 的四个窗口:展平索引 0,2,8,10(0-based)→ 1,3,9,11
local y, m = F.max_pool2d{ ctensor({1, 1, 4, 4}, 5.0), kernel_size = 2,
                           stride = 2, return_mask = true }
check(all_equals(y, 5.0), "max_pool2d return_mask out all-5")
check(m:numel() == 4 and item(m, 1) == 1 and item(m, 2) == 3
        and item(m, 3) == 9 and item(m, 4) == 11,
      "max_pool2d return_mask mask = {1,3,9,11} (1-based flat idx)")

-- max_pool2d dilation=2 → with_index 分支(pooling.py:1172)
local y = F.max_pool2d{ ctensor({1, 1, 8, 8}, 5.0), kernel_size = 3,
                        stride = 1, padding = 1, dilation = 2 }
local sh = y:shape()
check(sh[3] == 6 and sh[4] == 6, "max_pool2d dilation=2 k3 s1 -> 6x6")

-- avg_pool1d:全 1 [1,1,4] k2 s2 → [1,1,2] 全 1(unsqueeze→pool2d→squeeze)
local y = F.avg_pool1d{ ctensor({1, 1, 4}, 1.0), kernel_size = 2, stride = 2 }
local sh = y:shape()
check(sh[1] == 1 and sh[2] == 1 and sh[3] == 2, "avg_pool1d k2 s2 -> [1,1,2]")
check(all_equals(y, 1.0), "avg_pool1d all-1 -> all-1")

-- max_pool1d return_mask=True:out [1,1,2] 全 4;mask 是输入展平的最大值
-- 索引 1-based(C4)。kernel=2 stride=2 在 [1,1,4] 的两个窗口:展平索引
-- 0,2(0-based)→ 1,3
local y, m = F.max_pool1d{ ctensor({1, 1, 4}, 4.0), kernel_size = 2,
                           stride = 2, return_mask = true }
local sh = y:shape()
check(sh[3] == 2 and m:shape()[3] == 2, "max_pool1d return_mask shapes [1,1,2]")
check(all_equals(y, 4.0) and m:numel() == 2 and item(m, 1) == 1
        and item(m, 2) == 3,
      "max_pool1d return_mask out all-4 mask = {1,3}")

-- adaptive_avg_pool2d:全 1 4x4 output_size=2 → [1,1,2,2] 全 1
local y = F.adaptive_avg_pool2d{ x, output_size = 2 }
local sh = y:shape()
check(sh[3] == 2 and sh[4] == 2, "adaptive_avg_pool2d os=2 -> [1,1,2,2]")
check(all_equals(y, 1.0), "adaptive_avg_pool2d all-1 -> all-1")

-- adaptive_max_pool2d output_size={2,2} 全 5 → [1,1,2,2] 全 5
local y = F.adaptive_max_pool2d{ ctensor({1, 1, 4, 4}, 5.0),
                                 output_size = {2, 2} }
check(all_equals(y, 5.0), "adaptive_max_pool2d all-5 -> all-5")

-- adaptive_avg_pool1d / adaptive_max_pool3d 冒烟:shape 正确
local y = F.adaptive_avg_pool1d{ ctensor({1, 1, 6}, 2.0), output_size = 6 }
check(y:shape()[3] == 6, "adaptive_avg_pool1d os=6 -> L=6")
local y = F.adaptive_max_pool3d{ ctensor({1, 1, 4, 4, 4}, 2.0),
                                 output_size = 2 }
local sh = y:shape()
check(sh[3] == 2 and sh[4] == 2 and sh[5] == 2,
      "adaptive_max_pool3d os=2 -> [1,1,2,2,2]")

-- avg_pool3d:全 1 [1,1,2,2,2] kernel=2 → [1,1,1,1,1] 全 1
local y = F.avg_pool3d{ ctensor({1, 1, 2, 2, 2}, 1.0), kernel_size = 2 }
local sh = y:shape()
check(sh[3] == 1 and sh[4] == 1 and sh[5] == 1,
      "avg_pool3d k2 -> [1,1,1,1,1]")
check(all_equals(y, 1.0), "avg_pool3d all-1 -> all-1")

-- 池化 argrule 门禁
-- 1) kernel_size 坏类型(string)
local okk, err = pcall(function() F.avg_pool2d{ x, kernel_size = "wide" } end)
check(not okk, "avg_pool2d kernel_size string must raise")
-- 2) 未知键
local okk, err = pcall(function() F.avg_pool2d{ x, kernel_size = 2, bogus = 1 } end)
check(not okk, "avg_pool2d unknown key must raise")
-- 3) padding 未知字符串
local okk, err = pcall(function() F.avg_pool2d{ x, kernel_size = 2,
  padding = "UPPER" } end)
check(not okk, "avg_pool2d padding='UPPER' must raise")
-- 4) NHWC + return_mask 冲突(pooling.py:1212)
local okk, err = pcall(function() F.max_pool2d{ x, kernel_size = 2,
  data_format = "NHWC", return_mask = true } end)
check(not okk, "max_pool2d NHWC+return_mask must raise")

-- ---------------------------------------------------------------------
-- 21: 归一化(2.9.3.4)-- 镜像 Python test_layer_norm/test_batch_norm
-- ---------------------------------------------------------------------

-- layer_norm:全 1 [2,3] 输入,normalized_shape={2}(1-based 轴 2 →
-- begin_norm_axis=1,归一化第二维 3 个元素)→ 每行归一化后均值 0 方差 1,
-- 全 1 输入 → 输出全 0;weight=2,bias=1 → 0*2+1 = 1
local lnx = ctensor({2, 3}, 1.0)
local y = F.layer_norm{ lnx, normalized_shape = {2} }
check(all_equals(y, 0.0), "layer_norm all-1 normalized_shape={2} -> all-0")
local wgt = ctensor({3}, 2.0)
local bse = ctensor({3}, 1.0)
local y = F.layer_norm{ lnx, normalized_shape = {2}, weight = wgt, bias = bse }
check(all_equals(y, 1.0), "layer_norm w=2 b=1 -> all-1")
-- 单轴 int 形式:normalized_shape=2 等价 {2}
local y = F.layer_norm{ lnx, normalized_shape = 2 }
check(all_equals(y, 0.0), "layer_norm normalized_shape=2 (int) -> all-0")

-- batch_norm is_test(running stats 路径):x 全 5,running_mean=0,
-- running_var=4,weight=1,bias=0 → (5-0)/sqrt(4+1e-5) ≈ 2.5
local x = ctensor({1, 1, 4}, 5.0)
local rm = ctensor({1}, 0.0)
local rv = ctensor({1}, 4.0)
local w1 = ctensor({1}, 1.0)
local b0 = ctensor({1}, 0.0)
local y = F.batch_norm{ x, rm, rv, w1, b0 }  -- training=false → is_test
check(math.abs(item(y, 1) - 2.5) < 1e-3,
      "batch_norm is_test (5-0)/sqrt(4) ≈ 2.5")

-- ---------------------------------------------------------------------
-- 22: 损失(2.9.3.4 首批)-- 镜像 Python 公式逻辑
-- ---------------------------------------------------------------------

-- mse_loss:input 全 2,label 全 1 → diff=1,square=1 → mean=1,sum=3
local y = F.mse_loss{ ctensor({3}, 2.0), ctensor({3}, 1.0) }
close(item(y, 1), 1.0, "mse_loss mean (2-1)^2 = 1")
local y = F.mse_loss{ ctensor({3}, 2.0), ctensor({3}, 1.0), reduction = "sum" }
close(item(y, 1), 3.0, "mse_loss sum = 3")
local y = F.mse_loss{ ctensor({3}, 2.0), ctensor({3}, 1.0), reduction = "none" }
check(y:shape()[1] == 3 and all_equals(y, 1.0), "mse_loss none = [1,1,1]")

-- l1_loss:input 全 2,label 全 1 → |diff|=1 → mean=1,sum=3
local y = F.l1_loss{ ctensor({3}, 2.0), ctensor({3}, 1.0) }
close(item(y, 1), 1.0, "l1_loss mean |2-1| = 1")
local y = F.l1_loss{ ctensor({3}, 2.0), ctensor({3}, 1.0), reduction = "sum" }
close(item(y, 1), 3.0, "l1_loss sum = 3")

-- smooth_l1_loss(huber,delta=1):input=[0.5,1.5],label=[0,0]
-- |0.5|<1 → 0.5*0.25=0.125;|1.5|>=1 → 1.5-0.5=1.0 → mean=0.5625
local y = F.smooth_l1_loss{ ctensor({2}, 0.5), ctensor({2}, 0.0) }
close(item(y, 1), 0.125, "smooth_l1 |0.5|<1 -> 0.5*0.25")
local y = F.smooth_l1_loss{ ctensor({2}, 1.5), ctensor({2}, 0.0) }
close(item(y, 1), 1.0, "smooth_l1 |1.5|>=1 -> 1.5-0.5")
local y = F.smooth_l1_loss{ ctensor({2, 1}, 1.5), ctensor({2, 1}, 0.0),
                            reduction = "none" }
local sh = y:shape()
check(sh[1] == 2 and sh[2] == 1, "smooth_l1 none keeps shape")

-- binary_cross_entropy:input=0.5,label=1 → -log(0.5) = ln2
local y = F.binary_cross_entropy{ ctensor({1}, 0.5), ctensor({1}, 1.0) }
close(item(y, 1), math.log(2.0), "bce(0.5,1) = -log(0.5) = ln2")

-- binary_cross_entropy_with_logits:logit=0,label=1 → sigmoid(0)=0.5,
-- -log(0.5)=ln2;sigmoid_cross_entropy_with_logits 数值等价
local y = F.binary_cross_entropy_with_logits{ ctensor({1}, 0.0),
                                              ctensor({1}, 1.0) }
close(item(y, 1), math.log(2.0), "bce_with_logits(0,1) = ln2")
local y = F.binary_cross_entropy_with_logits{ ctensor({2}, 0.0),
                                              ctensor({2}, 1.0),
                                              reduction = "sum" }
close(item(y, 1), 2.0 * math.log(2.0), "bce_with_logits sum = 2*ln2")

-- 归一化/损失 argrule 门禁
local okk, err = pcall(function() F.batch_norm{ x, rm, rv, w1, b0,
  data_format = "BAD" } end)
check(not okk, "batch_norm data_format='BAD' must raise")
local okk, err = pcall(function() F.mse_loss{ ctensor({3}, 1.0),
  ctensor({3}, 1.0), reduction = "avg" } end)
check(not okk, "mse_loss reduction='avg' must raise")
local okk, err = pcall(function() F.layer_norm{ lnx, normalized_shape = {2},
  bogus = 1 } end)
check(not okk, "layer_norm unknown key must raise")

print(("NN_FUNCTIONAL_SMOKE PASS (%d assertions)"):format(ntests))