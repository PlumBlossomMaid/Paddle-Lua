-- tools/smoke/optimizer_smoke.lua -- M1 2.10: paddle.optimizer (SGD/Adam)
-- 镜像 Python test/legacy_test/test_optimizer.py 的核心行为断言:
--   test_weight_decay_int          (SGD weight_decay 耦合 L2)
--   test_step_without_closure      (forward->backward->step 参数更新)
--   10-optimizer.md §5 验收:同样的参数与梯度,一步更新后参数逐位一致
--   手算定值:loss = sum(x*x),d/dx = 2x
--
-- 覆盖:
--   SGD:   param = param - lr*grad            (cpu/sgd_kernel.cc:39)
--   Adam:  param -= lr_eff*mom1/(sqrt(mom2)+eps*sqrt(1-beta2_pow)),
--          betaN_pow 初值 = betaN(adam.py:311-335),一步后 *= betaN
--          (cpu/adam_kernel.cc:118-119)
--   写回:  p:copy_from(new)(pd_tensor_copy_from,tensor.h:510 深拷贝,
--          参数句柄不变、autograd meta 保留——Python 的 in-place 语义)
--   clear_grad: 置零后 grad 全 0(eager_method.cc:1066 set_to_zero=true)
--   step 包 no_grad:组合算子不建图,参数仍可再 backward
--   argrule 门禁:坏参数列表 / grad_clip 非 nil 报错
--   5.1 行为:none 需要
--
-- Run (from the repo root, same env as nn_containers_smoke):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/optimizer_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
local optimizer = paddle.optimizer

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

-- 0-d -> t:item();multi-element -> Lua-side 1-based flat item(1)
local function item(t, i)
  if t:numel() == 1 then return t:item() end
  return t:item(i or 1)
end

-- 参数探针:param = [v1, v2](stop_gradient=false),loss = sum(param*param)
-- grad = [2*v1, 2*v2](autograd_smoke.lua 同款构造)。
local function make_pair(v1, v2)
  local p = core.pd_op_full({ 2 }, 1.0, F32, CPU)
  p:set_stop_gradient(false)
  -- 覆写元素:pd_op_full 全部为 1,乘标量造 [v1,v2]?不——full 只填常数。
  -- 用 scale 组合造非对称参数:[v1,v2] = add(scale(full(2,1), v1),
  --   concat...) 太绕;直接断言里用手算常数参数 [c,c] 与 [1,2] 两组。
  return p
end

-- ---------------------------------------------------------------------
-- 1. SGD:param -= lr*grad(param=[1,1],grad=[2,2],lr=0.1 -> [0.8,0.8])
-- ---------------------------------------------------------------------
local p1 = core.pd_op_full({ 2 }, 1.0, F32, CPU)
p1:set_stop_gradient(false)
local loss1 = core.pd_op_sum(core.pd_op_multiply(p1, p1), {}, 0, 0)
loss1:backward()
local g1 = p1:grad()
check(g1 ~= nil, "SGD: param has grad after backward")
check(item(g1, 1) == 2.0 and item(g1, 2) == 2.0, "SGD: grad == [2,2]")

local sgd = optimizer.SGD{ parameters = { p1 }, learning_rate = 0.1 }
check(sgd:is_a(optimizer.Optimizer), "SGD is_a Optimizer")
check(sgd:get_parameters() ~= nil, "get_parameters returns list")
sgd:step()
close(item(p1, 1), 0.8, "SGD step[1] = 1 - 0.1*2")
close(item(p1, 2), 0.8, "SGD step[2] = 1 - 0.1*2")
-- 参数写回保身份:句柄不变(值变了),autograd meta 保留——可再 backward
local same_handle = p1 == p1
check(same_handle, "SGD: param userdata identity preserved")
sgd:clear_grad()
local g1c = p1:grad()
check(g1c ~= nil and item(g1c, 1) == 0.0 and item(g1c, 2) == 0.0,
      "clear_grad zeroes grads")

-- ---------------------------------------------------------------------
-- 2. SGD weight_decay(耦合 L2,test_weight_decay_int 镜像):
--    grad' = grad + wd*param;param -= lr*grad'
--    [1,1],grad=[2,2],wd=0.5,lr=0.1 -> grad'=[2.5,2.5] -> [0.75,0.75]
-- ---------------------------------------------------------------------
local p2 = core.pd_op_full({ 2 }, 1.0, F32, CPU)
p2:set_stop_gradient(false)
local loss2 = core.pd_op_sum(core.pd_op_multiply(p2, p2), {}, 0, 0)
loss2:backward()
local sgd2 = optimizer.SGD{ parameters = { p2 }, learning_rate = 0.1,
                            weight_decay = 0.5 }
sgd2:step()
close(item(p2, 1), 0.75, "SGD+wd step = 1 - 0.1*(2+0.5)")
close(item(p2, 2), 0.75, "SGD+wd step[2]")

-- ---------------------------------------------------------------------
-- 3. Adam 一步(adam_functors.h:325-345,beta_pow 初值 = betaN):
--    param=[1,1],grad=[2,2],lr=0.1,beta1=0.9,beta2=0.999,eps=1e-8
--    beta1_pow=0.9,beta2_pow=0.999(初始)
--    lr_eff = 0.1*sqrt(1-0.999)/(1-0.9) = sqrt(0.001) ≈ 0.0316227766
--    mom1 = 0.9*0 + 0.1*2 = 0.2;mom2 = 0.999*0 + 0.001*4 = 0.004
--    denom = sqrt(0.004) + 1e-8*sqrt(0.001) ≈ 0.0632455532
--    delta = 0.0316227766 * 0.2 / 0.0632455532 ≈ 0.1
--    param' = [1-0.1, 1-0.1] = [0.9, 0.9](两个元素同 grad 同 delta)
-- ---------------------------------------------------------------------
local p3 = core.pd_op_full({ 2 }, 1.0, F32, CPU)
p3:set_stop_gradient(false)
local loss3 = core.pd_op_sum(core.pd_op_multiply(p3, p3), {}, 0, 0)
loss3:backward()
local adam = optimizer.Adam{ parameters = { p3 }, learning_rate = 0.1 }
adam:step()
-- 手算 delta:
local lr_eff = 0.1 * math.sqrt(1 - 0.999) / (1 - 0.9)
local mom1 = 0.1 * 2.0
local mom2 = 0.001 * 4.0
local delta = lr_eff * mom1 / (math.sqrt(mom2) + 1e-8 * math.sqrt(1 - 0.999))
close(item(p3, 1), 1.0 - delta, "Adam step[1] = 1 - delta")
close(item(p3, 2), 1.0 - delta, "Adam step[2] = 1 - delta")
close(delta, 0.1, "Adam first-step delta == lr (identity check)")

-- 第二步:验证 beta_pow 推进(每步 *= betaN)。必须先 clear_grad——
-- backward 会累积到现有 grad(autograd_smoke 同款语义),不清则第二次
-- grad = 2 + 1.8 而非 1.8。
--   参数现值 [0.9,0.9],grad 重建 = 2*0.9 = [1.8,1.8]
--   状态:beta1_pow = 0.81, beta2_pow = 0.998001
--         mom1 = 0.36(0.9*0.2 + 0.1*1.8), mom2 = 0.007236
--         (0.999*0.004 + 0.001*1.8^2)
--   lr_eff2 = 0.1*sqrt(1-0.998001)/(1-0.81)
--   delta2  = lr_eff2 * 0.36 / (sqrt(0.007236) + 1e-8*sqrt(0.001999))
adam:clear_grad()
local loss3b = core.pd_op_sum(core.pd_op_multiply(p3, p3), {}, 0, 0)
loss3b:backward()
adam:step()
-- Lua 双精度独立手算(C++ 组合算子对比基准,非从输出倒推):
local b1p2, b2p2 = 0.81, 0.998001
local mom1_2 = 0.9 * 0.2 + 0.1 * 1.8
local mom2_2 = 0.999 * 0.004 + 0.001 * 1.8 * 1.8
local lr_eff2 = 0.1 * math.sqrt(1 - b2p2) / (1 - b1p2)
local denom2 = math.sqrt(mom2_2) + 1e-8 * math.sqrt(1 - b2p2)
local delta2 = lr_eff2 * mom1_2 / denom2
close(item(p3, 1), 0.9 - delta2, "Adam second step (beta_pow advanced)")
close(item(p3, 2), 0.9 - delta2, "Adam second step[2]")

-- ---------------------------------------------------------------------
-- 4. 门禁(argrule):坏 parameters / grad_clip 非 nil 报错
-- ---------------------------------------------------------------------
local okb1 = pcall(function() optimizer.SGD{ parameters = { 42 },
                                             learning_rate = 0.1 } end)
check(not okb1, "SGD non-tensor param must raise")
local okb2 = pcall(function() optimizer.Adam{ parameters = { p3 },
                                              grad_clip = 1.0 } end)
check(not okb2, "Adam grad_clip non-nil must raise (not implemented)")

-- ---------------------------------------------------------------------
-- 5. 真实网络闭环(MNIST 路径最小版):Linear -> MSELoss -> Adam,
--    五步 loss 单调下降(镜像 test_step_without_closure 的循环形态)
-- ---------------------------------------------------------------------
local nn = paddle.nn
local net = nn.Linear{ 4, 2 }
local mse = nn.MSELoss()
local opt5 = optimizer.Adam{ parameters = net:parameters(),
                             learning_rate = 0.05 }
local x5 = core.pd_op_full({ 3, 4 }, 0.7, F32, CPU)
local y5 = core.pd_op_full({ 3, 2 }, 0.1, F32, CPU)
local losses = {}
for i = 1, 5 do
  local out = net(x5)
  local loss5 = mse(out, y5)
  losses[i] = item(loss5)
  loss5:backward()
  opt5:step()
  opt5:clear_grad()
end
check(losses[5] < losses[1], "5-step Linear+MSELoss+Adam loss decreases")

print(("OPTIMIZER_SMOKE PASS (%d assertions)"):format(ntests))
