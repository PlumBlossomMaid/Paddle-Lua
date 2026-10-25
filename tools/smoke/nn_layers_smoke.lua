-- tools/smoke/nn_layers_smoke.lua -- M1 2.9.4: paddle.nn 层类 batch
-- (MNIST 验收路径)。镜像 Python test_* 的断言逻辑:定值输入/权重 →
-- 手算期望 → rtol 数值比较。层类 forward 走 functional(已 2.9.3 验证)
-- 或直调 paddle_core;本文件验证的是「层壳正确性」:构造参数落位、
-- 参数注册、forward 转发参数语义(含 1-based 转换、data_format 透传)。
--
-- Covered:
--   Conv2D(数值:定值权重手算卷积 + bias reshape→add)
--   ReLU / MaxPool2D / Flatten / Dropout / BatchNorm2D / Embedding /
--   CrossEntropyLoss(数值:softmax 交叉熵手算)
--   argrule 门禁:坏类型 / 未知键 / 冲突参数报错
--
-- NOTE:pd_op_arange 的 sol 绑定要求三个 LuaTensor(start/end/step,
-- gen_ops_a.cc:305),smoke 直接用数字会 0xC0000005 —— 非等值张量一律
-- 用 pd_op_concat(gen_ops_c.cc:254)/pd_op_stack 拼接 full 构造。
--
-- Run (from the repo root, same env as nn_linear_smoke):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/nn_layers_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
local nn = paddle.nn

local F32 = core.PD_DTYPE_FLOAT32
local I64 = core.PD_DTYPE_INT64
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

local function item(t, i)
  return t:item(i or 1)
end

-- 定值 N 维张量(1-based 表形状)
local function ctensor(shape, v, dtype)
  return core.pd_op_full(shape, v, dtype or F32, CPU)
end

local function shape_of(t)
  return t:shape()
end

-- 非等值 1-D 张量:逐元素 full + concat(arange 需 LuaTensor,不可用)
local function cvec(vals, dtype)
  local parts = {}
  for _, v in ipairs(vals) do
    parts[#parts + 1] = core.pd_op_full({1}, v, dtype or F32, CPU)
  end
  if #parts == 1 then return parts[1] end
  return core.pd_op_concat(parts, 0)
end

-- ---------------------------------------------------------------------
-- 1: Conv2D 构造 + 参数注册
-- ---------------------------------------------------------------------
check(nn.Conv2D ~= nil and nn.ReLU ~= nil and nn.MaxPool2D ~= nil
  and nn.Flatten ~= nil and nn.Dropout ~= nil and nn.BatchNorm2D ~= nil
  and nn.CrossEntropyLoss ~= nil and nn.Embedding ~= nil,
  "all 8 layer classes exported")

local conv = nn.Conv2D{ 1, 1, 2 }  -- in=1,out=1,kernel=2
local wsh = conv.weight:shape()
check(wsh[1] == 1 and wsh[2] == 1 and wsh[3] == 2 and wsh[4] == 2,
  "Conv2D weight shape [1,1,2,2]")
check(conv.bias ~= nil and conv.bias:shape()[1] == 1, "Conv2D bias [1]")
check(conv.weight:stop_gradient() == false, "Conv2D weight trainable")
-- state_dict 键名与 Python 一致
local sd = conv:state_dict()
check(sd["weight"] ~= nil and sd["bias"] ~= nil, "Conv2D state_dict keys")

-- ---------------------------------------------------------------------
-- 2: Conv2D 数值 —— 定值权重,手算卷积
--   x = 全 1 [1,1,3,3];W = 全 1 [1,1,2,2];bias = 1.0;kernel=2,stride=1,
--   padding=0 → 输出 [1,1,2,2],每元素 = ΣW(4) + bias(1) = 5
-- ---------------------------------------------------------------------
conv:set_state_dict{
  weight = ctensor({1, 1, 2, 2}, 1.0),
  bias = ctensor({1}, 1.0),
}
local x2 = ctensor({1, 1, 3, 3}, 1.0)
local y2 = conv(x2)
local osh = shape_of(y2)
check(osh[1] == 1 and osh[2] == 1 and osh[3] == 2 and osh[4] == 2,
  "Conv2D output shape [1,1,2,2]")
close(item(y2, 1), 5.0, "conv sum(W)=4 + bias=1 -> 5")
close(item(y2, 4), 5.0, "conv last element also 5")

-- SAME padding 走 padding_algorithm
local conv_s = nn.Conv2D{ 1, 1, 2, padding = "SAME" }
check(conv_s ~= nil, "Conv2D padding='SAME' accepted")
local y_s = conv_s(ctensor({1, 1, 3, 3}, 1.0))
check(shape_of(y_s)[3] == 3 and shape_of(y_s)[4] == 3,
  "Conv2D SAME keeps spatial size")

-- ---------------------------------------------------------------------
-- 3: ReLU —— max(x,0)(forward 转发 functional.relu)
-- ---------------------------------------------------------------------
local relu = nn.ReLU{}
local y = relu(ctensor({3}, -2.0))
close(item(y, 1), 0.0, "ReLU(-2) == 0")
local y = relu(ctensor({3}, 3.0))
close(item(y, 1), 3.0, "ReLU(3) == 3")

-- ---------------------------------------------------------------------
-- 4: MaxPool2D —— 定值 4x4,kernel=2 stride=2 → 每 2x2 窗口取 max
--   [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] → [[6,8],[14,16]]
-- ---------------------------------------------------------------------
local mp = nn.MaxPool2D{ kernel_size = 2, stride = 2 }
local xm = core.pd_op_reshape(cvec{
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
}, { 1, 1, 4, 4 })
local ym = mp(xm)
local msh = shape_of(ym)
check(msh[1] == 1 and msh[2] == 1 and msh[3] == 2 and msh[4] == 2,
  "MaxPool2D output [1,1,2,2]")
close(item(ym, 1), 6.0, "max of [1,2;5,6] == 6")
close(item(ym, 2), 8.0, "max of [3,4;7,8] == 8")
close(item(ym, 3), 14.0, "max of [9,10;13,14] == 14")
close(item(ym, 4), 16.0, "max of [11,12;15,16] == 16")

-- ---------------------------------------------------------------------
-- 5: Flatten —— 默认 (start_axis=2 1-based) 展平 [N, C, H, W] → [N, C*H*W]
--   (1-based 第 2 维起 = C++ 0-based 1,与 Python Flatten() 默认一致)
-- ---------------------------------------------------------------------
local fl = nn.Flatten{}
local yf = fl(ctensor({2, 3, 4, 5}, 0.0))
local fsh = shape_of(yf)
check(fsh[1] == 2 and fsh[2] == 60, "Flatten [2,3,4,5] -> [2,60]")

-- ---------------------------------------------------------------------
-- 6: Dropout —— eval 恒等;training 时 upscale 语义(0 或 x/(1-p))
-- ---------------------------------------------------------------------
local dp = nn.Dropout{ p = 0.5 }
dp:eval()
local yd = dp(ctensor({4}, 2.0))
close(item(yd, 1), 2.0, "Dropout eval keeps input")
close(item(yd, 4), 2.0, "Dropout eval keeps input (last)")
dp:train()
local yt = dp(ctensor({4}, 2.0))
local tsh = shape_of(yt)
check(tsh[1] == 4, "Dropout train keeps shape")
-- upscale_in_train:out ∈ {0, x/(1-p)} = {0, 4}
for i = 1, 4 do
  local v = item(yt, i)
  check(v == 0.0 or math.abs(v - 4.0) < 1e-6,
    ("Dropout train value %g in {0,4}"):format(v))
end

-- ---------------------------------------------------------------------
-- 7: BatchNorm2D —— eval 用 running stats:x=1, mean=0, var=1, weight=1,
--   bias=0 → out = 1/sqrt(1+eps) ≈ 0.999995
-- ---------------------------------------------------------------------
local bn = nn.BatchNorm2D{ num_features = 2 }
bn:set_state_dict{
  weight = ctensor({2}, 1.0),
  bias = ctensor({2}, 0.0),
  _mean = ctensor({2}, 0.0),
  _variance = ctensor({2}, 1.0),
}
bn:eval()
local yb = bn(ctensor({1, 2, 2, 2}, 1.0))
close(item(yb, 1), 1.0 / math.sqrt(1.0 + 1e-5),
  "BatchNorm2D eval x=1 -> 1/sqrt(1+eps)")

-- ---------------------------------------------------------------------
-- 8: Embedding —— 查表;x 是 index 语义,1-based 输入 id(C ABI -1)
--   weight 行 0..2 = [0.1,0.2]/[1.1,1.2]/[2.1,2.2];x=[1,2](1-based)
--   → 行 0、1 → [[0.1,0.2],[1.1,1.2]]
-- ---------------------------------------------------------------------
local emb = nn.Embedding{ num_embeddings = 3, embedding_dim = 2 }
local wrows = {}
for _, rowvals in ipairs({ {0.1, 0.2}, {1.1, 1.2}, {2.1, 2.2} }) do
  wrows[#wrows + 1] = core.pd_op_reshape(cvec(rowvals), { 1, 2 })
end
emb:set_state_dict{
  weight = core.pd_op_concat(wrows, 0),
}
local xe = core.pd_op_reshape(cvec({1, 2}, I64), { 2 })
local ye = emb(xe)
close(item(ye, 1), 0.1, "embed id1 -> row0[0]=0.1")
close(item(ye, 2), 0.2, "embed id1 -> row0[1]=0.2")
close(item(ye, 3), 1.1, "embed id2 -> row1[0]=1.1")
close(item(ye, 4), 1.2, "embed id2 -> row1[1]=1.2")

-- ---------------------------------------------------------------------
-- 9: CrossEntropyLoss —— logits [1,3] = [1,2,3],label=2(1-based 第 2 类)
--   loss = logsumexp(logits) - logit[label-1] = ln(e+e^2+e^3) - 2
-- ---------------------------------------------------------------------
local ce = nn.CrossEntropyLoss{}
local xl = core.pd_op_reshape(cvec{1.0, 2.0, 3.0}, { 1, 3 })
local lb = ctensor({1}, 2.0, I64)  -- 1-based 类别 2
local yl = ce(xl, lb)
local ls = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0)) - 2.0
close(item(yl, 1), ls, "CrossEntropyLoss logsumexp - logit2")

-- ---------------------------------------------------------------------
-- 10: argrule 门禁 —— 坏类型 / 未知键 / 冲突参数报错
-- ---------------------------------------------------------------------
local okk, err = pcall(function() nn.Conv2D{ 1, 1, 2, groups = "x" } end)
check(not okk, "Conv2D groups='x' must raise")
local okk, err = pcall(function() nn.Conv2D{ 1, 1, 2, bogus = 1 } end)
check(not okk, "Conv2D unknown key must raise")
local okk, err = pcall(function() nn.BatchNorm2D{ 2, data_format = "BAD" } end)
check(not okk, "BatchNorm2D data_format='BAD' must raise")
local okk, err = pcall(function() nn.MaxPool2D{ kernel_size = "x" } end)
check(not okk, "MaxPool2D kernel_size='x' must raise")
local okk, err = pcall(function() nn.Flatten{ start_axis = "x" } end)
check(not okk, "Flatten start_axis='x' must raise")
-- p 校验在 F.dropout(common.py:1404):构造不校验,forward 时报错
local dpbad = nn.Dropout{ p = -0.5 }
local okk, err = pcall(function() dpbad(ctensor({4}, 1.0)) end)
check(not okk, "Dropout p=-0.5 forward must raise")
-- reduction 校验在 F.cross_entropy(loss.py:3196):构造不校验,forward 报错
local cebad = nn.CrossEntropyLoss{ reduction = "avg" }
local okk, err = pcall(function() cebad(ctensor({1, 3}, 1.0),
                                        ctensor({1}, 1.0, I64)) end)
check(not okk, "CrossEntropyLoss reduction='avg' forward must raise")

-- ---------------------------------------------------------------------
-- 11: 池化层类(2.9.4.2)—— 层壳 forward 转发 functional(2.9.3.3 已验
--   数值),本段验证构造参数落位 + 转发语义。镜像 Python test_pool2d
--   定值手算(与 NN_FUNCTIONAL_SMOKE 池化段同款期望)。
-- ---------------------------------------------------------------------
-- 导出齐全
check(nn.AvgPool1D ~= nil and nn.AvgPool2D ~= nil and nn.AvgPool3D ~= nil
  and nn.MaxPool1D ~= nil and nn.MaxPool3D ~= nil
  and nn.AdaptiveAvgPool1D ~= nil and nn.AdaptiveAvgPool2D ~= nil
  and nn.AdaptiveAvgPool3D ~= nil and nn.AdaptiveMaxPool1D ~= nil
  and nn.AdaptiveMaxPool2D ~= nil and nn.AdaptiveMaxPool3D ~= nil,
  "all 11 pooling layer classes exported")

-- AvgPool1D:全 1 [1,1,4] k2 s2 → [1,1,2] 全 1
local ap1 = nn.AvgPool1D{ kernel_size = 2, stride = 2 }
local y = ap1(ctensor({1, 1, 4}, 1.0))
local sh = shape_of(y)
check(sh[1] == 1 and sh[2] == 1 and sh[3] == 2, "AvgPool1D k2 s2 -> [1,1,2]")
check(math.abs(item(y, 1) - 1.0) < 1e-6, "AvgPool1D all-1 -> all-1")

-- AvgPool2D:全 1 [1,1,4,4] k2 s2 → [1,1,2,2] 全 1;divisor_override=2 → 2
local ap2 = nn.AvgPool2D{ kernel_size = 2, stride = 2 }
local y = ap2(ctensor({1, 1, 4, 4}, 1.0))
local sh = shape_of(y)
check(sh[3] == 2 and sh[4] == 2, "AvgPool2D k2 s2 -> [1,1,2,2]")
check(math.abs(item(y, 1) - 1.0) < 1e-6, "AvgPool2D all-1 -> all-1")
local ap2d = nn.AvgPool2D{ kernel_size = 2, stride = 2, divisor_override = 2 }
local y = ap2d(ctensor({1, 1, 4, 4}, 1.0))
check(math.abs(item(y, 1) - 2.0) < 1e-6,
  "AvgPool2D divisor_override=2 -> 2 (pooling.py:407)")

-- AvgPool3D:全 1 [1,1,2,2,2] k2 → [1,1,1,1,1] 全 1
local ap3 = nn.AvgPool3D{ kernel_size = 2 }
local y = ap3(ctensor({1, 1, 2, 2, 2}, 1.0))
local sh = shape_of(y)
check(sh[3] == 1 and sh[4] == 1 and sh[5] == 1, "AvgPool3D k2 -> [1,1,1,1,1]")
check(math.abs(item(y, 1) - 1.0) < 1e-6, "AvgPool3D all-1 -> all-1")

-- MaxPool1D:全 4 [1,1,4] k2 s2 → [1,1,2] 全 4
local mp1 = nn.MaxPool1D{ kernel_size = 2, stride = 2 }
local y = mp1(ctensor({1, 1, 4}, 4.0))
check(shape_of(y)[3] == 2 and math.abs(item(y, 1) - 4.0) < 1e-6,
  "MaxPool1D all-4 k2 s2 -> [1,1,2] all-4")

-- MaxPool3D:全 5 [1,1,2,2,2] k2 → [1,1,1,1,1] 全 5
local mp3 = nn.MaxPool3D{ kernel_size = 2 }
local y = mp3(ctensor({1, 1, 2, 2, 2}, 5.0))
local sh = shape_of(y)
check(sh[3] == 1 and sh[4] == 1 and sh[5] == 1, "MaxPool3D k2 -> [1,1,1,1,1]")
check(math.abs(item(y, 1) - 5.0) < 1e-6, "MaxPool3D all-5 -> all-5")

-- AdaptiveAvgPool1D/2D/3D:全 1 → 目标形状全 1
local aa1 = nn.AdaptiveAvgPool1D{ output_size = 2 }
local y = aa1(ctensor({1, 1, 4}, 1.0))
check(shape_of(y)[3] == 2, "AdaptiveAvgPool1D os=2 -> L=2")
local aa2 = nn.AdaptiveAvgPool2D{ output_size = 2 }
local y = aa2(ctensor({1, 1, 4, 4}, 1.0))
local sh = shape_of(y)
check(sh[3] == 2 and sh[4] == 2, "AdaptiveAvgPool2D os=2 -> [1,1,2,2]")
local aa3 = nn.AdaptiveAvgPool3D{ output_size = 2 }
local y = aa3(ctensor({1, 1, 4, 4, 4}, 1.0))
local sh = shape_of(y)
check(sh[3] == 2 and sh[4] == 2 and sh[5] == 2,
  "AdaptiveAvgPool3D os=2 -> [1,1,2,2,2]")

-- AdaptiveMaxPool1D/2D/3D:全 5 → 目标形状全 5;return_mask 双输出
local am1 = nn.AdaptiveMaxPool1D{ output_size = 2 }
local y = am1(ctensor({1, 1, 4}, 5.0))
check(shape_of(y)[3] == 2 and math.abs(item(y, 1) - 5.0) < 1e-6,
  "AdaptiveMaxPool1D os=2 all-5")
local am2 = nn.AdaptiveMaxPool2D{ output_size = 2 }
local y = am2(ctensor({1, 1, 4, 4}, 5.0))
local sh = shape_of(y)
check(sh[3] == 2 and sh[4] == 2, "AdaptiveMaxPool2D os=2 -> [1,1,2,2]")
local am3 = nn.AdaptiveMaxPool3D{ output_size = 2, return_mask = true }
local y, m = am3(ctensor({1, 1, 4, 4, 4}, 5.0))
local sh = shape_of(y)
check(sh[3] == 2 and sh[4] == 2 and sh[5] == 2,
  "AdaptiveMaxPool3D os=2 return_mask -> [1,1,2,2,2]")
check(m ~= nil and shape_of(m)[3] == 2, "AdaptiveMaxPool3D mask output")

-- 池化层类 argrule 门禁
local okk, err = pcall(function() nn.AvgPool1D{ kernel_size = "x" } end)
check(not okk, "AvgPool1D kernel_size='x' must raise")
local okk, err = pcall(function() nn.MaxPool3D{ kernel_size = 2, bogus = 1 } end)
check(not okk, "MaxPool3D unknown key must raise")
local okk, err = pcall(function() nn.AdaptiveAvgPool2D{ output_size = "x" } end)
check(not okk, "AdaptiveAvgPool2D output_size='x' must raise")

-- ---------------------------------------------------------------------
-- 12: 2.9.4.2b 其余层类 —— Conv 变体数值(定值手算)+ 构造注册 +
--    norm/激活/损失/Dropout2D/3D 导出与门禁。镜像 Python test_conv、
--    test_norm、test_loss 的逻辑;池化/norm 数值在 functional 已验,
--    层类只验层壳转发 + 构造落位。
-- ---------------------------------------------------------------------
check(nn.Conv1D ~= nil and nn.Conv3D ~= nil and nn.Conv1DTranspose ~= nil
  and nn.Conv2DTranspose ~= nil and nn.Conv3DTranspose ~= nil,
  "all 5 conv variants exported")

-- Conv1D:全 1 [1,1,4],W 全 1 [1,1,2],bias=0,k2 s1 → [1,1,3] 全 2
local c1 = nn.Conv1D{ in_channels = 1, out_channels = 1, kernel_size = 2 }
c1:set_state_dict{ weight = ctensor({1, 1, 2}, 1.0),
                   bias = ctensor({1}, 0.0) }
local y = c1(ctensor({1, 1, 4}, 1.0))
local sh = shape_of(y)
check(sh[1] == 1 and sh[2] == 1 and sh[3] == 3,
  "Conv1D k2 s1 -> [1,1,3]")
close(item(y, 1), 2.0, "Conv1D sum(W)=2 (NCL conv2d 模拟链)")

-- Conv3D:全 1 [1,1,2,2,2],W 全 1 [1,1,2,2,2],bias=0,k2 → [1,1,1,1,1] = 8
local c3 = nn.Conv3D{ in_channels = 1, out_channels = 1, kernel_size = 2 }
c3:set_state_dict{ weight = ctensor({1, 1, 2, 2, 2}, 1.0),
                   bias = ctensor({1}, 0.0) }
local y = c3(ctensor({1, 1, 2, 2, 2}, 1.0))
local sh = shape_of(y)
check(sh[1] == 1 and sh[2] == 1 and sh[5] == 1,
  "Conv3D k2 -> [1,1,1,1,1]")
close(item(y, 1), 8.0, "Conv3D sum(W)=8")

-- Conv2DTranspose:全 1 [1,1,2,2],W 全 1 [1,1,2,2],bias=0,k2 s1 p0
-- → 输出 [1,1,3,3],中心加权 = 4(重叠全 1)
local ct = nn.Conv2DTranspose{ in_channels = 1, out_channels = 1,
                               kernel_size = 2 }
ct:set_state_dict{ weight = ctensor({1, 1, 2, 2}, 1.0),
                   bias = ctensor({1}, 0.0) }
local y = ct(ctensor({1, 1, 2, 2}, 1.0))
local sh = shape_of(y)
check(sh[3] == 3 and sh[4] == 3, "Conv2DTranspose k2 -> [1,1,3,3]")
close(item(y, 5), 4.0, "Conv2DTranspose center overlaps 4")

-- LayerNorm:全 2 [1,4],normalized_shape 4 → 归一化后全 0
local ln = nn.LayerNorm{ normalized_shape = 4 }
local y = ln(ctensor({1, 4}, 2.0))
close(item(y, 1), 0.0, "LayerNorm all-2 -> 0")
close(item(y, 4), 0.0, "LayerNorm last -> 0")

-- GroupNorm:全 2 [1,4],num_groups=2 → 每组内归一化 → 全 0
local gn = nn.GroupNorm{ num_groups = 2, num_channels = 4 }
local y = gn(ctensor({1, 4}, 2.0))
close(item(y, 1), 0.0, "GroupNorm g2 all-2 -> 0")
close(item(y, 4), 0.0, "GroupNorm last -> 0")

-- BatchNorm1D/3D:eval 用 running stats,同 BatchNorm2D 逻辑
local bn1 = nn.BatchNorm1D{ num_features = 2 }
bn1:set_state_dict{ weight = ctensor({2}, 1.0), bias = ctensor({2}, 0.0),
                    _mean = ctensor({2}, 0.0),
                    _variance = ctensor({2}, 1.0) }
bn1:eval()
local y = bn1(ctensor({1, 2, 4}, 1.0))
close(item(y, 1), 1.0 / math.sqrt(1.0 + 1e-5), "BatchNorm1D eval x=1")
local bn3 = nn.BatchNorm3D{ num_features = 2 }
bn3:set_state_dict{ weight = ctensor({2}, 1.0), bias = ctensor({2}, 0.0),
                    _mean = ctensor({2}, 0.0),
                    _variance = ctensor({2}, 1.0) }
bn3:eval()
local y = bn3(ctensor({1, 2, 2, 2, 2}, 1.0))
close(item(y, 1), 1.0 / math.sqrt(1.0 + 1e-5), "BatchNorm3D eval x=1")

-- 激活层类批量:恒定公式核验(与 NN_FUNCTIONAL_SMOKE 同款期望)
close(item(nn.LeakyReLU{ negative_slope = 0.01 }(
  ctensor({2}, -2.0)), 1), -0.02, "LeakyReLU(-2) = -0.02")
close(item(nn.Hardswish{}(ctensor({2}, 1.0)), 1), 1.0 * (1.0 + 3.0) / 6.0,
  "Hardswish(1) = x(x+3)/6")
close(item(nn.Sigmoid{}(ctensor({2}, 0.0)), 1), 0.5, "Sigmoid(0) = 0.5")
close(item(nn.Softplus{}(ctensor({2}, 10.0)), 1), 10.0,
  "Softplus(10) ~ x (threshold 20)")
local sm = nn.Softmax{ axis = -1 }
local y = sm(ctensor({1, 3}, 1.0))
close(item(y, 1), math.exp(1.0) / (3.0 * math.exp(1.0)),
  "Softmax all-equal -> 1/3")

-- 损失层类:数值镜像 functional 已验逻辑
local mse = nn.MSELoss{}
close(item(mse(ctensor({2}, 2.0), ctensor({2}, 1.0)), 1), 1.0,
  "MSELoss(2,1) mean = 1")
local l1 = nn.L1Loss{}
close(item(l1(ctensor({2}, 2.0), ctensor({2}, 1.0)), 1), 1.0,
  "L1Loss(2,1) mean = 1")
local sml = nn.SmoothL1Loss{}
close(item(sml(ctensor({2}, 0.5), ctensor({2}, 0.0)), 1), 0.125,
  "SmoothL1Loss(|0.5|<1) = 0.5*0.25")
local bce = nn.BCELoss{}
close(item(bce(ctensor({2}, 0.5), ctensor({2}, 1.0)), 1), -math.log(0.5),
  "BCELoss(0.5,1) = -log0.5")
local bcewl = nn.BCEWithLogitsLoss{}
close(item(bcewl(ctensor({2}, 0.0), ctensor({2}, 1.0)), 1), math.log(2.0),
  "BCEWithLogits(0,1) = ln2")

-- Dropout2D/3D:eval 恒等,构造门禁
local d2 = nn.Dropout2D{ p = 0.5 }
d2:eval()
local y = d2(ctensor({1, 1, 2, 2}, 3.0))
close(item(y, 1), 3.0, "Dropout2D eval keeps input")
local d3 = nn.Dropout3D{ p = 0.5 }
d3:eval()
local y = d3(ctensor({1, 1, 2, 2, 2}, 3.0))
close(item(y, 1), 3.0, "Dropout3D eval keeps input")
local okk, err = pcall(function() nn.Dropout2D{ p = 0.5,
                                                 data_format = "BAD" } end)
check(not okk, "Dropout2D data_format='BAD' must raise")

-- 2.9.4.2b 构造门禁(坏类型)
local okk, err = pcall(function() nn.Conv1D{ 1, 1, 2, groups = "x" } end)
check(not okk, "Conv1D groups='x' must raise")
local okk, err = pcall(function() nn.GroupNorm{ groups = 1 } end)
check(not okk, "GroupNorm missing num_channels must raise")

print(("NN_LAYERS_SMOKE PASS (%d assertions)"):format(ntests))
