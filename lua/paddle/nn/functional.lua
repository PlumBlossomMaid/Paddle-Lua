-- lua/paddle/nn/functional.lua -- paddle.nn.functional
-- Stage: P9 / M1 (WORKPLAN 2.9.3). Functional API: 参数整理 + 调一个算子
-- (09-nn.md sec.1.4 关键观察). 用户 2026-10-25: 几乎所有正式 API 用
-- argrule 包装,不手写解析器;API 完成必须补 CI 测试,镜像 Python 同逻辑。
--
-- Blueprint: python/paddle/nn/functional/{activation,common}.py.
-- Index semantics (C4): axis 等全部 1-based,负数不变(见 index_semantics)。
--
-- ★ 为什么 functional 层要显式传默认值(而不是依赖 _ops 生成物):
--   生成物的默认值来自 yaml,与 Python 契约不一致,实测三例:
--     leaky_relu  yaml 0.02 (ops.yaml:3185) / Python 0.01 (activation.py:475)
--                / _ops 生成物默认 0 (_ops/leaky_relu.lua)
--     softplus   yaml 20.0 (ops.yaml:5238) / _ops 生成物默认 2.0
--                (_ops/softplus.lua) -- 生成物默认值错误
--   argrule 规则表的 default 显式传值,生成物永远收不到 nil,不会触发
--   生成物默认。规则表是唯一契约来源。

local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"

-- Shared LuaTensor usertype metatable (tensor.lua augments the same one).
-- Structural tensor test, same pattern as nn/layer.lua.
local probe = paddle_core.tensor{1}
local TENSOR_MT = getmetatable(probe)
probe = nil
local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end
local Tensor = argrule.alias { function(v) return is_tensor(v) end }

local M = {}

-- in-place 变体随生成器(M2,skip_rules.lua trailing `_`),不手写(C8)。
-- 规则表仍带 inplace 参数保持与 Python 签名一致,true 时报清晰错误。
local function no_inplace_yet(fname)
  return function()
    error("paddle.nn.functional." .. fname
          .. ": inplace=true needs the in-place op (trailing `_`), "
          .. "lands with the op generator (M2, skip_rules.lua)", 2)
  end
end

-- ---------------------------------------------------------------------
-- 激活函数(第一批:签名与 Python 契约逐字核对过的核心激活)
-- ---------------------------------------------------------------------

-- relu{x, inplace} -- activation.py:803-836 (_C_ops.relu/relu_ 双路)
M.relu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, inplace)
  if inplace then return no_inplace_yet "relu"() end
  return paddle_core.pd_op_relu(x)   -- ops_gen.h:411
end)

-- leaky_relu{x, negative_slope, inplace} -- activation.py:472-518
M.leaky_relu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "negative_slope", type = argrule.number, opt = true, default = 0.01,
    doc = "slope for x < 0 (Python default 0.01, activation.py:475)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, negative_slope, inplace)
  if inplace then return no_inplace_yet "leaky_relu"() end
  return paddle_core.pd_op_leaky_relu(x, negative_slope)  -- ops_gen.h:282
end)

-- silu{x, inplace} -- activation.py:1121 (_C_ops.silu)
M.silu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, inplace)
  if inplace then return no_inplace_yet "silu"() end
  return paddle_core.pd_op_silu(x)   -- ops_gen.h:452
end)

-- sigmoid{x} -- activation.py:36 re-export from tensor/math (_C_ops.sigmoid)
M.sigmoid = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
}(function(x)
  return paddle_core.pd_op_sigmoid(x)  -- ops_gen.h:449
end)

-- tanh{x} -- activation.py:36 re-export from tensor/math (_C_ops.tanh)
M.tanh = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
}(function(x)
  return paddle_core.pd_op_tanh(x)     -- ops_gen.h:488
end)

-- gelu{x, approximate} -- activation.py:40 re-export from _C_ops;
-- yaml default approximate=false (ops.yaml:2579)
M.gelu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "approximate", type = argrule.boolean, opt = true, default = false,
    doc = "use tanh approximation (yaml default false, ops.yaml:2579)" },
}(function(x, approximate)
  return paddle_core.pd_op_gelu(x, (approximate and 1 or 0))  -- ops_gen.h:230
end)

-- softmax{x, axis, dtype} -- activation.py:1191-1378;axis 1-based(C4),
-- 负数不变;dtype 首批不支持(需 cast 算子链,见 nn-functional.md §7)。
-- ★ index 语义参数转发 `_ops` 生成物(它做 1-based->0-based 转换,
--   见 emit_lua 的 index_semantics;直调 pd_op 会漏转换,实测报
--   "axis only supported 1, -1 or 3" 类错误)。
M.softmax = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "softmax axis, 1-based, negative counts from the end (C4)" },
}(function(x, axis)
  local wrap = require("paddle._ops.softmax")
  return wrap.softmax(x, axis)
end)

-- log_softmax{x, axis} -- activation.py:1729 (_C_ops.log_softmax)
M.log_softmax = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "log-softmax axis, 1-based, negative from the end (C4)" },
}(function(x, axis)
  -- index 语义参数一律走生成物(1-based->0-based 转换在 _ops 层)
  local wrap = require("paddle._ops.log_softmax")
  return wrap.log_softmax(x, axis)
end)

-- softplus{x, beta, threshold} -- yaml (ops.yaml:5238): beta=1.0,
-- threshold=20.0。Python 从 _C_ops re-export(activation.py:41),默认即
-- yaml 默认。★ 生成物默认 threshold=2.0 是错的,规则表显式给 20.0。
M.softplus = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "beta", type = argrule.number, opt = true, default = 1.0,
    doc = "softplus beta (yaml default, ops.yaml:5238)" },
  { name = "threshold", type = argrule.number, opt = true, default = 20.0,
    doc = "softplus threshold (yaml default 20.0, NOT the _ops default 2.0)" },
}(function(x, beta, threshold)
  return paddle_core.pd_op_softplus(x, beta, threshold)  -- ops_gen.h:461
end)

-- hardswish{x} -- activation.py:400 (_C_ops.hardswish, 无参数)
M.hardswish = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
}(function(x)
  return paddle_core.pd_op_hardswish(x)  -- ops_gen.h:243
end)

-- elu{x, alpha, inplace} -- activation.py:114 (alpha=1.0)
M.elu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "alpha", type = argrule.number, opt = true, default = 1.0,
    doc = "ELU alpha (activation.py:117)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, alpha, inplace)
  if inplace then return no_inplace_yet "elu"() end
  return paddle_core.pd_op_elu(x, alpha)  -- ops_gen.h:154
end)

-- celu{x, alpha, inplace} -- activation.py:45 (alpha=1.0)
M.celu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "alpha", type = argrule.number, opt = true, default = 1.0,
    doc = "CELU alpha (activation.py:48)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, alpha, inplace)
  if inplace then return no_inplace_yet "celu"() end
  return paddle_core.pd_op_celu(x, alpha)  -- ops_gen.h:87
end)

-- relu6{x, inplace} -- activation.py:991
M.relu6 = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, inplace)
  if inplace then return no_inplace_yet "relu6"() end
  return paddle_core.pd_op_relu6(x)  -- ops_gen.h:412
end)

-- hardtanh{x, min, max} -- activation.py:257 (min=-1.0, max=1.0)
M.hardtanh = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "min", type = argrule.number, opt = true, default = -1.0,
    doc = "lower clamp (activation.py:258)" },
  { name = "max", type = argrule.number, opt = true, default = 1.0,
    doc = "upper clamp (activation.py:259)" },
}(function(x, min, max)
  return paddle_core.pd_op_hardtanh(x, min, max)  -- ops_gen.h:244
end)

-- hardsigmoid{x, slope, offset, inplace} -- activation.py:325
M.hardsigmoid = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "slope", type = argrule.number, opt = true, default = 0.1666667,
    doc = "hardsigmoid slope (activation.py:328)" },
  { name = "offset", type = argrule.number, opt = true, default = 0.5,
    doc = "hardsigmoid offset (activation.py:329)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, slope, offset, inplace)
  if inplace then return no_inplace_yet "hardsigmoid"() end
  return paddle_core.pd_op_hardsigmoid(x, slope, offset)  -- ops_gen.h:242
end)

-- thresholded_relu{x, threshold, value} -- activation.py:1651
M.thresholded_relu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "threshold", type = argrule.number, opt = true, default = 1.0,
    doc = "activation threshold (activation.py:1654)" },
  { name = "value", type = argrule.number, opt = true, default = 0.0,
    doc = "output value below threshold (activation.py:1655)" },
}(function(x, threshold, value)
  return paddle_core.pd_op_thresholded_relu(x, threshold, value)
  -- ops_gen.h:493
end)

-- selu{x, scale, alpha, inplace} -- activation.py:1038
M.selu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "scale", type = argrule.number, opt = true,
    default = 1.0507009873554805,
    doc = "selu scale (activation.py:1041)" },
  { name = "alpha", type = argrule.number, opt = true,
    default = 1.6732632423543772,
    doc = "selu alpha (activation.py:1042)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, scale, alpha, inplace)
  if inplace then return no_inplace_yet "selu"() end
  return paddle_core.pd_op_selu(x, scale, alpha)  -- ops_gen.h:433
end)

-- ---------------------------------------------------------------------
-- 第二批(2026-10-25):纯转发/单算子,Python 契约核对过的剩余激活。
-- glu/swiglu(y=nil 分支)是组合算子(chunk 未生成,v1 延后,见 tasks)。
-- ---------------------------------------------------------------------

-- softsign{x} -- activation.py:1459 (_C_ops.softsign)
M.softsign = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
}(function(x)
  return paddle_core.pd_op_softsign(x)  -- ops_gen.h:463
end)

-- tanhshrink{x} -- activation.py:1610 (_C_ops.tanh_shrink,算子名带下划线
-- 但 API 名是 tanhshrink -- 命名铁律:照搬 Python 函数名)
M.tanhshrink = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
}(function(x)
  return paddle_core.pd_op_tanh_shrink(x)  -- ops_gen.h:489
end)

-- swish{x, inplace} -- activation.py:1498 (silu 的另一种缩放;独立算子 swish)
M.swish = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, inplace)
  if inplace then return no_inplace_yet "swish"() end
  return paddle_core.pd_op_swish(x)  -- ops_gen.h:483
end)

-- log_sigmoid{x} -- activation.py:858 (_C_ops.logsigmoid,算子名无下划线)
M.log_sigmoid = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
}(function(x)
  return paddle_core.pd_op_logsigmoid(x)  -- gen_ops_l.cc:249
end)

-- mish{x, inplace} -- activation.py:1554;内部固定 lambda=20
-- (activation.py:1599 `_C_ops.mish(x, 20)`);yaml lambda 无默认
-- (ops.yaml:3852)→ functional 显式传 20,不暴露给用户。
M.mish = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, inplace)
  if inplace then return no_inplace_yet "mish"() end
  return paddle_core.pd_op_mish(x, 20.0)  -- ops_gen.h:342;lambda=20 fix
end)

-- softshrink{x, threshold} -- activation.py:1398 (threshold=0.5);
-- ★ 生成物默认 threshold=0 是错的(yaml 0.5 ops.yaml:5253)→ 显式 0.5
M.softshrink = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "threshold", type = argrule.number, opt = true, default = 0.5,
    doc = "shrink threshold, Python default 0.5 (activation.py:1399)" },
}(function(x, threshold)
  return paddle_core.pd_op_softshrink(x, threshold)  -- ops_gen.h:462
end)

-- maxout{x, groups, axis} -- activation.py:904 (groups 必选, axis 默认 1);
-- yaml axis 默认 1(ops.yaml:3721,0-based)→ Lua 1-based 默认 2;
-- index 语义(C4),生成物负责 1-based->0-based(显式传 2)。
M.maxout = rule {
  { name = "x", type = Tensor, doc = "input 4-D tensor [N,C,H,W]" },
  { name = "groups", type = argrule.number, doc = "channel groups (required)" },
  { name = "axis", type = argrule.number, opt = true, default = 2,
    doc = "channel axis, 1-based; Python default 1 (activation.py:905)" },
}(function(x, groups, axis)
  -- axis 是 index 语义:必须走生成物做 1-based->0-based(maxout 只
  -- 接受 1/-1/3,实测直调 pd_op 传 2 报错 unary.cc:2825)
  local wrap = require("paddle._ops.maxout")
  return wrap.maxout(x, groups, axis)
end)

-- prelu{x, weight, data_format} -- activation.py:548;Python 由 weight
-- numel 推导 mode:numel==1 → "all",否则 "channel"(activation.py:608-637);
-- data_format 默认 "NCHW"。★ 生成物默认 data_format/mode 带引号
-- ("\"NCHW\"" 生成器 bug)→ functional 显式传,绝不经过生成物默认。
M.prelu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "weight", type = Tensor,
    doc = "learnable slope; numel==1 -> 'all', else per-channel" },
  { name = "data_format", type = argrule.string, opt = true,
    default = "NCHW", doc = "NC/NCL/NCHW/NCDHW/NLC/NHWC/NDHWC" },
}(function(x, weight, data_format)
  local mode = "all"
  if weight:numel() > 1 then
    mode = "channel"
    local fmt = data_format
    if fmt == "NCL" or fmt == "NCHW" or fmt == "NCDHW" then
      data_format = "NCHW"
    elseif fmt == "NLC" or fmt == "NHWC" or fmt == "NDHWC" then
      data_format = "NHWC"
    end  -- activation.py:627 的归一化
  end
  return paddle_core.pd_op_prelu(x, weight, data_format, mode)
  -- ops_gen.h:391
end)

-- rrelu{x, lower, upper, training, inplace} -- activation.py:671;
-- Python is_test = not training(activation.py:777),算子收 is_test。
M.rrelu = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "lower", type = argrule.number, opt = true, default = 1 / 8,
    doc = "lower bound (activation.py:672)" },
  { name = "upper", type = argrule.number, opt = true, default = 1 / 3,
    doc = "upper bound (activation.py:673)" },
  { name = "training", type = argrule.boolean, opt = true, default = true,
    doc = "train mode; is_test = not training (activation.py:777)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, lower, upper, training, inplace)
  if inplace then return no_inplace_yet "rrelu"() end
  local is_test = 0
  if not training then is_test = 1 end
  return paddle_core.pd_op_rrelu(x, lower, upper, is_test)  -- ops_gen.h:426
end)

-- gumbel_softmax{x, temperature, hard, axis} -- activation.py:1925;
-- D35:装饰器(别名层)不移植,绑定被装饰函数 = 此签名。axis 1-based。
M.gumbel_softmax = rule {
  { name = "x", type = Tensor, doc = "logits tensor" },
  { name = "temperature", type = argrule.number, opt = true, default = 1.0,
    doc = "softmax temperature (activation.py:1926)" },
  { name = "hard", type = argrule.boolean, opt = true, default = false,
    doc = "hard one-hot sample (activation.py:1927)" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "softmax axis, 1-based, negative from the end (C4)" },
}(function(x, temperature, hard, axis)
  -- axis 是 index 语义:走生成物做 1-based->0-based(同 softmax 教训)
  local wrap = require("paddle._ops.gumbel_softmax")
  return wrap.gumbel_softmax(x, temperature, hard, axis)
end)

return M
