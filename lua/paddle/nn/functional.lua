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
-- axis 的 1-based->0-based 转换由 _ops/softmax.lua 生成物做(显式传值)。
M.softmax = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "softmax axis, 1-based, negative counts from the end (C4)" },
}(function(x, axis)
  return paddle_core.pd_op_softmax(x, axis)  -- ops_gen.h:460
end)

-- log_softmax{x, axis} -- activation.py:1729 (_C_ops.log_softmax)
M.log_softmax = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "log-softmax axis, 1-based, negative from the end (C4)" },
}(function(x, axis)
  return paddle_core.pd_op_log_softmax(x, axis)  -- ops_gen.h:297
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

return M
