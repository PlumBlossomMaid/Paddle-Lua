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

-- CPU place 缓存(dropout_nd 的 uniform/full 需要 place 参数)
local CPU_PLACE
local function cpu_place()
  if CPU_PLACE == nil then
    CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  end
  return CPU_PLACE
end

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

-- hardswish{x, inplace} -- activation.py:400-401 (inplace=False)
M.hardswish = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
}(function(x, inplace)
  if inplace then return no_inplace_yet "hardswish"() end
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

-- hardshrink{x, threshold} -- activation.py:302 (threshold=0.5)
M.hardshrink = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "threshold", type = argrule.number, opt = true, default = 0.5,
    doc = "shrink threshold, Python default 0.5 (activation.py:306)" },
}(function(x, threshold)
  return paddle_core.pd_op_hardshrink(x, threshold)  -- gen_ops_h.cc:17
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

-- ---------------------------------------------------------------------
-- 池化(pooling.py):avg/max 1d/2d/3d + adaptive_avg/max 1d/2d/3d
-- 直接绑定 pool2d/pool3d/max_pool2d_with_index/max_pool3d_with_index;
-- padding 三态解析 Lua 重写(pooling.py:135-193 的 _update_padding_nd 等价)。
-- ---------------------------------------------------------------------

-- 标量 → {v}*n(convert_to_list 等价,layers_utils.py:51)
local function pool_to_list(v, n)
  if type(v) == "number" then
    local t = {}
    for i = 1, n do t[i] = v end
    return t
  end
  return v
end

-- 对称 padding 检测:len==2n 且成对相等(layers_utils.py:361,
-- 0-based 比较 (0,1),(2,3)... → 1-based (1,2),(3,4)...)
local function pool_is_symmetric(padding, num_dims)
  if #padding ~= num_dims * 2 then return true end
  for i = 1, num_dims do
    if padding[i * 2 - 1] ~= padding[i * 2] then return false end
  end
  return true
end

-- 对偶列表 [[b,a],...]:batch/channel 维必须全 0(pooling.py:95)
local function pool_zero_bandc(padding, channel_last)
  if channel_last then
    return padding[1][1] == 0 and padding[1][2] == 0
       and padding[#padding][1] == 0 and padding[#padding][2] == 0
  end
  return padding[1][1] == 0 and padding[1][2] == 0
     and padding[2][1] == 0 and padding[2][2] == 0
end

-- 从对偶列表剔除 batch/channel 维并摊平(pooling.py:102)
local function pool_exclude_bandc(padding, channel_last)
  local out = {}
  local from = channel_last and 2 or 3
  local to = channel_last and (#padding - 1) or #padding
  for i = from, to do
    out[#out + 1] = padding[i][1]
    out[#out + 1] = padding[i][2]
  end
  return out
end

-- 取 [0],[2],... 下标(对偶 → 单值)
local function pool_even(t)
  local out = {}
  for i = 1, #t, 2 do out[#out + 1] = t[i] end
  return out
end

-- pool_update_padding:padding 三态 → (paddings, padding_algorithm)
-- 字符串 SAME/VALID / 数字 / 列表(一对、2n、n)(pooling.py:135)
local function pool_update_padding(padding, num_dims, channel_last, ceil_mode)
  if type(padding) == "string" then
    local up = string.upper(padding)
    if up ~= "SAME" and up ~= "VALID" then
      error(("Unknown padding: '%s'. It can only be 'SAME' or 'VALID'.")
        :format(padding), 2)
    end
    if up == "VALID" and ceil_mode then
      error('When Attr(padding) is "VALID", Attr(ceil_mode) must be False. '
        .. "Received ceil_mode: True.", 2)
    end
    local zero = {}
    for i = 1, num_dims do zero[i] = 0 end
    return zero, up
  end
  if type(padding) == "table" then
    -- 对偶列表 [[b,a],[b,a],...],含 batch/channel 维
    if #padding == 2 + num_dims and type(padding[1]) == "table" then
      if not pool_zero_bandc(padding, channel_last) then
        error("Non-zero padding in the batch or channel dimensions is "
          .. "not supported.", 2)
      end
      local flat = pool_exclude_bandc(padding, channel_last)
      if pool_is_symmetric(flat, num_dims) then
        return pool_even(flat), "EXPLICIT"
      end
      return flat, "EXPLICIT"
    end
    -- [a,b,a,b,...] 2n 或 [d1,d2,...] n
    if #padding == 2 * num_dims and type(padding[1]) == "number" then
      if pool_is_symmetric(padding, num_dims) then
        return pool_even(padding), "EXPLICIT"
      end
      return padding, "EXPLICIT"
    end
    if #padding == num_dims and type(padding[1]) == "number" then
      return padding, "EXPLICIT"
    end
    error("Invalid padding: " .. tostring(padding), 2)
  end
  -- 数字
  return pool_to_list(padding, num_dims), "EXPLICIT"
end

-- 1d 池化 padding 展开:2 值 → [0,0,*p],1 值 → [0,*p](pooling.py:196)
local function pool_expand_low(padding)
  if #padding == 2 then
    return { 0, 0, padding[1], padding[2] }
  elseif #padding == 1 then
    return { 0, padding[1] }
  end
  error("The size of padding's dimension should be 1 or 2. But got padding="
    .. tostring(padding), 2)
end

-- 校验 data_format 合法 + 是否 channel-last(pooling.py:108)
local function pool_channel_last(data_format, n)
  local map = {
    [1] = { "NCL", "NLC" },
    [2] = { "NCHW", "NHWC" },
    [3] = { "NCDHW", "NDHWC" },
  }
  local pair = map[n]
  if data_format ~= pair[1] and data_format ~= pair[2] then
    error(("Attr(data_format) should be '%s' or '%s'. Received %s")
      :format(pair[1], pair[2], tostring(data_format)), 2)
  end
  return data_format == pair[2]
end

-- 池化参数类型:kernel_size/stride = number 或 number 表
local PoolLen = argrule.Union { argrule.number, argrule.table }
-- padding = number / string / 表
local PoolPad = argrule.Union { argrule.number, argrule.string, argrule.table }

-- avg_pool1d{x, kernel_size, stride, padding, exclusive, ceil_mode}
-- pooling.py:246-293:NCL→NCHW 升维,pool2d 再 squeeze
M.avg_pool1d = rule {
  { name = "x", type = Tensor, doc = "3-D [N,C,L] tensor (NCL)" },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {L})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride (default kernel_size)" },
  { name = "padding", type = PoolPad, opt = true, default = 0,
    doc = "'SAME'/'VALID' / int / {L} / 2-list / pair-list" },
  { name = "exclusive", type = argrule.boolean, opt = true, default = true,
    doc = "exclude padding in avg (pooling.py:258)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:259)" },
}(function(x, kernel_size, stride, padding, exclusive, ceil_mode)
  local x2 = paddle_core.pd_op_unsqueeze(x, { 2 })  -- NCL→NCHW(0-based 轴2)
  local ks = { 1, pool_to_list(kernel_size, 1)[1] }
  local st
  if stride == nil then st = ks else st = { 1, pool_to_list(stride, 1)[1] } end
  local pad, alg = pool_update_padding(padding, 1, false, ceil_mode)
  pad = pool_expand_low(pad)
  local out = paddle_core.pd_op_pool2d(
    x2, ks, st, pad, (ceil_mode and 1 or 0), (exclusive and 1 or 0),
    "NCHW", "avg", 0, 0, alg)
  return paddle_core.pd_op_squeeze(out, { 2 })
end)

-- max_pool1d{x, kernel_size, stride, padding, return_mask, ceil_mode,
--            dilation} -- pooling.py:619-708
M.max_pool1d = rule {
  { name = "x", type = Tensor, doc = "3-D [N,C,L] tensor (NCL)" },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {L})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride (default kernel_size)" },
  { name = "padding", type = PoolPad, opt = true, default = 0,
    doc = "'SAME'/'VALID' / int / {L} / 2-list / pair-list" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (1-based, C4)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:637)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:638)" },
}(function(x, kernel_size, stride, padding, return_mask, ceil_mode, dilation)
  local x2 = paddle_core.pd_op_unsqueeze(x, { 2 })
  local d1 = pool_to_list(dilation, 1)
  local dl = { 1, d1[1] }
  local ks = { 1, pool_to_list(kernel_size, 1)[1] }
  local st
  if stride == nil then st = ks else st = { 1, pool_to_list(stride, 1)[1] } end
  local pad, alg = pool_update_padding(padding, 1, false, ceil_mode)
  pad = pool_expand_low(pad)
  local has_dilation = d1[1] ~= 1
  local out
  if return_mask or has_dilation then
    -- max_pool2d_with_index 双输出,out/mask(python:return (o,m) 或 o[0])
    local o, m = paddle_core.pd_op_max_pool2d_with_index(
      x2, ks, st, pad, dl, 0, 0, (ceil_mode and 1 or 0))
    if return_mask then
      return paddle_core.pd_op_squeeze(o, { 2 }),
             paddle_core.pd_op_squeeze(m, { 2 })
    end
    out = paddle_core.pd_op_squeeze(o, { 2 })
  else
    out = paddle_core.pd_op_pool2d(
      x2, ks, st, pad, (ceil_mode and 1 or 0), 1,
      "NCHW", "max", 0, 0, alg)
    out = paddle_core.pd_op_squeeze(out, { 2 })
  end
  return out
end)

-- avg_pool2d{x, kernel_size, stride, padding, ceil_mode, exclusive,
--            divisor_override, data_format} -- pooling.py:322-411
M.avg_pool2d = rule {
  { name = "x", type = Tensor, doc = "4-D [N,C,H,W] tensor" },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride (default kernel_size)" },
  { name = "padding", type = PoolPad, opt = true, default = 0,
    doc = "'SAME'/'VALID' / int / {H,W} / 4-list / pair-list" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:332)" },
  { name = "exclusive", type = argrule.boolean, opt = true, default = true,
    doc = "exclude padding in avg (pooling.py:333)" },
  { name = "divisor_override", type = argrule.number, opt = true,
    doc = "override division factor (pooling.py:407-411)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:335)" },
}(function(x, kernel_size, stride, padding, ceil_mode, exclusive,
          divisor_override, data_format)
  local cl = pool_channel_last(data_format, 2)
  local ks = pool_to_list(kernel_size, 2)
  local st = ks
  if stride ~= nil then st = pool_to_list(stride, 2) end
  local pad, alg = pool_update_padding(padding, 2, cl, ceil_mode)
  local out = paddle_core.pd_op_pool2d(
    x, ks, st, pad, (ceil_mode and 1 or 0), (exclusive and 1 or 0),
    data_format, "avg", 0, 0, alg)
  if divisor_override ~= nil then
    -- pooling.py:407:out * (k0*k1) / divisor_override
    out = paddle_core.pd_op_scale(
      out, ks[1] * ks[2] / divisor_override, 0, 1)
  end
  return out
end)

-- max_pool2d{x, kernel_size, stride, padding, return_mask, ceil_mode,
--            dilation, data_format} -- pooling.py:1172-1330
M.max_pool2d = rule {
  { name = "x", type = Tensor, doc = "4-D [N,C,H,W] tensor" },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride (default kernel_size)" },
  { name = "padding", type = PoolPad, opt = true, default = 0,
    doc = "'SAME'/'VALID' / int / {H,W} / 4-list / pair-list" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (1-based, C4);NCHW only" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:1187)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:1188)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:1189)" },
}(function(x, kernel_size, stride, padding, return_mask, ceil_mode,
          dilation, data_format)
  local cl = pool_channel_last(data_format, 2)
  local ks = pool_to_list(kernel_size, 2)
  local st = ks
  if stride ~= nil then st = pool_to_list(stride, 2) end
  local dl = pool_to_list(dilation, 2)
  local pad, alg = pool_update_padding(padding, 2, cl, ceil_mode)
  if data_format == "NHWC" and return_mask then
    error("When setting return_mask to true, data_format must be set to "
      .. "NCHW in API:max_pool2d", 2)
  end
  local has_dilation = dl[1] ~= 1 or dl[2] ~= 1
  if has_dilation and data_format == "NHWC" then
    error("When dilation != 1, data_format must be set to NCHW in "
      .. "API:max_pool2d", 2)
  end
  if return_mask or has_dilation then
    local o, m = paddle_core.pd_op_max_pool2d_with_index(
      x, ks, st, pad, dl, 0, 0, (ceil_mode and 1 or 0))
    if return_mask then return o, m end
    return o
  end
  return paddle_core.pd_op_pool2d(
    x, ks, st, pad, (ceil_mode and 1 or 0), 1,
    data_format, "max", 0, 0, alg)
end)

-- avg_pool3d{x, kernel_size, stride, padding, ceil_mode, exclusive,
--            divisor_override, data_format} -- pooling.py:452-535
M.avg_pool3d = rule {
  { name = "x", type = Tensor, doc = "5-D [N,C,D,H,W] tensor" },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {D,H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride (default kernel_size)" },
  { name = "padding", type = PoolPad, opt = true, default = 0,
    doc = "'SAME'/'VALID' / int / {D,H,W} / 6-list / pair-list" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:462)" },
  { name = "exclusive", type = argrule.boolean, opt = true, default = true,
    doc = "exclude padding in avg (pooling.py:463)" },
  { name = "divisor_override", type = argrule.number, opt = true,
    doc = "override division factor (pooling.py:534)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' or 'NDHWC' (pooling.py:465)" },
}(function(x, kernel_size, stride, padding, ceil_mode, exclusive,
          divisor_override, data_format)
  local cl = pool_channel_last(data_format, 3)
  local ks = pool_to_list(kernel_size, 3)
  local st = ks
  if stride ~= nil then st = pool_to_list(stride, 3) end
  local pad, alg = pool_update_padding(padding, 3, cl, ceil_mode)
  local out = paddle_core.pd_op_pool3d(
    x, ks, st, pad, (ceil_mode and 1 or 0), (exclusive and 1 or 0),
    data_format, "avg", 0, 0, alg)
  if divisor_override ~= nil then
    -- pooling.py:534:out * (k0*k1*k2) / divisor_override
    out = paddle_core.pd_op_scale(
      out, ks[1] * ks[2] * ks[3] / divisor_override, 0, 1)
  end
  return out
end)

-- max_pool3d{x, kernel_size, stride, padding, return_mask, ceil_mode,
--            dilation, data_format} -- pooling.py:1369-1535
M.max_pool3d = rule {
  { name = "x", type = Tensor, doc = "5-D [N,C,D,H,W] tensor" },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {D,H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride (default kernel_size)" },
  { name = "padding", type = PoolPad, opt = true, default = 0,
    doc = "'SAME'/'VALID' / int / {D,H,W} / 6-list / pair-list" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (1-based, C4);NCDHW only" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:1379)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:1380)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' or 'NDHWC' (pooling.py:1381)" },
}(function(x, kernel_size, stride, padding, return_mask, ceil_mode,
          dilation, data_format)
  local cl = pool_channel_last(data_format, 3)
  local ks = pool_to_list(kernel_size, 3)
  local st = ks
  if stride ~= nil then st = pool_to_list(stride, 3) end
  local dl = pool_to_list(dilation, 3)
  local pad, alg = pool_update_padding(padding, 3, cl, ceil_mode)
  if data_format == "NDHWC" and return_mask then
    error("When setting return_mask to true, data_format must be set to "
      .. "NCDHW in API:max_pool3d", 2)
  end
  local has_dilation = dl[1] ~= 1 or dl[2] ~= 1 or dl[3] ~= 1
  if has_dilation and data_format == "NDHWC" then
    error("When dilation != 1, data_format must be set to NCDHW in "
      .. "API:max_pool3d", 2)
  end
  if return_mask or has_dilation then
    local o, m = paddle_core.pd_op_max_pool3d_with_index(
      x, ks, st, pad, dl, 0, 0, (ceil_mode and 1 or 0))
    if return_mask then return o, m end
    return o
  end
  return paddle_core.pd_op_pool3d(
    x, ks, st, pad, (ceil_mode and 1 or 0), 1,
    data_format, "max", 0, 0, alg)
end)

-- adaptive_avg_pool1d{x, output_size} -- pooling.py:1536-1626
M.adaptive_avg_pool1d = rule {
  { name = "x", type = Tensor, doc = "3-D [N,C,L] tensor (NCL)" },
  { name = "output_size", type = PoolLen, doc = "target L (int or {L})" },
}(function(x, output_size)
  local os = pool_to_list(output_size, 1)
  local ps = { 1, os[1] }
  local x2 = paddle_core.pd_op_unsqueeze(x, { 2 })
  -- pooling.py:1596-1599:pool2d adaptive,EXPLICIT
  local out = paddle_core.pd_op_pool2d(
    x2, ps, { 1, 1 }, { 0, 0 }, 0, 1, "NCHW", "avg", 0, 1, "EXPLICIT")
  return paddle_core.pd_op_squeeze(out, { 2 })
end)

-- adaptive_avg_pool2d{x, output_size, data_format} -- pooling.py:1625-1738
M.adaptive_avg_pool2d = rule {
  { name = "x", type = Tensor, doc = "4-D [N,C,H,W] tensor" },
  { name = "output_size", type = PoolLen, doc = "target {H,W} (int = both)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:1643)" },
}(function(x, output_size, data_format)
  local cl = pool_channel_last(data_format, 2)
  local os = pool_to_list(output_size, 2)
  return paddle_core.pd_op_pool2d(
    x, os, { 1, 1 }, { 0, 0 }, 0, 1, data_format, "avg", 0, 1, "EXPLICIT")
end)

-- adaptive_avg_pool3d{x, output_size, data_format} -- pooling.py:1763-1898
M.adaptive_avg_pool3d = rule {
  { name = "x", type = Tensor, doc = "5-D [N,C,D,H,W] tensor" },
  { name = "output_size", type = PoolLen, doc = "target {D,H,W} (int = all)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' or 'NDHWC' (pooling.py:1776)" },
}(function(x, output_size, data_format)
  pool_channel_last(data_format, 3)
  local os = pool_to_list(output_size, 3)
  return paddle_core.pd_op_pool3d(
    x, os, { 1, 1, 1 }, { 0, 0, 0 }, 0, 1, data_format, "avg", 0, 1,
    "EXPLICIT")
end)

-- adaptive_max_pool1d{x, output_size, return_mask} -- pooling.py:1899-1990
M.adaptive_max_pool1d = rule {
  { name = "x", type = Tensor, doc = "3-D [N,C,L] tensor (NCL)" },
  { name = "output_size", type = PoolLen, doc = "target L (int or {L})" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (1-based, C4)" },
}(function(x, output_size, return_mask)
  local os = pool_to_list(output_size, 1)
  local ps = { 1, os[1] }
  local x2 = paddle_core.pd_op_unsqueeze(x, { 2 })
  local o, m = paddle_core.pd_op_max_pool2d_with_index(
    x2, ps, { 1, 1 }, { 0, 0 }, { 1, 1 }, 0, 1, 0)
  if return_mask then
    return paddle_core.pd_op_squeeze(o, { 2 }),
           paddle_core.pd_op_squeeze(m, { 2 })
  end
  return paddle_core.pd_op_squeeze(o, { 2 })
end)

-- adaptive_max_pool2d{x, output_size, return_mask} -- pooling.py:2004-2060
M.adaptive_max_pool2d = rule {
  { name = "x", type = Tensor, doc = "4-D [N,C,H,W] tensor" },
  { name = "output_size", type = PoolLen, doc = "target {H,W} (int = both)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (1-based, C4)" },
}(function(x, output_size, return_mask)
  local os = pool_to_list(output_size, 2)
  local o, m = paddle_core.pd_op_max_pool2d_with_index(
    x, os, { 1, 1 }, { 0, 0 }, { 1, 1 }, 0, 1, 0)
  if return_mask then return o, m end
  return o
end)

-- adaptive_max_pool3d{x, output_size, return_mask} -- pooling.py:2097-2202
M.adaptive_max_pool3d = rule {
  { name = "x", type = Tensor, doc = "5-D [N,C,D,H,W] tensor" },
  { name = "output_size", type = PoolLen, doc = "target {D,H,W} (int = all)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (1-based, C4)" },
}(function(x, output_size, return_mask)
  local os = pool_to_list(output_size, 3)
  local o, m = paddle_core.pd_op_max_pool3d_with_index(
    x, os, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 0, 1, 0)
  if return_mask then return o, m end
  return o
end)

-- ---------------------------------------------------------------------
-- 正则化(common.py):dropout。axis=nil 主路径直调 pd_op_dropout;
-- mode 语义转换 'downscale_in_infer'→'downgrade_in_infer'(common.py:1419)。
-- ---------------------------------------------------------------------

-- dropout{x, p, axis, training, inplace, mode}
-- common.py:1234-1431。F.dropout 主路径(axis=nil):
--   _C_ops.dropout(x, None, p, not training, mode, seed, fix_seed)
--   (common.py:1426-1428)。p==0 快速返回 x;p 校验 [0,1](1400-1407);
-- axis 非 nil 走 dropout_nd 组合(common.py:1477+),v2 随批。
M.dropout = rule {
  { name = "x", type = Tensor, doc = "input (fp16/fp32/fp64)" },
  { name = "p", type = argrule.number, opt = true, default = 0.5,
    doc = "zero-out probability (common.py:1235)" },
  { name = "axis", type = argrule.any, opt = true,
    doc = "dropout_nd axes; nil = element-wise (v2)" },
  { name = "training", type = argrule.boolean, opt = true, default = true,
    doc = "train phase; is_test = not training (common.py:1427)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (generator M2)" },
  { name = "mode", type = argrule.string, opt = true,
    default = "upscale_in_train",
    doc = "'upscale_in_train' | 'downscale_in_infer' (common.py:1238)" },
}(function(x, p, axis, training, inplace, mode)
  if inplace then return no_inplace_yet "dropout"() end
  if p == 0 then return x end                       -- common.py:1400
  if p < 0 or p > 1 then
    error(("paddle.nn.functional.dropout: p=%s out of [0,1]")
      :format(tostring(p)), 2)
  end
  if mode ~= "downscale_in_infer" and mode ~= "upscale_in_train" then
    error("paddle.nn.functional.dropout: mode must be "
      .. "'downscale_in_infer' or 'upscale_in_train'", 2)
  end
  if axis ~= nil then
    -- dropout_nd 组合(common.py:1473-1520):
    --   training: scale_input = x/(1-p)(upscale)或 x;mask 在 axis 维
    --   保持原尺寸、其余维为 1;uniform(mask_shape) >= p → keep_mask;
    --   scale_input * keep_mask
    --   test: downscale → x*(1-p),否则 x
    local keep_prob = 1.0 - p
    if not training then
      if mode == "downscale_in_infer" then
        return paddle_core.pd_op_scale(x, keep_prob, 0, 1)
      end
      return x
    end
    if p == 1.0 then
      return paddle_core.pd_op_scale(x, 0.0, 0, 1)  -- common.py:1481
    end
    local scale_input = x
    if mode == "upscale_in_train" then
      scale_input = paddle_core.pd_op_scale(x, 1.0 / keep_prob, 0, 1)
    end
    local input_shape = x:shape()
    local drop_axes = axis
    if type(axis) == "number" then drop_axes = { axis } end
    -- axis 1-based(C4)→ 0-based mask 索引;负数从尾数
    local mask_shape = {}
    for i = 1, #input_shape do mask_shape[i] = 1 end
    for _, a in ipairs(drop_axes) do
      local idx = a
      if idx < 0 then idx = #input_shape + idx + 1 end
      if idx < 1 or idx > #input_shape then
        error(("paddle.nn.functional.dropout: axis %s out of range for "
          .. "%d-D input"):format(tostring(a), #input_shape), 2)
      end
      mask_shape[idx] = input_shape[idx]
    end
    local random_tensor = paddle_core.pd_op_uniform(
      mask_shape, paddle_core.PD_DTYPE_FLOAT32, 0.0, 1.0, 0, cpu_place())
    local p_t = paddle_core.pd_op_full(
      { 1 }, p, paddle_core.PD_DTYPE_FLOAT32, cpu_place())
    local keep_mask = paddle_core.pd_op_greater_equal(random_tensor, p_t)
    local dt = x:dtype()
    scale_input = paddle_core.pd_op_cast(scale_input, dt)
    keep_mask = paddle_core.pd_op_cast(keep_mask, dt)
    return paddle_core.pd_op_multiply(scale_input, keep_mask)
  end
  local m = (mode == "downscale_in_infer") and "downgrade_in_infer" or mode
  return paddle_core.pd_op_dropout(x, nil, p, (training and 0 or 1), m, 0, 0)
  -- ops_gen.h:147
end)

-- ---------------------------------------------------------------------
-- 归一化(norm.py):batch_norm、layer_norm。直调 pd_op_batch_norm(6 输出
-- 取第 1)/pd_op_layer_norm(单出);scale/bias 均可空(sol optional)。
-- normalized_shape 是 1-based 轴范围(C4,nn-functional.md sec.3),转成
-- C++ 的 begin_norm_axis(norm.py:419-421 的等价:起点轴 - 1)。
-- ---------------------------------------------------------------------

-- 归一化/损失共用 reduction helper(sum/mean_all/原样)
local function reduce_out(out, reduction, fname)
  if reduction == "sum" then
    return paddle_core.pd_op_sum(out, {}, paddle_core.PD_DTYPE_UNDEFINED, 0)
  elseif reduction == "mean" then
    return paddle_core.pd_op_mean_all(out)
  elseif reduction == "none" then
    return out
  end
  error(("'reduction' in '%s' should be 'sum', 'mean' or 'none', but "
    .. "received %s"):format(fname, tostring(reduction)), 3)
end

-- batch_norm{x, running_mean, running_var, weight, bias, training,
--            momentum, epsilon, data_format, use_global_stats}
-- norm.py:162-285。is_test = not training;use_global_stats=None →
-- use_global_stats = not training + trainable_statistics=False,否则
-- trainable_statistics = not use_global_stats(norm.py:246-252)。
M.batch_norm = rule {
  { name = "x", type = Tensor, doc = "input [N,C,...] (float32/64)" },
  { name = "running_mean", type = Tensor, doc = "running mean (1-D, C)" },
  { name = "running_var", type = Tensor, doc = "running variance (1-D, C)" },
  { name = "weight", type = Tensor, opt = true,
    doc = "scale gamma (1-D, C); nil = identity" },
  { name = "bias", type = Tensor, opt = true,
    doc = "shift beta (1-D, C); nil = zero" },
  { name = "training", type = argrule.boolean, opt = true, default = false,
    doc = "train mode; is_test = not training (norm.py:258)" },
  { name = "momentum", type = argrule.number, opt = true, default = 0.9,
    doc = "running stats momentum (norm.py:256)" },
  { name = "epsilon", type = argrule.number, opt = true, default = 1e-5,
    doc = "variance epsilon (norm.py:257)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NC'/'NCL'/'NCHW'/'NCDHW' -> NCHW; 'NLC'/'NHWC'/'NDHWC' -> NHWC" },
  { name = "use_global_stats", type = argrule.boolean, opt = true,
    doc = "nil: is_test by training (norm.py:246)" },
}(function(x, running_mean, running_var, weight, bias, training, momentum,
          epsilon, data_format, use_global_stats)
  -- norm.py:241:按第 2 个字符 C → NCHW,否则 NHWC
  local norm_format = "NHWC"
  if string.sub(data_format, 2, 2) == "C" then norm_format = "NCHW" end
  local ugs, trainable
  if use_global_stats == nil then
    ugs = not training
    trainable = false
  else
    ugs = use_global_stats
    trainable = not use_global_stats
  end
  local out = paddle_core.pd_op_batch_norm(
    x, running_mean, running_var, weight, bias, (training and 0 or 1),
    momentum, epsilon, norm_format, (ugs and 1 or 0), (trainable and 1 or 0))
  return out  -- 6 输出取第 1(python 同样只取 batch_norm_out,norm.py:262)
end)

-- layer_norm{x, normalized_shape, weight, bias, epsilon}
-- norm.py:346-436。normalized_shape = 1-based 轴范围闭区间(C4,
-- nn-functional.md sec.3:{1,2} 归一化前两维);begin_norm_axis = 起点-1,
-- 即 C++ 的 0-based 起点(与 norm.py:421 input_ndim - normalized_ndim
-- 对"从尾部数"等价,这里按轴范围算起点)。
M.layer_norm = rule {
  { name = "x", type = Tensor, doc = "input tensor (bf16/f16/f32/f64)" },
  { name = "normalized_shape", type = argrule.Union { argrule.number,
    argrule.table },
    doc = "1-based 轴范围闭区间 {start,end} 或单轴 int(归一化到末尾)" },
  { name = "weight", type = Tensor, opt = true,
    doc = "per-element gain (norm.py:356); nil = identity" },
  { name = "bias", type = Tensor, opt = true,
    doc = "per-element shift (norm.py:357); nil = zero" },
  { name = "epsilon", type = argrule.number, opt = true, default = 1e-5,
    doc = "variance epsilon (norm.py:358)" },
}(function(x, normalized_shape, weight, bias, epsilon)
  local begin_axis
  if type(normalized_shape) == "number" then
    begin_axis = normalized_shape - 1      -- 单轴:该 1-based 轴 → 0-based
  else
    begin_axis = normalized_shape[1] - 1   -- 闭区间起点
  end
  return paddle_core.pd_op_layer_norm(x, weight, bias, epsilon, begin_axis)
  -- ops_gen.h:281
end)

-- ---------------------------------------------------------------------
-- 损失(loss.py,首批):mse/l1/smooth_l1/bce/bce_with_logits + 辅助。
-- 组合损失用基础算子链;singleton 算子直绑。label 非 index 语义(连续
-- target)除外 —— 本批均无 index label。reduction 三态共用 reduce_out。
-- ---------------------------------------------------------------------

-- mse_loss{input, label, reduction} -- loss.py:1988-2061
-- square(subtract(input,label));mean → mean_all,sum → sum,none → 原样
M.mse_loss = rule {
  { name = "input", type = Tensor, doc = "prediction (float32/64)" },
  { name = "label", type = Tensor, doc = "target, same shape" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:2046)" },
}(function(input, label, reduction)
  local d = paddle_core.pd_op_subtract(input, label)
  local sq = paddle_core.pd_op_square(d)
  return reduce_out(sq, reduction, "mse_loss")
end)

-- l1_loss{input, label, reduction} -- loss.py:1436-1514
-- abs(subtract(input,label));mean → mean_all,sum → sum,none → 原样
M.l1_loss = rule {
  { name = "input", type = Tensor, doc = "prediction (float32/64/int)" },
  { name = "label", type = Tensor, doc = "target, same shape" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:1508)" },
}(function(input, label, reduction)
  local d = paddle_core.pd_op_subtract(input, label)
  local a = paddle_core.pd_op_abs(d)
  return reduce_out(a, reduction, "l1_loss")
end)

-- smooth_l1_loss{input, label, reduction, delta, is_huber}
-- loss.py:1175-1263:huber_loss(input,label,delta);is_huber=False 时
-- out/delta(loss.py:1248),再 reduction。
M.smooth_l1_loss = rule {
  { name = "input", type = Tensor, doc = "prediction (float32/64)" },
  { name = "label", type = Tensor, doc = "target, same shape" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:1251)" },
  { name = "delta", type = argrule.number, opt = true, default = 1.0,
    doc = "huber threshold (loss.py:1186)" },
  { name = "is_huber", type = argrule.boolean, opt = true, default = true,
    doc = "False divides loss by delta (loss.py:1200)" },
}(function(input, label, reduction, delta, is_huber)
  local out = paddle_core.pd_op_huber_loss(input, label, delta)
  if not is_huber then
    -- loss.py:1248:out / delta → scale(out, 1/delta, 0, 1)
    out = paddle_core.pd_op_scale(out, 1.0 / delta, 0, 1)
  end
  return reduce_out(out, reduction, "smooth_l1_loss")
end)

-- binary_cross_entropy{input, label, weight, reduction}
-- loss.py:627-738:bce_loss(input,label) → *weight → reduction
M.binary_cross_entropy = rule {
  { name = "input", type = Tensor,
    doc = "probabilities (sigmoid output), in [0,1]" },
  { name = "label", type = Tensor, doc = "target, same shape, in [0,1]" },
  { name = "weight", type = Tensor, opt = true,
    doc = "per-element rescale (loss.py:646); nil = none" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:651)" },
}(function(input, label, weight, reduction)
  local out = paddle_core.pd_op_bce_loss(input, label)  -- ops_gen.h:56
  if weight ~= nil then
    out = paddle_core.pd_op_multiply(out, weight)  -- loss.py:710
  end
  return reduce_out(out, reduction, "binary_cross_entropy")
end)

-- binary_cross_entropy_with_logits{logit, label, weight, reduction,
--   pos_weight} -- loss.py:773-960,sigmoid_cross_entropy_with_logits 单出
M.binary_cross_entropy_with_logits = rule {
  { name = "logit", type = Tensor, doc = "raw logits (pre-sigmoid)" },
  { name = "label", type = Tensor, doc = "target, same shape, in [0,1]" },
  { name = "weight", type = Tensor, opt = true,
    doc = "per-element rescale (loss.py:819); nil = none" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:824)" },
  { name = "pos_weight", type = Tensor, opt = true,
    doc = "positive-class weight (loss.py:821); nil = none" },
}(function(logit, label, weight, reduction, pos_weight)
  -- loss.py:916:normalize=False, ignore_index=-100
  local out = paddle_core.pd_op_sigmoid_cross_entropy_with_logits(
    logit, label, pos_weight, 0, -100)  -- ops_gen.h:450
  if weight ~= nil then
    out = paddle_core.pd_op_multiply(out, weight)  -- loss.py:921
  end
  return reduce_out(out, reduction, "binary_cross_entropy_with_logits")
end)

-- ---------------------------------------------------------------------
-- 损失续(loss.py):cross_entropy(hard-label 主路径)。直调
-- pd_op_cross_entropy_with_softmax(2 输出取 loss,sol gen_ops_c.cc:392;
-- C ABI 层对 label 内容做 pd_idx_minus1,ops_c.cc:433)。INDEX 语义:
--   label 内容 1-based→0-based 由 C ABI 完成;
--   ignore_index 是 label 值(1-based,C4),C ABI 原样透传 → 正数 -1;
--   axis 正数 -1(C4),负数不变。
-- soft_label / weight / label_smoothing 后续批(tasks.md 剩余清单)。
-- ---------------------------------------------------------------------

-- cross_entropy{input, label, weight, ignore_index, reduction,
--               soft_label, axis, use_softmax, label_smoothing}
-- loss.py:2902-3453 主路径(hard label,use_softmax=true):
--   _, out = _C_ops.cross_entropy_with_softmax(input, label, soft_label,
--     use_softmax, True, ignore_index, axis)(loss.py:3218)
M.cross_entropy = rule {
  { name = "input", type = Tensor, doc = "logits [N..., C] (fp32/64)" },
  { name = "label", type = Tensor, doc = "class ids, 1-based (C4)" },
  { name = "weight", type = Tensor, opt = true,
    doc = "class weights (v2, tasks 剩余)" },
  { name = "ignore_index", type = argrule.number, opt = true, default = -100,
    doc = "class id to ignore, 1-based (C4, nn.md sec.3)" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:2965)" },
  { name = "soft_label", type = argrule.boolean, opt = true, default = false,
    doc = "one-hot target (v2, tasks 剩余)" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "class axis, 1-based (C4)" },
  { name = "use_softmax", type = argrule.boolean, opt = true, default = true,
    doc = "softmax inside op (loss.py:2910)" },
  { name = "label_smoothing", type = argrule.number, opt = true, default = 0.0,
    doc = "smoothing epsilon (v2, tasks 剩余)" },
}(function(input, label, weight, ignore_index, reduction, soft_label,
          axis, use_softmax, label_smoothing)
  if soft_label or weight ~= nil or label_smoothing > 0 then
    error("paddle.nn.functional.cross_entropy: soft_label / weight / "
      .. "label_smoothing land in a later batch -- tracked in tasks.md", 2)
  end
  if reduction ~= "none" and reduction ~= "mean" and reduction ~= "sum" then
    error(("'reduction' in 'cross_entropy' should be 'sum', 'mean' or "
      .. "'none', but received %s"):format(tostring(reduction)), 2)
  end
  -- loss.py:3204-3208:input rank == label rank + 1 → label unsqueeze(axis)
  local idim = #input:shape()
  local ldim = #label:shape()
  local ax = (axis > 0) and (axis - 1) or axis  -- 1-based → 0-based
  if idim - 1 == ldim then
    label = paddle_core.pd_op_unsqueeze(label, { ax })
  end
  local ig = ignore_index
  if ig > 0 then ig = ig - 1 end  -- 1-based → 0-based,负数不变(C4)
  local _, out = paddle_core.pd_op_cross_entropy_with_softmax(
    input, label, 0, (use_softmax and 1 or 0), 1, ig, ax)
  -- gen_ops_c.cc:392(2 输出取 loss);label 内容 C ABI 已 -1
  if reduction == "none" then
    if idim - 1 == ldim then
      out = paddle_core.pd_op_squeeze(out, { ax })  -- loss.py:3477
    end
    return out
  elseif reduction == "sum" then
    return paddle_core.pd_op_sum(out, {}, paddle_core.PD_DTYPE_UNDEFINED, 0)
  end
  -- mean:ignore_index<0(默认)→ mean_all;>=0 的 mask 计数后续批
  if ignore_index >= 0 then
    error("paddle.nn.functional.cross_entropy: mean reduction with "
      .. "ignore_index >= 0 needs the mask-count path -- tracked in "
      .. "tasks.md", 2)
  end
  return paddle_core.pd_op_mean_all(out)  -- loss.py:3395
end)

return M
