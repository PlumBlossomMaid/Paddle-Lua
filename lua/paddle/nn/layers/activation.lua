-- lua/paddle/nn/layers/activation.lua -- paddle.nn 激活层类批量
-- Stage: P9 / M1 (WORKPLAN 2.9.4.2b)。激活层类 = 极薄壳:构造存参,
-- forward 一行转发 functional(已 argrule 包装,2.9.3.1/2.9.3.2)。
--
-- Upstream blueprint: python/paddle/nn/layer/activation.py
--   CELU:33 / ELU:91 / GLU:158 / GELU:219 / Hardshrink:300 /
--   Hardswish:361 / Tanh:419 / Hardtanh:460 / PReLU:515 /
--   RReLU:615 / ReLU:726 / ReLU6:773 / SELU:821 / LeakyReLU:887 /
--   Sigmoid:953 / Hardsigmoid:996 / Softplus:1057 / Softshrink:1105 /
--   Softsign:1166 / Swish:1207 / Mish:1255 / Tanhshrink:1310 /
--   ThresholdedReLU:1351 / Silu:1408 / LogSigmoid:1463 /
--   Softmax:1505 / LogSoftmax:1642 / Maxout:1703 / Softmax2D:1769
--
-- 说明:
--   - PReLU 有可学习 weight 参数(Constant(init)),RReLU forward 透传
--     self.training;这两个与其他纯转发不同。
--   - Softmax2D = softmax(axis=-3)(activation.py:1808)。
--   - GLU 依赖 functional.glu(chunk 未生成,tasks 剩余)→ 本批不做,
--     在 Common.do_not_implement 报清晰错误见 setup_gates(如有)。
--   - inplace 参数保留与 Python 签名一致,true → functional 报错。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

local CPU_PLACE

local function cpu_place()
  if CPU_PLACE == nil then
    CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  end
  return CPU_PLACE
end

-- 通用薄壳工厂:构造存参(去 name)+ forward 转发 fn。
-- fields = { {name=, type=, opt=, default=, doc=} ... }(排除 self/name)。
local function act_layer(cname, fields, fwd)
  local Cls = class(Layer)
  Cls._name = cname
  local rules = { { name = "self", type = Layer } }
  for _, f in ipairs(fields) do rules[#rules + 1] = f end
  rules[#rules + 1] = {
    name = "name", type = argrule.Optional { argrule.string }, opt = true,
  }
  Cls._init = rule(rules)(function(self, ...)
    self:super()
    local i = 1
    for _, f in ipairs(fields) do
      self[f.name] = select(i, ...)
      i = i + 1
    end
    self.name = select(i, ...)
  end)
  Cls.forward = fwd
  return Cls
end

local inplace_opt = {
  name = "inplace", type = argrule.boolean, opt = true, default = false,
  doc = "in-place variant (functional 报错,生成器 M2)",
}

-- CELU(activation.py:33):alpha=1.0, inplace
local CELU = act_layer("CELU", {
  { name = "alpha", type = argrule.number, opt = true, default = 1.0,
    doc = "CELU alpha (activation.py:48)" },
  inplace_opt,
}, function(self, x)
  return functional.celu(x, self.alpha, self.inplace)
end)

-- ELU(activation.py:91):alpha=1.0, inplace
local ELU = act_layer("ELU", {
  { name = "alpha", type = argrule.number, opt = true, default = 1.0,
    doc = "ELU alpha (activation.py:94)" },
  inplace_opt,
}, function(self, x)
  return functional.elu(x, self.alpha, self.inplace)
end)

-- GELU(activation.py:219):approximate=False
local GELU = act_layer("GELU", {
  { name = "approximate", type = argrule.boolean, opt = true, default = false,
    doc = "tanh approximation (activation.py:284)" },
}, function(self, x)
  return functional.gelu(x, self.approximate)
end)

-- Hardshrink(activation.py:300):threshold=0.5
local Hardshrink = act_layer("Hardshrink", {
  { name = "threshold", type = argrule.number, opt = true, default = 0.5,
    doc = "shrink threshold (activation.py:306)" },
}, function(self, x)
  return functional.hardshrink(x, self.threshold)
end)

-- Hardswish(activation.py:361):inplace
local Hardswish = act_layer("Hardswish", { inplace_opt },
  function(self, x)
    return functional.hardswish(x, self.inplace)
  end)

-- Tanh(activation.py:419)
local Tanh = act_layer("Tanh", {}, function(self, x)
  return functional.tanh(x)
end)

-- Hardtanh(activation.py:460):min=-1, max=1
local Hardtanh = act_layer("Hardtanh", {
  { name = "min", type = argrule.number, opt = true, default = -1.0,
    doc = "lower clamp (activation.py:461)" },
  { name = "max", type = argrule.number, opt = true, default = 1.0,
    doc = "upper clamp (activation.py:462)" },
}, function(self, x)
  return functional.hardtanh(x, self.min, self.max)
end)

-- ReLU6(activation.py:773):inplace
local ReLU6 = act_layer("ReLU6", { inplace_opt }, function(self, x)
  return functional.relu6(x, self.inplace)
end)

-- SELU(activation.py:821):scale, alpha, inplace
local SELU = act_layer("SELU", {
  { name = "scale", type = argrule.number, opt = true,
    default = 1.0507009873554805, doc = "selu scale (activation.py:866)" },
  { name = "alpha", type = argrule.number, opt = true,
    default = 1.6732632423543772, doc = "selu alpha (activation.py:867)" },
  inplace_opt,
}, function(self, x)
  return functional.selu(x, self.scale, self.alpha, self.inplace)
end)

-- LeakyReLU(activation.py:887):negative_slope=0.01, inplace
local LeakyReLU = act_layer("LeakyReLU", {
  { name = "negative_slope", type = argrule.number, opt = true, default = 0.01,
    doc = "slope for x<0 (activation.py:927)" },
  inplace_opt,
}, function(self, x)
  return functional.leaky_relu(x, self.negative_slope, self.inplace)
end)

-- Sigmoid(activation.py:953)
local Sigmoid = act_layer("Sigmoid", {}, function(self, x)
  return functional.sigmoid(x)
end)

-- Hardsigmoid(activation.py:996):inplace
local Hardsigmoid = act_layer("Hardsigmoid", { inplace_opt },
  function(self, x)
    -- M.hardsigmoid 规则 {x, slope, offset, inplace}:具名表调用,
    -- 只传 inplace,slope/offset 走 functional 默认(0.1666667/0.5)
    return functional.hardsigmoid{ x = x, inplace = self.inplace }
  end)

-- Softplus(activation.py:1057):beta=1, threshold=20
local Softplus = act_layer("Softplus", {
  { name = "beta", type = argrule.number, opt = true, default = 1.0,
    doc = "softplus beta (activation.py:1089)" },
  { name = "threshold", type = argrule.number, opt = true, default = 20.0,
    doc = "softplus threshold (activation.py:1090)" },
}, function(self, x)
  return functional.softplus(x, self.beta, self.threshold)
end)

-- Softshrink(activation.py:1105):threshold=0.5
local Softshrink = act_layer("Softshrink", {
  { name = "threshold", type = argrule.number, opt = true, default = 0.5,
    doc = "shrink threshold (activation.py:1144)" },
}, function(self, x)
  return functional.softshrink(x, self.threshold)
end)

-- Softsign(activation.py:1166)
local Softsign = act_layer("Softsign", {}, function(self, x)
  return functional.softsign(x)
end)

-- Swish(activation.py:1207):inplace
local Swish = act_layer("Swish", { inplace_opt }, function(self, x)
  return functional.swish(x, self.inplace)
end)

-- Mish(activation.py:1255):inplace
local Mish = act_layer("Mish", { inplace_opt }, function(self, x)
  return functional.mish(x, self.inplace)
end)

-- Tanhshrink(activation.py:1310)
local Tanhshrink = act_layer("Tanhshrink", {}, function(self, x)
  return functional.tanhshrink(x)
end)

-- ThresholdedReLU(activation.py:1351):threshold=1.0
local ThresholdedReLU = act_layer("ThresholdedReLU", {
  { name = "threshold", type = argrule.number, opt = true, default = 1.0,
    doc = "activation threshold (activation.py:1389)" },
}, function(self, x)
  return functional.thresholded_relu(x, self.threshold)
end)

-- Silu(activation.py:1408):inplace
local Silu = act_layer("Silu", { inplace_opt }, function(self, x)
  return functional.silu(x, self.inplace)
end)

-- LogSigmoid(activation.py:1463)
local LogSigmoid = act_layer("LogSigmoid", {}, function(self, x)
  return functional.log_sigmoid(x)
end)

-- ReLU(activation.py:726):inplace(2.9.4.1 原类,重写时用工厂重建)
local ReLU = act_layer("ReLU", { inplace_opt }, function(self, x)
  return functional.relu(x, self.inplace)
end)

-- Softmax(activation.py:1505):axis=-1(1-based,C4)
local Softmax = class(Layer)
Softmax._name = "Softmax"
Softmax._init = rule {
  { name = "self", type = Layer },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "softmax axis, 1-based (activation.py:1628)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, axis, name)
  self:super()
  self.axis = axis
  self.name = name
end)
function Softmax:forward(x)
  return functional.softmax(x, self.axis)
end

-- LogSoftmax(activation.py:1642):axis=-1(1-based,C4)
local LogSoftmax = class(Layer)
LogSoftmax._name = "LogSoftmax"
LogSoftmax._init = rule {
  { name = "self", type = Layer },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "log-softmax axis, 1-based (activation.py:1690)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, axis, name)
  self:super()
  self.axis = axis
  self.name = name
end)
function LogSoftmax:forward(x)
  return functional.log_softmax(x, self.axis)
end

-- Maxout(activation.py:1703):groups(必选), axis=1(1-based→functional
-- 默认 2 等价 Python 1)
local Maxout = act_layer("Maxout", {
  { name = "groups", type = argrule.number, doc = "channel groups" },
  { name = "axis", type = argrule.number, opt = true, default = 2,
    doc = "channel axis, 1-based; Python default 1 (activation.py:1753)" },
}, function(self, x)
  return functional.maxout(x, self.groups, self.axis)
end)

-- Softmax2D(activation.py:1769):softmax(axis=-3)
local Softmax2D = act_layer("Softmax2D", {}, function(self, x)
  local ndim = #x:shape()
  if ndim ~= 3 and ndim ~= 4 then
    error("paddle.nn.Softmax2D requires a 3D or 4D tensor as input. "
      .. "Received: " .. ndim .. "D.", 2)
  end
  return functional.softmax(x, -3)  -- activation.py:1808
end)

-- PReLU(activation.py:515):有可学习 weight(Constant(init));layer 与
-- functional.prelu 共享同一 weight tensor。
local PReLU = class(Layer)
PReLU._name = "PReLU"
PReLU._init = rule {
  { name = "self", type = Layer },
  { name = "num_parameters", type = argrule.number, opt = true, default = 1,
    doc = "1 or num_channels (activation.py:579)" },
  { name = "init", type = argrule.number, opt = true, default = 0.25,
    doc = "init value of weight (activation.py:580)" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "initializer (v2, 同 Linear)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "NC/NCL/NCHW/NCDHW/NLC/NHWC/NDHWC (activation.py:581)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, num_parameters, init, weight_attr, data_format, name)
  self:super()
  self._num_parameters = num_parameters
  self._data_format = data_format
  self.name = name
  local place = cpu_place()
  local w = paddle_core.pd_op_full({ num_parameters }, init,
    paddle_core.PD_DTYPE_FLOAT32, place)
  w:set_stop_gradient(false)
  self:add_parameter("weight", w)
end)
function PReLU:forward(x)
  -- activation.py:579-587:forward → F.prelu(x, self._weight, data_format)
  return functional.prelu(x, self.weight, self._data_format)
end

-- RReLU(activation.py:615):lower=1/8, upper=1/3, inplace;forward 透传
-- self.training(与 Python 一致:正反例都看 training)
local RReLU = act_layer("RReLU", {
  { name = "lower", type = argrule.number, opt = true, default = 1 / 8,
    doc = "lower bound (activation.py:700)" },
  { name = "upper", type = argrule.number, opt = true, default = 1 / 3,
    doc = "upper bound (activation.py:701)" },
  inplace_opt,
}, function(self, x)
  return functional.rrelu(x, self.lower, self.upper, self.training,
    self.inplace)
end)

return {
  CELU = CELU,
  ELU = ELU,
  GELU = GELU,
  Hardshrink = Hardshrink,
  Hardswish = Hardswish,
  Tanh = Tanh,
  Hardtanh = Hardtanh,
  PReLU = PReLU,
  RReLU = RReLU,
  ReLU = ReLU,
  ReLU6 = ReLU6,
  SELU = SELU,
  LeakyReLU = LeakyReLU,
  Sigmoid = Sigmoid,
  Hardsigmoid = Hardsigmoid,
  Softplus = Softplus,
  Softshrink = Softshrink,
  Softsign = Softsign,
  Swish = Swish,
  Mish = Mish,
  Tanhshrink = Tanhshrink,
  ThresholdedReLU = ThresholdedReLU,
  Silu = Silu,
  LogSigmoid = LogSigmoid,
  Softmax = Softmax,
  LogSoftmax = LogSoftmax,
  Maxout = Maxout,
  Softmax2D = Softmax2D,
}