-- lua/paddle/nn/layers/norm.lua -- paddle.nn 归一化层类
-- Stage: P9 / M1 (WORKPLAN 2.9.4.2b)。本文件重写为表导出:
--   BatchNorm1D / BatchNorm2D / BatchNorm3D(_BatchNormBase 模板,
--   norm.py:759,data_format 差异)
--   LayerNorm(norm.py:597,forward → functional.layer_norm)
--   GroupNorm(norm.py:424,直调 pd_op_group_norm)
--
-- Upstream blueprint: python/paddle/nn/layer/norm.py
--   _BatchNormBase.__init__:759-933;BatchNorm1D:1240+ / BatchNorm2D:
--   1360+ / BatchNorm3D:1453(仅 data_format 默认与 _check_* 不同)
--   LayerNorm:597-757(weight Constant(1.0)/bias Constant(0.0),
--   forward → F.layer_norm(input, normalized_shape, weight, bias, eps))
--   GroupNorm:424-591(weight Constant(1.0)/bias Constant(0.0),
--   forward → F.group_norm)
--
-- BatchNorm 的 momentum=None 动态语义与 buffers 槽沿用 BatchNorm2D
-- 既有实现(2.9.4.1 已验证)。functional.batch_norm/layer_norm 已在
-- 2.9.3.4 包装并对 data_format 归一化(按第 2 字符 C)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

local CPU_PLACE_CACHE

local function cpu_place()
  if CPU_PLACE_CACHE == nil then
    CPU_PLACE_CACHE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  end
  return CPU_PLACE_CACHE
end

-- ---------------------------------------------------------------------
-- _BatchNormBase 模板工厂:BatchNorm1D/2D/3D 只有 data_format 集合不同
-- (norm.py:759;1D:1346 / 2D:1441 / 3D:1563 的 _check_data_format)。
-- ---------------------------------------------------------------------

local function build_batch_norm(ndim, default_fmt, valid_fmts)
  local Cls = class(Layer)
  Cls._name = "BatchNorm" .. ndim .. "D"

  Cls._init = rule {
    { name = "self", type = Layer },
    { name = "num_features", type = argrule.number,
      doc = "channel count C of input" },
    { name = "momentum", type = argrule.number, opt = true, default = 0.9,
      doc = "running stats momentum (norm.py:763)" },
    { name = "epsilon", type = argrule.number, opt = true, default = 1e-5,
      doc = "variance epsilon (norm.py:764)" },
    { name = "weight_attr", type = argrule.any, opt = true,
      doc = "false = no scale; default Constant(1.0)" },
    { name = "bias_attr", type = argrule.any, opt = true,
      doc = "false = no shift; default Constant(0.0)" },
    { name = "data_format", type = argrule.string, opt = true,
      default = default_fmt, doc = table.concat(valid_fmts, "/") },
    { name = "use_global_stats", type = argrule.boolean, opt = true,
      doc = "force global stats; nil = not training (norm.py:768)" },
    { name = "name", type = argrule.Optional { argrule.string }, opt = true,
      doc = "optional layer name" },
  }(function(self, num_features, momentum, epsilon, weight_attr, bias_attr,
            data_format, use_global_stats, name)
    local ok_fmt = false
    for _, f in ipairs(valid_fmts) do
      if data_format == f then ok_fmt = true end
    end
    if not ok_fmt then
      error("paddle.nn." .. Cls._name .. ": data_format must be one of "
        .. table.concat(valid_fmts, "/"), 2)
    end
    self:super()
    self._dtype = paddle_core.PD_DTYPE_FLOAT32
    local shape = { num_features }
    local place = cpu_place()
    if weight_attr ~= false then
      -- Constant(1.0):pd_op_full 造全 1,add_parameter 注册(参数槽)
      local w = paddle_core.pd_op_full(shape, 1.0, self._dtype, place)
      w:set_stop_gradient(false)
      self:add_parameter("weight", w)  -- norm.py:789
    else
      self.weight = nil
    end
    if bias_attr ~= false then
      local b = paddle_core.pd_op_full(shape, 0.0, self._dtype, place)
      b:set_stop_gradient(false)
      self:add_parameter("bias", b)  -- norm.py:801
    else
      self.bias = nil
    end
    -- 运行统计是 buffer(不参与训练):直接赋 tensor 走 buffers 槽
    -- (is_tensor → buffers,与 Python trainable=False 对应)。
    self._mean = paddle_core.pd_op_full(shape, 0.0, self._dtype, place)
    self._variance = paddle_core.pd_op_full(shape, 1.0, self._dtype, place)
    self._momentum = momentum
    self._epsilon = epsilon
    self._data_format = data_format
    self._use_global_stats = use_global_stats
    self._num_batches_tracked = 0
    self.name = name
  end)

  Cls.forward = function(self, input)
    -- norm.py:895-912 的 momentum 语义
    local momentum = self._momentum
    if self.training then
      if self._momentum == nil then
        self._num_batches_tracked = self._num_batches_tracked + 1
        momentum = 1.0 - 1.0 / self._num_batches_tracked
      end
    end
    if momentum == nil then momentum = 0.0 end
    return functional.batch_norm{
      x = input,
      running_mean = self._mean,
      running_var = self._variance,
      weight = self.weight,
      bias = self.bias,
      training = self.training,
      momentum = momentum,
      epsilon = self._epsilon,
      data_format = self._data_format,
      use_global_stats = self._use_global_stats,
    }
  end

  return Cls
end

local BatchNorm1D = build_batch_norm(1, "NCL", { "NCL", "NLC" })
local BatchNorm2D = build_batch_norm(2, "NCHW", { "NCHW", "NHWC" })
local BatchNorm3D = build_batch_norm(3, "NCDHW", { "NCDHW", "NDHWC" })

-- ---------------------------------------------------------------------
-- LayerNorm(norm.py:597)。normalized_shape 是形状(int 或 list,归一
-- 化最后 n 维);forward 按输入 ndim 换算成 1-based 轴范围闭区间
-- 传给 functional.layer_norm(C4,nn-functional.md sec.3)。weight 默认
-- Constant(1.0),bias 默认 Constant(0.0),elementwise_affine=False /
-- bias=False 时不创建。
-- ---------------------------------------------------------------------

local LayerNorm = class(Layer)
LayerNorm._name = "LayerNorm"

LayerNorm._init = rule {
  { name = "self", type = Layer },
  { name = "normalized_shape", type = argrule.Union { argrule.number,
    argrule.table },
    doc = "归一化形状:int 或 {n1,n2,...}(最后 n 维)" },
  { name = "epsilon", type = argrule.number, opt = true, default = 1e-5,
    doc = "variance epsilon (norm.py:598)" },
  { name = "elementwise_affine", type = argrule.boolean, opt = true,
    default = true, doc = "False → 不创建 weight/bias" },
  { name = "bias", type = argrule.boolean, opt = true, default = true,
    doc = "False → 不创建 bias(仍可创建 weight)" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "false = no weight; default Constant(1.0)" },
  { name = "bias_attr", type = argrule.any, opt = true,
    doc = "false = no bias; default Constant(0.0)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, normalized_shape, epsilon, elementwise_affine, bias,
          weight_attr, bias_attr, name)
  if not elementwise_affine then
    weight_attr = false
    bias_attr = false
  elseif not bias then
    bias_attr = false
  end
  self:super()
  local shape = normalized_shape
  if type(shape) == "number" then shape = { shape } end
  self._normalized_shape = shape
  self._epsilon = epsilon
  self._dtype = paddle_core.PD_DTYPE_FLOAT32
  local place = cpu_place()
  local n = 1
  for _, s in ipairs(shape) do n = n * s end
  if weight_attr ~= false then
    local w = paddle_core.pd_op_full({ n }, 1.0, self._dtype, place)
    w:set_stop_gradient(false)
    self:add_parameter("weight", w)
  else
    self.weight = nil
  end
  if bias_attr ~= false then
    local b = paddle_core.pd_op_full({ n }, 0.0, self._dtype, place)
    b:set_stop_gradient(false)
    self:add_parameter("bias", b)
  else
    self.bias = nil
  end
  self.name = name
end)

function LayerNorm:forward(input)
  -- 归一化最后 n 维 → 1-based 轴范围 {ndim-n+1, ndim}
  local ndim = #input:shape()
  local n = #self._normalized_shape
  local start = ndim - n + 1
  return functional.layer_norm{
    x = input,
    normalized_shape = { start, ndim },
    weight = self.weight,
    bias = self.bias,
    epsilon = self._epsilon,
  }
end

-- ---------------------------------------------------------------------
-- GroupNorm(norm.py:424)。weight Constant(1.0)/bias Constant(0.0),
-- data_format 支持一/二/三维布局并归一化成 NCHW/NHWC(norm.py:456)。
-- forward 直调 pd_op_group_norm(gen_ops_g.cc:149)。
-- ---------------------------------------------------------------------

local GroupNorm = class(Layer)
GroupNorm._name = "GroupNorm"

GroupNorm._init = rule {
  { name = "self", type = Layer },
  { name = "num_groups", type = argrule.number,
    doc = "number of groups (norm.py:426)" },
  { name = "num_channels", type = argrule.number,
    doc = "channel count C (norm.py:426)" },
  { name = "epsilon", type = argrule.number, opt = true, default = 1e-5,
    doc = "variance epsilon (norm.py:426)" },
  { name = "affine", type = argrule.boolean, opt = true, default = true,
    doc = "False → 不创建 weight/bias" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "false = no scale; default Constant(1.0)" },
  { name = "bias_attr", type = argrule.any, opt = true,
    doc = "false = no shift; default Constant(0.0)" },
  { name = "data_format", type = argrule.string, opt = true,
    default = "NCHW",
    doc = "'NCL'/'NCHW'/'NCDHW'/'NLC'/'NHWC'/'NDHWC'" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, num_groups, num_channels, epsilon, affine, weight_attr,
          bias_attr, data_format, name)
  if data_format ~= "NCL" and data_format ~= "NCHW"
    and data_format ~= "NCDHW" and data_format ~= "NLC"
    and data_format ~= "NHWC" and data_format ~= "NDHWC" then
    error("paddle.nn.GroupNorm: unsupported data layout: "
      .. tostring(data_format), 2)
  end
  if not affine then
    weight_attr = false
    bias_attr = false
  end
  self:super()
  self._epsilon = epsilon
  self._num_groups = num_groups
  -- norm.py:456:按第 2 个字符 C → NCHW,否则 NHWC
  self._data_format = "NCHW"
  if string.sub(data_format, 2, 2) ~= "C" then self._data_format = "NHWC" end
  self._dtype = paddle_core.PD_DTYPE_FLOAT32
  local shape = { num_channels }
  local place = cpu_place()
  if weight_attr ~= false then
    local w = paddle_core.pd_op_full(shape, 1.0, self._dtype, place)
    w:set_stop_gradient(false)
    self:add_parameter("weight", w)  -- norm.py:534
  else
    self.weight = nil
  end
  if bias_attr ~= false then
    local b = paddle_core.pd_op_full(shape, 0.0, self._dtype, place)
    b:set_stop_gradient(false)
    self:add_parameter("bias", b)  -- norm.py:548
  else
    self.bias = nil
  end
  self.name = name
end)

function GroupNorm:forward(input)
  -- ops_gen.h:237:pd_op_group_norm(x, scale, bias, epsilon, groups,
  -- data_format)
  return paddle_core.pd_op_group_norm(
    input, self.weight, self.bias, self._epsilon, self._num_groups,
    self._data_format)
end

return {
  BatchNorm1D = BatchNorm1D,
  BatchNorm2D = BatchNorm2D,
  BatchNorm3D = BatchNorm3D,
  LayerNorm = LayerNorm,
  GroupNorm = GroupNorm,
}