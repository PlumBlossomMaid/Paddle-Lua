-- lua/paddle/nn/layers/norm.lua -- paddle.nn.BatchNorm2D
-- Stage: P9 / M1 (WORKPLAN 2.9.4)。归一化层类:构造创建
-- weight/bias(参数)+ _mean/_variance(buffer),forward 组装参数后一行
-- 转发 functional.batch_norm(已 argrule 包装,2.9.3.4)。
--
-- Upstream blueprint: python/paddle/nn/layer/norm.py:759 (_BatchNormBase)
-- + norm.py:1365 (BatchNorm2D)。forward 语义 norm.py:895-912:
--   momentum=None 且 training → 1.0 - 1.0/num_batches_tracked;
--   momentum=None → 0.0;否则用用户 momentum(默认 0.9)。
-- 训练时运行统计更新由 pd_op_batch_norm 内部完成(running mean/var
-- 就地更新,与上游 batch_norm(input, self._mean, ...) 一致)。

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

local BatchNorm2D = class(Layer)
BatchNorm2D._name = "BatchNorm2D"

BatchNorm2D._init = rule {
  { name = "self", type = Layer },
  { name = "num_features", type = argrule.number,
    doc = "channel count C of [N,C,H,W]" },
  { name = "momentum", type = argrule.number, opt = true, default = 0.9,
    doc = "running stats momentum (norm.py:763)" },
  { name = "epsilon", type = argrule.number, opt = true, default = 1e-5,
    doc = "variance epsilon (norm.py:764)" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "false = no scale; default Constant(1.0)" },
  { name = "bias_attr", type = argrule.any, opt = true,
    doc = "false = no shift; default Constant(0.0)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (norm.py:767)" },
  { name = "use_global_stats", type = argrule.boolean, opt = true,
    doc = "force global stats; nil = not training (norm.py:768)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, num_features, momentum, epsilon, weight_attr, bias_attr,
          data_format, use_global_stats, name)
  if data_format ~= "NCHW" and data_format ~= "NHWC" then
    error("paddle.nn.BatchNorm2D: data_format must be 'NCHW' or 'NHWC'", 2)
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

function BatchNorm2D:forward(input)
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

return BatchNorm2D