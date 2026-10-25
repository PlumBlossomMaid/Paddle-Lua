-- lua/paddle/nn/layers/pooling.lua -- paddle.nn.MaxPool2D
-- Stage: P9 / M1 (WORKPLAN 2.9.4)。池化层类 = 极薄壳:构造存参,
-- forward 一行转发 functional.max_pool2d(已 argrule 包装,2.9.3.3)。
--
-- Upstream blueprint: python/paddle/nn/layer/pooling.py:722 (MaxPool2D);
-- forward → F.max_pool2d(x, kernel_size, stride, padding, return_mask,
-- ceil_mode, dilation, data_format)(pooling.py:701-712 同构)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

local PoolLen = argrule.Union { argrule.number, argrule.table }

local MaxPool2D = class(Layer)
MaxPool2D._name = "MaxPool2D"

MaxPool2D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:747)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:748)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (pooling.py:753)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:754)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:755)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:756)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, kernel_size, stride, padding, return_mask, ceil_mode,
          dilation, data_format, name)
  self:super()
  self.kernel_size = kernel_size
  self.stride = stride
  self.padding = padding
  self.return_mask = return_mask
  self.ceil_mode = ceil_mode
  self.dilation = dilation
  self.data_format = data_format
  self.name = name
end)

function MaxPool2D:forward(x)
  return functional.max_pool2d(
    x, self.kernel_size, self.stride, self.padding, self.return_mask,
    self.ceil_mode, self.dilation, self.data_format)
end

return MaxPool2D