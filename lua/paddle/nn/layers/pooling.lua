-- lua/paddle/nn/layers/pooling.lua -- paddle.nn 池化层类(12 个)
-- Stage: P9 / M1 (WORKPLAN 2.9.4)。池化层类 = 极薄壳:构造存参,
-- forward 转发 functional.avg/max_pool* + adaptive_*(已 argrule 包装,
-- 2.9.3.3)。上游 blueprint: python/paddle/nn/layer/pooling.py(全部类
-- 见文件内注释)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

local PoolLen = argrule.Union { argrule.number, argrule.table }

-- ---------------------------------------------------------------------
-- AvgPool1D (pooling.py:50,__init__ 115)
-- ---------------------------------------------------------------------

local AvgPool1D = class(Layer)
AvgPool1D._name = "AvgPool1D"

AvgPool1D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {L})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:116)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:117)" },
  { name = "exclusive", type = argrule.boolean, opt = true, default = true,
    doc = "exclude padding in avg (pooling.py:118)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:119)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, kernel_size, stride, padding, exclusive, ceil_mode, name)
  self:super()
  self.kernel_size = kernel_size
  self.stride = stride
  self.padding = padding
  self.exclusive = exclusive
  self.ceil_mode = ceil_mode
  self.name = name
end)

function AvgPool1D:forward(x)
  return functional.avg_pool1d(
    x, self.kernel_size, self.stride, self.padding, self.exclusive,
    self.ceil_mode)
end

-- ---------------------------------------------------------------------
-- AvgPool2D (pooling.py:150,__init__ 231)
-- ---------------------------------------------------------------------

local AvgPool2D = class(Layer)
AvgPool2D._name = "AvgPool2D"

AvgPool2D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:232)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:233)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:234)" },
  { name = "exclusive", type = argrule.boolean, opt = true, default = true,
    doc = "exclude padding in avg (pooling.py:235)" },
  { name = "divisor_override", type = argrule.number, opt = true,
    doc = "override division factor (pooling.py:236)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:237)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, kernel_size, stride, padding, ceil_mode, exclusive,
          divisor_override, data_format, name)
  self:super()
  self.kernel_size = kernel_size
  self.stride = stride
  self.padding = padding
  self.ceil_mode = ceil_mode
  self.exclusive = exclusive
  self.divisor_override = divisor_override
  self.data_format = data_format
  self.name = name
end)

function AvgPool2D:forward(x)
  return functional.avg_pool2d{
    x = x, kernel_size = self.kernel_size, stride = self.stride,
    padding = self.padding, ceil_mode = self.ceil_mode,
    exclusive = self.exclusive, divisor_override = self.divisor_override,
    data_format = self.data_format,
  }
end

-- ---------------------------------------------------------------------
-- AvgPool3D (pooling.py:271,__init__ 341)
-- ---------------------------------------------------------------------

local AvgPool3D = class(Layer)
AvgPool3D._name = "AvgPool3D"

AvgPool3D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen,
    doc = "pool window (int or {D,H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:342)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:343)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:344)" },
  { name = "exclusive", type = argrule.boolean, opt = true, default = true,
    doc = "exclude padding in avg (pooling.py:345)" },
  { name = "divisor_override", type = argrule.number, opt = true,
    doc = "override division factor (pooling.py:346)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' or 'NDHWC' (pooling.py:347)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, kernel_size, stride, padding, ceil_mode, exclusive,
          divisor_override, data_format, name)
  self:super()
  self.kernel_size = kernel_size
  self.stride = stride
  self.padding = padding
  self.ceil_mode = ceil_mode
  self.exclusive = exclusive
  self.divisor_override = divisor_override
  self.data_format = data_format
  self.name = name
end)

function AvgPool3D:forward(x)
  return functional.avg_pool3d{
    x = x, kernel_size = self.kernel_size, stride = self.stride,
    padding = self.padding, ceil_mode = self.ceil_mode,
    exclusive = self.exclusive, divisor_override = self.divisor_override,
    data_format = self.data_format,
  }
end

-- ---------------------------------------------------------------------
-- MaxPool1D (pooling.py:605,__init__ 684)
-- ---------------------------------------------------------------------

local MaxPool1D = class(Layer)
MaxPool1D._name = "MaxPool1D"

MaxPool1D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {L})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:685)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:686)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (pooling.py:687)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:688)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:689)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, kernel_size, stride, padding, return_mask, ceil_mode,
          dilation, name)
  self:super()
  self.kernel_size = kernel_size
  self.stride = stride
  self.padding = padding
  self.return_mask = return_mask
  self.ceil_mode = ceil_mode
  self.dilation = dilation
  self.name = name
end)

function MaxPool1D:forward(x)
  return functional.max_pool1d{
    x = x, kernel_size = self.kernel_size, stride = self.stride,
    padding = self.padding, return_mask = self.return_mask,
    ceil_mode = self.ceil_mode, dilation = self.dilation,
  }
end

-- ---------------------------------------------------------------------
-- MaxPool2D (pooling.py:722,__init__ 816)
-- ---------------------------------------------------------------------

local MaxPool2D = class(Layer)
MaxPool2D._name = "MaxPool2D"

MaxPool2D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen, doc = "pool window (int or {H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:817)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:818)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (pooling.py:819)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:820)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:821)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:822)" },
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
  return functional.max_pool2d{
    x = x, kernel_size = self.kernel_size, stride = self.stride,
    padding = self.padding, return_mask = self.return_mask,
    ceil_mode = self.ceil_mode, dilation = self.dilation,
    data_format = self.data_format,
  }
end

-- ---------------------------------------------------------------------
-- MaxPool3D (pooling.py:857,__init__ 939)
-- ---------------------------------------------------------------------

local MaxPool3D = class(Layer)
MaxPool3D._name = "MaxPool3D"

MaxPool3D._init = rule {
  { name = "self", type = Layer },
  { name = "kernel_size", type = PoolLen,
    doc = "pool window (int or {D,H,W})" },
  { name = "stride", type = PoolLen, opt = true,
    doc = "stride; nil = kernel_size (pooling.py:940)" },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID' (pooling.py:941)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices;NCDHW only (pooling.py:942)" },
  { name = "ceil_mode", type = argrule.boolean, opt = true, default = false,
    doc = "ceil output shape (pooling.py:943)" },
  { name = "dilation", type = PoolLen, opt = true, default = 1,
    doc = "dilation (pooling.py:944)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' or 'NDHWC' (pooling.py:945)" },
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

function MaxPool3D:forward(x)
  return functional.max_pool3d{
    x = x, kernel_size = self.kernel_size, stride = self.stride,
    padding = self.padding, return_mask = self.return_mask,
    ceil_mode = self.ceil_mode, dilation = self.dilation,
    data_format = self.data_format,
  }
end

-- ---------------------------------------------------------------------
-- AdaptiveAvgPool1D (pooling.py:980,__init__ 1034)
-- ---------------------------------------------------------------------

local AdaptiveAvgPool1D = class(Layer)
AdaptiveAvgPool1D._name = "AdaptiveAvgPool1D"

AdaptiveAvgPool1D._init = rule {
  { name = "self", type = Layer },
  { name = "output_size", type = PoolLen, doc = "target L (int or {L})" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, output_size, name)
  self:super()
  self.output_size = output_size
  self.name = name
end)

function AdaptiveAvgPool1D:forward(x)
  return functional.adaptive_avg_pool1d(x, self.output_size)
end

-- ---------------------------------------------------------------------
-- AdaptiveAvgPool2D (pooling.py:1046,__init__ 1114)
-- ---------------------------------------------------------------------

local AdaptiveAvgPool2D = class(Layer)
AdaptiveAvgPool2D._name = "AdaptiveAvgPool2D"

AdaptiveAvgPool2D._init = rule {
  { name = "self", type = Layer },
  { name = "output_size", type = PoolLen, doc = "target {H,W} (int = both)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC' (pooling.py:1115)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, output_size, data_format, name)
  self:super()
  self.output_size = output_size
  self.data_format = data_format
  self.name = name
end)

function AdaptiveAvgPool2D:forward(x)
  return functional.adaptive_avg_pool2d(
    x, self.output_size, self.data_format)
end

-- ---------------------------------------------------------------------
-- AdaptiveAvgPool3D (pooling.py:1146,__init__ 1221)
-- ---------------------------------------------------------------------

local AdaptiveAvgPool3D = class(Layer)
AdaptiveAvgPool3D._name = "AdaptiveAvgPool3D"

AdaptiveAvgPool3D._init = rule {
  { name = "self", type = Layer },
  { name = "output_size", type = PoolLen,
    doc = "target {D,H,W} (int = all)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' or 'NDHWC' (pooling.py:1222)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, output_size, data_format, name)
  self:super()
  self.output_size = output_size
  self.data_format = data_format
  self.name = name
end)

function AdaptiveAvgPool3D:forward(x)
  return functional.adaptive_avg_pool3d(
    x, self.output_size, self.data_format)
end

-- ---------------------------------------------------------------------
-- AdaptiveMaxPool1D (pooling.py:1253,__init__ 1326)
-- ---------------------------------------------------------------------

local AdaptiveMaxPool1D = class(Layer)
AdaptiveMaxPool1D._name = "AdaptiveMaxPool1D"

AdaptiveMaxPool1D._init = rule {
  { name = "self", type = Layer },
  { name = "output_size", type = PoolLen, doc = "target L (int or {L})" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (pooling.py:1327)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, output_size, return_mask, name)
  self:super()
  self.output_size = output_size
  self.return_mask = return_mask
  self.name = name
end)

function AdaptiveMaxPool1D:forward(x)
  return functional.adaptive_max_pool1d(
    x, self.output_size, self.return_mask)
end

-- ---------------------------------------------------------------------
-- AdaptiveMaxPool2D (pooling.py:1354,__init__ 1421)
-- ---------------------------------------------------------------------

local AdaptiveMaxPool2D = class(Layer)
AdaptiveMaxPool2D._name = "AdaptiveMaxPool2D"

AdaptiveMaxPool2D._init = rule {
  { name = "self", type = Layer },
  { name = "output_size", type = PoolLen, doc = "target {H,W} (int = both)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices (pooling.py:1422)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, output_size, return_mask, name)
  self:super()
  self.output_size = output_size
  self.return_mask = return_mask
  self.name = name
end)

function AdaptiveMaxPool2D:forward(x)
  return functional.adaptive_max_pool2d(
    x, self.output_size, self.return_mask)
end

-- ---------------------------------------------------------------------
-- AdaptiveMaxPool3D (pooling.py:1455,__init__ 1527 附近)
-- ---------------------------------------------------------------------

local AdaptiveMaxPool3D = class(Layer)
AdaptiveMaxPool3D._name = "AdaptiveMaxPool3D"

AdaptiveMaxPool3D._init = rule {
  { name = "self", type = Layer },
  { name = "output_size", type = PoolLen,
    doc = "target {D,H,W} (int = all)" },
  { name = "return_mask", type = argrule.boolean, opt = true, default = false,
    doc = "also return max indices" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, output_size, return_mask, name)
  self:super()
  self.output_size = output_size
  self.return_mask = return_mask
  self.name = name
end)

function AdaptiveMaxPool3D:forward(x)
  return functional.adaptive_max_pool3d(
    x, self.output_size, self.return_mask)
end

return {
  AvgPool1D = AvgPool1D,
  AvgPool2D = AvgPool2D,
  AvgPool3D = AvgPool3D,
  MaxPool1D = MaxPool1D,
  MaxPool2D = MaxPool2D,
  MaxPool3D = MaxPool3D,
  AdaptiveAvgPool1D = AdaptiveAvgPool1D,
  AdaptiveAvgPool2D = AdaptiveAvgPool2D,
  AdaptiveAvgPool3D = AdaptiveAvgPool3D,
  AdaptiveMaxPool1D = AdaptiveMaxPool1D,
  AdaptiveMaxPool2D = AdaptiveMaxPool2D,
  AdaptiveMaxPool3D = AdaptiveMaxPool3D,
}
