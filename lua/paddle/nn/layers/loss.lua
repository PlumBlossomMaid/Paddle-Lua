-- lua/paddle/nn/layers/loss.lua -- paddle.nn.CrossEntropyLoss
-- Stage: P9 / M1 (WORKPLAN 2.9.4)。损失层类:构造存参,forward 一行
-- 转发 functional.cross_entropy(已 argrule 包装,本批补)。
--
-- Upstream blueprint: python/paddle/nn/layer/loss.py:494
-- (CrossEntropyLoss.__init__)+ loss.py:517-527 (forward →
-- F.cross_entropy(input, label, weight, ignore_index, reduction,
-- soft_label, axis, use_softmax, label_smoothing))。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

local CrossEntropyLoss = class(Layer)
CrossEntropyLoss._name = "CrossEntropyLoss"

CrossEntropyLoss._init = rule {
  { name = "self", type = Layer },
  { name = "weight", type = argrule.any, opt = true,
    doc = "class weights (v2, functional 报错)" },
  { name = "ignore_index", type = argrule.number, opt = true, default = -100,
    doc = "class id to ignore, 1-based (C4, nn.md sec.3)" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:498)" },
  { name = "soft_label", type = argrule.boolean, opt = true, default = false,
    doc = "one-hot target (v2, functional 报错)" },
  { name = "axis", type = argrule.number, opt = true, default = -1,
    doc = "class axis, 1-based (C4)" },
  { name = "use_softmax", type = argrule.boolean, opt = true, default = true,
    doc = "softmax inside (loss.py:501)" },
  { name = "label_smoothing", type = argrule.number, opt = true, default = 0.0,
    doc = "smoothing epsilon (v2, functional 报错)" },
}(function(self, weight, ignore_index, reduction, soft_label, axis,
          use_softmax, label_smoothing)
  self:super()
  self.weight = weight
  self.ignore_index = ignore_index
  self.reduction = reduction
  self.soft_label = soft_label
  self.axis = axis
  self.use_softmax = use_softmax
  self.label_smoothing = label_smoothing
end)

function CrossEntropyLoss:forward(input, label)
  -- loss.py:517-527
  return functional.cross_entropy{
    input = input,
    label = label,
    weight = self.weight,
    ignore_index = self.ignore_index,
    reduction = self.reduction,
    soft_label = self.soft_label,
    axis = self.axis,
    use_softmax = self.use_softmax,
    label_smoothing = self.label_smoothing,
  }
end

return CrossEntropyLoss