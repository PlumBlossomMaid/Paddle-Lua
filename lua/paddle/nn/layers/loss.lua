-- lua/paddle/nn/layers/loss.lua -- paddle.nn 损失层类
-- Stage: P9 / M1 (WORKPLAN 2.9.4.2b)。本文件重写为表导出:
--   CrossEntropyLoss(2.9.4.1 已有,保留语义)
--   MSELoss / L1Loss / SmoothL1Loss / BCELoss / BCEWithLogitsLoss
--
-- Upstream blueprint: python/paddle/nn/layer/loss.py
--   CrossEntropyLoss:494 / MSELoss:684 / L1Loss:773 / BCELoss:865 /
--   BCEWithLogitsLoss:83 / SmoothL1Loss:1614
--
-- 构造存参(全部 argrule 包装,self 规则),forward 一行转发 functional
-- 已验 API(2.9.3.4):mse_loss/l1_loss/smooth_l1_loss/binary_cross_entropy/
-- binary_cross_entropy_with_logits。label 均非 index 语义(连续 target)。
-- reduction 三态校验在 functional 层(reduce_out),构造不重复校验 ——
-- 镜像 Python(BCELoss 构造校验 reduction?loss.py:865 无校验,forward
-- 转发;MSELoss 构造校验 loss.py:704)。构造校验只做 MSELoss/L1Loss
-- 镜像,其余靠 functional 兜底。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

-- CrossEntropyLoss(2.9.4.1 原实现,语义不变)
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

-- ---------------------------------------------------------------------
-- MSELoss(loss.py:684)。构造校验 reduction(loss.py:704 镜像)。
-- ---------------------------------------------------------------------

local MSELoss = class(Layer)
MSELoss._name = "MSELoss"

MSELoss._init = rule {
  { name = "self", type = Layer },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:697)" },
}(function(self, reduction)
  if reduction ~= "sum" and reduction ~= "mean" and reduction ~= "none" then
    error("'reduction' in 'MSELoss' should be 'sum', 'mean' or 'none', "
      .. "but received " .. tostring(reduction), 2)
  end
  self:super()
  self.reduction = reduction
end)

function MSELoss:forward(input, label)
  return functional.mse_loss{ input = input, label = label,
    reduction = self.reduction }
end

-- ---------------------------------------------------------------------
-- L1Loss(loss.py:773)
-- ---------------------------------------------------------------------

local L1Loss = class(Layer)
L1Loss._name = "L1Loss"

L1Loss._init = rule {
  { name = "self", type = Layer },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:783)" },
}(function(self, reduction)
  self:super()
  self.reduction = reduction
end)

function L1Loss:forward(input, label)
  return functional.l1_loss{ input = input, label = label,
    reduction = self.reduction }
end

-- ---------------------------------------------------------------------
-- SmoothL1Loss(loss.py:1614)
-- ---------------------------------------------------------------------

local SmoothL1Loss = class(Layer)
SmoothL1Loss._name = "SmoothL1Loss"

SmoothL1Loss._init = rule {
  { name = "self", type = Layer },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:1636)" },
  { name = "delta", type = argrule.number, opt = true, default = 1.0,
    doc = "huber threshold (loss.py:1637)" },
  { name = "is_huber", type = argrule.boolean, opt = true, default = true,
    doc = "False divides loss by delta (loss.py:1638)" },
}(function(self, reduction, delta, is_huber)
  self:super()
  self.reduction = reduction
  self.delta = delta
  self.is_huber = is_huber
end)

function SmoothL1Loss:forward(input, label)
  return functional.smooth_l1_loss{ input = input, label = label,
    reduction = self.reduction, delta = self.delta,
    is_huber = self.is_huber }
end

-- ---------------------------------------------------------------------
-- BCELoss(loss.py:865)。label 非 index(连续 target)。
-- ---------------------------------------------------------------------

local BCELoss = class(Layer)
BCELoss._name = "BCELoss"

BCELoss._init = rule {
  { name = "self", type = Layer },
  { name = "weight", type = argrule.any, opt = true,
    doc = "per-element rescale tensor (loss.py:868); nil = none" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:869)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, weight, reduction, name)
  self:super()
  self.weight = weight
  self.reduction = reduction
  self.name = name
end)

function BCELoss:forward(input, label)
  -- loss.py:908-911
  return functional.binary_cross_entropy{ input = input, label = label,
    weight = self.weight, reduction = self.reduction }
end

-- ---------------------------------------------------------------------
-- BCEWithLogitsLoss(loss.py:83)
-- ---------------------------------------------------------------------

local BCEWithLogitsLoss = class(Layer)
BCEWithLogitsLoss._name = "BCEWithLogitsLoss"

BCEWithLogitsLoss._init = rule {
  { name = "self", type = Layer },
  { name = "weight", type = argrule.any, opt = true,
    doc = "per-element rescale tensor (loss.py:86); nil = none" },
  { name = "reduction", type = argrule.string, opt = true, default = "mean",
    doc = "'none' | 'mean' | 'sum' (loss.py:87)" },
  { name = "pos_weight", type = argrule.any, opt = true,
    doc = "positive-class weight (loss.py:88); nil = none" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true },
}(function(self, weight, reduction, pos_weight, name)
  self:super()
  self.weight = weight
  self.reduction = reduction
  self.pos_weight = pos_weight
  self.name = name
end)

function BCEWithLogitsLoss:forward(logit, label)
  -- loss.py:168-175
  return functional.binary_cross_entropy_with_logits{
    logit = logit,
    label = label,
    weight = self.weight,
    reduction = self.reduction,
    pos_weight = self.pos_weight,
  }
end

return {
  CrossEntropyLoss = CrossEntropyLoss,
  MSELoss = MSELoss,
  L1Loss = L1Loss,
  SmoothL1Loss = SmoothL1Loss,
  BCELoss = BCELoss,
  BCEWithLogitsLoss = BCEWithLogitsLoss,
}