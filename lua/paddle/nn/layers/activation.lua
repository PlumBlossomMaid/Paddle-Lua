-- lua/paddle/nn/layers/activation.lua -- paddle.nn 激活层类
-- Stage: P9 / M1 (WORKPLAN 2.9.4)。激活层类 = 极薄壳:构造存参,
-- forward 一行转发 functional(已 argrule 包装,2.9.3)。
--
-- Upstream blueprint: python/paddle/nn/layer/activation.py
-- (ReLU 726 附近,forward → F.relu(x, inplace, name) 756-766)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

local ReLU = class(Layer)
ReLU._name = "ReLU"

ReLU._init = rule {
  { name = "self", type = Layer },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (functional 报错,生成器 M2)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, inplace, name)
  self:super()
  self._inplace = inplace
  self.name = name
end)

function ReLU:forward(x)
  return functional.relu(x, self._inplace)
end

return ReLU