-- lua/paddle/nn/layers/common.lua -- paddle.nn.Flatten / Dropout / Embedding
--                                        + Dropout2D / Dropout3D (2.9.4.2b)
-- Stage: P9 / M1 (WORKPLAN 2.9.4)。上游全部在 python/paddle/nn/layer/
-- common.py(Flatten 2761-2790、Dropout 813-910、Embedding 2291+、
-- Dropout2D 911-994、Dropout3D 996-1082)。
--
-- Flatten:start_axis/stop_axis 是轴(C4 1-based)。Python 默认
-- (start_axis=1, stop_axis=-1) 0-based = 从第 2 维展平到末尾(保留
-- batch)。Lua 1-based 等价默认 = (2, -1)。生成物 _ops/flatten.lua 对
-- 轴无 1-based 转换(flatten=true 在 no_index 段),层内自处理:
--   正数 n → n-1;负数不变(与 C4 表一致)。
--
-- Dropout:forward → F.dropout(input, p, axis, training=self.training,
-- inplace, mode)(common.py:840-851)。functional.M.dropout 已 argrule
-- 包装(本批补),axis=nil 主路径。
--
-- Embedding:forward 直调 paddle_core.pd_op_embedding(x, weight,
-- padding_idx, sparse)(ops_gen.h:155)。输入 id x 是 index 语义,
-- C ABI 层已做 pd_idx_minus1(ops_e.cc:104);padding_idx 也是 index
-- 语义标量 → 正数 -1(C4),负数不变;nil → -1(无 padding,生成物默认)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local Layer = require("paddle.nn.layer").Layer
local functional = require "paddle.nn.functional"

-- ---------------------------------------------------------------------
-- Flatten
-- ---------------------------------------------------------------------

local Flatten = class(Layer)
Flatten._name = "Flatten"

Flatten._init = rule {
  { name = "self", type = Layer },
  { name = "start_axis", type = argrule.number, opt = true, default = 2,
    doc = "first dim to flatten, 1-based (C4; default 2 ≡ Python 1)" },
  { name = "stop_axis", type = argrule.number, opt = true, default = -1,
    doc = "last dim to flatten, 1-based (C4; default -1 = last)" },
}(function(self, start_axis, stop_axis)
  self:super()
  self.start_axis = start_axis
  self.stop_axis = stop_axis
end)

function Flatten:forward(input)
  -- common.py:2783:out = paddle.flatten(input, start_axis, stop_axis)。
  -- 层内 1-based → 0-based(正数 -1,负数不变)。
  local sa = (self.start_axis > 0) and (self.start_axis - 1)
    or self.start_axis
  local so = (self.stop_axis > 0) and (self.stop_axis - 1)
    or self.stop_axis
  return paddle_core.pd_op_flatten(input, sa, so)  -- gen_ops_f.cc:351
end

-- ---------------------------------------------------------------------
-- Dropout
-- ---------------------------------------------------------------------

local Dropout = class(Layer)
Dropout._name = "Dropout"

Dropout._init = rule {
  { name = "self", type = Layer },
  { name = "p", type = argrule.number, opt = true, default = 0.5,
    doc = "zero-out probability (common.py:821)" },
  { name = "axis", type = argrule.any, opt = true,
    doc = "dropout_nd axes; nil = element-wise" },
  { name = "mode", type = argrule.string, opt = true,
    default = "upscale_in_train",
    doc = "'upscale_in_train' | 'downscale_in_infer' (common.py:824)" },
  { name = "inplace", type = argrule.boolean, opt = true, default = false,
    doc = "in-place variant (functional 报错,生成器 M2)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, p, axis, mode, inplace, name)
  self:super()
  self.p = p
  self.axis = axis
  self.mode = mode
  self.inplace = inplace
  self.name = name
end)

function Dropout:forward(input)
  -- common.py:840-851:training 透传 self.training(train/eval 切换)
  return functional.dropout{
    x = input,
    p = self.p,
    axis = self.axis,
    training = self.training,
    inplace = self.inplace,
    mode = self.mode,
  }
end

-- ---------------------------------------------------------------------
-- Dropout2D(common.py:911-994):整通道置零。forward → F.dropout2d →
-- dropout(x, p, axis=[0,1] NCHW / [0,3] NHWC, training,
-- mode='upscale_in_train')(common.py:1743-1747)。axis 是 1-based(C4):
-- NCHW {1,2} = Python [0,1];NHWC {1,4} = [0,3]。
-- ---------------------------------------------------------------------

local Dropout2D = class(Layer)
Dropout2D._name = "Dropout2D"

Dropout2D._init = rule {
  { name = "self", type = Layer },
  { name = "p", type = argrule.number, opt = true, default = 0.5,
    doc = "zero-out probability (common.py:917)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' | 'NHWC' (common.py:918)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, p, data_format, name)
  if data_format ~= "NCHW" and data_format ~= "NHWC" then
    error("paddle.nn.Dropout2D: data_format must be 'NCHW' or 'NHWC'", 2)
  end
  self:super()
  self.p = p
  self.data_format = data_format
  self.name = name
end)

function Dropout2D:forward(input)
  local axis
  if self.data_format == "NCHW" then
    axis = { 1, 2 }
  else
    axis = { 1, 4 }
  end
  return functional.dropout{
    x = input,
    p = self.p,
    axis = axis,
    training = self.training,
    mode = "upscale_in_train",
  }
end

-- ---------------------------------------------------------------------
-- Dropout3D(common.py:996-1082):整通道置零(3D 特征面)。forward →
-- F.dropout3d → dropout(x, p, axis=[0,1] NCDHW / [0,4] NDHWC,
-- training, mode='upscale_in_train')(common.py:1810-1812)。1-based:
-- NCDHW {1,2} = [0,1];NDHWC {1,5} = [0,4]。
-- ---------------------------------------------------------------------

local Dropout3D = class(Layer)
Dropout3D._name = "Dropout3D"

Dropout3D._init = rule {
  { name = "self", type = Layer },
  { name = "p", type = argrule.number, opt = true, default = 0.5,
    doc = "zero-out probability (common.py:1001)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCDHW",
    doc = "'NCDHW' | 'NDHWC' (common.py:1002)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, p, data_format, name)
  if data_format ~= "NCDHW" and data_format ~= "NDHWC" then
    error("paddle.nn.Dropout3D: data_format must be 'NCDHW' or 'NDHWC'", 2)
  end
  self:super()
  self.p = p
  self.data_format = data_format
  self.name = name
end)

function Dropout3D:forward(input)
  local axis
  if self.data_format == "NCDHW" then
    axis = { 1, 2 }
  else
    axis = { 1, 5 }
  end
  return functional.dropout{
    x = input,
    p = self.p,
    axis = axis,
    training = self.training,
    mode = "upscale_in_train",
  }
end

-- ---------------------------------------------------------------------
-- Embedding
-- ---------------------------------------------------------------------

local Embedding = class(Layer)
Embedding._name = "Embedding"

Embedding._init = rule {
  { name = "self", type = Layer },
  { name = "num_embeddings", type = argrule.number,
    doc = "vocab size (common.py:2292)" },
  { name = "embedding_dim", type = argrule.number,
    doc = "embedding vector size (common.py:2293)" },
  { name = "padding_idx", type = argrule.number, opt = true,
    doc = "padding token id, 1-based (C4); nil = none" },
  { name = "sparse", type = argrule.boolean, opt = true, default = false,
    doc = "sparse gradient update (common.py:2295)" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "initializer (v2, 同 Linear)" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, num_embeddings, embedding_dim, padding_idx, sparse,
          weight_attr, name)
  self:super()
  self._dtype = paddle_core.PD_DTYPE_FLOAT32
  self.weight = self:create_parameter{
    shape = { num_embeddings, embedding_dim },
    dtype = self._dtype,
  }
  self._padding_idx = padding_idx
  self._sparse = sparse
  self.name = name
end)

function Embedding:forward(x)
  -- 输入 id 内容 1-based → C ABI 层 pd_idx_minus1(ops_e.cc:104)。
  -- padding_idx 标量 index:正数 -1,负数不变,nil → -1。
  local pid = self._padding_idx
  if pid == nil then
    pid = -1
  elseif pid > 0 then
    pid = pid - 1
  end
  return paddle_core.pd_op_embedding(
    x, self.weight, pid, (self._sparse and 1 or 0))  -- ops_gen.h:155
end

return {
  Flatten = Flatten,
  Dropout = Dropout,
  Dropout2D = Dropout2D,
  Dropout3D = Dropout3D,
  Embedding = Embedding,
}