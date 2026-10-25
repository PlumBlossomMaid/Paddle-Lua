-- lua/paddle/nn/layers/linear.lua -- paddle.nn.Linear
-- Stage: P9 / M1 (WORKPLAN 2.9.2). First real layer: proves the
-- auto-registration + create_parameter + forward loop works end to end.
--
-- Design: docs/migration/plan/api/nn.md sec.2.3 + modules/09-nn.md.
-- Upstream blueprint: python/paddle/nn/layer/common.py:210 (Linear
-- __init__) + nn/functional/common.py:2469 (F.linear dispatch).
--
-- Parameter shapes (common.py:224-229): weight = [in_features,
-- out_features], bias = [out_features]; both created via
-- Layer:create_parameter (auto-registered, stop_gradient=false).
--
-- Forward dispatch: upstream F.linear on CPU/dygraph calls
-- _C_ops.linear(x, weight, bias) (common.py:2541); the generated op is
-- linear_v2 (ops.yaml:3246, `Tensor input, Tensor weight, Tensor bias,
-- bool transpose_weight=false`), bound as pd_op_linear_v2
-- (gen_ops_l.cc:113-117). With NO bias upstream falls back to
-- _C_ops.matmul(x, weight) (common.py:2596); linear_v2's bias is a
-- REQUIRED tensor in the yaml, so the no-bias path must use matmul
-- (pd_op_matmul, gen_ops_m.cc:72-75).
--
-- SIGNATURE DISCIPLINE (D31 + 用户 2026-10-25): 正式 API 一律用
-- argrule 包装,不手写解析器。The ctor is a rule-wrapped function whose
-- first rule is named `self` -- this makes pl.class's ctor chain
-- (`Linear{2, 3}` -> call_ctor(obj, {2,3})) normalize through argrule's
-- parse_self_call into (self, in_features, out_features, ...). All
-- three call shapes work (api/README.md sec.2.1.5):
--   Linear{2, 3}                          -- table positional
--   Linear{ in_features=2, out_features=3 }  -- named
--   Linear(2, 3)                           -- positional
--   Linear{2, 3, bias_attr=false}          -- mixed tail

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local Layer = require("paddle.nn.layer").Layer

local Linear = class(Layer)
Linear._name = "Linear"

-- v1: weight_attr/bias_attr initializer OBJECTS land with 2.9.3
-- (nn.initializer); the only branch honored today is bias_attr == false
-- (no bias, upstream docstring common.py:149). Default fill is
-- Constant(0) via create_parameter (Xavier/Constant default arrives with
-- 2.9.3). weight_attr/bias_attr accept `any` for now so initializer
-- objects pass the type gate before 2.9.3 wires them.
Linear._init = rule {
  { name = "self", type = Layer,
    doc = "pl.class instance (auto) -- first rule is self (argrule QWEN.md)" },
  { name = "in_features", type = argrule.number, doc = "number of input units" },
  { name = "out_features", type = argrule.number, doc = "number of output units" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "initializer for weight (2.9.3)" },
  { name = "bias_attr", type = argrule.any, opt = true,
    doc = "false = no bias; initializer lands with 2.9.3" },
  { name = "name", type = argrule.Optional { argrule.string }, opt = true,
    doc = "optional layer name" },
}(function(self, in_features, out_features, weight_attr, bias_attr, name)
  self:super()
  self._dtype = paddle_core.PD_DTYPE_FLOAT32  -- v1: get_default_dtype pending
  self.weight = self:create_parameter{
    shape = { in_features, out_features },
    dtype = self._dtype,
  }
  if bias_attr ~= false then
    self.bias = self:create_parameter{
      shape = { out_features },
      dtype = self._dtype,
    }
  end
  self.name = name
end)

function Linear:forward(x)
  if self.bias ~= nil then
    return paddle_core.pd_op_linear_v2(x, self.weight, self.bias, 0)
  end
  return paddle_core.pd_op_matmul(x, self.weight, 0, 0)
end

return Linear
