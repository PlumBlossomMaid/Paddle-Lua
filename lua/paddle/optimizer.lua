-- lua/paddle/optimizer.lua -- paddle.optimizer: Optimizer base + SGD + Adam
-- Stage: P10 / M1 (WORKPLAN 2.10). First batch per contract
-- (api/optimizer.md §2): Optimizer / SGD / Adam; Momentum/AdamW and the
-- lr schedulers land in the 2.10 follow-ups.
--
-- ★ M1 has NO fused optimizer ops (10-optimizer.md §2's premise is
-- rewritten): _C_ops.sgd_/adam_ are all in-place variants (trailing `_`)
-- skipped by the generator (skip_rules.lua:45) until M2. So this module
-- COMPOSES basic ops and copies the kernel math verbatim:
--   SGD : param -= lr * grad            (cpu/sgd_kernel.cc:39)
--   Adam: adam_functors.h:325-345
--         lr_eff    = lr * sqrt(1-beta2_pow) / (1-beta1_pow)
--         mom1      = beta1*mom1 + (1-beta1)*g
--         mom2      = beta2*mom2 + (1-beta2)*g*g
--         param    -= lr_eff * mom1 / (sqrt(mom2) + eps*sqrt(1-beta2_pow))
--         betaN_pow *= betaN            (cpu/adam_kernel.cc:118-119)
-- betaN_pow INITIAL = betaN (adam.py:311-335 with fill_value=betaN, NOT 1).
--
-- Write-back discipline (10-optimizer.md sec.4): composed ops produce a
-- NEW tensor; the update is written INTO the parameter's existing storage
-- via p:copy_from(new) -- the new C ABI pd_tensor_copy_from
-- (paddle_c.h, -> paddle::Tensor::copy_, tensor_method.cc:98, deep copy,
-- keeps the param's autograd meta / leaf identity), exactly the in-place
-- semantics Python gets from _C_ops.sgd_/adam_ ParamOut.
--
-- step() runs under no_grad (Python's @imperative_base.no_grad(),
-- optimizer.py:2060): composed ops must NOT build a graph off the leaf
-- parameters. pcall-guard restores the flag even on error (same pattern
-- as lua/paddle/autograd.lua no_grad).
--
-- Lua 5.1 subset (C3); every Paddle symbol below was read from
-- $PADDLE_ROOT sources (paths given at each use).

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"

-- CPU place cache (full/zeros_like need a place argument)
local CPU_PLACE
local function cpu_place()
  if CPU_PLACE == nil then
    CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  end
  return CPU_PLACE
end

-- Structural LuaTensor test (same probe pattern as nn/functional.lua)
local probe = paddle_core.tensor{1}
local TENSOR_MT = getmetatable(probe)
probe = nil
local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end
local Tensor = argrule.alias { function(v) return is_tensor(v) end }

-- parameters contract (api/optimizer.md §2.1): net:parameters() returns
-- a pl.List of Tensors; also accept a plain Lua array. # works for both
-- (pl.List keeps elements in its array part).
local ParamList = argrule.alias { function(v)
  if type(v) ~= "table" then return false end
  local n = #v
  for i = 1, n do
    if not is_tensor(v[i]) then return false end
  end
  return true
end }

-- ---------------------------------------------------------------------
-- Optimizer base (api/optimizer.md §2.1)
-- ---------------------------------------------------------------------

local Optimizer = class()          -- pl.class: class() not class("name")
Optimizer._name = "Optimizer"

Optimizer._init = rule {
  { name = "self", type = Optimizer, doc = "pl.class instance (auto)" },
  { name = "parameters", type = ParamList,
    doc = "net:parameters() -- pl.List of Tensor" },
  { name = "learning_rate", type = argrule.number, opt = true, default = 0.1,
    doc = "learning rate (scheduler objects land with later batches)" },
  { name = "weight_decay", type = argrule.number, opt = true,
    doc = "L2 coefficient (coupled: added into grad, see _grad_with_wd)" },
  { name = "grad_clip", type = argrule.any, opt = true,
    doc = "gradient clipping -- NOT implemented in M1, keep nil" },
}(function(self, parameters, learning_rate, weight_decay, grad_clip)
  if grad_clip ~= nil then
    error("paddle.optimizer: grad_clip is not implemented yet (M1); "
          .. "leave it nil", 2)
  end
  self.parameters = parameters          -- pl.List (value contract)
  self.learning_rate = learning_rate
  self.weight_decay = weight_decay
  -- Plain array mirror; avoids iterator surprises (D27 spirit) in step().
  -- # works for pl.List and plain arrays alike (same as ParamList).
  self._params = {}
  for i = 1, #parameters do
    self._params[i] = parameters[i]
  end
end)

-- Coupled L2 (api/optimizer.md §6③): SGD/Adam weight_decay merges into
-- the gradient (grad + wd*param), unlike AdamW's decoupled form.
function Optimizer:_grad_with_wd(p, g)
  if self.weight_decay == nil then return g end
  return paddle_core.pd_op_add(g, paddle_core.pd_op_scale(p, self.weight_decay, 0, 1))
end

-- Subclasses implement the per-parameter update (param, grad, state idx).
function Optimizer:_update_param(_p, _g, _i)
  error("paddle.optimizer: _update_param not implemented in subclass", 2)
end

Optimizer.get_parameters = rule {
  { name = "opt", type = Optimizer },
}(function(self)
  return self.parameters
end)

-- step{...} -- one update over all parameters (optimizer.py:2060).
-- Runs under has_grad=false so composed ops never build a graph on the
-- leaf params; restores the flag via pcall even when an update errors.
Optimizer.step = rule {
  { name = "opt", type = Optimizer },
  { name = "set_gradients", type = argrule.any, opt = true,
    doc = "kept for signature parity; not used in M1" },
}(function(self, _set_gradients)
  local old = paddle_core.has_grad()
  paddle_core.set_has_grad(false)
  local ok, err = pcall(function()
    for i, p in ipairs(self._params) do
      local g = p:grad()
      if g ~= nil then                  -- stop_gradient/skipped params have nil grad
        self:_update_param(p, g, i)
      end
    end
  end)
  paddle_core.set_has_grad(old)
  if not ok then error(err, 0) end
end)

-- clear_grad{} -- zero every parameter's accumulated grad
-- (set_to_zero=true, the eager default; eager_method.cc:1066).
Optimizer.clear_grad = rule {
  { name = "opt", type = Optimizer },
  { name = "set_to_zero", type = argrule.boolean, opt = true, default = true,
    doc = "zero grads (true, default); release form (false) not in M1" },
}(function(self, set_to_zero)
  if not set_to_zero then
    error("paddle.optimizer: clear_grad(set_to_zero=false) is not "
          .. "implemented yet (M1)", 2)
  end
  for _, p in ipairs(self._params) do
    local g = p:grad()
    if g ~= nil then
      -- g is a shell SHARING the autograd grad storage (autograd.cc:167),
      -- so writing zeros through it zeroes the real grad.
      local z = paddle_core.pd_op_zeros_like(g, 0, cpu_place())
      g:copy_from(z)
    end
  end
end)

-- ---------------------------------------------------------------------
-- SGD (sgd.py:40; no state)
-- ---------------------------------------------------------------------

local SGD = class(Optimizer)        -- class(parent), like nn Linear
SGD._name = "SGD"

SGD._init = rule {
  { name = "self", type = Optimizer, doc = "pl.class instance (auto)" },
  { name = "parameters", type = ParamList,
    doc = "net:parameters() -- pl.List of Tensor" },
  { name = "learning_rate", type = argrule.number, opt = true, default = 0.001,
    doc = "learning rate (Python SGD default 0.001, sgd.py:96)" },
  { name = "weight_decay", type = argrule.number, opt = true,
    doc = "L2 coefficient (coupled)" },
  { name = "grad_clip", type = argrule.any, opt = true,
    doc = "gradient clipping -- NOT implemented in M1, keep nil" },
}(function(self, parameters, learning_rate, weight_decay, grad_clip)
  self:super(parameters, learning_rate, weight_decay, grad_clip)
  self.type = "sgd"
end)

-- param -= lr * grad'   (cpu/sgd_kernel.cc:39 sgd_dense_kernel)
function SGD:_update_param(p, g, _i)
  local t = self:_grad_with_wd(p, g)
  local delta = paddle_core.pd_op_scale(t, self.learning_rate, 0, 1)
  local new_p = paddle_core.pd_op_subtract(p, delta)
  p:copy_from(new_p)
end

-- ---------------------------------------------------------------------
-- Adam (adam.py:58; per-param mom1/mom2 + betaN_pow state)
-- ---------------------------------------------------------------------

local Adam = class(Optimizer)
Adam._name = "Adam"

Adam._init = rule {
  { name = "self", type = Optimizer, doc = "pl.class instance (auto)" },
  { name = "parameters", type = ParamList,
    doc = "net:parameters() -- pl.List of Tensor" },
  { name = "learning_rate", type = argrule.number, opt = true, default = 0.001,
    doc = "learning rate (Python Adam default 0.001, adam.py:201)" },
  { name = "beta1", type = argrule.number, opt = true, default = 0.9,
    doc = "first moment decay (adam.py:204)" },
  { name = "beta2", type = argrule.number, opt = true, default = 0.999,
    doc = "second moment decay (adam.py:205)" },
  { name = "epsilon", type = argrule.number, opt = true, default = 1e-8,
    doc = "denominator epsilon (adam.py:206)" },
  { name = "weight_decay", type = argrule.number, opt = true,
    doc = "L2 coefficient (coupled)" },
  { name = "grad_clip", type = argrule.any, opt = true,
    doc = "gradient clipping -- NOT implemented in M1, keep nil" },
}(function(self, parameters, learning_rate, beta1, beta2, epsilon,
           weight_decay, grad_clip)
  self:super(parameters, learning_rate, weight_decay, grad_clip)
  self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
  self.type = "adam"
  -- Per-param state built in _init (api/optimizer.md §6①): mom1/mom2 are
  -- zeros_like(param); betaN_pow start at betaN (adam.py:311-335, NOT 1)
  -- and multiply by betaN every step (cpu/adam_kernel.cc:118-119).
  -- v1: betaN_pow kept as Lua numbers (state_dict wiring is later, §7).
  self._state = {}
  for i, p in ipairs(self._params) do
    self._state[i] = {
      mom1 = paddle_core.pd_op_zeros_like(p, 0, cpu_place()),
      mom2 = paddle_core.pd_op_zeros_like(p, 0, cpu_place()),
      beta1_pow = beta1,
      beta2_pow = beta2,
    }
  end
end)

-- Verbatim adam_functors.h:325-345 (dense, no amsgrad):
--   lr_eff   = lr * sqrt(1-beta2_pow) / (1-beta1_pow)
--   mom1'    = beta1*mom1 + (1-beta1)*g
--   mom2'    = beta2*mom2 + (1-beta2)*g*g
--   param   -= lr_eff * mom1' / (sqrt(mom2') + eps*sqrt(1-beta2_pow))
function Adam:_update_param(p, g, i)
  local st = self._state[i]
  local t = self:_grad_with_wd(p, g)
  local b1, b2, eps = self.beta1, self.beta2, self.epsilon

  local lr_eff = self.learning_rate * math.sqrt(1 - st.beta2_pow)
                                        / (1 - st.beta1_pow)

  local m1 = paddle_core.pd_op_add(
      paddle_core.pd_op_scale(st.mom1, b1, 0, 1),
      paddle_core.pd_op_scale(t, 1 - b1, 0, 1))

  local gg = paddle_core.pd_op_multiply(t, t)
  local m2 = paddle_core.pd_op_add(
      paddle_core.pd_op_scale(st.mom2, b2, 0, 1),
      paddle_core.pd_op_scale(gg, 1 - b2, 0, 1))

  -- sqrt(m2) + eps*sqrt(1-beta2_pow): scale(x,1,bias=true) = x + bias
  local denom = paddle_core.pd_op_scale(
      paddle_core.pd_op_sqrt(m2), 1, eps * math.sqrt(1 - st.beta2_pow), 1)

  local delta = paddle_core.pd_op_scale(
      paddle_core.pd_op_divide(m1, denom), lr_eff, 0, 1)
  local new_p = paddle_core.pd_op_subtract(p, delta)
  p:copy_from(new_p)

  st.mom1, st.mom2 = m1, m2
  st.beta1_pow = st.beta1_pow * b1
  st.beta2_pow = st.beta2_pow * b2
end

-- ---------------------------------------------------------------------
-- Momentum (momentum.py:40; per-param velocity state)
-- 非 nesterov 主路径(momentum.py:73-79 + momentum_kernel_impl.h:47-51):
--   velocity = mu*velocity + grad'(grad' 已含 coupled wd)
--   param   -= lr*velocity
-- nesterov(use_nesterov=true):param -= lr*(grad' + mu*velocity)(momentum.py:77)
-- 状态在 _init 建(zeros_like,api/optimizer.md §6①);weight_decay 走
-- 基类 _grad_with_wd(耦合 L2,与上游 l2_decay 并入 grad 一致)。
-- ---------------------------------------------------------------------

local Momentum = class(Optimizer)
Momentum._name = "Momentum"

Momentum._init = rule {
  { name = "self", type = Optimizer, doc = "pl.class instance (auto)" },
  { name = "parameters", type = ParamList,
    doc = "net:parameters() -- pl.List of Tensor" },
  { name = "learning_rate", type = argrule.number, opt = true, default = 0.001,
    doc = "learning rate (Python Momentum default 0.001, momentum.py:151)" },
  { name = "momentum", type = argrule.number, opt = true, default = 0.9,
    doc = "momentum factor (momentum.py:152)" },
  { name = "weight_decay", type = argrule.number, opt = true,
    doc = "L2 coefficient (coupled)" },
  { name = "use_nesterov", type = argrule.boolean, opt = true, default = false,
    doc = "nesterov momentum (momentum.py:154)" },
  { name = "grad_clip", type = argrule.any, opt = true,
    doc = "gradient clipping -- NOT implemented in M1, keep nil" },
}(function(self, parameters, learning_rate, momentum, weight_decay,
           use_nesterov, grad_clip)
  self:super(parameters, learning_rate, weight_decay, grad_clip)
  self.momentum = momentum
  self.use_nesterov = use_nesterov
  self.type = "momentum"
  self._state = {}
  for i, p in ipairs(self._params) do
    self._state[i] = { velocity = paddle_core.pd_op_zeros_like(p, 0, cpu_place()) }
  end
end)

-- velocity = mu*velocity + grad'; param -= lr*velocity (非 nesterov)
--   momentum.py:75-78 + cpu/momentum_kernel.cc → momentum_kernel_impl.h:47-51
-- nesterov: param -= lr*(grad' + mu*velocity)   momentum_kernel_impl.h:48-49
function Momentum:_update_param(p, g, i)
  local t = self:_grad_with_wd(p, g)
  local st = self._state[i]
  local mu = self.momentum
  local v = paddle_core.pd_op_add(
      paddle_core.pd_op_scale(st.velocity, mu, 0, 1), t)     -- velocity'
  local delta
  if self.use_nesterov then
    delta = paddle_core.pd_op_scale(
        paddle_core.pd_op_add(t, paddle_core.pd_op_scale(v, mu, 0, 1)),
        self.learning_rate, 0, 1)
  else
    delta = paddle_core.pd_op_scale(v, self.learning_rate, 0, 1)
  end
  local new_p = paddle_core.pd_op_subtract(p, delta)
  p:copy_from(new_p)
  st.velocity = v
end

return { Optimizer = Optimizer, SGD = SGD, Adam = Adam, Momentum = Momentum }