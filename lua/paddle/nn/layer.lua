-- lua/paddle/nn/layer.lua -- paddle.nn.Layer base class
-- Stage: P9 / M1 (WORKPLAN 2.9.1). First file of the nn module.
--
-- Design: docs/migration/plan/modules/09-nn.md sec.3.2 (auto-registration
-- via FIELDS private key + pl.class _create/_class_init hooks) +
-- foundations.md sec.1.4 (the hook wiring) / sec.2 (pl.List returns).
-- Upstream blueprint: python/paddle/nn/layer/layers.py (Layer base).
--
-- KEY INVARIANT (09-nn.md sec.3.2): the instance raw table stays EMPTY
-- except for the single private FIELDS key. Because __newindex only fires
-- when rawget(self,k)==nil, keeping user-visible fields out of the raw
-- table makes __newindex fire on EVERY assignment -- that is the whole
-- auto-registration trick. Any rawset(self,k,v) in this module (or user
-- code) would permanently disable it for that key (CI grep: only FIELDS
-- rawset allowed under lua/paddle/nn/).
--
-- Parameter vs buffer discrimination: upstream checks
-- isinstance(value, framework.Parameter) (layers.py:2467); Lua has no
-- Tensor subclass, so Layer:create_parameter registers the returned
-- tensor in a module-level weak table (09-nn.md sec.3.2 is_parameter
-- slot). Weak keys (__mode="k") so a dropped tensor leaves no leak.

local class = require "pl.class"
local List = require "pl.List"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"

-- Shared LuaTensor usertype metatable (tensor.lua augments the same one).
-- A throwaway instance reaches it; the shell is dropped right after.
local probe = paddle_core.tensor{1}
local TENSOR_MT = getmetatable(probe)
probe = nil

-- Private key: a fresh table, users cannot spell it (09-nn.md sec.3.2).
local FIELDS = {}

-- Parameter registry: tensor -> true for tensors created by
-- Layer:create_parameter / Layer:add_parameter (upstream Parameter
-- instances, layers.py:2467). Weak keys so GC reclaims dropped tensors.
local PARAMETER_REGISTRY = setmetatable({}, { __mode = "k" })

-- ---------------------------------------------------------------------
-- classification predicates (09-nn.md sec.3.2 newindex branches)
-- ---------------------------------------------------------------------

local function is_parameter(v)
  return type(v) == "userdata" and PARAMETER_REGISTRY[v] == true
end

local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end

local Layer  -- forward decl; is_layer needs it, defined below

local function is_layer(v)
  if type(v) ~= "table" then return false end
  local mt = getmetatable(v)
  if mt == nil or type(mt.is_a) ~= "function" then return false end
  return v:is_a(Layer)
end

-- ---------------------------------------------------------------------
-- argrule type contracts for the public Layer methods below (用户
-- 2026-10-25: 几乎所有正式 API 用 argrule,不手写解析器)。shape-style
-- params are IntList (api/README.md sec.2.1.2: 容器 + 整数)。Tensor/Layer
-- matched structurally via the local predicates above (argrule accepts a
-- function as a predicate contract).
-- ---------------------------------------------------------------------

local IntList = argrule.List { argrule.number }
local TensorContract = argrule.alias { function(v) return is_tensor(v) end }
local LayerContract = argrule.alias { function(v) return is_layer(v) end }

-- ---------------------------------------------------------------------
-- order book: insertion order of registered names (Python dict is
-- insertion-ordered; Lua tables are not -- without this, parameters() /
-- state_dict() order would be nondeterministic per run, breaking seed
-- reproducibility and byte-stable dumps -- 09-nn.md sec.3.2 "order 表
-- 不是可选的"). Replace keeps the ORIGINAL position; nil removes.
-- ---------------------------------------------------------------------

local function find_order(f, k)
  for i, name in ipairs(f.order) do
    if name == k then return i end
  end
  return nil
end

local function remove_order(f, k)
  local i = find_order(f, k)
  if i ~= nil then table.remove(f.order, i) end
end

local function push_order(f, k)
  if find_order(f, k) == nil then
    f.order[#f.order + 1] = k
  end
end

-- ---------------------------------------------------------------------
-- unregister / register (single source of truth for all four slots)
-- ---------------------------------------------------------------------
-- Order-book discipline (09-nn.md sec.3.2 三态边界表):
--   replace  (old value then NEW value) -> order position PRESERVED
--   set nil  (old value then nil)       -> removed from order too
-- So unregister_data only clears the four value slots; the ORDER removal
-- is decided by layer_newindex based on whether the new value is nil.
local function unregister_data(f, k)
  f.params[k] = nil
  f.sublayers[k] = nil
  f.buffers[k] = nil
  f.plain[k] = nil
end

local function register(f, k, v)
  if is_parameter(v) then
    f.params[k] = v
    push_order(f, k)
  elseif is_layer(v) then
    f.sublayers[k] = v
    push_order(f, k)
  elseif is_tensor(v) then
    f.buffers[k] = v
    push_order(f, k)
  else
    f.plain[k] = v        -- plain fields NOT in the order book (09-nn.md
  end                     -- sec.3.2: push_order only on the three slots)
end

-- ---------------------------------------------------------------------
-- __newindex / __index (foundations.md sec.1.4 wiring)
-- ---------------------------------------------------------------------

local function layer_newindex(self, k, v)
  local f = rawget(self, FIELDS)
  if f == nil then
    error("paddle.nn: broken Layer instance (no FIELDS) -- class.cast "
          .. "bypasses _create and is forbidden (foundations.md sec.1.5 ②)", 2)
  end
  unregister_data(f, k)
  if v == nil then
    remove_order(f, k)    -- nil: out of the order book entirely
    return
  end
  register(f, k, v)       -- push_order keeps the ORIGINAL position
end

local function layer_index(self, k)
  local f = rawget(self, FIELDS)
  if f ~= nil then
    local v = f.params[k]
    if v ~= nil then return v end
    v = f.sublayers[k]
    if v ~= nil then return v end
    v = f.buffers[k]
    if v ~= nil then return v end
    v = f.plain[k]
    if v ~= nil then return v end
  end
  -- class chain: getmetatable(self) IS the class table c; c[k] walks
  -- c's own metatable ({__index = base}, pl/class.lua setmetatable(c,mt))
  -- so inheritance resolves here. NEVER write self.__index(...) -- that
  -- would recurse into the instance dispatcher (09-nn.md sec.3.2).
  return getmetatable(self)[k]
end

-- ---------------------------------------------------------------------
-- the class
-- ---------------------------------------------------------------------

local Layer = class()

Layer._name = "Layer"

-- ① _create hook: pl/class.lua mt.__call runs `if rawget(c,'_create')
-- then obj = c._create(...) end` BEFORE _init -- the instance raw table
-- gets exactly one key: FIELDS.
Layer._create = function()
  local o = {}
  rawset(o, FIELDS, {
    params     = {},
    sublayers  = {},
    buffers    = {},
    plain      = {},
    order      = {},
  })
  return o
end

-- ② _class_init hook: pl/class.lua `c.__index = c` is hard-coded and
-- would clobber our inherited __index on every subclass creation; the
-- hook runs AFTER that line, so it re-wires the dispatchers per class
-- (foundations.md sec.1.4 "必须走这里").
Layer._class_init = function(c)
  c.__index = layer_index
  c.__newindex = layer_newindex
end

-- ③ the base class itself never goes through _class_init (it has no
-- base), so wire it once here.
Layer.__index = layer_index
Layer.__newindex = layer_newindex

-- Empty ctor that seeds the TRAINING flag. Subclasses call self:super()
-- then assign fields (auto-registration takes over).
--
-- NOTE: self.training must be set HERE, not in _create -- _create runs
-- on the BARE table before setmetatable, so `o.training = true` there
-- would rawset into the raw table and permanently short-circuit
-- __newindex for that key. Setting it in _init goes through the
-- dispatcher into the plain slot (09-nn.md sec.3.2 plain category).
function Layer:_init()
  self.training = true
end

-- forward is abstract: subclasses must implement (layers.py abstract).
function Layer:forward(x)
  error(("paddle.nn: %s.forward() not implemented")
    :format(getmetatable(self)._name or "Layer"), 2)
end

-- __call on the CLASS-table slot: instances dispatch `net(x)` here
-- (getmetatable(net).__call == c.__call, inherited via tupdate). The
-- pl.class mt.__call that CREATES instances lives on the class's OWN
-- metatable -- the two never collide.
Layer.__call = function(self, ...)
  return self:forward(...)
end

-- ---------------------------------------------------------------------
-- train / eval (09-nn.md sec.3.4): training must propagate to ALL
-- sublayers recursively -- a missed sublayer turns inference into
-- random-looking outputs with NO error. Explicit-stack traversal (09-nn.md
-- pitfall ③): Lua's C stack is shallow, 100+ layer nets would overflow
-- recursion.
-- ---------------------------------------------------------------------

function Layer:train(mode)
  local m = (mode == nil) and true or mode
  if type(m) ~= "boolean" then
    error("paddle.nn: training mode is expected to be boolean", 2)
  end
  local stack = { self }
  while #stack > 0 do
    local l = table.remove(stack)
    -- self.training is a PLAIN field: it must go through __newindex so
    -- layer_index can read it back (writing f.training directly would
    -- land in a place layer_index never looks -- measured failure
    -- 2026-10-25: `net.training` returned nil right after train()).
    l.training = m
    local f = rawget(l, FIELDS)
    if f ~= nil then
      for _, name in ipairs(f.order) do
        local sub = f.sublayers[name]
        if sub ~= nil then stack[#stack + 1] = sub end
      end
    end
  end
  return self
end

function Layer:eval()
  return self:train(false)
end

-- ---------------------------------------------------------------------
-- apply(fn): call fn on self and every sublayer (layers.py:872)
-- ---------------------------------------------------------------------

function Layer:apply(fn)
  if type(fn) ~= "function" then
    error("paddle.nn: apply() expects a function", 2)
  end
  local stack = { self }
  while #stack > 0 do
    local l = table.remove(stack)
    fn(l)
    local f = rawget(l, FIELDS)
    if f ~= nil then
      for _, name in ipairs(f.order) do
        local sub = f.sublayers[name]
        if sub ~= nil then stack[#stack + 1] = sub end
      end
    end
  end
  return self
end

-- ---------------------------------------------------------------------
-- full_name (layers.py:929). Upstream builds name_scope/class paths;
-- v1 returns the class name (enough for debugging). NOTE: state_dict
-- keys do NOT come from full_name -- they come from the named_*
-- structured prefix (09-nn.md sec.3.5), so this simplification does not
-- touch the interchange contract.
-- ---------------------------------------------------------------------

function Layer:full_name()
  return getmetatable(self)._name or "Layer"
end

-- ---------------------------------------------------------------------
-- named_* iterators (api/README.md sec.2.3: generators -> iterator
-- functions) -- explicit-stack DFS, ordered by the order book.
-- yield (name, value) pairs, name = prefix + "." + key with empty-prefix
-- elision (upstream layers.py:1639 `layer_prefix + ('.' if layer_prefix
-- else '') + key`).
--
-- Pre-order DFS: a layer's OWN params come before its sublayers
-- (upstream named_parameters iterates named_sublayers(include_self=True)
-- and yields each layer's _parameters -- layers.py:1568-1639). The
-- iterator keeps a `cur` record (current layer + its order index) ACROSS
-- calls so a layer's remaining params are not skipped after the first
-- yield (measured bug, 2026-10-25: b/fc.b vanished with a naÃ¯ve
-- "pop-from-stack each call" loop).
-- ---------------------------------------------------------------------

local function named_iter(slots, stack, seen)
  -- cur = { layer, order_index } -- the layer whose params we are
  -- mid-way through; nil means "pop the next layer from the stack".
  local cur = nil
  return function()
    while true do
      if cur == nil then
        if #stack == 0 then
          return nil
        end
        local entry = table.remove(stack)
        local prefix, l = entry[1], entry[2]
        local f = rawget(l, FIELDS)
        if f == nil then
          return nil
        end
        -- push sublayers (reversed so the order book's first sublayer
        -- sits on top and pops first)
        for i = #f.order, 1, -1 do
          local name = f.order[i]
          local sub = f.sublayers[name]
          if sub ~= nil then
            local p = (prefix == "") and name or (prefix .. "." .. name)
            stack[#stack + 1] = { p, sub }
          end
        end
        cur = { l, 1, prefix }
      end
      local l, i, prefix = cur[1], cur[2], cur[3]
      local f = rawget(l, FIELDS)
      -- resume this layer's params from where we left off
      for idx = i, #f.order do
        local name = f.order[idx]
        for _, slot in ipairs(slots) do
          local v = f[slot][name]
          if v ~= nil and (seen == nil or seen[v] == nil) then
            if seen ~= nil then seen[v] = true end
            cur[2] = idx + 1           -- resume AFTER this one
            local full = (prefix == "") and name or (prefix .. "." .. name)
            return full, v
          end
        end
      end
      -- this layer exhausted; next call pops a sublayer (or ends)
      cur = nil
    end
  end
end

-- Generic DFS iterator factory. The returned closure OWNS the stack and
-- seen set -- they MUST be created once per iterator, not per call:
-- rebuilding them inside the returned function would restart the DFS at
-- the first parameter forever (measured hang, 2026-10-25 -- probe_nn_step2
-- reproduced `named:` w w w ... without termination).
function Layer:_named_iter_factory(slots, prefix, remove_duplicate)
  local stack = { { prefix or "", self } }
  local seen = (remove_duplicate ~= false) and {} or nil
  return named_iter(slots, stack, seen)
end

function Layer:named_parameters(prefix, remove_duplicate)
  return self:_named_iter_factory({ "params" }, prefix, remove_duplicate)
end

function Layer:named_buffers(prefix, remove_duplicate)
  return self:_named_iter_factory({ "buffers" }, prefix, remove_duplicate)
end

function Layer:named_sublayers(prefix, remove_duplicate)
  return self:_named_iter_factory({ "sublayers" }, prefix, remove_duplicate)
end

-- ---------------------------------------------------------------------
-- collection returns (api/README.md sec.2.3: user-facing collections ->
-- pl.List; the hot-path exceptions like x:shape() stay bare tables)
-- ---------------------------------------------------------------------

function Layer:parameters()
  local lst = List()
  for _, p in self:named_parameters() do
    lst:append(p)
  end
  return lst
end

function Layer:buffers()
  local lst = List()
  for _, b in self:named_buffers() do
    lst:append(b)
  end
  return lst
end

function Layer:sublayers()
  local lst = List()
  for _, s in self:named_sublayers() do
    lst:append(s)
  end
  return lst
end

-- ---------------------------------------------------------------------
-- state_dict (layers.py _state_dict_impl:2709). KEY FORMAT MUST MATCH
-- PYTHON BYTE-FOR-BYTE: destination[structured_name_prefix + name],
-- sublayer recursion appends layer_name + "." -- the interchange
-- contract (09-nn.md sec.3.5). v1 includes ALL buffers (upstream
-- excludes non-persistable ones via _non_persistable_buffer_names_set;
-- register_buffer is the persistable path, and v1 has no
-- non-persistable API yet -- revisit with BatchNorm's running stats).
-- Explicit-stack traversal: no recursion, 200-layer safe (09-nn.md ③).
-- ---------------------------------------------------------------------

function Layer:state_dict()
  local d = {}
  local stack = { { "", self } }
  while #stack > 0 do
    local entry = table.remove(stack)
    local prefix, l = entry[1], entry[2]
    local f = rawget(l, FIELDS)
    if f ~= nil then
      for _, name in ipairs(f.order) do
        local p = f.params[name]
        if p ~= nil then d[prefix .. name] = p end
      end
      for _, name in ipairs(f.order) do
        local b = f.buffers[name]
        if b ~= nil then d[prefix .. name] = b end
      end
      for _, name in ipairs(f.order) do
        local sub = f.sublayers[name]
        if sub ~= nil then
          stack[#stack + 1] = { prefix .. name .. ".", sub }
        end
      end
    end
  end
  return d
end

-- set_state_dict(d): overwrite parameters/buffers from a dict produced
-- by state_dict() (or loaded from a Python-saved .pdparams). v1 assigns
-- via pd_op_assign (new tensor sharing the data) and keeps the slot's
-- trainability (parameter -> stop_gradient=false). Missing keys error:
-- silent misses hide a broken interchange (09-nn.md 坑, and the same
-- discipline as load-time strictness).
function Layer:set_state_dict(d)
  if type(d) ~= "table" then
    error("paddle.nn: set_state_dict() expects a dict", 2)
  end
  local stack = { { "", self } }
  local nset = 0
  while #stack > 0 do
    local entry = table.remove(stack)
    local prefix, l = entry[1], entry[2]
    local f = rawget(l, FIELDS)
    if f ~= nil then
      for _, name in ipairs(f.order) do
        local key = prefix .. name
        local is_param = f.params[name] ~= nil
        local is_buffer = f.buffers[name] ~= nil
        if (is_param or is_buffer) and d[key] ~= nil then
          local v = paddle_core.pd_op_assign(d[key])
          if is_param then
            v:set_stop_gradient(false)  -- parameters stay trainable
            f.params[name] = v
          else
            f.buffers[name] = v
          end
          nset = nset + 1
        end
      end
      for _, name in ipairs(f.order) do
        local sub = f.sublayers[name]
        if sub ~= nil then
          stack[#stack + 1] = { prefix .. name .. ".", sub }
        end
      end
    end
  end
  return nset
end

-- ---------------------------------------------------------------------
-- to(place) / cpu(): move parameters and buffers to a place (layers.py
-- to:3284 / cpu:3889). pd_op_copy_to returns a NEW tensor on the target
-- place (sol binding gen_ops_c.cc:319-323; the C ABI pd_tensor_copy_to
-- exists but is not sol-exposed); slots are swapped in place.
--
-- place is the PUBLIC value type (device.lua PlaceMT table), so it must
-- become a LuaPlace handle at the op boundary -- same conversion ops do
-- at the creation APIs (device.lua sec.3.3 _to_core_place).
-- ---------------------------------------------------------------------

local device_mod = require "paddle.device"

local function to_one(v, place)
  local ph = device_mod._to_core_place(place)
  return paddle_core.pd_op_copy_to(v, ph, 1)
end

function Layer:to(place)
  local stack = { self }
  while #stack > 0 do
    local l = table.remove(stack)
    local f = rawget(l, FIELDS)
    if f ~= nil then
      for _, name in ipairs(f.order) do
        local p = f.params[name]
        if p ~= nil then f.params[name] = to_one(p, place) end
        local b = f.buffers[name]
        if b ~= nil then f.buffers[name] = to_one(b, place) end
      end
      for _, name in ipairs(f.order) do
        local sub = f.sublayers[name]
        if sub ~= nil then stack[#stack + 1] = sub end
      end
    end
  end
  return self
end

function Layer:cpu()
  return self:to(device_mod.CPUPlace())
end

-- ---------------------------------------------------------------------
-- explicit registration APIs (09-nn.md sec.3.2 "add_sublayer /
-- add_parameter 仍然保留"): zero-cost compat for migrated Python code
-- that calls them. Both just route through the __newindex dispatcher.
-- argrule-wrapped (用户 2026-10-25: 正式 API 不手写解析器).
-- ---------------------------------------------------------------------

Layer.add_parameter = rule {
  { name = "self", type = Layer },
  { name = "name", type = argrule.string },
  { name = "tensor", type = TensorContract },
}(function(self, name, tensor)
  PARAMETER_REGISTRY[tensor] = true
  self[name] = tensor
  return tensor
end)

Layer.add_sublayer = rule {
  { name = "self", type = Layer },
  { name = "name", type = argrule.string },
  { name = "sublayer", type = LayerContract },
}(function(self, name, sublayer)
  self[name] = sublayer
  return sublayer
end)

-- ---------------------------------------------------------------------
-- create_parameter (layers.py:1160). v1: shape + dtype + optional
-- default_initializer (callable, nn.initializer objects land in 2.9.3).
-- Default fill is Constant(0) for now -- the upstream Xavier/Constant
-- default (layers.py:1193-1217) arrives with 2.9.3 initializers.
-- Registered as a parameter (trainable, appears in state_dict).
-- argrule-wrapped: shape is IntList (api/README.md sec.2.1.2),
-- callable as create_parameter{shape={...}, dtype=...} (table) or
-- create_parameter(shape, dtype) (positional) -- both normalize.
-- ---------------------------------------------------------------------

local CPU_PLACE_CACHE

Layer.create_parameter = rule {
  { name = "self", type = Layer },
  { name = "shape", type = IntList },
  { name = "dtype", type = argrule.any, opt = true },
  { name = "default_initializer", type = argrule.Callable, opt = true },
}(function(self, shape, dtype, default_initializer)
  if dtype == nil then dtype = paddle_core.PD_DTYPE_FLOAT32 end
  if CPU_PLACE_CACHE == nil then
    CPU_PLACE_CACHE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  end
  local v = paddle_core.pd_op_full(shape, 0.0, dtype, CPU_PLACE_CACHE)
  v:set_stop_gradient(false)
  if default_initializer ~= nil then
    -- 初始化在 no_grad 下跑(gaussian_inplace 等 in-place 初始化算子带
    -- 自己的反向节点,如 gaussian_inplace_grad —— yaml ops.yaml:2564
    -- `backward: gaussian_inplace_grad`)。has_grad 下执行会把节点挂进
    -- autograd 图,训练时再次 backward 到同一参数即崩(FatalError:
    -- "TensorWrappers of GaussianInplaceGradNode do not exist",
    -- backward.cc:215,2026-10-27 MNIST 验收实证)。上游 Python 的
    -- initializer 同样不建图。与 optimizer.step 的 pcall 恢复同款
    -- (10-optimizer.md sec.4)。
    local old = paddle_core.has_grad()
    paddle_core.set_has_grad(false)
    local ok, err = pcall(default_initializer, v)
    paddle_core.set_has_grad(old)
    if not ok then error(err, 0) end
  end
  PARAMETER_REGISTRY[v] = true
  return v
end)

return { Layer = Layer, FIELDS = FIELDS, PARAMETER_REGISTRY = PARAMETER_REGISTRY }
