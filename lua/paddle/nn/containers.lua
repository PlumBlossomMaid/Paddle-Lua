-- lua/paddle/nn/containers.lua -- paddle.nn 容器(Sequential / LayerList /
-- LayerDict)。WORKPLAN 2.9.5。
--
-- Upstream blueprint: python/paddle/nn/layer/container.py
--   Sequential:750-872 / LayerList:566-750 / LayerDict:39-330
--
-- 设计约束(09-nn.md §3.2/D27/D-R15):
--   - 三个容器都继承 Layer(pl.class)——is_a(nn.Layer) 为真,
--     parameters()/state_dict()/train() 的递归遍历不需要特例。
--   - 实例 raw 表保持空:子层经 add_sublayer 走 __newindex 自动注册
--     (FIELDS.sublayers + order 书),绝不 rawset 整数键。
--   - Lua 5.1 上 #ml / ipairs(ml) 静默失效 → 提供 ml:len() / ml:iter(),
--     文档与生成代码一律只用这两个。
--   - 用户下标 ml[i] 是 1-based(C4 破例的唯一例外);负数从尾数
--     (Python _get_abs_idx container.py:606);state_dict 键名是
--     0-based 字符串 "0","1",...(D-R15,与 Python 逐字符一致)。
--     转换发生在容器方法层:按 order 书序取第 k 个子层(命名对容器
--     的键名不是 "0","1",但 order 书给出插入顺序,下标语义不变)。
--
-- 构造的 argrule 形态(变长参数):
--   pl.class 构造链把 `Sequential{ l1, l2 }` 变成 _init(obj, {l1,l2});
--   argrule 的 parse_whole_first 把「self + 整个调用表」合并成一张表
--   作为唯一实参传给回调 —— 回调收到 whole,whole[1] = self 实例,
--   whole[2..] = 层列表。所以规则表只有一个 self 槽(type 放宽为
--   table,真正实例在 whole[1]),回调里解包后 self:super()。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local layer_module = require "paddle.nn.layer"
local Layer = layer_module.Layer
-- FIELDS 是私有键(layer.lua:36 local FIELDS = {}),经模块返回表导出
-- (layer.lua:602 return { Layer, FIELDS, ... })——不在 Layer 类表上。
local FIELDS = layer_module.FIELDS

local is_layer = function(v)
  if type(v) ~= "table" then return false end
  local mt = getmetatable(v)
  if mt == nil or type(mt.is_a) ~= "function" then return false end
  return v:is_a(Layer)
end

local LayerContract = argrule.alias { is_layer }

-- ---------------------------------------------------------------------
-- LayerList(container.py:566)
-- ---------------------------------------------------------------------

local LayerList = class(Layer)
LayerList._name = "LayerList"

-- 变长构造。argrule 有两种形态(探针 2026-10-26 实测):
--   带子层 → parse_whole_first:whole = {self, l1, l2, ...}(普通表,无 FIELDS)
--   空构造 → parse_position_table:直接传 self 实例(有 FIELDS)
-- 用 rawget(whole, FIELDS) 区分;空构造时数字长度 #whole = 0 循环不执行。
LayerList._init = rule {
  { name = "self", type = argrule.table,
    doc = "whole call table ([1]=self) or self instance (empty ctor)" },
}(function(whole)
  local f = rawget(whole, FIELDS)
  local self = f and whole or whole[1]
  self:super()
  self._len = 0
  for i = (f and 1 or 2), #whole do
    self:append(whole[i])
  end
end)

-- 按 order 书序取第 k 个子层(1-based;负数从尾数,Python _get_abs_idx)。
-- 返回 nil 表示越界(Lua 数组惯例;Python 抛 IndexError,差异记
-- nn.md §4 —— v1 以 nil 返回,Lua 生态的数组越界读就是 nil)。
local function layer_at(self, k)
  local f = rawget(self, FIELDS)
  if f == nil then return nil end
  local n = self._len
  if k >= 1 and k <= n then
    -- 正数:第 k 个(按 order 书序)
  elseif k < 0 and -k <= n then
    k = n + k + 1
  else
    return nil
  end
  local idx = 0
  for _, name in ipairs(f.order) do
    local v = f.sublayers[name]
    if v ~= nil then
      idx = idx + 1
      if idx == k then return v end
    end
  end
  return nil
end

function LayerList:len()
  return self._len
end

-- (i, layer) 迭代器,i 从 1 开始(与下标一致)
function LayerList:iter()
  local i = 0
  local n = self._len
  return function()
    i = i + 1
    if i > n then return nil end
    return i, layer_at(self, i)
  end
end

-- 数字下标访问(1-based,容器方法层转换)。__index 特判数字键 →
-- layer_at;字符串/其他键走 Layer.__index(方法、参数、缓冲)。
local base_index = Layer.__index
function LayerList:__index(k)
  if type(k) == "number" and k == math.floor(k) then
    local v = layer_at(self, k)
    if v ~= nil then return v end
  end
  return base_index(self, k)
end

-- 方法规则表第一规则不叫 "self":argrule 的 parse_self_call 只对
-- 「第一规则名 self + 第二参数是裸表」做 pl.class 构造链展开
-- (table_with_self)。方法调用 self:extend({l1,l2}) 的第二参数恰是
-- 裸表列表 —— 若叫 self 会被误判成构造调用(探针 2026-10-26 实测:
-- "too many positional table arguments")。规则名不重要,type 校验
-- 仍挡住非 Layer 的第一个实参。
LayerList.append = rule {
  { name = "container", type = Layer, doc = "the LayerList itself" },
  { name = "sublayer", type = LayerContract },
}(function(self, sublayer)
  -- 键名 0-based 字符串(container.py:666 str(len) → Lua tostring(n))
  local name = tostring(self._len)
  self:add_sublayer(name, sublayer)
  self._len = self._len + 1
  return self
end)

LayerList.insert = rule {
  { name = "container", type = Layer, doc = "the LayerList itself" },
  { name = "index", type = argrule.number },
  { name = "sublayer", type = LayerContract },
}(function(self, index, sublayer)
  local n = self._len
  -- container.py:689-695:允许 -n <= index <= n,负数 +n
  if not (-n <= index and index <= n) then
    error(("index should be an integer in range [%d, %d]")
      :format(-n, n), 2)
  end
  local idx = index
  if idx < 0 then idx = idx + n end
  local f = rawget(self, FIELDS)
  -- 值搬移(container.py:693):键名 tostring(i-1) 的值 → tostring(i),
  -- 从 n 到 idx+1 递减。键名不变,order 书只需末尾补新键 tostring(n)。
  for i = n, idx + 1, -1 do
    f.sublayers[tostring(i)] = f.sublayers[tostring(i - 1)]
  end
  -- 搬移引入的新键 tostring(n) 进 order 书末尾
  -- (idx < n 时才有搬移;idx == n 即追加,add_sublayer 的 push_order 处理)
  if idx < n then
    f.order[#f.order + 1] = tostring(n)
  end
  self:add_sublayer(tostring(idx), sublayer)
  self._len = n + 1
  return self
end)

LayerList.extend = rule {
  { name = "container", type = Layer, doc = "the LayerList itself" },
  { name = "sublayers", type = argrule.List { LayerContract } },
}(function(self, sublayers)
  for _, l in ipairs(sublayers) do self:append(l) end
  return self
end)

-- ---------------------------------------------------------------------
-- Sequential(container.py:750)
-- ---------------------------------------------------------------------

local Sequential = class(Layer)
Sequential._name = "Sequential"

-- 变长构造(形态同 LayerList:空构造直接收 self 实例,带子层收 whole 表)。
Sequential._init = rule {
  { name = "self", type = argrule.table,
    doc = "whole call table ([1]=self) or self instance (empty ctor)" },
}(function(whole)
  local f = rawget(whole, FIELDS)
  local self = f and whole or whole[1]
  self:super()
  self._len = 0
  -- 镜像 container.py:799-816:单一 OrderedDict / (name,layer) 对列表 /
  -- 纯 Layer 列表。Lua 侧:whole[2..] 的元素,若是 Layer → 纯层;
  -- 若是 {name, layer} 两元素表 → 命名对。空构造时 whole 就是 self
  -- 实例(raw 表空,不能读 whole[2]),first 保持 nil 走纯层分支且
  -- 循环 #whole=0 不执行。
  local first = (f == nil) and whole[2] or nil
  local named_pairs = type(first) == "table" and getmetatable(first) == nil
    and first[1] ~= nil and type(first[1]) == "string"
    and is_layer(first[2])
  if named_pairs then
    for i = 2, #whole do
      local pair = whole[i]
      if type(pair) ~= "table" or type(pair[1]) ~= "string"
        or not is_layer(pair[2]) then
        error("paddle.nn.Sequential: expected (name, layer) pairs, got "
          .. tostring(pair), 2)
      end
      self:add_sublayer(pair[1], pair[2])
      self._len = self._len + 1
    end
  else
    for i = 2, #whole do
      if not is_layer(whole[i]) then
        error("paddle.nn.Sequential: expected Layers or (name, layer) "
          .. "pairs, got " .. tostring(whole[i]), 2)
      end
      self:add_sublayer(tostring(self._len), whole[i])
      self._len = self._len + 1
    end
  end
end)

function Sequential:__index(k)
  if type(k) == "number" and k == math.floor(k) then
    local v = layer_at(self, k)
    if v ~= nil then return v end
  elseif type(k) == "string" then
    local f = rawget(self, FIELDS)
    local v = f and f.sublayers[k]
    if v ~= nil then return v end
  end
  return base_index(self, k)
end

function Sequential:len()
  return self._len
end

function Sequential:iter()
  local i = 0
  local n = self._len
  return function()
    i = i + 1
    if i > n then return nil end
    return i, layer_at(self, i)
  end
end

function Sequential:forward(input)
  for _, layer in self:iter() do
    input = layer(input)
  end
  return input
end

Sequential.append = LayerList.append
Sequential.insert = LayerList.insert
Sequential.extend = LayerList.extend
-- Sequential.append/insert 键名与 LayerList 相同(tostring(len))

-- ---------------------------------------------------------------------
-- LayerDict(container.py:39)
-- ---------------------------------------------------------------------

local LayerDict = class(Layer)
LayerDict._name = "LayerDict"

-- 变长构造(形态同 LayerList:空构造直接收 self 实例,带子层收 whole 表)。
LayerDict._init = rule {
  { name = "self", type = argrule.table,
    doc = "whole call table ([1]=self) or self instance (empty ctor)" },
}(function(whole)
  local f = rawget(whole, FIELDS)
  local self = f and whole or whole[1]
  self:super()
  self._len = 0
  for i = 2, #whole do
    local pair = whole[i]
    if type(pair) ~= "table" or type(pair[1]) ~= "string"
      or not is_layer(pair[2]) then
      error("paddle.nn.LayerDict: expected (key, layer) pairs, got "
        .. tostring(pair), 2)
    end
    self:update(pair[1], pair[2])
  end
end)

function LayerDict:__index(k)
  if type(k) == "string" then
    local f = rawget(self, FIELDS)
    local v = f and f.sublayers[k]
    if v ~= nil then return v end
  end
  return base_index(self, k)
end

function LayerDict:len()
  return self._len
end

function LayerDict:iter()
  local f = rawget(self, FIELDS)
  local names = {}
  for _, name in ipairs(f.order) do
    if f.sublayers[name] ~= nil then names[#names + 1] = name end
  end
  local i = 0
  return function()
    i = i + 1
    if i > #names then return nil end
    return names[i], f.sublayers[names[i]]
  end
end

LayerDict.update = rule {
  { name = "container", type = Layer, doc = "the LayerDict itself" },
  { name = "key", type = argrule.string },
  { name = "sublayer", type = LayerContract },
}(function(self, key, sublayer)
  local f = rawget(self, FIELDS)
  local existed = f.sublayers[key] ~= nil
  self:add_sublayer(key, sublayer)
  if not existed then self._len = self._len + 1 end
  return self
end)

LayerDict.keys = function(self)
  local f = rawget(self, FIELDS)
  local out = {}
  for _, name in ipairs(f.order) do
    if f.sublayers[name] ~= nil then out[#out + 1] = name end
  end
  return out
end

LayerDict.values = function(self)
  local out = {}
  for _, name in ipairs(self:keys()) do
    out[#out + 1] = self[name]
  end
  return out
end

return {
  Sequential = Sequential,
  LayerList = LayerList,
  LayerDict = LayerDict,
}