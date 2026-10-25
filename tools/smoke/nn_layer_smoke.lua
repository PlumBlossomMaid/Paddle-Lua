-- tools/smoke/nn_layer_smoke.lua -- M1 2.9.1: paddle.nn.Layer base class.
-- Covers 09-nn.md sec.5 acceptance subset that belongs to the Layer base:
--   - auto-registration on a THREE-LEVEL class chain (Layer -> MyBlock ->
--     MyNet), proving _class_init rewires __index/__newindex on every
--     subclass (foundations.md sec.1.4 "必须走这里")
--   - order book: insertion order, replace keeps position, nil removes
--   - state_dict key format matches Python byte-for-byte (09-nn.md sec.3.5)
--   - train/eval recursion + training flag consistency across sublayers
--   - print(layer) does not crash on any of the five Lua versions
--     (Penlight's default __tostring reads obj._class -- 坑④)
--   - 200-layer state_dict does not overflow (explicit stack, 坑③)
--   - class.cast forbidden (broken instance without FIELDS)
--
-- Build first:  cmake -S . -B build-test
--               cmake --build build-test --target paddle_core -j 6
-- Run (from the repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/nn_layer_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
assert(paddle.nn ~= nil, "paddle.nn lazy module loads")
assert(paddle.nn.Layer ~= nil, "paddle.nn.Layer exported")

local class = require "pl.class"
local Layer = paddle.nn.Layer

-- --- helpers -----------------------------------------------------------

-- returns the shared LuaTensor metatable (same probe as tensor.lua)
local function tensor_mt()
  local t = core.pd_op_full({1}, 0.0, core.PD_DTYPE_FLOAT32,
                            core.pd_place_new(core.PD_PLACE_CPU, 0))
  return getmetatable(t)
end

local FIELDS = paddle.nn._FIELDS

local function fields_of(l)
  return rawget(l, FIELDS)
end

local function assert_close(a, b, msg)
  assert(math.abs(a - b) < 1e-6, msg)
end

-- --- 1: three-level class chain + auto-registration --------------------

-- MyLeaf: leaf block with one param
local MyLeaf = class(Layer)
MyLeaf._name = "MyLeaf"
function MyLeaf:_init()
  self:super()
  self.w = self:create_parameter{ shape = {1, 1} }
end

-- MyBlock2: intermediate layer, one param + one bias + one sublayer
local MyBlock2 = class(Layer)
MyBlock2._name = "MyBlock2"
function MyBlock2:_init()
  self:super()
  self.w = self:create_parameter{ shape = {2, 2} }
  self.b = self:create_parameter{ shape = {2} }
  self.leaf = MyLeaf()
end

-- MyNet: top-level, three levels deep
local MyNet = class(MyBlock2)
MyNet._name = "MyNet"
function MyNet:_init()
  self:super()
  self.fc = MyBlock2()
end

local net = MyNet()

-- _class_init rewired every level: __index is the dispatcher, not `c`
-- (pl/class.lua hardcodes c.__index = c; our hook re-wires after it)
assert(getmetatable(MyNet).__index == getmetatable(MyBlock2).__index,
       "MyNet and MyBlock2 share the same __index dispatcher")
assert(getmetatable(MyLeaf).__index == getmetatable(Layer).__index,
       "MyLeaf (second level) also rewired")

-- instance raw table is EMPTY except FIELDS (09-nn.md sec.3.2 invariant)
local nraw = 0
for _ in pairs(net) do nraw = nraw + 1 end
assert(nraw == 1, "raw table has exactly one key (FIELDS), got " .. nraw)
assert(fields_of(net) ~= nil, "FIELDS present on instance")

-- auto-registration: sublayer + params visible through named_*
local names = {}
for name in net:named_parameters() do names[#names + 1] = name end
assert(names[1] == "w", ("first param name == 'w', got %s"):format(
  tostring(names[1])))
assert(names[2] == "b", "second param == 'b'")
assert(names[3] == "leaf.w", ("third == 'leaf.w', got %s"):format(
  tostring(names[3])))
assert(names[4] == "fc.w", "fourth == 'fc.w'")
assert(names[5] == "fc.b", "fifth == 'fc.b'")
assert(names[6] == "fc.leaf.w", "sixth == 'fc.leaf.w'")
assert(#names == 6, ("6 named params, got %d"):format(#names))

-- --- 2: order book -- replace keeps position, nil removes --------------
local M = class(Layer)
M._name = "M"
function M:_init()
  self:super()
  self.a = self:create_parameter{ shape = {1} }
  self.b = self:create_parameter{ shape = {1} }
  self.c = self:create_parameter{ shape = {1} }
end
local m = M()
local f = fields_of(m)
assert(f.order[1] == "a" and f.order[2] == "b" and f.order[3] == "c",
       "order = a,b,c")
-- replace b: position preserved. NOTE: build the replacement tensor
-- BEFORE the assignment -- `m.b = self and ...` would be nil at top
-- level, which DELETES b (nil removes).
local w2 = m:create_parameter{ shape = {1} }
m.b = w2
assert(f.order[1] == "a" and f.order[2] == "b" and f.order[3] == "c",
       "replace keeps order position")
-- set nil: removed entirely
m.c = nil
assert(f.order[1] == "a" and f.order[2] == "b" and f.order[3] == nil,
       "nil removes from order")
assert(f.params.c == nil, "nil removes from params")

-- --- 3: plain fields not in order book, do not leak into state_dict ----
local P = class(Layer)
P._name = "P"
function P:_init()
  self:super()
  self.w = self:create_parameter{ shape = {1} }
  self.notes = "hello"   -- plain, not a tensor/layer/param
end
local p = P()
assert(p.notes == "hello", "plain field readable")
assert(fields_of(p).order[1] == "w" and fields_of(p).order[2] == nil,
       "plain field not in order book")
local sd = p:state_dict()
assert(sd.w ~= nil and sd.notes == nil, "plain field absent from state_dict")

-- --- 4: state_dict key format (Python byte-for-byte, 09-nn.md sec.3.5) --
local sd = net:state_dict()
assert(sd["w"] ~= nil, "state_dict has 'w'")
assert(sd["leaf.w"] ~= nil, "state_dict has 'leaf.w'")
assert(sd["fc.w"] ~= nil, "state_dict has 'fc.w'")
assert(sd["fc.leaf.w"] ~= nil, "state_dict has 'fc.leaf.w'")
local nkeys = 0
for _ in pairs(sd) do nkeys = nkeys + 1 end
assert(nkeys == 6, ("6 keys, got %d"):format(nkeys))

-- --- 5: set_state_dict round trip --------------------------------------
local sd2 = {}
for k, v in pairs(sd) do
  sd2[k] = core.pd_op_full({1}, 2.0, core.PD_DTYPE_FLOAT32,
                           core.pd_place_new(core.PD_PLACE_CPU, 0))
end
local nset = net:set_state_dict(sd2)
assert(nset == 6, ("set_state_dict set 6, got %d"):format(nset))

-- --- 6: train/eval recursion (09-nn.md sec.3.4) ------------------------
-- training is a PLAIN field (`self.training = m` goes through __newindex
-- into the plain slot; layer_index reads it back) -- assert via the
-- public field, not FIELDS internals.
net:train()
assert(net.training == true, "root training == true after train()")
local all_train = true
for _, sub in net:named_sublayers() do
  if sub.training ~= true then all_train = false end
end
assert(all_train, "all sublayers training == true after train()")
net:eval()
for _, sub in net:named_sublayers() do
  assert(sub.training == false,
         "sublayer training == false after eval()")
end

-- --- 7: print(layer) does not crash (坑④: default __tostring reads
-- obj._class, must resolve through layer_index to the class table) ------
local okp, serr = pcall(function() return tostring(net) end)
assert(okp, "print(layer) must not crash: " .. tostring(serr))
local okp2 = pcall(function() print(net) end)
assert(okp2, "print(net) no crash")

-- --- 8: 200-layer chain, state_dict with explicit stack (坑③) ---------
local Deep = class(Layer)
Deep._name = "Deep"
function Deep:_init(depth)
  self:super()
  self.w = self:create_parameter{ shape = {1} }
  if depth > 1 then
    self.next = Deep(depth - 1)
  end
end
local deep = Deep(200)
local okd, sddeep = pcall(function() return deep:state_dict() end)
assert(okd, "200-layer state_dict must not overflow: " .. tostring(sddeep))
assert(type(sddeep) == "table" and sddeep["w"] ~= nil, "deep sd has w")
assert(sddeep["next.w"] ~= nil, "deep sd has next.w")

-- --- 9: class.cast forbidden (broken instance without FIELDS) ----------
-- NOTE: `class.cast` is NOT a function -- the pl.class table's __index
-- metamethod turns `class.cast` into "make a class named cast" (pl/class.lua
-- __index branch). The cast METHOD lives on each class: `MyLeaf:cast(obj)`
-- (pl/class.lua `c.cast = cast`). Foundations sec.1.5 ② forbids it
-- because it swaps the metatable without running _create.
local fake = {}
local okc, cerr = pcall(function()
  local broken = MyLeaf:cast(fake)
  broken.w = 1   -- must error: no FIELDS
end)
assert(not okc, "assignment on class.cast'd instance must error")
assert(tostring(cerr):find("FIELDS") ~= nil,
       "error mentions FIELDS: " .. tostring(cerr))

-- --- 10: named_sublayers / sublayers() List ----------------------------
-- MyNet structure: net(leaf, fc) and fc has fc.leaf -- named_sublayers
-- is RECURSIVE (upstream layers.py named_sublayers includes all
-- descendants), so the full list is leaf, fc, fc.leaf = 3.
local sub_names = {}
for name, sub in net:named_sublayers() do
  sub_names[#sub_names + 1] = name
end
assert(sub_names[1] == "leaf" and sub_names[2] == "fc"
       and sub_names[3] == "fc.leaf",
       ("sublayers == leaf,fc,fc.leaf got %s"):format(
         table.concat(sub_names, ",")))
local subs = net:sublayers()
assert(type(subs) == "table" and subs:is_a(require "pl.List"),
       "sublayers() returns pl.List")
assert(subs:len() == 3, ("3 sublayers, got %d"):format(subs:len()))

-- --- 11: parameters() returns pl.List ----------------------------------
local params = net:parameters()
assert(params:is_a(require "pl.List"), "parameters() returns pl.List")
assert(params:len() == 6, ("6 params, got %d"):format(params:len()))

-- --- 12: forward + __call dispatch -------------------------------------
local Lin = class(Layer)
Lin._name = "Lin"
function Lin:_init()
  self:super()
  self.w = self:create_parameter{ shape = {1} }
end
function Lin:forward(x)
  return core.pd_op_multiply(x, self.w)
end
local lin = Lin()
local x = core.pd_op_full({1}, 3.0, core.PD_DTYPE_FLOAT32,
                          core.pd_place_new(core.PD_PLACE_CPU, 0))
local y = lin(x)              -- __call -> forward
assert(y:numel() == 1, "forward through __call")
assert_close(y:item(1), 0.0, "w initialized to 0 -> 3*0 == 0")
local y2 = lin:forward(x)
assert_close(y2:item(1), 0.0, "explicit :forward same result")

-- --- 13: apply(fn) visits self + every sublayer ------------------------
-- Layer instances: net, net.leaf, net.fc, net.fc.leaf = 4.
local visited = 0
net:apply(function() visited = visited + 1 end)
assert(visited == 4, ("apply visits 4 layers, got %d"):format(visited))

-- --- 14: to(place)/cpu() on CPU are identity round trips ---------------
local okto = pcall(function() return net:to(paddle.CPUPlace()) end)
assert(okto, "to(CPUPlace) works")
local okcpu = pcall(function() return net:cpu() end)
assert(okcpu, "cpu() works")

-- --- 15: add_sublayer / add_parameter explicit APIs --------------------
local E = class(Layer)
E._name = "E"
function E:_init()
  self:super()
  local w = self:create_parameter{ shape = {1} }
  self:add_parameter("w", w)    -- explicit registration path
  self:add_sublayer("sub", MyLeaf())
end
local e = E()
assert(e.w ~= nil, "add_parameter registered")
assert(e.sub ~= nil, "add_sublayer registered")
local enames = {}
for name in e:named_parameters() do enames[#enames + 1] = name end
assert(enames[1] == "w" and enames[2] == "sub.w",
       ("explicit reg names == w,sub.w got %s"):format(
         table.concat(enames, ",")))

print("NN_LAYER_SMOKE PASS")
