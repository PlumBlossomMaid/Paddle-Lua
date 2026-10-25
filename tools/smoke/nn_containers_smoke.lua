-- tools/smoke/nn_containers_smoke.lua -- M1 2.9.5: paddle.nn 容器
-- (Sequential / LayerList / LayerDict)。镜像 Python 容器测试的行为
-- 断言:
--   test/legacy_test/test_layerlist.py        (空表 insert 边界)
--   test/legacy_test/test_imperative_container_layerlist.py
--       (append/insert/extend/负数下标/len/forward 依序 shape)
--   nn.md §2.2                                (构造、len、iter 形态)
--   container.py LayerDict(构造/update/keys/values/len/iter)
--
-- 覆盖:
--   LayerList:变长构造、len、iter(1-based)、数字下标(正/负)、
--     append/insert(边界与搬移序)/extend、state_dict 键名 0-based
--     (D-R15,与 Python 逐字符一致)、is_a(nn.Layer)
--   Sequential:纯构造 + 命名对构造、forward 依序执行(probe 层)、
--     append/insert/extend、数字/字符串下标、state_dict 键名前缀
--   LayerDict:构造、update(新键/覆盖键)、keys/values/iter 顺序、len
--   5.1 行为:#ml == 0(ipairs 跑空的镜像,见 nn.md §4 diff 4)
--   argrule 门禁:坏构造(非 Layer 元素 / 命名对缺 name)报错
--
-- Run (from the repo root, same env as nn_layers_smoke):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/nn_containers_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
local nn = paddle.nn

local F32 = core.PD_DTYPE_FLOAT32
local CPU = core.pd_place_new(core.PD_PLACE_CPU, 0)

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

local function close(a, b, msg)
  ntests = ntests + 1
  local err = math.abs(a - b) / math.max(1.0, math.abs(b))
  if not (err < 1e-5) then
    error(("assertion #%d failed: %s (got %.7g, want %.7g, rel %.2e)"
      ):format(ntests, msg, a, b, err), 2)
  end
end

local function item(t, i)
  return t:item(i or 1)
end

local function ctensor(shape, v, dtype)
  return core.pd_op_full(shape, v, dtype or F32, CPU)
end

local function shape_of(t)
  return t:shape()
end

-- 层探针:forward 把 tag 追加到输入表,返回同一个表 —— 验证容器
-- 「依序执行 + 传参」,不依赖任何 tensor 数值(纯 Lua 层)。
-- 构造必须用位置调用 SeqProbe("a"):pl.class 的表调用 SeqProbe{ "a" }
-- 会把 { "a" } 整表当唯一参数传给 _init(tag),tag 变成表而非字符串。
local class = require "pl.class"
local SeqProbe = class(nn.Layer)
SeqProbe._name = "SeqProbe"
function SeqProbe:_init(tag)
  self:super()
  self.tag = tag
end
function SeqProbe:forward(x)
  x[#x + 1] = self.tag
  return x
end

-- ---------------------------------------------------------------------
-- 1. LayerList:构造 / len / iter / 下标
-- ---------------------------------------------------------------------
local l1, l2, l3 = nn.Linear{2, 4}, nn.Linear{4, 8}, nn.Linear{8, 2}
local ml = nn.LayerList{ l1, l2, l3 }
check(ml:len() == 3, "LayerList len == 3")
check(ml[1] == l1 and ml[2] == l2 and ml[3] == l3, "LayerList index 1-based")
check(ml[-1] == l3, "LayerList negative index -1 -> last")
check(ml[-3] == l1, "LayerList negative index -3 -> first")
check(ml[4] == nil, "LayerList out-of-range index -> nil")
check(ml:is_a(nn.Layer), "LayerList is_a(nn.Layer) (D27)")
check(#ml == 0, "5.1: #ml == 0 (ipairs runs empty, nn.md §4 diff 4)")
local seen = {}
for i, l in ml:iter() do
  seen[#seen + 1] = { i, l }
end
check(#seen == 3 and seen[1][1] == 1 and seen[3][2] == l3,
  "LayerList iter is 1-based in order")

-- ---------------------------------------------------------------------
-- 2. LayerList:append / insert / extend(镜像 container.py 语义)
-- ---------------------------------------------------------------------
local ml2 = nn.LayerList{}
ml2:append(l1)
ml2:append(l2)
check(ml2:len() == 2 and ml2[2] == l2, "LayerList append")
local lx = nn.Linear{2, 2}
ml2:insert(1, lx)
check(ml2:len() == 3 and ml2[1] == l1 and ml2[2] == lx and ml2[3] == l2,
  "LayerList insert(1) shifts right")
local ly = nn.Linear{2, 2}
ml2:insert(-1, ly)
-- Python [l1,lx,l2].insert(-1, ly) → index=-1+3=2 → [l1,lx,ly,l2]
-- (0-based);Lua 1-based:ml2[3]==ly, ml2[4]==l2(container.py:689-695)
check(ml2:len() == 4 and ml2[3] == ly and ml2[4] == l2,
  "LayerList insert(-1) == before last (container.py:689-695)")
-- 空表 insert(0) 成功、insert(1) 报错(test_layerlist.py 镜像)
local mt0 = nn.LayerList{}
mt0:insert(0, l1)
check(mt0:len() == 1 and mt0[1] == l1, "LayerList empty insert(0) ok")
local okk = pcall(function() nn.LayerList{}:insert(1, l1) end)
check(not okk, "LayerList empty insert(1) must raise (test_layerlist.py)")
-- extend
local ml3 = nn.LayerList{}
ml3:extend({ l1, l2 })
check(ml3:len() == 2 and ml3[1] == l1 and ml3[2] == l2, "LayerList extend")
-- state_dict 键名 0-based(D-R15,与 Python 逐字符一致)
local mlsd = nn.LayerList{ nn.Linear{2, 3} }
local sd = mlsd:state_dict()
check(sd["0.weight"] ~= nil and sd["0.bias"] ~= nil,
  "LayerList state_dict keys are 0-based (\"0.weight\")")
check(sd["1.weight"] == nil, "LayerList state_dict has no phantom key")

-- ---------------------------------------------------------------------
-- 3. Sequential:构造 + forward 依序(探针层)+ 命名对
-- ---------------------------------------------------------------------
-- SeqProbe 是普通 pl.class 类(无 argrule):必须用位置调用 SeqProbe("a")
-- ——表调用 SeqProbe{ "a" } = SeqProbe({ "a" }),pl.class 会把整表当
-- 唯一参数传进 _init(tag),tag 变成表而非字符串。
local seq = nn.Sequential{ SeqProbe("a"), SeqProbe("b"), SeqProbe("c") }
local acc = seq({})
check(acc[1] == "a" and acc[2] == "b" and acc[3] == "c",
  "Sequential forward runs layers in order")
check(seq:len() == 3, "Sequential len")
check(seq[1] ~= nil and seq[3] ~= nil, "Sequential int index works")
-- 命名对构造(容器.py:808-816 的 (name, layer) pair 列表形态)
local seq2 = nn.Sequential{ { "fc1", SeqProbe("a") },
                             { "fc2", SeqProbe("b") } }
check(seq2:len() == 2, "Sequential named pairs len")
check(seq2["fc1"] ~= nil and seq2["fc2"] ~= nil,
  "Sequential named access by string")
check(seq2[1] ~= nil and seq2[2] ~= nil,
  "Sequential named pairs still indexable by order")
local acc2 = seq2({})
check(acc2[1] == "a" and acc2[2] == "b",
  "Sequential named forward runs in order")
check(seq2:iter() ~= nil, "Sequential iter exists")
-- append 命名对容器也可用(键名 str(len),与 Python container.py:848 一致)
seq2:append(l3)
check(seq2:len() == 3 and seq2[3] == l3, "Sequential append on named pairs")
-- insert / extend 只在纯层构造上测:命名对容器键名不是连续数字
-- (container.py:679-695 用 str(i) 搬移,Python 对命名对容器 insert 同样
-- 无效/报错)——镜像上游语义,不在命名对上跑 insert。
local seq3 = nn.Sequential{ SeqProbe("x"), SeqProbe("y") }
seq3:insert(1, SeqProbe("z"))
check(seq3:len() == 3 and seq3[2] ~= nil, "Sequential insert shifts")
seq3:extend({ SeqProbe("w") })
check(seq3:len() == 4, "Sequential extend")
-- 坏构造报错(非 Layer / 命名对缺 name)
local okk = pcall(function() nn.Sequential{ l1, "not-a-layer" } end)
check(not okk, "Sequential non-Layer element must raise")
local okk = pcall(function() nn.Sequential{ { l1, l1 } } end)
check(not okk, "Sequential pair without string name must raise")
-- state_dict 键名前缀(命名对 "fc1",纯构造 "0")
local seqsd = nn.Sequential{ { "fc1", nn.Linear{2, 3} } }
local sd2 = seqsd:state_dict()
check(sd2["fc1.weight"] ~= nil, "Sequential state_dict uses pair name")
local seqsd2 = nn.Sequential{ nn.Linear{2, 3} }
local sd3 = seqsd2:state_dict()
check(sd3["0.weight"] ~= nil, "Sequential state_dict uses 0-based name")
check(seq:is_a(nn.Layer) and seq2:is_a(nn.Layer), "Sequential is_a(nn.Layer)")

-- ---------------------------------------------------------------------
-- 4. LayerDict:构造 / update / keys / values / iter / len
-- ---------------------------------------------------------------------
local ld = nn.LayerDict{ { "conv1", l1 }, { "conv2", l2 }, { "conv3", l3 } }
check(ld:len() == 3, "LayerDict len == 3")
check(ld["conv1"] == l1 and ld["conv3"] == l3, "LayerDict string access")
local keys = ld:keys()
check(#keys == 3 and keys[1] == "conv1" and keys[3] == "conv3",
  "LayerDict keys in insertion order")
local vals = ld:values()
check(#vals == 3 and vals[2] == l2, "LayerDict values in order")
local dseen = {}
for k, l in ld:iter() do
  dseen[#dseen + 1] = { k, l }
end
check(#dseen == 3 and dseen[2][1] == "conv2" and dseen[3][2] == l3,
  "LayerDict iter yields (key, layer) in order")
-- update:新键 len+1,覆盖键 len 不变(容器.py:39-330)
ld:update("conv4", lx)
check(ld:len() == 4 and ld["conv4"] == lx, "LayerDict update adds key")
ld:update("conv2", lx)
check(ld:len() == 4 and ld["conv2"] == lx, "LayerDict update overwrites key")
local okk = pcall(function() nn.LayerDict{ { 1, l1 } } end)
check(not okk, "LayerDict non-string key must raise")
check(ld:is_a(nn.Layer), "LayerDict is_a(nn.Layer)")
-- state_dict(D-R15:键名是给 sublayer 的名字)
local ldsd = nn.LayerDict{ { "fc", nn.Linear{2, 3} } }
local sd4 = ldsd:state_dict()
check(sd4["fc.weight"] ~= nil, "LayerDict state_dict uses key name")

-- ---------------------------------------------------------------------
-- 5. 真实 tensor 路径:Sequential(Linear) 数值 shape(镜像
--    test_imperative_container_layerlist.py 的 shape 断言)
-- ---------------------------------------------------------------------
local net = nn.Sequential{ nn.Linear{4, 4}, nn.Linear{4, 2} }
local x = ctensor({ 5, 4 }, 1.0)
local y = net(x)
local sh = shape_of(y)
check(sh[1] == 5 and sh[2] == 2,
  "Sequential(Linear,Linear) forward shape [5,4]->[5,2]")

print(("NN_CONTAINERS_SMOKE PASS (%d assertions)"):format(ntests))