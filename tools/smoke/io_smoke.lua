-- tools/smoke/io_smoke.lua -- M1 2.11: paddle.io Dataset/DataLoader (single worker)
-- 镜像 11-io.md §5 验收 + api/io.md 契约:
--   1. DataLoader 迭代 batch 序号从 1 开始
--   2. shuffle=true 固定种子下顺序与 Python numpy 2.4.6 一致
--      (期望值 build-test/diag_io_expected.py,seed=42,Insight choice 逐位对拍)
--   3. drop_last 两种取值尾批行为正确
--   4. 中途 break 后重新开始迭代行为正确
--   5. num_workers>0 报"尚未实现"而非静默降级
--   6. IterableDataset 路径可用
--   7. random_split 划分(绝对数 / 比例)不重叠、并集覆盖全集
--   8. default_collate:数字 -> Tensor;单字段 -> stack;多字段 -> pl.List
--
-- Run (from the repo root, same env as optimizer_smoke):
--   PATH=<paddle runtime dll dirs>;$PATH lua tools/smoke/io_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
-- Insight7 (shuffle RNG, soft dep): Lua 绑定 + _insight 原生 + CPU backend
package.path = "E:/code/Insight7/bindings/lua/?.lua;E:/code/Insight7/bindings/lua/?/init.lua;" .. package.path
package.cpath = "E:/code/Insight7/build/bindings/lua/?.dll;" .. package.cpath
package.cpath = "E:/code/Insight7/build/backends/cpu/?.dll;" .. package.cpath

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
local io = paddle.io

-- 映 9.1 numpy 期望值的 +1 换算(0-based -> 1-based,diag_io_expected.py)
local SEQ_RAW = { 9, 2, 6, 1, 8, 3, 10, 5, 4, 7 }       -- choice(10,rep=false)
local RAND_RAW = { 7, 4, 8, 5, 7, 10 }                   -- choice(10,size=6,rep=true)
local W_RAW = { 3, 5, 4, 2, 1 }                          -- choice(5,rep=false,p=w)
local B_ALL = { { 9, 2, 6, 1 }, { 8, 3, 10, 5 }, { 4, 7 } }
local B_DROP = { { 9, 2, 6, 1 }, { 8, 3, 10, 5 } }

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

local function eq_list(a, b, msg)
  ntests = ntests + 1
  if #a ~= #b then
    error(("assertion #%d failed: %s (len %d vs %d)"):format(
      ntests, msg, #a, #b), 2)
  end
  for i = 1, #a do
    if a[i] ~= b[i] then
      error(("assertion #%d failed: %s ([%d]=%s vs %s)"):format(
        ntests, msg, i, tostring(a[i]), tostring(b[i])), 2)
    end
  end
end

-- 把 collate 出来的 batch 全读成 Lua 数组(1-based 平铺)
local function tensor_to_list(t)
  local out = {}
  for i = 1, t:numel() do out[i] = t:item(i) end
  return out
end

-- ---------------------------------------------------------------------
-- 简易 map-style dataset:get(i) 返回数字 i(len 个样本)
-- ---------------------------------------------------------------------
local NumDS = require("pl.class")(io.Dataset)
function NumDS:_init(n)
  self.n = n
end
function NumDS:len()
  return self.n
end
function NumDS:get(i)
  return i
end

local ds10 = NumDS(10)
check(ds10:is_a(io.Dataset), "NumDS is_a io.Dataset")

-- ---------------------------------------------------------------------
-- 1. DataLoader 迭代:batch 序号从 1 开始,无 shuffle 顺序 1..n
-- ---------------------------------------------------------------------
local dl = io.DataLoader{ dataset = ds10, batch_size = 4 }
local no, batch = dl:iter()()
check(no == 1, "first batch number == 1")
check(batch:numel() == 4, "first batch has 4 samples")
eq_list(tensor_to_list(batch), { 1, 2, 3, 4 }, "no-shuffle batch[1]")
local rows = {}
local ii = 0
for n_, b in dl:iter() do
  ii = ii + 1
  rows[ii] = tensor_to_list(b)
end
check(ii == 3, "10 samples / bs 4 -> 3 batches (drop_last=false)")
eq_list(rows[3], { 9, 10 }, "last partial batch kept")

-- ---------------------------------------------------------------------
-- 2. shuffle 固定种子与 numpy 逐位一致(seed=42,Insight choice)
-- ---------------------------------------------------------------------
local dlr = io.DataLoader{ dataset = ds10, batch_size = 10, shuffle = true, seed = 42 }
local _, b1 = dlr:iter()()
eq_list(tensor_to_list(b1), SEQ_RAW, "shuffle seed42 == np.random.choice(10,rep=false)+1")

-- RandomSampler 直测(num_samples 缺省 = len)
local rs = io.RandomSampler{ dataset = ds10, seed = 42 }
local got = {}
for v in rs:iter() do got[#got + 1] = v end
eq_list(got, SEQ_RAW, "RandomSampler seed42 order")

-- replacement=true + num_samples
local rsr = io.RandomSampler{ dataset = ds10, num_samples = 6,
                              replacement = true, seed = 42 }
local gotr = {}
for v in rsr:iter() do gotr[#gotr + 1] = v end
eq_list(gotr, RAND_RAW, "RandomSampler replace=true num_samples=6")

-- WeightedRandomSampler(p = {0.1,0.1,0.2,0.2,0.4})
local rsw = io.WeightedRandomSampler{
  weights = { 0.1, 0.1, 0.2, 0.2, 0.4 },
  num_samples = 5, replacement = false, seed = 42 }
local gotw = {}
for v in rsw:iter() do gotw[#gotw + 1] = v end
eq_list(gotw, W_RAW, "WeightedRandomSampler seed42 == np weighted choice +1")

-- 不可复现参数报错
local okb = pcall(function() io.RandomSampler{ dataset = ds10,
  num_samples = 20 } end)  -- replace=false 但 num_samples > len
check(not okb, "RandomSampler num_samples > len without replacement raises")

-- ---------------------------------------------------------------------
-- 3. drop_last 两种取值
-- ---------------------------------------------------------------------
local dld = io.DataLoader{ dataset = ds10, batch_size = 4, drop_last = true }
local n_batch = 0
for _ in dld:iter() do n_batch = n_batch + 1 end
check(n_batch == 2, "drop_last=true -> 2 full batches")

-- batch 面值验证(diag b_all / b_drop,shuffle seed 42)
local dlb = io.DataLoader{ dataset = ds10, batch_size = 4,
                           shuffle = true, seed = 42 }
local bi = 0
for _, b in dlb:iter() do
  bi = bi + 1
  eq_list(tensor_to_list(b), B_ALL[bi], ("batch[%d] == b_all"):format(bi))
end
check(bi == 3, "shuffle b_all -> 3 batches")

local dlbd = io.DataLoader{ dataset = ds10, batch_size = 4,
                            shuffle = true, seed = 42, drop_last = true }
local bd = 0
for _, b in dlbd:iter() do
  bd = bd + 1
  eq_list(tensor_to_list(b), B_DROP[bd], ("drop_last batch[%d]"):format(bd))
end
check(bd == 2, "shuffle drop_last -> 2 batches")

-- ---------------------------------------------------------------------
-- 4. 中途 break 后重新开始迭代
-- ---------------------------------------------------------------------
local dllo = io.DataLoader{ dataset = ds10, batch_size = 4, shuffle = true, seed = 42 }
local it1 = dllo:iter()
local n1, bfirst = it1()
check(n1 == 1, "break: first iter batch no == 1")
-- 重开 = 重新 :iter():batch 序号从 1 重启(同 Python __iter__)。
-- 顺序不比较:构造期播一次、epoch 间继续消耗 Insight 流(numpy 全局流
-- 语义),第二次顺序本来就和第一次不同(11-io.md §6 ② 只承诺同 seed
-- 从头构造的两次运行逐位一致,不承诺同一 loader 内 epoch 间相同)。
local it2 = dllo:iter()
local n2, b2 = it2()
check(n2 == 1, "reopen: batch no restarts at 1")
check(b2 ~= nil and (b2:numel() == 4), "reopen: first batch is a valid size-4 batch")

-- ---------------------------------------------------------------------
-- 5. num_workers>0 且 dataset 是实例 -> 报错(需工厂函数,不静默降级)
-- ---------------------------------------------------------------------
local okn = pcall(function()
  io.DataLoader{ dataset = ds10, batch_size = 4, num_workers = 4 }
end)
check(not okn, "num_workers=4 with a Dataset instance must raise (factory required)")

-- ---------------------------------------------------------------------
-- 6. IterableDataset 路径
-- ---------------------------------------------------------------------
local IterDS = require("pl.class")(io.IterableDataset)
function IterDS:_init(n)
  self.n = n
end
function IterDS:iter()
  local i = 0
  return function()
    i = i + 1
    if i > self.n then return nil end
    return i
  end
end
local ids = IterDS(10)
check(ids:is_a(io.IterableDataset), "IterDS is_a IterableDataset")
local dli = io.DataLoader{ dataset = ids, batch_size = 4 }
local irows = {}
local ci = 0
for nnum, b in dli:iter() do
  ci = ci + 1
  irows[ci] = tensor_to_list(b)
  check(nnum == ci, "IterableDataset batch numbers consecutive")
end
check(ci == 3, "IterableDataset 10 items / bs4 -> 3 batches")
eq_list(irows[1], { 1, 2, 3, 4 }, "IterableDataset batch[1]")
eq_list(irows[3], { 9, 10 }, "IterableDataset last partial batch")

-- IterableDataset + shuffle/sampler/batch_sampler 全部报错
local ok1 = pcall(function() io.DataLoader{ dataset = ids, shuffle = true } end)
local ok2 = pcall(function() io.DataLoader{ dataset = ids,
  sampler = io.SequenceSampler(ds10) } end)
local ok3 = pcall(function() io.DataLoader{ dataset = ids,
  batch_sampler = io.BatchSampler{ sampler = io.SequenceSampler(ds10),
                                   batch_size = 2 } } end)
check(not ok1 and not ok2 and not ok3,
  "IterableDataset excludes shuffle/sampler/batch_sampler (reader.py 互斥)")

-- ---------------------------------------------------------------------
-- 7. random_split:绝对数 + 比例,不重叠、并集覆盖全集
-- ---------------------------------------------------------------------
local splits = io.random_split(ds10, { 4, 6 })
check(splits[1]:len() == 4 and splits[2]:len() == 6,
  "random_split absolute sizes {4,6}")
local seen = {}
for s = 1, 2 do
  for k = 1, splits[s]:len() do
    local v = splits[s]:get(k)   -- Subset:get -> dataset:get 返回数字 i
    check(seen[v] == nil, "random_split: no duplicate index")
    seen[v] = true
  end
end
check(#seen == 10, "random_split: union covers all 10 samples")

local fsplits = io.random_split(ds10, { 0.8, 0.2 })  -- 10*0.8=8,余数交替
check(fsplits[1]:len() == 8 and fsplits[2]:len() == 2,
  "random_split fraction {0.8,0.2} -> {8,2}")

-- ---------------------------------------------------------------------
-- 8. 其它采样器 / 批次组织:SequenceSampler / BatchSampler / Subset
-- ---------------------------------------------------------------------
local seq = io.SequenceSampler(ds10)
local sv = {}
for v in seq:iter() do sv[#sv + 1] = v end
eq_list(sv, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, "SequenceSampler 1..10")

local bs = io.BatchSampler{ sampler = seq, batch_size = 3, drop_last = true }
local blen = bs:len()
check(blen == 3, "BatchSampler len(10/3 drop) == 3")
local bsv = {}
for b in bs:iter() do bsv[#bsv + 1] = b end
check(#bsv == 3, "BatchSampler iter -> 3 batches")
eq_list(bsv[2], { 4, 5, 6 }, "BatchSampler batch[2] == {4,5,6}")

local sub = io.Subset{ dataset = ds10, indices = { 3, 7, 1 } }
check(sub:len() == 3, "Subset len == 3")
check(sub:get(2) == 7, "Subset:get(2) == 7 (dataset:get(7))")

-- ---------------------------------------------------------------------
-- 9. default_collate:数字/多字段
-- ---------------------------------------------------------------------
local coll1 = io.default_collate({ 1, 2, 3 })
eq_list(tensor_to_list(coll1), { 1, 2, 3 }, "default_collate numbers -> Tensor")

-- TensorDataset 多字段 -> pl.List of 每字段 stack
local TD = require "paddle.tensor"  -- 确保 tensor 元表
local tdset = io.TensorDataset{
  tensors = { core.pd_op_full({ 3, 1 }, 1.0, core.PD_DTYPE_FLOAT32,
                              core.pd_place_new(core.PD_PLACE_CPU, 0)),
              core.pd_op_full({ 3, 1 }, 2.0, core.PD_DTYPE_FLOAT32,
                              core.pd_place_new(core.PD_PLACE_CPU, 0)) } }
check(tdset:len() == 3, "TensorDataset len == 3")
local x, y = tdset:get(2)
check(x:numel() == 1 and y:numel() == 1, "TensorDataset:get multi-return slices")

local dltd = io.DataLoader{ dataset = tdset, batch_size = 2 }
local _, btd = dltd:iter()()
check(type(btd) == "table" and btd.is_a ~= nil
      and (btd:is_a(require("pl.List"))), "multi-field collate -> pl.List")
check(btd[1]:numel() == 2 and btd[2]:numel() == 2,
  "pl.List has stacked per-field tensors [2]")

print(("IO_SMOKE PASS (%d assertions)"):format(ntests))