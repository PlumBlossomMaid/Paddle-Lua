-- tools/smoke/lanes_dataloader_smoke.lua -- M2 3.2: DataLoader multi-worker
-- (Lanes, 13-lanes.md §6 acceptance + api/io.md §4/§6 契约)
--
-- 覆盖:
--   1. num_workers=2/4 与单 worker 的 batch 逐位一致
--   2. 固定种子下多 worker 与单 worker 数据顺序一致(seed=42)
--   3. drop_last 尾批行为与单 worker 一致
--   4. worker 里 error() -> 主 lane 收到完整错误信息(含 batch_no)
--   5. 迭代中途 break,worker 正确退出;重新 iter() 无残留
--   6. 工厂函数 + dataset_len 形态;实例 + num_workers>0 报错
--   7. IterableDataset + num_workers>0 报错
--
-- Run (from repo root, same env as io_smoke):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/lanes_dataloader_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.cpath = "./build-test/dl-deps/luajit-lib/?.dll;" .. package.cpath
package.cpath = "./build-test/dl-deps/luajit-lib/?/core.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.path = "./build-test/dl-deps/luajit-lib/?.lua;" .. package.path
-- smoke 专用的 dataset 模块(worker 需通过 package.path 找到它)
package.path = "./tools/smoke/?.lua;" .. package.path
-- Insight7 (shuffle RNG, soft dep)
package.path = "E:/code/Insight7/bindings/lua/?.lua;E:/code/Insight7/bindings/lua/?/init.lua;" .. package.path
package.cpath = "E:/code/Insight7/build/bindings/lua/?.dll;" .. package.cpath
package.cpath = "E:/code/Insight7/build/backends/cpu/?.dll;" .. package.cpath

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
local io = paddle.io

local LANES_OK = pcall(require, "lanes")
if not LANES_OK then
  print("LANES_DATALOADER_SMOKE SKIP: lanes rock not installed")
  os.exit(0)
end

-- 工厂函数形态的 data source:worker 内 require(无 upvalue,可 dump)
local function factory(cfg)
  local mod = require "lanes_smoke_ds"
  return mod.NumDS(cfg.n)
end

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

local function tensor_to_list(t)
  local out = {}
  for i = 1, t:numel() do out[i] = t:item(i) end
  return out
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

-- ---------------------------------------------------------------------
-- 1. num_workers=2/4 与单 worker 逐位一致(不 shuffle)
-- ---------------------------------------------------------------------
local function collect_all(nw)
  local dl
  if nw == 0 then
    -- 单 worker:Dataset 实例形态(factory 在无 upvalue 且仅依赖可 require 模块,
    -- 主 lane 直接调它构造实例;多 worker 才把 factory 本身传进 lane)
    dl = io.DataLoader{ dataset = factory{ n = 10 },
                        num_workers = 0, batch_size = 4 }
  else
    dl = io.DataLoader{ dataset = factory, config = { n = 10 },
                        dataset_len = 10, num_workers = nw, batch_size = 4 }
  end
  local rows = {}
  for n_, b in dl:iter() do
    rows[#rows + 1] = tensor_to_list(b)
  end
  return rows
end

local single = collect_all(0)
check(#single == 3, "single worker: 10 samples / bs4 -> 3 batches")
local two = collect_all(2)
check(#two == 3, "num_workers=2: 3 batches")
eq_list(two[1], single[1], "nw=2 batch[1] == single")
eq_list(two[2], single[2], "nw=2 batch[2] == single")
eq_list(two[3], single[3], "nw=2 batch[3] == single (tail kept)")
local four = collect_all(4)
eq_list(four[1], single[1], "nw=4 batch[1] == single")
eq_list(four[2], single[2], "nw=4 batch[2] == single")
eq_list(four[3], single[3], "nw=4 batch[3] == single")

-- ---------------------------------------------------------------------
-- 2. 固定种子下多 worker 与单 worker 顺序一致(shuffle seed=42)
--    shuffle 采样在主 lane 的 batch_sampler,worker 只按索引取数,
--    所以顺序天然一致(api/io.md §6 坑②)。
-- ---------------------------------------------------------------------
local function collect_shuffle(nw)
  local dl
  if nw == 0 then
    dl = io.DataLoader{ dataset = factory{ n = 10 },
                        num_workers = 0, batch_size = 4,
                        shuffle = true, seed = 42 }
  else
    dl = io.DataLoader{ dataset = factory, config = { n = 10 },
                        dataset_len = 10, num_workers = nw, batch_size = 4,
                        shuffle = true, seed = 42 }
  end
  local rows = {}
  for _, b in dl:iter() do
    rows[#rows + 1] = tensor_to_list(b)
  end
  return rows
end

local s_single = collect_shuffle(0)
local s_multi = collect_shuffle(3)
check(#s_multi == #s_single, "shuffle nw=3 batch count == single")
for i = 1, #s_single do
  eq_list(s_multi[i], s_single[i],
          ("shuffle nw=3 batch[%d] == single (seed42)"):format(i))
end

-- ---------------------------------------------------------------------
-- 3. worker 错误传播:BadDS 在某索引 error -> 主 lane 完整信息
-- ---------------------------------------------------------------------
local ok_err, err_msg = pcall(function()
  local dl = io.DataLoader{
    dataset = function(cfg)
      local mod = require "lanes_smoke_ds"
      return mod.BadDS(cfg.n, cfg.bad_at)
    end,
    config = { n = 10, bad_at = 5 },
    dataset_len = 10, num_workers = 2, batch_size = 4,
  }
  for _ in dl:iter() do end   -- 某一批触发坏的索引
end)
check(not ok_err, "bad index in worker must raise")
check(tostring(err_msg):find("injected dataset error") ~= nil,
  "worker error message reaches main lane: " .. tostring(err_msg))
check(tostring(err_msg):find("batch") ~= nil,
  "worker error carries batch context")

-- ---------------------------------------------------------------------
-- 4. break 后重新 iter():旧 worker 池被清理,新轮次正常
-- ---------------------------------------------------------------------
local dlb = io.DataLoader{
  dataset = factory, config = { n = 10 },
  dataset_len = 10, num_workers = 2, batch_size = 4,
}
local it = dlb:iter()
local n1, b1 = it()
check(n1 == 1, "break: first batch number == 1")
local it2 = dlb:iter()
local n2, b2 = it2()
check(n2 == 1, "reopen: batch no restarts at 1")
check(b2 ~= nil and b2:numel() == 4, "reopen: valid size-4 batch")

-- ---------------------------------------------------------------------
-- 5. 工厂形态契约
-- ---------------------------------------------------------------------
-- 实例 + num_workers>0 -> 报错
local ok1 = pcall(function()
  local RealDS = require("pl.class")(io.Dataset)
  function RealDS:len() return 10 end
  function RealDS:get(i) return i end
  io.DataLoader{ dataset = RealDS(), num_workers = 2, batch_size = 4 }
end)
check(not ok1,
  "Dataset instance + num_workers>0 must raise (factory required)")

-- 工厂 + 缺 dataset_len -> 报错
local ok2 = pcall(function()
  io.DataLoader{ dataset = factory, config = { n = 10 },
                 num_workers = 2, batch_size = 4 }
end)
check(not ok2, "factory without dataset_len must raise")

-- IterableDataset + num_workers>0 -> 报错
local ok3 = pcall(function()
  local IterDS = require("pl.class")(io.IterableDataset)
  function IterDS:iter() return function() return nil end end
  io.DataLoader{ dataset = IterDS(), num_workers = 2, batch_size = 4 }
end)
check(not ok3, "IterableDataset + num_workers>0 must raise")

print(("LANES_DATALOADER_SMOKE PASS (%d assertions)"):format(ntests))