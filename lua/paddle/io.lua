-- lua/paddle/io.lua -- paddle.io: Dataset/Sampler/DataLoader (single worker)
-- Stage: P11 / M1 (WORKPLAN 2.11).
--
-- Contract: plan/api/io.md; design: plan/modules/11-io.md.
-- Upstream: python/paddle/io/dataloader/{dataset,sampler,batch_sampler,
-- collate}.py + python/paddle/io/reader.py.
--
-- ★ shuffle 的随机源 = Insight7 的 choice(用户 2026-10-22 定案):固定
--   种子下 Insight 与 numpy 2.4.6 逐位一致(本会话已对齐),因此固定
--   seed 下 Lua 的 shuffle 顺序与 Python np.random.choice 完全相同,
--   "训练曲线对拍"前提成立。Insight 是软依赖(foundations.md §3.7):
--   不可用时 shuffle=true 报清晰错误,不静默降级。
--
-- index 语义(api/io.md §3,整条链 1-based):Dataset:get(i)、采样器产出
-- 的索引、Subset.indices、weights 下标全部 1-based。Insight choice 返回
-- 0..n-1 的取值数组 → 采样层 +1(边界在库里,用户代码零换算)。
--
-- collate 输出 pl.List(D-R21);collate 内部中间缓冲用裸表(热路径)。
--
-- Lua 5.1 subset (C3).

local class = require "pl.class"
local List = require "pl.List"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local unpack = table.unpack or unpack

-- Lanes (P13 multi-worker; D8 hard dependency). Loaded lazily so that
-- single-worker users never pay the price of starting lanes.core, but the
-- error is actionable (rockspec declares lanes >= 3.16, C6).
local LANES
local function lanes()
  if LANES == nil then
    local ok, m = pcall(require, "lanes")
    if not ok then
      error("paddle.io: num_workers>0 needs Lua Lanes (C6; "
            .. "luarocks install lanes)", 3)
    end
    LANES = m
  end
  return LANES
end

-- Shared LuaTensor metatable probe (same pattern as nn/functional.lua).
local probe = paddle_core.tensor{1}
local TENSOR_MT = getmetatable(probe)
probe = nil
local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end
local Tensor = argrule.alias { function(v) return is_tensor(v) end }

-- CPU place cache (default_collate's number->tensor needs a place).
local CPU_PLACE
local function cpu_place()
  if CPU_PLACE == nil then
    CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  end
  return CPU_PLACE
end

local F64 = paddle_core.PD_DTYPE_FLOAT64
local I64 = paddle_core.PD_DTYPE_INT64

-- ---------------------------------------------------------------------
-- Insight7 soft dependency (shuffle RNG; user 2026-10-22 decision)
-- ---------------------------------------------------------------------

local INSIGHT = false   -- false = not probed yet

local function insight()
  if INSIGHT == false then
    local ok, mod = pcall(require, "insight")
    INSIGHT = ok and mod or nil
  end
  if INSIGHT == nil then
    error("paddle.io: shuffle needs Insight7 (numpy-aligned RNG; soft "
          .. "dependency, foundations.md §3.7). Install it or pass "
          .. "shuffle=false.", 3)
  end
  return INSIGHT
end

-- Insight int64 Array, 1-based Lua positions, values in [0, n). Read the
-- k-th slot (k 1-based) and return the 0-based value.
local function ins_at(arr, k)
  return arr:get(k - 1)     -- Insight get() is 0-based (insight_lua.cpp:685)
end

-- Draw num_samples indices in [1, n] from the numpy-aligned stream.
-- 播种不在这里:构造时播一次(见 RandomSampler/WeightedRandomSampler
-- ._init),epoch 间继续消耗流,同 numpy 全局流——每轮 iter 重播种会让
-- 每 epoch 同序。seed==nil 时同样不播种(用 Insight 当前全局种子)。
local function choice_indices(n, num_samples, replace)
  local ins = insight()
  local chosen = ins.choice(ins.arange(n), num_samples, replace)
  local out = {}
  for k = 1, num_samples do
    out[k] = ins_at(chosen, k) + 1   -- 0-based value -> 1-based sample index
  end
  return out
end

-- ---------------------------------------------------------------------
-- Dataset (dataset.py: Dataset 协议; 11-io.md §3.1)
-- ---------------------------------------------------------------------

local Dataset = class()
Dataset._name = "Dataset"
Dataset._is_iterable = false   -- map-style base

function Dataset:len()
  error(("'%s' not implement in class %s (must override :len)"
    ):format("len", self._name or "Dataset"), 2)
end

function Dataset:get(i)
  error(("'%s' not implement in class %s (must override :get)"
    ):format("get", self._name or "Dataset"), 2)
end

-- 多 worker(P13)工厂形态下的 len-only 占位数据集:工厂函数无法在主
-- lane 获得真实 Dataset 实例去计算 len(实例在 worker 里才构造),但
-- 采样器(SequenceSampler/RandomSampler)只需要 len() 来生成索引序列
-- —— 它们从不调 get()。用户通过 DataLoader 的 dataset_len 提供长度,
-- 与单 worker 的 sampler 行为逐位一致(api/io.md §6 坑②)。
local LenOnlyDS = class(Dataset)
LenOnlyDS._name = "LenOnlyDS"

function LenOnlyDS:_init(n)
  self._len = n
end

function LenOnlyDS:len()
  return self._len
end

function LenOnlyDS:get(i)
  error("LenOnlyDS:get should never be called (P13 placeholder)", 2)
end

local IterableDataset = class(Dataset)
IterableDataset._name = "IterableDataset"
IterableDataset._is_iterable = true

function IterableDataset:iter()
  error(("'%s' not implement in class %s (must override :iter)"
    ):format("iter", self._name or "IterableDataset"), 2)
end

function IterableDataset:get(i)
  error(("'%s' should not be called for %s"):format("get",
    self._name or "IterableDataset"), 2)
end

function IterableDataset:len()
  error(("'%s' should not be called for %s"):format("len",
    self._name or "IterableDataset"), 2)
end

-- ---------------------------------------------------------------------
-- TensorDataset (dataset.py: TensorDataset)
-- ---------------------------------------------------------------------

local TensorDataset = class(Dataset)
TensorDataset._name = "TensorDataset"

-- Accept a list of tensors: TensorDataset({t1,t2}) or
-- TensorDataset{ tensors = {t1,t2} }.
TensorDataset._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "tensors", type = argrule.table,
    doc = "list/pl.List of Tensors, same 1st-dim size" },
}(function(self, tensors)
  if tensors ~= nil and type(tensors) == "table"
     and tensors.tensors ~= nil then
    tensors = tensors.tensors
  end
  local n = #tensors
  assert(n > 0, "TensorDataset: tensors should not be empty")
  local first = tensors[1]:shape()[1]
  for i = 1, n do
    assert(is_tensor(tensors[i]),
      ("TensorDataset: element %d is not a Tensor"):format(i))
    assert(tensors[i]:shape()[1] == first,
      "TensorDataset: tensors not have same shape of the 1st dimension")
  end
  self._tensors = tensors
end)

function TensorDataset:get(i)
  -- i 1-based; each tensor[i] slices along dim 0 (tensor.lua index_number).
  local fields = {}
  for k = 1, #self._tensors do
    fields[k] = self._tensors[k][i]
  end
  return unpack(fields)
end

function TensorDataset:len()
  return self._tensors[1]:shape()[1]
end

-- ---------------------------------------------------------------------
-- ConcatDataset (dataset.py: ConcatDataset, cumulative_sizes + bisect)
-- ---------------------------------------------------------------------

local ConcatDataset = class(Dataset)
ConcatDataset._name = "ConcatDataset"

ConcatDataset._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "datasets", type = argrule.table,
    doc = "list/pl.List of Dataset (no IterableDataset)" },
}(function(self, datasets)
  assert(#datasets > 0, "ConcatDataset: datasets should not be empty")
  for i = 1, #datasets do
    assert(datasets[i]:is_a(Dataset),
      "ConcatDataset: each element should be paddle.io.Dataset")
    assert(not datasets[i]:is_a(IterableDataset),
      "ConcatDataset does not support IterableDataset")
  end
  self._datasets = datasets
  local cum = {}
  local s = 0
  for i = 1, #datasets do
    s = s + datasets[i]:len()
    cum[i] = s
  end
  self._cum = cum
end)

function ConcatDataset:len()
  return self._cum[#self._cum]
end

function ConcatDataset:get(i)
  -- i 1-based; linear bisect-right over cumulative sizes (small n).
  local ds_i = 1
  while ds_i <= #self._cum and self._cum[ds_i] < i do
    ds_i = ds_i + 1
  end
  local sample_i
  if ds_i == 1 then
    sample_i = i
  else
    sample_i = i - self._cum[ds_i - 1]
  end
  return self._datasets[ds_i]:get(sample_i)
end

-- ---------------------------------------------------------------------
-- Subset + random_split (dataset.py: Subset / random_split)
-- ---------------------------------------------------------------------

local Subset = class(Dataset)
Subset._name = "Subset"

Subset._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "dataset", type = Dataset, doc = "the whole Dataset" },
  { name = "indices", type = argrule.table,
    doc = "list/pl.List of 1-based indices into the whole set" },
}(function(self, dataset, indices)
  if indices ~= nil and type(indices) == "table"
     and indices.indices ~= nil then
    indices = indices.indices
  end
  self._dataset = dataset
  self._indices = indices
end)

function Subset:get(i)
  return self._dataset:get(self._indices[i])
end

function Subset:len()
  return #self._indices
end

-- Read the i-th element (1-based) of a 1-D LuaTensor as a scalar.
-- t[i] -> index_number -> pd_op_slice with decrease (0-d, numel==1);
-- :item() reads the scalar (tensor.lua / tensor_binding.cc "item").
local function tensor_at(t, i)
  return t[i]:item()
end

-- random_split(dataset, lengths, seed) -> pl.List of Subset
-- lengths are absolute sizes, or fractions summing to ~1 (remainder
-- distributed round-robin like upstream dataset.py:334-344, NOT all to
-- the last split).
-- RNG: seed==nil -> paddle.randperm (pd_op_randperm, 1-based, upstream
--   dataset.py 的 randperm 对齐);seed given -> Insight randperm
--   (numpy 2.4.6 对齐,固定种子可复现 —— api/io.md §2.5 契约)。
-- perm 统一为 Lua 表:pd_op_randperm 经 emit_capi returns_index +1 已
--   1-based,Insight randperm 0-based 手动 +1 —— 两条路径结果一致。
local function random_split(dataset, lengths, seed)
  assert(dataset:is_a(Dataset), "random_split: dataset should be a Dataset")
  local n_total = dataset:len()
  local sum = 0
  for i = 1, #lengths do sum = sum + lengths[i] end
  local is_frac = math.abs(sum - 1) < 1e-6 and sum <= 1
  local sizes = {}
  if is_frac then
    local acc = 0
    for i = 1, #lengths do
      sizes[i] = math.floor(n_total * lengths[i])
      acc = acc + sizes[i]
    end
    local remainder = n_total - acc
    for r = 0, remainder - 1 do
      sizes[(r % #lengths) + 1] = sizes[(r % #lengths) + 1] + 1
    end
  else
    for i = 1, #lengths do sizes[i] = lengths[i] end
  end
  local total = 0
  for i = 1, #sizes do total = total + sizes[i] end
  assert(total == n_total,
    "random_split: sum of lengths does not equal the length of the dataset")
  local perm = {}
  if seed ~= nil then
    local ins = insight()
    ins.seed(seed)
    local chosen = ins.randperm(n_total)   -- 0-based, numpy-aligned
    for k = 1, n_total do
      perm[k] = ins_at(chosen, k) + 1      -- -> 1-based
    end
  else
    local pt = paddle_core.pd_op_randperm(n_total, I64, cpu_place())
    for k = 1, n_total do
      perm[k] = pt[k]:item()               -- already 1-based (+1 in C ABI)
    end
  end
  local splits = List()
  local offset = 0
  for i = 1, #sizes do
    local idx = {}
    for k = 1, sizes[i] do
      idx[k] = perm[offset + k]
    end
    splits:append(Subset{ dataset = dataset, indices = idx })
    offset = offset + sizes[i]
  end
  return splits
end

-- ---------------------------------------------------------------------
-- Sampler base (sampler.py: Sampler)
-- ---------------------------------------------------------------------

local Sampler = class()
Sampler._name = "Sampler"

function Sampler:iter()
  error(("'%s' not implement in class %s (must override :iter)"
    ):format("iter", self._name or "Sampler"), 2)
end

-- SequenceSampler (sampler.py: 1..n, 1-based)
local SequenceSampler = class(Sampler)
SequenceSampler._name = "SequenceSampler"

SequenceSampler._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "data_source", type = Dataset, doc = "dataset to sample" },
}(function(self, data_source)
  self._data_source = data_source
end)

function SequenceSampler:iter()
  local n = self._data_source:len()
  local i = 0
  return function()
    i = i + 1
    if i > n then return nil end
    return i
  end
end

function SequenceSampler:len()
  return self._data_source:len()
end

-- RandomSampler (sampler.py: replacement/num_samples; RNG = Insight choice)
local RandomSampler = class(Sampler)
RandomSampler._name = "RandomSampler"

-- seed: 固定种子下与 Python np.random.choice 逐位一致(用户定案)。
-- seed==nil -> 用 Insight 全局种子(继续消耗已播种的流,同 numpy)。
RandomSampler._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "dataset", type = Dataset, doc = "dataset to sample" },
  { name = "replacement", type = argrule.boolean, opt = true,
    default = false, doc = "sample with replacement" },
  { name = "num_samples", type = argrule.number, opt = true,
    default = nil, doc = "samples to draw (default: len(dataset))" },
  { name = "seed", type = argrule.number, opt = true, default = nil,
    doc = "reproducible seed (Insight/numpy-aligned)" },
}(function(self, dataset, replacement, num_samples, seed)
  self._dataset = dataset
  self._replacement = replacement
  local n = dataset:len()
  self._num_samples = num_samples or n
  assert(self._num_samples > 0, "num_samples should be a positive integer")
  if not replacement then
    assert(self._num_samples <= n,
      "num_samples should be <= len(dataset) when replacement=false")
  end
  -- seed 非 nil:构造期播一次,epoch 间继续消耗流(numpy 语义对齐)。
  if seed ~= nil then insight().seed(seed) end
end)

function RandomSampler:iter()
  local idx = choice_indices(self._dataset:len(), self._num_samples,
                             self._replacement)
  local i = 0
  return function()
    i = i + 1
    if i > #idx then return nil end
    return idx[i]
  end
end

function RandomSampler:len()
  return self._num_samples
end

-- WeightedRandomSampler (sampler.py: _weighted_sample + np.random.choice)
local WeightedRandomSampler = class(Sampler)
WeightedRandomSampler._name = "WeightedRandomSampler"

WeightedRandomSampler._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "weights", type = argrule.table,
    doc = "list of non-negative weights (1-based per sample)" },
  { name = "num_samples", type = argrule.number,
    doc = "number of samples to draw" },
  { name = "replacement", type = argrule.boolean, opt = true,
    default = true, doc = "sample with replacement" },
  { name = "seed", type = argrule.number, opt = true, default = nil,
    doc = "reproducible seed" },
}(function(self, weights, num_samples, replacement, seed)
  if weights ~= nil and type(weights) == "table"
     and weights.weights ~= nil then
    weights = weights.weights
  end
  assert(num_samples > 0, "num_samples should be a positive integer")
  local d = #weights
  local nz = 0
  for i = 1, d do
    assert(weights[i] >= 0.0, "weights should be non-negative")
    if weights[i] > 0.0 then nz = nz + 1 end
  end
  assert(nz > 0, "weights should have positive values")
  if not replacement then
    assert(nz >= num_samples,
      "weights positive value count < num_samples when replacement=false")
  end
  self._weights = weights
  self._num_samples = num_samples
  self._replacement = replacement
  -- seed 非 nil:构造期播一次,epoch 间继续消耗流(numpy 语义对齐)。
  if seed ~= nil then insight().seed(seed) end
end)

function WeightedRandomSampler:iter()
  local ins = insight()
  local w = self._weights
  local s = 0
  for i = 1, #w do s = s + w[i] end
  local p = {}
  for i = 1, #w do p[i] = w[i] / s end
  -- insight.choice(d, num, replace, p) == np.random.choice(p=...) bitwise
  local chosen = ins.choice(ins.arange(#w), self._num_samples,
                            self._replacement, ins.from_table(p))
  local idx = {}
  for k = 1, self._num_samples do
    idx[k] = ins_at(chosen, k) + 1
  end
  local i = 0
  return function()
    i = i + 1
    if i > #idx then return nil end
    return idx[i]
  end
end

function WeightedRandomSampler:len()
  return self._num_samples
end

-- ---------------------------------------------------------------------
-- BatchSampler (batch_sampler.py: __iter__ + __len__ formulas)
-- ---------------------------------------------------------------------

local BatchSampler = class(Sampler)
BatchSampler._name = "BatchSampler"

BatchSampler._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "sampler", type = Sampler, doc = "indices sampler" },
  { name = "batch_size", type = argrule.number, opt = true, default = 1,
    doc = "samples per batch" },
  { name = "drop_last", type = argrule.boolean, opt = true, default = false,
    doc = "drop the last incomplete batch" },
}(function(self, sampler, batch_size, drop_last)
  assert(batch_size > 0, "batch_size should be a positive integer")
  self._sampler = sampler
  self._batch_size = batch_size
  self._drop_last = drop_last
end)

function BatchSampler:iter()
  local it = self._sampler:iter()
  local bs = self._batch_size
  return function()
    local batch = {}
    local k = 0
    while k < bs do
      local idx = it()
      if idx == nil then break end
      k = k + 1
      batch[k] = idx
    end
    if k == 0 then return nil end
    if k < bs and self._drop_last then return nil end
    return batch
  end
end

-- upstream __len__: num += int(not drop_last)*(bs-1); num // bs
function BatchSampler:len()
  local num = self._sampler:len()
  if not self._drop_last then num = num + self._batch_size - 1 end
  return math.floor(num / self._batch_size)
end

-- ---------------------------------------------------------------------
-- default_collate (collate.py: default_collate_fn; api/io.md §2.6)
-- batch = bare Lua array of samples; returns stacked structure
-- (pl.List for multi-field samples)
-- ---------------------------------------------------------------------

local function number_to_tensor(numbers, dtype, place)
  local cols = {}
  for i = 1, #numbers do
    cols[i] = paddle_core.pd_op_full({1}, numbers[i], dtype, place)
  end
  -- concat n x [1] -> [n] (stack would give [n,1])
  return paddle_core.pd_op_concat(cols, 0)
end

local function default_collate(batch)
  local n = #batch
  if n == 0 then return batch end
  local sample = batch[1]
  local t = type(sample)
  if is_tensor(sample) then
    local cols = {}
    for i = 1, n do cols[i] = batch[i] end
    -- 直接调 C ABI 不经过 _ops/stack.lua 的 1-based->0-based wrapper
    -- (stack.lua: `if axis > 0 then axis = axis - 1 end`),axis 已是
    -- C++ 0-based:沿新轴 0 拼 batch -> [n, ...](np.stack axis=0)。
    return paddle_core.pd_op_stack(cols, 0)
  elseif t == "number" then
    local all_int = true
    for i = 1, n do
      if batch[i] % 1 ~= 0 then all_int = false break end
    end
    return number_to_tensor(batch, all_int and I64 or F64, cpu_place())
  elseif t == "string" then
    return batch   -- bare table, no Tensor conversion (api/io.md §2.6)
  elseif t == "table" then
    local is_map = false
    for k in pairs(batch[1]) do
      if type(k) == "string" then is_map = true break end
    end
    if is_map then
      local out = {}
      for k in pairs(batch[1]) do
        local col = {}
        for i = 1, n do col[i] = batch[i][k] end
        out[k] = default_collate(col)
      end
      return out
    else
      local nf = #batch[1]
      for i = 1, n do
        assert(#batch[i] == nf,
          "default_collate: fields number not same among samples in a batch")
      end
      local out = List()
      for k = 1, nf do
        local col = {}
        for i = 1, n do col[i] = batch[i][k] end
        out:append(default_collate(col))
      end
      return out
    end
  end
  error(("default_collate: unsupported sample type %s"):format(t), 2)
end

-- ---------------------------------------------------------------------
-- DataLoader (reader.py: __init__ 互斥全部前置到构造;单 worker)
-- ---------------------------------------------------------------------

local DataLoader = class()
DataLoader._name = "DataLoader"

-- 构造参数互斥(api/io.md §2.3,全部在 _init 报错):
--   batch_sampler ↔ batch_size/shuffle/drop_last/sampler
--   sampler        ↔ shuffle
--   IterableDataset + sampler/shuffle/batch_sampler -> 报错
--   num_workers > 0 -> "尚未实现" (api/io.md §7, P13)
DataLoader._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "dataset", type = argrule.Union{ Dataset, argrule.func },
    doc = "Dataset instance (num_workers=0) or pure factory fn (P13)" },
  { name = "batch_size", type = argrule.number, opt = true, default = 1,
    doc = "samples per batch" },
  { name = "shuffle", type = argrule.boolean, opt = true, default = false,
    doc = "shuffle each epoch (RNG = Insight choice)" },
  { name = "drop_last", type = argrule.boolean, opt = true, default = false,
    doc = "drop the last incomplete batch" },
  { name = "num_workers", type = argrule.number, opt = true, default = 0,
    doc = "0 = single worker; >0 = Lanes worker pool (P13)" },
  { name = "config", type = argrule.table, opt = true, default = nil,
    doc = "pure-data config passed to the dataset factory (multi-worker)" },
  { name = "dataset_len", type = argrule.number, opt = true, default = nil,
    doc = "sample count for factory datasets (multi-worker; sampler needs it)" },
  { name = "collate_fn", type = argrule.func, opt = true,
    default = nil, doc = "default collate if nil" },
  { name = "sampler", type = Sampler, opt = true, default = nil,
    doc = "custom indices sampler (mutually exclusive with shuffle)" },
  { name = "batch_sampler", type = BatchSampler, opt = true, default = nil,
    doc = "custom batch sampler (overrides batch_size/shuffle/drop_last)" },
  { name = "prefetch", type = argrule.number, opt = true, default = 2,
    doc = "reserved for P13 (Lanes)" },
  { name = "seed", type = argrule.number, opt = true, default = nil,
    doc = "reproducible shuffle seed (Insight/numpy aligned)" },
}(function(self, dataset, batch_size, shuffle, drop_last, num_workers,
          config, dataset_len, collate_fn, sampler, batch_sampler,
          prefetch, seed)
  self._dataset = dataset
  self._prefetch = prefetch
  self._config = config

  assert(num_workers >= 0, "num_workers should be a non-negative value")
  self._num_workers = num_workers

  -- P13 工厂形态:dataset 是函数(num_workers>0, worker 内构造)。
  -- 主 lane 不执行工厂——采样器只需要 len,用 LenOnlyDS 占位。
  local is_factory = type(dataset) == "function"
  if is_factory then
    assert(num_workers > 0,
      "paddle.io: a dataset factory function requires num_workers > 0 "
      .. "(P13 multi-worker; single worker takes a Dataset instance)")
    assert(dataset_len ~= nil and dataset_len > 0,
      "paddle.io: num_workers>0 with a dataset factory needs "
      .. "dataset_len (sample count for the sampler)")
    dataset = LenOnlyDS(dataset_len)
  else
    assert(num_workers == 0,
      "paddle.io: num_workers>0 requires a dataset factory function "
      .. "(P13; got a Dataset instance instead)")
  end

  local is_iter = dataset:is_a(IterableDataset)
  if is_iter then
    assert(num_workers == 0,
      "paddle.io: IterableDataset does not support num_workers > 0")
    assert(not shuffle, "IterableDataset not support shuffle")
    assert(sampler == nil, "IterableDataset expect unspecified sampler")
    assert(batch_sampler == nil,
      "IterableDataset expect unspecified batch_sampler")
    self._batch_size = batch_size
    self._collate_fn = collate_fn or default_collate
    self._batch_sampler = nil
    return
  end

  if batch_sampler ~= nil then
    assert(batch_size == 1 and not shuffle and not drop_last,
      "batch_size/shuffle/drop_last should not be set when "
        .. "batch_sampler is given")
    assert(sampler == nil,
      "sampler should not be set when batch_sampler is given")
    self._batch_sampler = batch_sampler
  else
    if sampler ~= nil then
      assert(not shuffle,
        "shuffle should be False when sampler is set")
    else
      sampler = shuffle
          and RandomSampler{ dataset = dataset, seed = seed }
          or SequenceSampler(dataset)
    end
    self._batch_sampler = BatchSampler{
      sampler = sampler, batch_size = batch_size, drop_last = drop_last,
    }
  end
  self._collate_fn = collate_fn or default_collate
end)

function DataLoader:len()
  return self._batch_sampler:len()
end

-- :iter() 返回迭代器函数;第一返回值是 batch 序号(1-based,永不为 nil),
-- 第二返回值是 batch(11-io.md §4 坑①)。
function DataLoader:iter()
  if self._num_workers > 0 then
    if type(self._dataset) == "function" then
      -- factory dataset:worker 内构造,主 lane 不检查类型
    elseif self._dataset:is_a(IterableDataset) then
      error("paddle.io: IterableDataset does not support num_workers > 0",
            2)
    end
    return self:_make_multiworker_iter()
  end
  if self._dataset:is_a(IterableDataset) then
    return self:_make_iterable_iter()
  end
  local bs_iter = self._batch_sampler:iter()
  local batch_no = 0
  local collate_fn = self._collate_fn
  local ds = self._dataset
  return function()
    local indices = bs_iter()
    if indices == nil then return nil end
    batch_no = batch_no + 1
    local samples = {}
    for k = 1, #indices do
      local fields = { ds:get(indices[k]) }
      if #fields == 1 then
        samples[k] = fields[1]
      else
        samples[k] = fields
      end
    end
    return batch_no, collate_fn(samples)
  end
end

-- IterableDataset 单 worker 路径:顺序取样本,攒满 batch_size 出一批
-- (不足的最后一批保留,同 map 的 drop_last=false 语义)。
function DataLoader:_make_iterable_iter()
  local ds = self._dataset
  local gen = ds:iter()
  local bs = self._batch_size
  local collate_fn = self._collate_fn
  local batch_no = 0
  return function()
    local samples = {}
    local k = 0
    while k < bs do
      local fields = { gen() }
      if fields[1] == nil then break end
      k = k + 1
      if #fields == 1 then
        samples[k] = fields[1]
      else
        samples[k] = fields
      end
    end
    if k == 0 then return nil end
    batch_no = batch_no + 1
    return batch_no, collate_fn(samples)
  end
end

-- ---------------------------------------------------------------------
-- DataLoader 多 worker 协议(P13 / M2 3.2,13-lanes.md §4.2)
--
-- 数据流(主 lane 采样器保持不变 → 顺序一致):
--   主 lane 把 batch_sampler 产出的每个批次索引表经 Linda "index"
--   分发给 worker;worker 在新 lua_State 里重新 require paddle,构造
--   dataset(工厂函数 + 纯数据 config,见下),对每个索引 ds:get(i),
--   collate 成一个 batch,把 batch 里的每个 Tensor registry_put 换
--   成整数 id,经 Linda "batch" 送回主 lane 有序收集;主 lane
--   registry_take 还原 Tensor。
--
-- 跨 lane 传什么(13-lanes.md §4.2/§4.3):
--   数字/字符串/纯数据表   ✅
--   Tensor              ✅ 句柄表退路(只传整数 id,registry.cc Group10)
--   Lua 函数             ✅ 仅 string.dump 可序列化(工厂函数禁止 upvalue)
--   pl.class 实例        ❌ metatable 复杂对象(文档禁止)
--   Insight Array       ❌ userdata(文档禁止 → transform 留在 get 内)
--
-- worker 节点本身是 string.dump 序列化进 lane 的纯函数,禁止引用本
-- 模块任何局部变量;一切依赖(package 路径、工厂、config、linda)都从
-- 参数进。const/参数名带 _ 前缀,避免 5.1 的 upvalue 捕获歧义。
-- ---------------------------------------------------------------------

-- worker 数量上限与超时(秒),避免毫秒级 busy-loop
-- Worker cold-start may need to construct a heavy dataset (e.g. Cifar10:
-- ~7s pure-Lua pickle parse of the 150MB archive + DLPack) before the
-- first batch can be produced. Must exceed that, else the main lane times
-- out waiting for the first reply. Genuine worker errors still surface via
-- the {batch_no,err} path, so a large value only relaxes the first-reply wait.
local _WORKER_STOP_TIMEOUT = 120.0

-- worker 节点:在 lane 的全新 lua_State 里运行。函数体经 Lanes
-- string.dump 传输到 lane,因此不能引用本模块任何 upvalue —— 所有
-- 外部依赖都从参数进,辅助函数定义在函数体内(运行时创建,不参与
-- dump)。collate_fn 为 nil 时用 worker 内 require 的
-- paddle.io.default_collate(io.lua 是同一份源码,行为与主 lane 一致)。
--
-- 协议:
--   linda:receive 等 "index" (主 lane 发 {batch_no, indices})
--   batch_no = 主 lane 的批序号(顺序一致性的锚点)
--   成功后 linda:send("batch", { batch_no=.., packed=.. })
--   失败后 linda:send("batch", { batch_no=.., err=.. })
--   收到 "stop" 则退出
local function _worker_node(ppath, pcpath, factory, config, collate_fn,
                            linda)
  package.path = ppath
  package.cpath = pcpath
  local core = require "paddle_core"
  core.init()
  -- 构造 dataset;factory 必须是无 upvalue 的纯函数(用户契约,
  -- dataloader.md §4),这样它的字节码可被 Lanes 序列化跨 state 执行。
  local ds = factory(config)
  -- collate:传 nil 时用默认(worker 内重新 require paddle.io,拿默认
  -- collate;不从主 lane 带,避免 dump 时 upvalue 断裂)
  local collate = collate_fn
  if collate == nil then
    collate = require("paddle.io").default_collate
  end
  -- 递归 pack:LuaTensor -> registry id;表 -> 逐键递归;其余原样。
  -- 定义在函数体内:递归 self-reference 是运行期闭包,不经 dump。
  local function pack(v)
    local t = type(v)
    if t == "userdata" then
      local id = core.tensor_registry_put(v)  -- 非 Tensor 在此抛错
      return { __t = id }
    elseif t == "table" then
      local out = {}
      for k, sub in pairs(v) do
        out[k] = pack(sub)
      end
      return out
    end
    return v
  end
  while true do
    local key, msg = linda:receive(nil, "index", "stop")
    if key == nil or key == "stop" then return end
    local batch_no, indices = msg[1], msg[2]
    local ok, batch = pcall(function()
      local samples = {}
      for k = 1, #indices do
        local fields = { ds:get(indices[k]) }
        if #fields == 1 then
          samples[k] = fields[1]
        else
          samples[k] = fields
        end
      end
      return collate(samples)
    end)
    if not ok then
      linda:send("batch", { batch_no = batch_no, err = tostring(batch) })
    else
      local ok2, packed = pcall(pack, batch)
      if not ok2 then
        linda:send("batch",
                   { batch_no = batch_no, err = tostring(packed) })
      else
        linda:send("batch", { batch_no = batch_no, packed = packed })
      end
    end
  end
end

-- 主 lane 收包还原:pack 的逆映射(id -> registry_take)。
-- 只在主 lane 用,不需 dump,可以引用本模块 upvalue。
local function _unpack_batch(packed)
  if type(packed) == "table" then
    if packed.__t ~= nil then
      local t = paddle_core.tensor_registry_take(packed.__t)
      if t == nil then
        error("paddle.io: multi-worker batch id " .. tostring(packed.__t)
              .. " already taken", 3)
      end
      return t
    end
    local out = {}
    for k, v in pairs(packed) do
      out[k] = _unpack_batch(v)
    end
    return out
  end
  return packed
end

-- 多 worker 迭代器:主 lane 侧。每轮 iter() 清理旧池再建新池。
-- 顺序保证:batch_no 由主 lane 从 1 递增分配,收集后按 batch_no
-- 缓存、按序产出 —— 与单 worker 路径产出顺序逐位一致
-- (api/io.md §6 坑②)。
function DataLoader:_make_multiworker_iter()
  local lns = lanes()
  if self._lanes then
    self._lanes.shutdown()
    self._lanes = nil
  end
  local nw = self._num_workers
  local linda = lns.linda()
  local factory = self._dataset
  assert(type(factory) == "function",
    "paddle.io: num_workers>0 requires dataset to be a factory function "
    .. "taking a pure-data config (dataloader.md §4); got "
    .. type(factory))
  local config = self._config or {}
  local ppath, pcpath = package.path, package.cpath

  -- 默认 collate 不跨 lane 传(_worker_node 契约:传 nil -> worker 内
  -- require paddle.io 拿默认 collate;带 paddle_core C 函数 upvalue 的
  -- 函数无法按名跨 state 传 —— 目标 lane 尚未 require paddle_core)。
  local collate_arg = self._collate_fn
  if collate_arg == default_collate then collate_arg = nil end
  local handles = {}
  for w = 1, nw do
    handles[w] = lns.gen("*", _worker_node)(ppath, pcpath, factory,
                                            config, collate_arg, linda)
  end

  local bs_iter = self._batch_sampler:iter()
  local expected, pending = 1, {}
  local sent = 0          -- 已发出的批数(也是下一个 batch_no)
  local in_flight = 0     -- 已发出未回收的批数
  local exhausted = false

  local function shutdown()
    for _ = 1, nw do linda:send("stop", true) end
    for w = 1, nw do
      pcall(function() handles[w]:join(_WORKER_STOP_TIMEOUT) end)
    end
    self._lanes = nil
  end

  -- 从 pending 按 expected 序取一包,返回 (batch_no, batch) 或 nil
  local function drain()
    local p = pending[expected]
    if p == nil then return nil end
    pending[expected] = nil
    expected = expected + 1
    if p.err then
      error(("paddle.io: worker error at batch %s: %s"):format(
        tostring(p.batch_no), p.err), 2)
    end
    return p.batch_no, _unpack_batch(p.packed)
  end

  local next_batch
  next_batch = function()
    while true do
      -- 补发直到窗口满或耗尽(窗口 = nw*2,防主 lane 空等)
      if not exhausted then
        while in_flight < nw * 2 do
          local indices = bs_iter()
          if indices == nil then
            exhausted = true
            break
          end
          sent = sent + 1
          linda:send("index", { sent, indices })
          in_flight = in_flight + 1
        end
      end
      local bno, batch = drain()
      if bno then return bno, batch end
      if exhausted and in_flight == 0 then
        shutdown()
        self._lanes = nil
        return nil
      end
      -- 收一个 worker 结果
      local key, res = linda:receive(_WORKER_STOP_TIMEOUT, "batch")
      if key == nil or res == nil then
        error("paddle.io: worker lane timed out (no batch reply)", 2)
      end
      in_flight = in_flight - 1
      pending[res.batch_no] = res   -- {batch_no=, packed=|err=}
    end
  end

  self._lanes = { shutdown = shutdown }
  return next_batch
end

-- ---------------------------------------------------------------------
-- module exports (api/io.md §1, P11 范围)
-- ---------------------------------------------------------------------

return {
  Dataset = Dataset,
  IterableDataset = IterableDataset,
  TensorDataset = TensorDataset,
  ConcatDataset = ConcatDataset,
  Subset = Subset,
  random_split = random_split,
  Sampler = Sampler,
  SequenceSampler = SequenceSampler,
  RandomSampler = RandomSampler,
  WeightedRandomSampler = WeightedRandomSampler,
  BatchSampler = BatchSampler,
  DataLoader = DataLoader,
  default_collate = default_collate,
}