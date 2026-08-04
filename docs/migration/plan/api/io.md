# paddle.io

| | |
|---|---|
| 阶段 | P11(单 worker)/ P13(多 worker) |
| 依赖模块 | `paddle`(Tensor)、`pl.class`、`pl.List` |
| 可选依赖 | Lanes(`num_workers > 0` 时才用;**未装则 `num_workers > 0` 报错**,不静默降级) |
| 稳定性 | **稳定** |
| 实现文档 | `plan/modules/11-io.md`(单 worker)、`plan/modules/13-lanes.md`(多 worker) |

> 📌 本文件是 `plan/api/` 的**样板**。新写模块 api 文档时照着这份的详细程度。

---

## 1. 导出清单

| 导出 | 类型 | 阶段 | 稳定性 |
|---|---|---|---|
| `paddle.io.Dataset` | 基类 | P11 | 稳定 |
| `paddle.io.IterableDataset` | 基类 | P11 | 稳定 |
| `paddle.io.TensorDataset` | 类 | P11 | 稳定 |
| `paddle.io.ConcatDataset` | 类 | P11 | 稳定 |
| `paddle.io.Subset` | 类 | P11 | 稳定 |
| `paddle.io.random_split` | 函数 | P11 | 稳定 |
| `paddle.io.Sampler` | 基类 | P11 | 稳定 |
| `paddle.io.SequenceSampler` | 类 | P11 | 稳定 |
| `paddle.io.RandomSampler` | 类 | P11 | 稳定 |
| `paddle.io.WeightedRandomSampler` | 类 | P11 | 稳定 |
| `paddle.io.BatchSampler` | 类 | P11 | 稳定 |
| `paddle.io.DistributedBatchSampler` | 类 | P17 | 实验 |
| `paddle.io.DataLoader` | 类 | P11/P13 | 稳定 |
| `paddle.io.default_collate` | 函数 | P11 | 稳定 |
| `paddle.io.get_worker_info` | 函数 | P13 | 稳定 |

---

## 2. 详细签名

### 2.1 `Dataset`(映射式)

```lua
local class = require "pl.class"
local MyDS = class(paddle.io.Dataset)
MyDS._name = "MyDS"

function MyDS:_init(root)
  self:super()
  self.items = paddle.utils.fs.list(root)   -- pl.List
end

function MyDS:len()  return #self.items end          -- 必须实现
function MyDS:get(i) return load_one(self.items[i]) end  -- 必须实现,i 是 1-based
```

| 方法 | 必须? | 说明 |
|---|:-:|---|
| `:len()` | ✅ | 返回样本数。**不是 `#ds`** —— 见 §4 |
| `:get(i)` | ✅ | `i ∈ [1, len]`。返回一个样本(Tensor / table / 多返回值) |

**`get` 的返回值形状决定 collate 的行为**,见 §2.6。

### 2.2 `IterableDataset`(流式)

```lua
function MyStream:iter()      -- 返回一个迭代器函数
  return function() ... end   -- 每次调用返回一个样本,耗尽返回 nil
end
```

**没有 `len` / `get`。** 与 Python 的 `__iter__` 对应。
`DataLoader` 检测到 `IterableDataset` 时**忽略 `sampler` / `shuffle`**,
和 Python 行为一致 —— 但**我们要显式报错而不是静默忽略**,见 §6。

### 2.3 `DataLoader`

```lua
local loader = paddle.io.DataLoader{
  dataset      = ds,          -- 必填
  batch_size   = 64,          -- 默认 1;与 batch_sampler 互斥
  shuffle      = false,
  drop_last    = false,
  num_workers  = 0,           -- > 0 需要 Lanes(P13)
  collate_fn   = nil,         -- 默认 default_collate
  sampler      = nil,         -- 与 shuffle 互斥
  batch_sampler= nil,         -- 与 batch_size/shuffle/drop_last/sampler 互斥
  prefetch     = 2,           -- 每个 worker 预取的 batch 数(P13)
  seed         = nil,         -- 不给则用全局种子
}

for batch in loader:iter() do ... end     -- ★ 推荐写法
print(loader:len())                        -- batch 数
```

**互斥关系必须在构造时报错,不能后置到迭代时。**
Python 侧有几组参数组合是运行到一半才炸的,我们在 `_init` 里一次性查完。

### 2.4 采样器

```lua
paddle.io.SequenceSampler(ds)                      -- 1,2,...,n
paddle.io.RandomSampler{ dataset = ds,
                         replacement = false,
                         num_samples = nil,
                         seed = nil }
paddle.io.WeightedRandomSampler{ weights = {...},  -- 1-based 对应样本 1..n
                                 num_samples = 100,
                                 replacement = true }
paddle.io.BatchSampler{ sampler = s, batch_size = 64, drop_last = false }
```

所有采样器都实现 `:iter()`(产出**索引**)与 `:len()`。

### 2.5 `random_split`

```lua
local tr, va = paddle.io.random_split(ds, {0.8, 0.2}, seed)  -- 比例
local a, b   = paddle.io.random_split(ds, {50000, 10000}, seed) -- 绝对数
```

返回 **`pl.List` of `Subset`**。比例形式的取整余数**全部给最后一份**,
并在文档里写死这条规则 —— 不写死会导致不同版本切出来的划分不同,
**破坏"固定种子可复现"**。

### 2.6 `default_collate`

| `get(i)` 返回 | collate 后 |
|---|---|
| 单个 Tensor | 形状 `[B, ...]` 的 Tensor |
| `x, y`(多返回值) | 两个 Tensor,各自 stack |
| `{x = t1, y = t2}` | `{x = [B,...], y = [B,...]}`,**保持键** |
| `pl.List`(定长) | 逐位置 stack |
| number / string | number -> Tensor;**string -> 裸表,不转 Tensor** |

**string 不转 Tensor 是有意的** —— 变长文本没有无损的张量表示,
静默 pad 会制造难查的 bug。需要的人自己写 `collate_fn`。

---

## 3. index 语义参数

**本模块是全项目 index 语义最密集的地方**(`overview.md` §6.1.1、`ci.md` §6 ④)。

| 位置 | 语义 | 基准 |
|---|---|---|
| `Dataset:get(i)` 的 `i` | 样本序号 | **1-based** |
| 采样器产出的索引 | 样本序号 | **1-based** |
| `Subset` 的 indices | 样本序号 | **1-based** |
| `WeightedRandomSampler.weights` 的下标 | 对应样本序号 | **1-based** |
| `get_worker_info().id` | worker 编号 | **1-based**(⚠️ Python 是 0-based,见 §4) |
| `DataLoader` 产出的 batch 内部下标 | 张量维度 | 张量语义,同样 1-based |

**整条链上没有一处需要换算** —— `pl.List`、Lua 表、采样器、Dataset 全是 1-based。
这正是 D-R6 "边界在库里不在用户代码里" 的兑现点。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `len(ds)` / `ds[i]` | **`ds:len()` / `ds:get(i)`** | `__index` 会与实例字段冲突(11-io.md §3.1);`#` 对我们的实例不可靠(D27) |
| 2 | 索引从 0 | **从 1** | D-R6 |
| 3 | `for batch in loader:` | **`for batch in loader:iter() do`** | Lua 的泛型 for 需要显式迭代器 |
| 4 | `worker_info.id ∈ [0, n)` | **`∈ [1, n]`** | D-R6。⚠️ **迁移时最容易漏的一条** —— 用它做数据分片的代码会静默错位 |
| 5 | `DataLoader(ds, batch_size=64)` | **`DataLoader{ dataset=ds, batch_size=64 }`** | `_wrap` 三模式(D-R8)。位置参数形式也支持,但示例一律用表 |
| 6 | `num_workers` 默认按 CPU 数 | **默认 0** | 显式优于隐式;多 worker 需要 Lanes |
| 7 | 返回 list | 返回 **`pl.List`** | D-R21 |
| 8 | `IterableDataset` + sampler 静默忽略 | **报错** | 静默忽略会让人以为 shuffle 生效了 |

第 4 行单独强调:**`worker_info.id` 的差异是会静默产生错误结果的那一类。**
Python 侧常见写法 `if idx % num_workers == worker_id` 直译过来会漏掉一个分片、重复另一个,
而**训练照常收敛,只是数据用错了**。文档必须给出对照写法。

---

## 5. 用法示例

```lua
local paddle = require "paddle"
local class  = require "pl.class"

local MNIST = class(paddle.io.Dataset)
MNIST._name = "MNIST"
function MNIST:_init(split)
  self:super()
  self.x, self.y = load_mnist(split)      -- [N,1,28,28], [N]
end
function MNIST:len()  return self.y:shape()[1] end
function MNIST:get(i) return self.x[i], self.y[i] end   -- i 是 1-based

local loader = paddle.io.DataLoader{
  dataset = MNIST("train"), batch_size = 64,
  shuffle = true, num_workers = 0, seed = 1234,
}

for x, y in loader:iter() do
  local loss = criterion(net(x), y)
  loss:backward()
  opt:step(); opt:clear_grad()
end
```

---

## 6. 本模块特有的坑

**① `#ds` 不等于 `ds:len()`。** 用户会本能地写 `#ds`,得到 0。
`Dataset` 应当**在 5.2+ 上设 `__len`**,让 `#ds` 能用;
但 5.1/LuaJIT 上 `__len` 对表无效(C3),所以**文档和示例一律用 `:len()`**。
这与 `LayerList` 是同一个坑的两次出现(D27)。

**② 迭代顺序的确定性是可验收的承诺。** 同一 `seed` + 同一 `num_workers`,
产出的 batch 序列必须逐位一致。多 worker 下这要求**每个 worker 的 RNG 由主种子派生**,
而不是各自随机 —— 否则 `num_workers` 一改,训练结果就变,而且看起来"只是随机性"。

**③ `collate_fn` 跑在 worker 线程里(P13)。** 它捕获的 upvalue 会被 Lanes 处理,
**捕获了 Tensor 以外的 userdata 就会炸**。文档必须写明 `collate_fn` 应当是纯函数。

**④ 单 worker 与多 worker 必须是同一条代码路径。** `num_workers = 0` 不应该走
一条"简化实现",否则两条路径的行为会慢慢分叉,而用户是在 0 上开发、在 4 上训练的。

**⑤ 异常要能跨 worker 传回来。** worker 里 `get(i)` 抛错时,主线程必须收到
**原始错误信息 + 是哪个样本索引**,而不是一句 "worker died"。
这条不做的话,数据集里一张坏图能让人查一整天。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `num_workers > 0` | **P13**。P11 阶段传 > 0 直接报错,不静默降级为 0 |
| `DistributedBatchSampler` | P17,⛔ 无多卡环境无法验收 |
| `pin_memory` | **延后**,需要 P14 的显存机制先落地 |
| `timeout` / `worker_init_fn` | **延后**,P13 时按需要再定 |
| 持久 worker(`persistent_workers`) | **延后** |
