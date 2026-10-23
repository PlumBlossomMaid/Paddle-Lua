# paddle.distributed

| | |
|---|---|
| 阶段 | **P17**(⛔ **无多卡环境,写得了测不了**——验收挂起,见 §7) |
| 依赖模块 | `paddle`、`paddle.Tensor`、`paddle.io`(shard_dataloader) |
| 稳定性 | **实验**(全程) |
| 实现文档 | `plan/modules/17-distributed.md`(待写) |
| 上游来源 | `python/paddle/distributed/__init__.py`(导出,见 §1);`auto_parallel/api.py`(DistAttr/DistModel/Strategy);`io.py`(shard_dataloader) |

> 📌 **⛔ ⛔ ⛔ 本模块的验收前提:至少一台多卡机器。**
> 按 `overview.md` §2.1.4「没有多卡测试环境前一行都不写」——
> **本文只是契约** (api/README 的「写得了测不了」),实现与验收全部挂起。
> 写本文的目的:把「要做什么」钉下来,避免 P17 时重新论证。

---

## 1. 导出清单

`python/paddle/distributed/__init__.py` 的真实导出(摘核心):

| 导出 | 来源 | 说明 |
|---|---|---|
| `paddle.distributed.init_parallel_env` | `__init__.py` | 并行环境初始化 |
| `paddle.distributed.get_rank` / `get_world_size` | `__init__.py` | 进程 rank / 总进程数 |
| `paddle.distributed.get_backend` | `__init__.py` | 后端(NCCL/GLOO 等) |
| `paddle.distributed.all_reduce` / `all_gather` / `broadcast` / `scatter` / `gather` / `reduce` / `send` / `recv` | `__init__.py` | 集合通信算子 |
| `paddle.distributed.ParallelMode` | `:22` | 并行模式枚举 |
| `paddle.distributed.create_nccl_config` | `:23` | NCCL 配置 |
| `paddle.distributed.DistAttr` / `DistModel` / `Strategy` | `auto_parallel/api.py`(`:34-39`) | 自动并行 API |
| `paddle.distributed.ShardingStage1/2/3` | `:36-38` | 参数切片 |
| `paddle.distributed.reshard` | `:43` | 张量重分片 |
| `paddle.distributed.shard_dataloader` | `:44` | 数据并行 loader |
| `paddle.distributed.spawn` | `__init__.py` | 多进程启动器 |
| `paddle.distributed.io` / `rpc` / `cloud_utils` | `:28-30` | 子模块 |

---

## 2. 详细签名(契约,未验证)

### 2.1 环境与 rank

```lua
paddle.distributed.init_parallel_env{}            -- 初始化
local rank = paddle.distributed.get_rank()        -- 当前进程号(0-based,Python 语义)
local size = paddle.distributed.get_world_size()  -- 总进程数
```

⚠️ **rank 语义待拍板**:Python 侧 `get_rank` 返回 0-based(进程号)。
Lua 侧按 C4 或按上游 —— **P17 与用户确认后定**,本文先按上游 0-based 记。

### 2.2 集合通信

```lua
paddle.distributed.all_reduce{ tensor, group=nil, use_calc_stream=true }
paddle.distributed.all_gather{ output_list, input, group=nil }
paddle.distributed.broadcast{ tensor, src, group=nil }
```

绑定形态 = 顶层算子风格(一张参数表 + `_ops` 组合);集合通信走
Paddle 的 NCCL/GLOO 后端(T-M0 构建含 `PADDLE_WITH_NCCL` 与否待核)。

### 2.3 数据并行

```lua
local loader = paddle.distributed.shard_dataloader{
  dataloader, world_size = ..., rank = ...
}
```

数据并行分片 loader。⚠️ 与 `paddle.io.DataLoader` 的关系:
shard_key 按 rank 切分,P17 时定(D7:线程模型,不搞多进程——这条约束
对 distributed 同样成立,见 §6 ③)。

---

## 3. index 语义参数

| 参数 | 语义 | 基准 |
|---|---|---|
| `get_rank()` 返回值 | 进程号 | ⚠️ **待拍板**(上游 0-based,§2.1;与 C4 冲突点) |
| `broadcast.src` | 源进程号 | 同上 |
| `all_gather.output_list` 下标 | 张量列表 | 非索引(pl.List) |

**其余参数非索引**(group/tensor 等)。⚠️ 本模块的 rank/进程号语义是
**唯一可能突破 C4 的候选**——因为 rank 是「外部系统的身份标识」(像
`worker_info.id`),不是库内部索引。**P17 拍板**,本文不预判。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.distributed.init_parallel_env()` | `init_parallel_env{}` | 一张参数表 |
| 2 | 集合通信同名同参 | 同名同参 | 照搬 |
| 3 | `rank` 0-based | **待拍板** | 外部身份 vs C4(§3) |
| 4 | `spawn` 多进程 | **不移植(Lua 用线程)** | D7:lua_State 天然隔离,无 GIL 无 spawn 坑 |
| 5 | `shard_dataloader` | 同名 | 分片逻辑照搬 |

---

## 5. 用法示例(契约草稿,未验证)

```lua
local paddle = require "paddle"
local dist = paddle.distributed

dist.init_parallel_env{}
local rank = dist.get_rank()

-- 数据并行训练(多线程,见 §6 ③)
local loader = dist.shard_dataloader{ dataloader = dl, world_size = dist.get_world_size(), rank = rank }
for x, y in loader:iter() do
  local loss = criterion(net(x), y)
  loss:backward()
  local grad = net:parameters()   -- 参数梯度
  dist.all_reduce{ tensor = ... } -- 或走优化器的分布式变体
  opt:step(); opt:clear_grad()
end
```

⚠️ **实现细节(P17)**:参数梯度同步通常由优化器封装(如
`paddle.distributed.fleet` 的做法),用户不手写 all_reduce。上面只是
示意集合通信的形态,不是推荐写法。

---

## 6. 本模块特有的坑

**① 没有环境,一切都是纸上谈兵。** 本文标注「实验 + 未验证」。
P17 开工前必须先有一台双卡(哪怕 2×CPU 卡)机器做验收基准;
没有就继续挂起,不降级用单卡模拟(单卡 all_reduce 是恒等运算,测不出错)。

**② rank 语义是 C4 的唯一冲突候选。** `get_rank` 0-based(上游)vs
C4 全 1-based。本文倾向:**rank/进程号按上游 0-based**(外部标识,同
`worker_info.id` 的处理要拍板但不一定是 1-based)——**P17 与用户确认,
不预判**。

**③ 线程模型(D7)对 distributed 的推论。** DataLoader 不搞多进程
(D7:lua_State 天然隔离)。distributed 的「多进程」同理:
`paddle.distributed.spawn` 在 Lua 侧**没有等价物**——多卡训练要么
多进程(用户自己起,每个进程一个 lua),要么走 Lane。**P17 定**,本文
记:不移植 `spawn`,提供 `init_parallel_env` 供多进程用户自己拼。

**④ `use_calc_stream` 等底层参数照搬是陷阱。** `all_reduce{
use_calc_stream=true }` 是性能参数,语义与 back cast 流相关。Lua 侧
默认值照抄,但文档不强调(用户基本不需要碰)。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| **全部** | **⛔ P17 且须有多卡环境**;本文仅是契约 |
| 验收基准(2×卡冒烟) | P17 前置,未排期 |
| rank 语义拍板 | P17 与用户确认(C4 冲突点) |
| `spawn` 等价物 | P17 定(不移植 spawn,§6 ③) |
| `auto_parallel`(DistAttr/Strategy/Sharding) | P17 后期(依赖手动集合通信先稳) |