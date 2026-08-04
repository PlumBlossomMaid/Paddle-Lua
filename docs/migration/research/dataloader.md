# DataLoader:为什么 Lua 侧不需要多进程

> 判断:Python 那边 multi-worker 是多进程,Lua 里似乎没必要。
> **结论:这个判断是对的,而且理由比"没必要"更强 —— 线程模型在这里是严格更优,不是勉强够用。**

---

## 1. 先说清楚 Python 为什么被迫用多进程

**不是因为多进程好,是 GIL 逼的。** PyTorch/Paddle 的 DataLoader 多进程是一个纯粹的 workaround:
Python 线程跑不了并行的 CPU 密集预处理,只能开进程。

代价极其昂贵,而且**大部分复杂度都花在偿还这个代价上**:

- worker 与主进程不共地址空间 → tensor 必须走 shared memory / fd 传递
- 要序列化 dataset 对象到 worker
- Windows 没有 fork,只能 spawn + 重新 import + pickle,坑极多
  (**正是当前 Paddle 分支 `windows_dataloader_multiprocess` 在处理的事**)
- 僵尸进程、文件描述符泄漏、信号处理、shm 清理……

**Lua 没有 GIL,这个 workaround 的前提直接消失了。**

## 2. 但要说准确:Lua 也没有"共享内存多线程"

Lua 的并发模型是 **one `lua_State` per OS thread,状态完全隔离**
(`lua_lanes` / `effil` / LuaJIT+pthread 都是这个模型)。
两个 lua_State 之间不能共享 closure、upvalue、table。

所以准确的表述不是"Lua 不需要多进程",而是:

> **Lua 的「N 个 lua_State + N 个 OS 线程」同时具备
> 多进程的隔离性 和 多线程的零拷贝 —— 它比 Python 的两个选项都好。**

| | Python 多进程 | Python 多线程 | **Lua 多 lua_State + 线程** |
|---|---|---|---|
| 真并行 | ✅ | ❌ GIL | ✅ 本来就没 GIL |
| 解释器状态隔离 | ✅ | ❌ 共享,要加锁 | ✅ lua_State 天然隔离 |
| 启动成本 | fork/spawn,Windows 上极贵 | 便宜 | **CreateThread,微秒级** |
| **数据回传** | **序列化 + shm/fd** | 零拷贝 | **零拷贝(同地址空间)** |
| Windows | spawn+pickle,坑多 | ok | **无差别** |
| 崩溃隔离 | ✅ | ❌ | ❌ |

**最大的收益是"数据回传零拷贝"。**
Python DataLoader 复杂度的主要来源就是把 tensor 传回主进程(也是 Windows 上问题最多的地方)。
Lua 线程模型下,worker 直接在**同一个 Paddle allocator** 里分配输出 tensor,
主线程拿指针即可 —— **整个 shared memory 层可以删掉。**

---

## 3. 三个必须验证的前提(已查两个)

### 3.1 ✅ Paddle eager 状态本来就是 thread_local —— 好消息,但也是硬要求

```cpp
// paddle/fluid/eager/api/utils/global_utils.h:162
static thread_local std::shared_ptr<paddle::imperative::Tracer> tracer_;

// paddle/fluid/imperative/tracer.cc
:49  static thread_local std::shared_ptr<Tracer> g_current_tracer(nullptr);
:51  static thread_local std::shared_ptr<AmpAttrs> g_current_amp_attrs = ...;
:54  static thread_local bool g_has_grad = true;
:45  thread_local std::string Tracer::python_stack_;
:47  thread_local bool Tracer::use_layout_autotune_;
```

**好消息**:tracer / grad mode / AMP 全是 `thread_local` ——
说明 Paddle 的 eager 栈本来就是按线程隔离设计的,多线程使用被考虑过。

**但这是一个硬要求,不是免费的**:

> `g_current_tracer` 初值是 **`nullptr`**。
> **每个 worker 线程必须自己初始化 tracer,否则第一个 op 就会崩。**

顺带一个红利:worker 应设 `g_has_grad = false`(预处理不需要 autograd),
而它正好也是 `thread_local` —— **天然不会污染主线程的 grad 状态**,不需要加锁或保存/恢复。

### 3.2 ✅ 分配器线程安全 —— 已确认

```cpp
// retry_allocator.h
bool IsAllocThreadSafe() const override { return true; }
// 构造时强制校验底层分配器
PADDLE_ENFORCE_EQ(underlying_allocator_->IsAllocThreadSafe(), true, ...);
```

Paddle **强制要求**分配器链路线程安全。worker 线程直接分配 tensor 是安全的。

### 3.3 ⚠️ 待验证:phi kernel / DeviceContext 的多线程共享

`paddle/fluid/jit/layer.h` 里有 `std::shared_ptr<Layer> Clone(void* stream)`,
暗示 Paddle 官方对多线程推理的建议是**每线程一份 clone**。
需要在 M0 确认:多个线程同时调用 CPU kernel(共享 `CPUContext`)是否安全,
还是需要 per-thread context。

---

## 4. 唯一比 Python 麻烦的地方:worker 代码怎么过去

**这是真正的 API 设计约束,必须在设计阶段定死,后面改不了。**

lua_State 之间不能共享 closure/upvalue,所以不能照抄 Python 的写法:

```python
# Python:dataset 对象连闭包一起被 pickle 到 worker
ds = MyDataset(root="/data", transform=my_transform)
loader = DataLoader(ds, num_workers=4)
```

Lua 必须改成"**worker 自己构造 dataset**":

```lua
local loader = paddle.io.DataLoader{
  num_workers = 4,
  -- 这个函数在每个 worker 的独立 lua_State 里各执行一次
  -- 约束:不能有 upvalue(string.dump 不携带 upvalue)
  dataset = function(cfg)
      local MyDataset = require("mydataset")
      return MyDataset.new(cfg.root)
  end,
  config = { root = "/data" },   -- 纯数据 table,可跨 state 深拷贝
}
```

必须写进文档第一页的约束:

| 约束 | 说明 |
|---|---|
| `dataset` 工厂函数不能有 upvalue | `string.dump` 不携带 upvalue |
| `config` 只能是纯数据 | 无 function / userdata / 循环引用(或我们自己实现处理) |
| worker 需要的模块必须可 `require` | `package.path` / `package.cpath` 要传给 worker state |

> 好处:这个约束比 Python 的 pickle 限制**更清晰、更可预测**。
> Python 的"哪些对象能被 pickle"是出了名的玄学。

---

## 5. 什么时候仍然需要多进程(诚实的保留)

两个真实场景:

1. **崩溃隔离** —— 图像/音频解码库处理损坏文件时段错误的概率不低。
   多进程下 worker 挂了能重启,线程模型下整个训练进程死。
2. **第三方库非线程安全** —— 某些老 codec、某些 BLAS 配置。

但这两个是"可选加固",不是"默认必需"。建议:

```lua
paddle.io.DataLoader{ worker_mode = "thread" }   -- 默认
paddle.io.DataLoader{ worker_mode = "process" }  -- 后期可选,甚至可以不做
```

**这跟 Python 正好相反** —— Python 被迫默认多进程,线程模式几乎没用;
我们默认线程,多进程降级为可选加固。

---

## 6. 实现路径(⚠️ 本节已被第 9 节推翻,保留作为 fallback 方案)

那类库的核心工作是**在 state 之间深拷贝 Lua 值**,而我们恰恰不需要 ——
我们要跨线程传的是 `paddle::Tensor`,不是 Lua 值。

```
C 侧线程池 (N 个 OS 线程)
  每个 worker 线程启动时:
    1. luaL_newstate()                     独立 lua_State
    2. 初始化 Paddle tracer                 <- 3.1 的硬要求,别忘
    3. g_has_grad = false
    4. 设置 package.path,执行用户的 dataset 工厂 bytecode
  主循环:
    index queue  --取-->  调 Lua 侧 dataset[i] / collate
                 --出-->  paddle::Tensor(shared_ptr)  推入 result queue

主线程:从 result queue 取 Tensor,直接用(零拷贝)
```

**关键简化:Lua 值不跨 state,只有 C++ Tensor 跨线程。**
队列在 C 侧,传的是 `shared_ptr<TensorBase>`,完全绕开了跨 state 传 Lua 值的所有难题。

### 工作量对比

| | 行数 |
|---|---|
| Python 版 `python/paddle/io/` | **4,219 行**(还不含 C++ 侧 shared memory / IPC) |
| Lua 版预估:C 侧线程池 + 队列 | 600-900 行 |
| Lua 版预估:Lua 侧 API | 300-500 行 |

**约为 Python 版的 1/4,而且没有 Windows 特有的那一堆坑。**

---

## 7. 与 GC 方案的一个交互(容易漏)

`RegisterOOMCallback` 注册的是**全局**回调:

```cpp
// retry_allocator.cc:24
static std::function<size_t(Place, size_t)> g_oom_callback;   // 不是 thread_local
```

但 `lua_gc(L, ...)` 需要知道**是哪个 `lua_State`**。
如果 OOM 发生在 worker 线程,不能去 GC 主线程的 state。

对策:维护 `thread_local lua_State* t_current_L`,回调里取当前线程的 state:

```cpp
RegisterOOMCallback([](phi::Place p, size_t size) -> size_t {
    if (t_current_L == nullptr) return 0;      // 非 Lua 线程,放弃
    size_t before = t_live_bytes;
    lua_gc(t_current_L, LUA_GCCOLLECT, 0);
    return before - t_live_bytes;
});
```

> 又一次印证抄 Torch7 是对的:`THGeneral.c:136-140` 的
> `heapSize` / `heapSoftmax` 本来就是 `__thread` 的 —— 它当年就是按多线程设计的。
> 我们的堆追踪也应**按线程统计**,而不是全局。

---

## 8. 净结论

| 项 | 结论 |
|---|---|
| Lua 侧是否需要多进程 | **默认不需要,线程模型严格更优** |
| 理由 | Python 用多进程是 GIL 的 workaround;Lua 没有 GIL,且线程模型额外白拿零拷贝 |
| 前提 1 | worker 线程**必须自己初始化 tracer**(`g_current_tracer` 初值为 nullptr) |
| 前提 2 | 分配器线程安全 ✅ 已确认 |
| 前提 3 | phi kernel 多线程共享 DeviceContext ⚠️ M0 待验证 |
| API 约束 | dataset 必须是**无 upvalue 的工厂函数 + 纯数据 config**,要写在文档第一页 |
| 多进程 | 降级为可选加固(崩溃隔离 / 非线程安全的第三方库) |
| 工作量 | 约为 Python 版 4,219 行的 **1/4** |

---

## 9. 修正:采用 Lua Lanes(推翻第 6 节)

> https://github.com/LuaLanes/lanes

### 9.1 先澄清名字

**没有叫 luna / lune 的 Lua 线程库。**

- **Lune** = `lune-org/lune`,926 stars,**Rust 写的 Luau 独立运行时**,和 Lua 5.x 无关
- **Luna** = 无相关项目

你要找的是 **Lua Lanes**。

### 9.2 ❌ 我第 6 节的建议是错的

第 6 节我写"不要用 lua_lanes / effil,因为它们的核心工作是在 state 之间深拷贝 Lua 值,
而我们不需要"。

**这个理由只对 Linda(值传递通道)成立,完全忽略了 Lanes 的另一半功能。**

`src/deep.hpp` 开头就写着:

```cpp
/*
 * public 'deep' API to be used by external modules if they want to implement
 * Lanes-aware userdata
 * said modules can either link against lanes, or embed
 * compat.cpp/h deep.cpp/h tools.cpp/h universe.cpp/h
 */

class DeepPrelude
{
    UniqueKey const magic{ kDeepVersion };
    DeepFactory& factory;
    protected:
    // data is destroyed when refcount is 0
    std::atomic<int> refcount{ 0 };      // <- 原子引用计数
};

class DeepFactory
{
    virtual DeepPrelude* newDeepObjectInternal(lua_State* L_) const = 0;
    virtual void deleteDeepObjectInternal(lua_State* L_, DeepPrelude* o_) const = 0;
    virtual void createMetatable(lua_State* L_) const = 0;
};
```

**"deep userdata" 是 Lanes 专门为外部 C 模块提供的公共 API:
让一个 C/C++ 对象以「按引用共享 + 原子引用计数」的方式跨多个 `lua_State` 使用。**

**这精确地就是 `paddle::Tensor` 的需求。** 仓库根目录甚至有整个
`deep_userdata_example/`(`deep_userdata_example.cpp` + `deeptest.lua`)作为官方示例。

我第 6 节手画的那张架构图,**Lanes 基本已经全部实现了。**

### 9.3 Lanes 尽调数据(已核实)

| 项 | 数据 |
|---|---|
| Stars | 538 |
| 许可 | **MIT**(`COPYRIGHT`: "same MIT license as the Lua 5.1 source code") |
| 最近提交 | **2026-03-12**(活跃) |
| 最新 release | **v4.0.0 / 2026-02-27**;上一个稳定 v3.17.2 / 2025-10-23 |
| Lua 版本 | **5.1 – 5.5**(`compat.hpp` 有 `==501` / `<504` / `<505` 显式分支) |
| 实现语言 | **C++**(`.hpp`、`std::atomic`、`[[nodiscard]]`、`if constexpr`) |
| 平台 | Windows / Linux / macOS,有 `.vcxproj` 和 `.sln`,**Windows 是一等公民** |

**一个有意思的不对称:Lanes 支持 Lua 5.5,而 sol2 不支持。**
所以将来若要加 5.5,瓶颈在 sol2,不在线程层 ——
这反过来再次印证了「砍掉 5.5」和「中间层保留纯 C ABI 退路」两个决定。

### 9.4 Lanes 直接替我们解决了什么

| 需求 | Lanes 提供 | 省下 |
|---|---|---|
| lua_State 生命周期 / 错误传播 | `lane.cpp` | 自建线程池的大头 |
| 跨平台线程 | Windows/POSIX 都有 | Windows 适配 |
| **Tensor 跨线程按引用共享** | **deep userdata + 原子 refcount** | **最关键的一块** |
| 队列 / 通信原语 | **Linda**(`linda.cpp`) | index queue / result queue 自己写 |
| 取消 / 超时 | `cancel.cpp` | worker 优雅退出 |
| 传 config table 给 worker | Linda 的值拷贝(第 4 节的约束正好用它) | 序列化 |
| 自定义分配器 | `allocator.cpp` | 可能可与堆追踪协调 |

**工作量修正:第 6 节估的 C 侧 600-900 行 → 实现一个 `DeepFactory` + 组装,预计 200-300 行。**

> ⚠️ **已被 R18 推翻(2026-08-03)。** 最终**不用** deep userdata,改为普通 sol2 usertype +
> `__lanesclone`。理由:deep userdata 的价值是给共享 C 对象做原子引用计数,
> 而 `paddle::Tensor` 本来就持有 `std::shared_ptr<phi::TensorBase>`(`tensor.h:142`),
> 计数已经是原子的。工作量随之下调到 100-150 行。
> 现结论见 `plan/modules/13-lanes.md` §4.1;本节保留作为论证痕迹。

### 9.5 但有五个必须验证的点(诚实)

**(a) ~~会不会强制所有用户依赖 Lanes?~~ → ✅ 已决策,本项关闭**

> **决策:Lanes 列为强制依赖。不装 Lanes 就没有多 worker。**

原先纠结「单线程用户不该被迫装 Lanes」,既然定为强制依赖,
`DeepPrelude` 依赖 `Universe` 这条就不再是障碍 —— 我们**无条件**嵌入
`compat/deep/tools/universe`,Tensor **始终**是 deep userdata,**不做双表示**。

这个决策带来的是净收益,不只是"绕开问题":

| 项 | 可选依赖方案 | 强制依赖方案 |
|---|---|---|
| Tensor 表示 | 双表示(普通 / deep userdata) | **单一表示** |
| 代码路径 | 两条,各自测试 | 一条 |
| 运行时检测逻辑 | 需要 | 不需要 |
| 用户 bug 复现 | "我这边是普通模式" → 环境发散 | 环境一致 |
| M0 验证项 | 多一项 | **少一项** |

**唯一代价:发行包体积 + 构建复杂度。** Lanes 本体很小(MIT、纯 C++、无外部依赖),
我们本来就要静态链接分发,可接受。

**(b) v4.0.0 太新。** 2026-02-27 发布,大版本跳跃通常伴随 API 变动。
建议先按 **v3.17.2** 开发,4.x 观望。
⚠️ 注意:上面引用的 `deep.hpp` 我读的是 **master(4.x)**,3.17 的 deep API 可能不同,**需核对**。

**(c) C++ ABI 一致性。** Lanes 4.x 是 C++,deep userdata 接口是**虚函数**,
意味着 Lanes 和我们的绑定必须用同一编译器/同一 C++ 运行时编译(MSVC 上尤其敏感)。
这与主报告 B.3「中间层用纯 C ABI」是同一类问题 —— 但这里无法回避,虚函数就是虚函数。

**(d) 每个 lane 里的 Paddle tracer 初始化。**
第 3.1 节查到 `g_current_tracer` 初值为 `nullptr`,每个 worker 线程必须自己初始化。
Lanes 不知道 Paddle 存在 → 需要用 Lanes 的 `on_state_create` 钩子注入(**待确认该钩子存在**)。

**(e) 与 GC 堆追踪的协调。** Lanes 有 `allocator.cpp`,支持自定义分配器;
`research/gc.md` §4 的堆追踪要按线程统计。两者需要对齐,可能是协同也可能是冲突。

### 9.6 修正后的方案

```
采用 Lua Lanes v3.17.x
  ├── paddle.Tensor  普通 sol2 usertype,跨 lane 走 __lanesclone(R18 推翻了原 deep userdata 方案)
  ├── DataLoader     worker = lane,队列 = Linda
  ├── dataset 工厂函数 + config table 经 Linda 传入(第 4 节约束不变)
  └── lane 启动钩子里初始化 Paddle tracer + g_has_grad=false
```

**Fallback**:若 9.5(b)(c) 的版本/ABI 问题无解,退回第 6 节的自建线程池方案,
600-900 行,不是灾难。但 (a) 已关闭 → **Lanes 是默认路径,不是候选路径。**

### 9.7 净变化

| 项 | 原结论 | 修正后 |
|---|---|---|
| 是否用现成线程库 | ❌ 不用,自建 | ✅ **用 Lanes** |
| Tensor 跨线程 | 自己在 C 侧建队列传 shared_ptr | **Lanes deep userdata,官方支持** |
| 队列 | 自己写 | **Linda** |
| C 侧工作量 | 600-900 行 | **200-300 行** |
| 新增依赖 | 无 | **Lanes(强制依赖,MIT,活跃,Windows 一等公民)** |
| 新增 M0 验证项 | — | ~~deep userdata 能否做成可选依赖~~ **已决策关闭**;剩 (b)~(e) |
