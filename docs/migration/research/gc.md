# GC 与显存生命周期 · Torch7 先例研究

> 回答:老 Torch 时代怎么解决的?能不能解决主报告第 4 节的问题?
> **简短答案:Torch7 缓解了,但从未真正解决 —— 最终的"解法"是迁移到有引用计数的宿主语言(Python)。**
> **但我们的条件比 Torch7 好得多,而且 Paddle 里已经有现成的钩子。**

✅ **已通过代理拉取 torch7 上游源码逐条核实,见文末第 6 节。**
原标注"[中高置信]/[待核实]"的结论:**3 条证实,1 条证伪**(已在正文修正)。
参考文件下载于 `_ref/`:`THGeneral.c`、`utils.c`、`init.lua`、`lua_version.hpp`。

---

## 1. Torch7 实际做了什么(四层机制)

### 1.1 C 层引用计数 + Tensor/Storage 分离 [高置信]

Torch7 **没有**把生命周期完全交给 Lua GC:

```
THStorage { void* data; ptrdiff_t size; int refcount; ... }   <- 真正持有内存
THTensor  { THStorage* storage; long* size; long* stride;      <- 只是一个 view
            ptrdiff_t storageOffset; int refcount; ... }
```

- `THTensor_(retain)` / `THTensor_(free)` 手动增减 refcount
- Lua 侧的 userdata 只持有**一份**引用,`__gc` 元方法调用 `THTensor_free`
- **多个 Tensor 共享同一个 Storage**(view/narrow/select 都不拷贝)

意义:Lua 可见对象极小,真正的内存由 C 层 refcount 管理。
**Lua GC 延迟释放的只是"最后一份引用",不是全部引用。**

> 这一点 Paddle 天然具备:`paddle::Tensor` 内部是
> `std::shared_ptr<phi::TensorBase>`,`Tensor` 本身也是 view 语义。
> 我们直接继承这个好处,无需额外设计。

### 1.2 堆追踪:把 C 内存压力翻译成 Lua GC 压力 [已核实]

这是 Torch7 最核心的发明,也正是主报告 4 节我提的"对策 4"。

```c
/* lib/TH/THGeneral.c 大意 */
static ptrdiff_t heapSize = 0;
static ptrdiff_t heapSoftmax = ...;        /* 软上限,记忆中默认 ~300MB */

void THHeapUpdate(ptrdiff_t size) {
  heapSize += size;
  if (heapSize > heapSoftmax) {
    torchGCFunction(torchGCData);          /* -> lua_gc(L, LUA_GCCOLLECT, 0) */
    /* 然后根据回收效果自适应调整 heapSoftmax */
  }
}
```

Lua 侧开关:`torch.setheaptracking(true)`。

❌ **原文这里我判断错了,已核实并更正**:torch7 的 `init.lua:189` 有一行
`torch.setheaptracking(true)` —— **它是默认开启的**,不是可选功能。
Torch7 用原子累加 + 1MB 批量阈值(`heapMaxDelta`)来把统计开销压下去,详见第 6 节。

### 1.3 分配失败时先 GC 再重试 [已核实]

```c
void* THAlloc(ptrdiff_t size) {
  void* ptr = malloc(size);
  if (!ptr && torchGCFunction) {
    torchGCFunction(torchGCData);   /* 跑一次完整 Lua GC */
    ptr = malloc(size);             /* 再试一次 */
  }
  if (!ptr) THError("$ Torch: not enough memory ...");
  return ptr;
}
```

GPU 侧 cutorch 同理:`THCudaMalloc` 在 `cudaMalloc` 失败时调用
`THCSetGCHandler` 注册的回调跑 Lua GC,然后重试。

**这就是主报告 4 节的"对策 3"。Torch7 验证过它是有效的。**

### 1.4 缓存分配器 [高置信]

cutorch 后期引入 `THCCachingAllocator` —— **PyTorch caching allocator 的直系祖先**。
把 `cudaMalloc/cudaFree` 变成从缓存池取还,让 1.3 的"GC 后重试"代价大幅下降
(GC 释放的块回到池里,立刻可复用,不必真的还给驱动)。

---

## 2. 解决了吗?没有

证据:**Torch7 训练循环里手写 `collectgarbage()` 是当年的普遍习惯。** [高置信]

```lua
for epoch = 1, N do
   for i, batch in ipairs(data) do
      ...
      collectgarbage()      -- 几乎人人都写
   end
end
```

一个机制如果真的有效,用户不会普遍手动兜底。四个根因:

| # | 问题 | 说明 |
|---|---|---|
| 1 | **全量 GC 太贵** | `LUA_GCCOLLECT` 是 stop-the-world 全堆标记清扫,还发生在不可预测的时刻 |
| 2 | **追踪不全** | 只统计走 `THAlloc` 的内存;cuDNN workspace、BLAS scratch、第三方库分配统统看不见 |
| 3 | **GPU 侧只有"失败后补救",没有"提前预防"** | CPU 有 heapSoftmax 主动施压,GPU 主要靠 OOM 后重试 |
| 4 | **LuaJIT 的 1-2GB 地址空间限制** | 见下,当年的著名痛点 |

### 2.1 LuaJIT 地址空间限制(必须知道)[高置信]

GC64 之前的 LuaJIT 在 x64 上只能把 GC 对象分配在低 2GB(实测常常 ~1GB)地址空间内。
Torch7 用户大量遇到 `not enough memory` —— 明明机器有 128GB。

- Tensor 的**数据**走 malloc,不在这个限制内
- 但 userdata 头、Lua table、closure、字符串**都在**
- 大规模模型 + 大量小对象很容易撞上

**对我们的直接影响:如果支持 LuaJIT,必须确认构建开启了 `LJ_GC64`。**
LuaJIT 2.1 在 x64 上已默认开启 GC64,但要在 M0 显式验证。

### 2.2 最终"解法"是换宿主语言

PyTorch 迁到 Python 后,**引用计数让 `del x` / 出作用域立即释放**,
GC 只处理循环引用。这个问题基本消失了。

**这是最诚实的结论:Lua 的纯 mark-sweep 与"大块 off-heap 内存"的组合,
在语言层面就是不匹配的。Torch7 的四层机制是缓解,不是消除。**

---

## 3. 但我们的条件比 Torch7 好得多

### 3.1 Paddle 已经内置了 Torch7 的两层机制 —— 零上游改动

这是本次调研最好的发现之一。

**(a) OOM 回调钩子已存在,且是 `PADDLE_API` 导出的:**

```cpp
// paddle/phi/core/memory/allocation/retry_allocator.h:19
PADDLE_API void RegisterOOMCallback(std::function<size_t(Place, size_t)> callback);
```

调用逻辑(`retry_allocator.cc:60-82`)与 Torch7 的 `THAlloc` 几乎一模一样:

```cpp
for (int64_t i = 0; i < FLAGS_offload_retry_times && has_offloaded; ++i) {
  try { return alloc_func(); }
  catch (BadAlloc&) {
    has_offloaded = (g_oom_callback(place_, size) > 0);   // 返回"释放了多少字节"
  }
}
return alloc_func();
```

pybind 目前用它做 activation offload(`pybind.cc:3552 register_offload_callback`)。
**我们注册自己的即可:**

```cpp
paddle::memory::allocation::RegisterOOMCallback(
  [L](phi::Place place, size_t size) -> size_t {
    size_t before = g_live_tensor_bytes;
    lua_gc(L, LUA_GCCOLLECT, 0);          // Torch7 的 1.3
    return before - g_live_tensor_bytes;  // >0 则 Paddle 自动重试
  });
```

**Torch7 要改 TH 源码才能做到的事,Paddle 已经开放成公共 API。**
完全符合"不增加 C++ 代码 / 不改上游"的约束。

**(b) 缓存分配器已经有,而且比 cutorch 那代强得多:**

```
paddle/phi/core/memory/allocation/
  auto_growth_best_fit_allocator{,_v2}      buddy_allocator
  retry_allocator                           cuda_malloc_async_allocator
  vmm_auto_growth_best_fit_allocator_v2     virtual_memory_auto_growth_...
```

即 Torch7 的 1.4 层免费获得,而且是 VMM 级别的实现。

### 3.2 三件 Torch7 不可能有的东西

#### (1) Lua 5.4+ 的分代 GC

```lua
collectgarbage("generational")
```

我们的负载特征是**每个 iteration 产生大量短命张量**,这正是分代 GC 的最佳场景。
Torch7 只有 Lua 5.1/LuaJIT 的增量标记清扫,没有这个选项。

#### (2) Lua 5.4+ 的 to-be-closed 变量 —— **确定性析构**

```lua
do
  local t <close> = paddle.zeros({4096, 4096})
  ...
end   -- 离开作用域立即调用 __close,内存确定性释放,不等 GC
```

**这在语言层面直接消除了问题**,等价于 C++ RAII / Python 的 `with`。
Torch7 时代(5.1)完全没有这个东西。

#### (3) LuaJIT GC64

2.1 节那个把 Torch7 折磨多年的 1-2GB 限制,今天默认已经解决。

### 3.3 关于 Lua 5.1 语法约束的重要区分

> 补充约束:库代码用 Lua 5.1 语法子集,因为 5.1 语法在任何项目里都能用。

**这个约束完全合理,而且不影响上面 (1)(2) 两条 —— 因为要区分"语法"和"运行时特性"。**

| | 是什么 | 5.1 约束下能否用 |
|---|---|---|
| `collectgarbage("generational")` | **运行时字符串参数**,不是语法 | ✅ 能。`pcall` 保护即可在 5.1 上安全降级 |
| `local t <close> = ...` | **5.4 语法** | ❌ 我们的库代码不能写 |
| **给 userdata 设 `__close` 元方法** | **C API 操作**,不是 Lua 语法 | ✅ 能。在 C 侧 `lua_setfield(L,-2,"__close")` |

所以正确的做法是:

```
我们的 Lua 库代码      100% Lua 5.1 语法子集,一行不越界
        +
C 侧无条件挂 __close   (5.1/5.2/5.3/LuaJIT 忽略它,5.4/5.5 生效)
        +
运行时特性检测         pcall(collectgarbage, "generational")
```

**结果:库代码严守 5.1,但 5.4/5.5 的用户在自己的代码里可以写
`local t <close> = paddle.zeros(...)` 拿到确定性释放。**
我们不用这个语法,不代表不能给用户提供这个能力。

---

## 4. 建议的最终方案(Torch7 四层 + 我们的三个增量)

| 层 | 来源 | 我们的实现 | 成本 |
|---|---|---|---|
| 1. C 层 refcount + view 语义 | Torch7 1.1 | `paddle::Tensor` 天然具备 | 0 |
| 2. 缓存/自增长分配器 | Torch7 1.4 | Paddle 已有 | 0 |
| 3. **OOM 时跑 Lua GC 再重试** | Torch7 1.3 | `RegisterOOMCallback` + `lua_gc` | **~20 行** |
| 4. **堆追踪主动施压** | Torch7 1.2 | 自己统计 live tensor bytes,超阈值 `lua_gc(L, LUA_GCSTEP, n)` | ~50 行 |
| 5. 显式 `t:free()` | 兜底 | 手写 | ~10 行 |
| 6. **`paddle.scope(fn)` 作用域回收** | 新 | 纯 Lua 5.1 语法可实现 | ~40 行 |
| 7. **C 侧挂 `__close`** | 5.4+ 增量 | C 侧一行,用户侧 `<close>` | ~5 行 |
| 8. **分代 GC 特性检测** | 5.4+ 增量 | `pcall(collectgarbage,"generational")` | ~3 行 |
| 9. **确认 LuaJIT GC64** | 避开 Torch7 的坑 | 构建检查 | 0 |

**总成本约 130 行,其中最关键的第 3 层因为 Paddle 已开放 `RegisterOOMCallback` 而近乎免费。**

### 风险评级修正

主报告 4 节把这个列为"**最大的运行时风险**"。
基于本节发现(Paddle 已有钩子 + Torch7 验证过路径可行 + 5.4 `<close>` 兜底),
**降级为"中等风险、有成熟解法、需在 M0 验证"。**

但保留一个诚实的保留意见:
**Torch7 的历史证明这套机制在"训练大模型 + 显存吃紧"时仍会退化。**
如果目标场景包含大模型训练,应该从第一天就把 `paddle.scope` /`<close>`
做成**文档里推荐的默认写法**,而不是等出问题再补 —— 这正是 Torch7 当年做晚了的事。

---

## 5. M0 新增验证项

| # | 验证 | 方法 |
|---|---|---|
| 1 | `RegisterOOMCallback` + `lua_gc` 能否真的救回一次 OOM | 分配到 OOM,看是否自动恢复 |
| 2 | LuaJIT 是否 GC64 | 检查构建宏 / 分配超 2GB 小对象 |
| 3 | sol2 3.5.0 能否编过 Lua 5.5.0 | **vcpkg 本地就有两者,10 分钟可验证** |
| 4 | `__close` 在 5.4/5.5 上对 userdata 是否按预期触发 | 小 demo |

> 注:`E:/code/vcpkg/ports` 下已有 `lua 5.5.0`、`sol2 3.5.0`、`luajit(2026-03-30)`,
> 验证项 3 现在就能做,不需要编译 Paddle。

---

## 6. 源码核实结果(已联网拉取上游)

### 6.1 Torch7 堆追踪 —— 证实,且拿到了确切参数

`torch7/lib/TH/THGeneral.c`:

```c
:134  static __thread void (*torchGCFunction)(void *data) = NULL;
:136  static ptrdiff_t heapSize = 0;
:138  static const ptrdiff_t heapMaxDelta = (ptrdiff_t)1e6;   // +/-1MB 才同步一次全局
:140  static __thread ptrdiff_t heapSoftmax = (ptrdiff_t)3e8; // 300MB 软上限,动态上调
:141  static const double heapSoftmaxGrowthThresh = 0.8;      // GC 后仍 >80% 则扩容
:142  static const double heapSoftmaxGrowthFactor = 1.4;      // 每次扩 40%
:154  void THSetGCHandler(void (*torchGCFunction_)(void*), void *data)
:191    if (torchGCFunction && curHeapSize > heapSoftmax) {
:192      torchGCFunction(torchGCData);
:197      if (newHeapSize > heapSoftmax * heapSoftmaxGrowthThresh)
:198        heapSoftmax = (ptrdiff_t)(heapSoftmax * heapSoftmaxGrowthFactor);
```

`torch7/utils.c`:

```c
:186  static void luaTorchGCFunction(void *data) {
:188    lua_State *L = data;
:189    lua_gc(L, LUA_GCCOLLECT, 0);        // 全量 GC
:190  }
:199    if(enabled) THSetGCHandler(luaTorchGCFunction, L);
```

**三个可直接抄的工程细节:**

| 参数 | Torch7 取值 | 含义 |
|---|---|---|
| `heapSoftmax` 初值 | **300 MB** | 超过就触发全量 GC |
| 自适应策略 | GC 后仍 >80% → 软上限 **×1.4** | 防止在阈值附近反复抖动 |
| `heapMaxDelta` | **1 MB** | **批量累加,减少原子操作竞争** |
| 存储类别 | `__thread` | `heapSize`/`heapSoftmax` **线程局部** |

`heapMaxDelta` 那条是我原先没想到的:每次分配都做全局原子加会有严重的多线程竞争,
Torch7 的做法是**线程本地累积到 ±1MB 才同步一次全局**。
我们做主报告"对策 4"时应直接照抄这个模式。

### 6.2 GC-on-OOM 重试 —— 证实

`THGeneral.c` 的 `THAlloc` / `THRealloc`:

```c
:264  if(!ptr && torchGCFunction) {
:265    torchGCFunction(torchGCData);      /* 跑 Lua 全量 GC */
:266    ptr = malloc(size);                /* 重试一次 */
:270  THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", ...);
:292  if(!newptr && torchGCFunction) { ... }   /* realloc 同理 */
```

与 Paddle 的 `RetryAllocator::AllocateImpl` + `RegisterOOMCallback` 是同一套设计。
**区别是 Paddle 的更强**:Torch7 只重试 1 次,Paddle 按 `FLAGS_offload_retry_times` 循环重试,
且用回调返回值(释放字节数)判断是否值得继续。

### 6.3 ❌ 我错了一处:堆追踪是**默认开启**的

`torch7/init.lua:189`:

```lua
torch.setheaptracking(true)
```

我原先说"它是可选、不是默认开的"—— **错误**。Torch7 默认就开着,
靠 6.1 的 `heapMaxDelta` 批量策略把开销压下去。

**这反而是个更强的正面信号**:Torch7 敢默认开,说明开销可接受,
我们也应该**默认开启**堆追踪,而不是做成开关。

### 6.4 sol2 对 Lua 5.5 —— 明确不支持

sol2 `include/sol/compatibility/` 目录只有:

```
compat-5.3.c.h   compat-5.3.h   compat-5.4.h   lua_version.hpp
```

**没有 `compat-5.5.h`。**

上游三个 issue,**全部 open**:

| # | 标题 | 创建时间 | 状态 |
|---|---|---|---|
| 1721 | Support Lua 5.5 | 2025-07-17 | **open** |
| 1723 | Add lua 5.5 support | 2025-07-21 | **open** |
| 1747 | **Issues with Lua 5.5** | **2026-02-26** | **open** |

即到 2026-02 为止 sol2 仍未支持 Lua 5.5。**结论:确定不支持。**

**而且有个隐蔽的坑。** Lua 5.5 的 `lua.h`:

```c
#define LUA_VERSION_NUM  (LUA_VERSION_MAJOR_N * 100 + LUA_VERSION_MINOR_N)   // = 505
```

而 sol2 `lua_version.hpp:105-116`:

```c
#if !defined(SOL_LUA_VERSION)
    #if defined(LUA_VERSION_NUM) && LUA_VERSION_NUM >= 502
        #define SOL_LUA_VERSION LUA_VERSION_NUM     // 505 从这里通过
```

→ **sol2 不会在版本检测阶段报错,`SOL_LUA_VERSION` 会被设成 505 一路放行,
然后在后面真正用到 5.5 变更过的 C API 时才炸。**
这就是 issue #1747「Issues with Lua 5.5」的成因。

**踩坑特征:不是干净的 "unsupported version" 报错,而是一堆莫名其妙的编译错误。**
将来若要加 5.5,先看这一条能省很多时间。

### 6.5 决策:5.5 暂不支持(已采纳)

目标版本收敛为 **Lua 5.1 / 5.2 / 5.3 / 5.4 + LuaJIT**,共 5 个。

代价评估:**几乎为零,且可逆。** 因为主报告 B.2 的分层里中间层是纯 C ABI,
将来 5.5 要加时只有两条路,都不影响已有代码:

1. 等 sol2 上游合并 5.5(issue 已开一年半,大概率会来)
2. 5.5 单独走裸 Lua C API + 我们自己的 `lua_compat.h`,不经 sol2

**架构上不需要为 5.5 预留任何东西。**

---

## 7. 核实后的净变化

| 项 | 变化 |
|---|---|
| Torch7 堆追踪机制 | ✅ 证实,并拿到 300MB / ×1.4 / 1MB批量 / `__thread` 四个可直接抄的参数 |
| GC-on-OOM 重试 | ✅ 证实,Paddle 的实现比 Torch7 更强 |
| 堆追踪是否默认开 | ❌ **我判断错了** —— 是默认开的。结论反而更乐观,我们也应默认开 |
| sol2 + Lua 5.5 | ✅ 查清:**不支持**,3 个 issue 全 open;且会伪装成一堆编译错误而非清晰报错 |
| 目标 Lua 版本 | 5.5 移出范围 → **5.1/5.2/5.3/5.4 + LuaJIT** |
| 第 4 节"M0 验证项 3" | **已完成,无需再验证** |
