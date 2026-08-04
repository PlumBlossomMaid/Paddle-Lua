# P13 · Lanes 多 worker

| | |
|---|---|
| 阶段 | P13 |
| 类别 | 手写 C++ —— **风险最高的一块** |
| 开工条件 | P11 完工 + M0 第 3/4/5/6 项验证通过 |
| 预估 | 3 周 |

---

## 1. 做什么 / 不做什么

**做:** `DataLoader` 的 `num_workers > 0` 路径,基于 Lua Lanes。

**不做:** 不用多进程。**这是与 Python 的根本分歧,也是我们的优势。**

---

## 2. 为什么线程胜过进程

Python 的 DataLoader 用 fork,是因为 **GIL 让多线程无法真并行**。
Lua 没有 GIL —— 每个 `lua_State` 是独立的,天然真并行。

这带来一个 Python 永远做不到的好处:

| | Python(多进程) | Lua(多线程) |
|---|---|---|
| 传一个 batch | pickle 序列化 -> 管道/共享内存 -> 反序列化 | **传指针** |
| 一个 224×224×3 float32 batch(64) | ~38MB 的拷贝 | 8 字节 |
| 启动开销 | fork 整个进程 | 建一个 `lua_State` |
| Windows 上 | 没有 fork,要 spawn + 重新 import,极慢 | 一样快 |

**这不是微优化,是数量级差异。** 也是这个项目值得做的理由之一。

---

## 3. 上游/第三方有什么可以用

| 出处 | 内容 |
|---|---|
| Lanes `deep.hpp`(见 `research/_ref/lanes_deep.hpp`) | `DeepPrelude` + `DeepFactory` —— **调研过,最终未采用**,见 §4.1 |
| Lanes 可克隆 userdata(`__lanesclone`) | **采用**。⚠️ 签名待核实 |
| Lanes Linda | 跨 lane 队列 |
| `research/dataloader.md` §9 | 完整的设计论证 |

**明确不用的东西:**

> `paddle/fluid/framework/new_executor/workqueue/` ——
> 它调度的是 `std::function<void()>`,**不涉及 `lua_State`**。
> 我们需要的是"每个 worker 有独立 Lua 解释器"的模型,它给不了。
> 已记入 `process/decisions.md` R14,**不要重新论证**。

---

## 4. 设计

### 4.1 Tensor 怎么跨 lane:sol2 usertype + `__lanesclone`

> ✅ **Q-11 已收敛。** 表示形式**与 sol2 对齐**,不改成 Lanes 的 deep userdata。
> 原先"P2 要改成 deep userdata 布局"的要求**已撤销**,`02-binding.md` 不需要返工。

#### 为什么不需要 deep userdata

Lanes 的 deep userdata 解决的是"多个 `lua_State` 共享一个 C 对象、并对它做原子引用计数"。

**但 `paddle::Tensor` 自己已经是引用计数的。**
`tensor.h:142` 的构造函数收 `std::shared_ptr<phi::TensorBase>` ——
底层缓冲区由 `shared_ptr` 管理,而 `std::shared_ptr` 的计数**是原子的**。

所以拷贝一个 `paddle::Tensor` 就是:

- 引用计数 +1(原子操作,跨线程安全)
- **底层显存/内存一份都不拷**

我们要的"零拷贝跨线程传递"**已经免费拿到了**,再套一层 `DeepPrelude` 的引用计数
是在给同一件事做两遍。

#### 实际结构

```
worker lane                       main lane
 +----------------+               +----------------+
 | sol2 usertype  |               | sol2 usertype  |
 | LuaTensor      |               | LuaTensor      |
 |  paddle::Tensor|               |  paddle::Tensor|   各自一个壳,各自 __gc
 +-------+--------+               +-------+--------+
         |  shared_ptr 原子计数            |
         +---------------+----------------+
                         v
              +---------------------+
              | phi::TensorBase     |   缓冲区只有一份
              +---------------------+
```

**每个 lane 有自己的壳,各自被自己 `lua_State` 的 GC 管**,
最后一个壳析构时缓冲区才真正释放。这比共享一个 proxy 更简单,也更符合 Lua 的 GC 模型。

#### 传递机制:`__lanesclone`

Lanes 除了 deep userdata,还支持**可克隆 userdata**:
在 userdata 的元表上挂 `__lanesclone`,Lanes 跨 lane 传递时会在目标状态里
分配同样大小的 userdata,然后调用该元方法完成内容复制。

我们的实现就是一次 C++ 拷贝构造:

```cpp
// 目标侧收到一块生内存 → placement new 一个副本
// 注意:绝不能让 Lanes 直接 memcpy —— 那会按位复制 shared_ptr 而不增加引用计数
usertype["__lanesclone"] = [](void* dst, const LuaTensor& src) {
  new (dst) LuaTensor(src);      // 拷贝构造 → shared_ptr 计数 +1
};
```

**"绝不能 memcpy"这一条是本阶段最容易崩的地方。** 按位复制 `shared_ptr`
会让两个 lane 各自持有计数为 1 的副本,先析构的那个就把缓冲区释放了,
另一个变成悬空指针 —— 而且崩溃点离出错点很远。

⚠️ **`__lanesclone` 的存在性与确切签名待核实**(Q-11 转为核实项,M0 第 5 项一并做)。

#### 退路:句柄表

若目标 Lanes 版本没有 `__lanesclone`,退路是**只传一个整数**:

```
worker: id = registry_put(tensor)   -- 全局表,加锁,持有一份 shared_ptr
        linda:send("batch", id)
main:   tensor = registry_take(id)  -- 取出并从表中移除
```

跨 lane 传的是 `lua_Integer`,Lanes 天然支持。
代价是多一个全局锁和一次查表(纳秒级,batch 粒度上完全不可见),
**但同样是零数据拷贝**,而且**完全不依赖 Lanes 的任何高级特性**。

**这条退路的存在意味着:即使 `__lanesclone` 核实下来不可用,多线程 DataLoader 依然成立。**
多 worker 不是一个建立在未验证特性上的赌注。

### 4.2 数据流

```
主 lane                     Linda 队列              worker lanes (N)
  │                                                      │
  ├─ 分发索引 [1..60000] ──► "index" ──────────────────► │ 取索引
  │                                                      │ ds:get(i)
  │                                                      │ transform
  │                                                      │ collate
  │ ◄──────────────── "batch" ◄──────────────────────────┤ 送 Tensor 句柄
  └─ for 循环取 batch
```

### 4.3 worker 里能用什么、不能用什么

**这是本阶段最需要写清楚的约定**(P11 §3.3 已提前埋伏):

| | 能否跨 lane |
|---|---|
| 数字、字符串、布尔、表(深拷贝) | ✅ |
| Tensor(sol2 usertype + `__lanesclone`) | ✅ 传指针,`shared_ptr` 计数 +1 |
| 纯函数(`string.dump` 可序列化) | ✅ |
| **带 upvalue 的闭包** | ⚠️ upvalue 会被深拷贝,**不共享** |
| 文件句柄、C 库句柄 | ❌ |
| coroutine | ❌ |
| metatable 复杂的对象 | ⚠️ 取决于是否注册了 deep factory |

**违反时的报错必须清晰。** Lanes 默认的错误信息很晦涩,
我们要在传递前主动检查并给出"你的 Dataset 里第 3 个字段是 userdata,无法跨 worker 传递"
这种信息。

### 4.4 每个 lane 的初始化

每个 worker lane 是一个全新的 `lua_State`,**需要重新 `require "paddle"`**。
Lanes 提供 `on_state_create` 钩子做这件事。

⚠️ **Q-03:该钩子在目标 Lanes 版本上是否存在、签名如何,未核实。** M0 第 6 项。

---

## 5. 已知的坑

**① phi kernel / DeviceContext 的多线程安全性未验证。** M0 第 3 项。
如果 DeviceContext 不是线程安全的,worker 里就不能做任何张量运算,
只能做纯 CPU 的数据读取与解码 —— **这会大幅削减多 worker 的收益**,方案要改。

**② Lanes 版本差异。** M0 第 4 项。
`__lanesclone` 在哪个版本引入、签名如何,必须锁定一个版本并在 rockspec 里写死下限。
**核实不通过就走 §4.1 的句柄表退路**,不要为了用上它去升级到不稳定的版本。

**③ C++ ABI 一致性。** Lanes 是 C 库,我们的绑定是 C++。
两边必须用同一个编译器、同一套运行时。Windows 上 MSVC 的
debug/release CRT 混用会直接崩。M0 第 5 项。

**④ 堆追踪(P14)与 Lanes 的自定义分配器可能冲突。** M0 第 10 项。
Torch7 的堆追踪用 `__thread` 存储(见 `research/_ref/THGeneral.c`),
而 Lanes 可能替换分配器。两者叠加的行为要实测。

**⑤ 崩溃的调试难度是单线程的十倍。** 一开始就要:
- 每个 lane 有编号,所有日志带编号
- 提供 `num_workers = 0` 的等价路径,**任何 bug 先用它复现**
- worker 里的错误必须完整传回主 lane 并带 traceback

---

## 6. 验收

- [ ] `num_workers = 4` 的吞吐 >= `num_workers = 0` 的 3 倍
- [ ] 传递的 batch 与单 worker 路径**逐位一致**
- [ ] 固定种子下,多 worker 与单 worker 的数据顺序一致
- [ ] 连跑 1 小时,无内存泄漏、无显存泄漏、无崩溃
- [ ] worker 里主动 `error()`,主 lane 收到完整错误信息与 traceback
- [ ] 迭代中途 `break`,所有 worker 正确退出,无僵尸线程
- [ ] Dataset 里塞一个文件句柄,报出可读的错误而不是崩
- [ ] Linux 与 Windows 各验一遍

---

## 7. 未解问题

- Q-02 Lanes 版本差异:`__lanesclone` 的可用性与签名(原 deep API 差异问题的收窄版)
- Q-03 `on_state_create` 钩子是否存在
- M0 第 3 项(kernel 多线程安全)的结论会**决定本方案是否需要重做**
