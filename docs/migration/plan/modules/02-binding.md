# P2 · sol2 绑定层

| | |
|---|---|
| 阶段 | P2 |
| 类别 | 手写 C++ |
| 开工条件 | P1 完工 |
| 预估 | 2 周 |

---

## 1. 做什么 / 不做什么

**做:** 用 sol2 把 P1 的 C 接口暴露成 Lua 模块 `paddle_core`,一份源码编出五个二进制。

**不做:** 不写任何算子绑定(那是 P3 生成的);不做 API 语义设计(那是 Lua 层的事)。
这一层要薄到"看一眼就知道对不对"。

---

## 2. 上游有什么可以用

这一层**不直接接触 Paddle**,只接触 `paddle_c.h`。这是 P1 分层的意义。

外部依赖:

| 依赖 | 版本 | 出处 |
|---|---|---|
| sol2 | 3.5.0 | vcpkg,已在 `E:\code\vcpkg` |
| Lua | 5.1 / 5.2 / 5.3 / 5.4 | 各自的 `lua.h` |
| LuaJIT | 2.1 | 兼容 5.1 C API |

---

## 3. 设计

### 3.1 一份源码,五个二进制

```
csrc/lua/binding.cpp        ← 唯一的源文件,不做 #if LUA_VERSION 分支
        │
        ├── -DLUA_VERSION=51 → paddle_core.so  (链 liblua5.1)
        ├── -DLUA_VERSION=52 → ...
        ├── -DLUA_VERSION=53 → ...
        ├── -DLUA_VERSION=54 → ...
        └── LuaJIT           → ...
```

**版本差异全部交给 sol2 处理,我们的代码里不应该出现 `#if LUA_VERSION_NUM` 分支。**
出现了就说明抽象漏了,要么改设计,要么把差异隔离进一个 `compat.hpp`。

> 参考:`research/_ref/lua_version.hpp` 是 sol2 自己的版本适配方式。
> (原本还引了 `sol_compat.hpp`,但那次抓取拿到的是 404 页面,**没有真读过**,已删。
> 需要时重新抓 `sol2/include/sol/compatibility/` 下的文件。)

### 3.2 userdata 布局

```cpp
struct LuaTensor {
  pd_tensor h;          // C 句柄
  ~LuaTensor() { pd_tensor_free(h); }   // 交给 Lua GC
};
```

用 sol2 的 `usertype`,析构挂在 `__gc` 上。

**注意 Lua 5.1 的一个真实陷阱:** 5.1 里 `newuserdata` 创建的 userdata 默认**没有** `__gc`,
必须显式在 metatable 上设置。sol2 会处理,但如果我们手写任何一处 userdata,必须记住这条。
LuaJIT 同 5.1。

#### 这个布局不需要为 Lanes 做任何改动

`LuaTensor` 就是一个普通的 sol2 usertype,**不是 Lanes 的 deep userdata**。

跨 lane 传递靠的是 `paddle::Tensor` 自带的 `shared_ptr` 原子引用计数
(`tensor.h:142` 构造函数收 `std::shared_ptr<phi::TensorBase>`),
每个 lane 各持一个壳、共享同一个缓冲区。完整论证见 `13-lanes.md` §4.1。

P2 唯一要为 P13 预留的是**一个元方法槽位**:

```cpp
usertype["__lanesclone"] = [](void* dst, const LuaTensor& src) {
  new (dst) LuaTensor(src);       // 拷贝构造,绝不能 memcpy
};
```

P13 之前它不会被调用,但**现在挂上去比将来回头改 usertype 便宜**。
若 M0 核实下来 `__lanesclone` 不可用,P13 走句柄表退路,这个槽位闲置即可,
**不会反过来要求 P2 返工** —— 这正是选 sol2 表示而非 deep userdata 的价值。

### 3.3 错误映射

C 层返回 `pd_status`,Lua 层要变成 `error()`。写一个统一的检查宏:

```cpp
inline void pd_check() {
  if (pd_last_status() != PD_OK) throw sol::error(pd_last_error());
}
```

**约定:所有对外 Lua API 都用 `error()` 抛,不用 `return nil, msg`。**
理由:Paddle 的 Python API 也是抛异常,行为对齐;而且 `nil, msg` 在链式调用
(`net(x):sum():backward()`)里完全无法使用。

需要 `pcall` 语义的地方,由用户自己 `pcall`。

### 3.4 这一层手写的内容清单

| 内容 | 行数(估) |
|---|---|
| `Tensor` usertype + 元方法(`__gc` `__tostring` `__index` `__newindex` 算术) | 600 |
| dtype / place 枚举表 | 150 |
| 自动微分入口(`backward` / `grad` / `no_grad`) | 200 |
| 设备与显存统计 | 150 |
| 模块注册与版本信息 | 100 |
| 错误映射与工具 | 200 |
| 版本兼容层 `compat.hpp` | 100 |
| **合计** | **~1500** |

算子绑定不在此列。

### 3.5 `__index` 的三重职责

`x.shape`(属性)、`x:sum()`(方法)、`x["1:3, :"]`(切片)走的是**同一个** `__index`。
分派逻辑:

```
__index(t, k):
  type(k) == "string" and 是已知属性名  → 返回属性
  type(k) == "string" and 是已知方法名  → 返回方法
  type(k) == "string"                   → 当作切片表达式(P5 处理)
  type(k) == "table"                    → 当作多维下标(P5 处理)
  type(k) == "number"                   → 当作第一维下标(P5 处理)
```

**顺序很重要:属性和方法必须优先于切片。**
否则用户写 `x.shape` 会被当成切片表达式 `"shape"` 送去解析,报出莫名其妙的错。

---

## 4. 已知的坑

**① sol2 的编译时间。** 单个 TU 可能几分钟。**不要把所有绑定塞进一个 `.cpp`**,
按 §3.4 的清单拆成 6–7 个 TU,能并行编译,改一处也不用全量重编。

**② Lua 5.1 没有 `__len` 对 userdata 生效。** `#x` 在 5.1 上对 userdata 不调用 `__len`
(5.2+ 才支持)。所以**不要设计 `#tensor` 这种 API**,统一用 `x:numel()` / `x:size(1)`。
这是 C3(Lua 5.1 子集)在 API 设计上的直接后果。

**③ 5.3 开始 integer 与 float 是不同子类型。** `lua_Integer` 在 5.3+ 是 64 位整数,
在 5.1/5.2 是 `ptrdiff_t`。传 shape 时必须走 `lua_Number` -> `int64_t` 的显式转换并检查精度,
不能假设。

**④ LuaJIT 走 sol2 而不是 FFI,是已定决策(D-R1),不要重新论证。**
理由在 `process/decisions.md` R1:绑定开销 50–200ns vs kernel 1–10µs。

---

## 5. 验收

- [ ] 五个 Lua 版本**全部**能 `require "paddle_core"`
- [ ] 五个版本上 `core.version()` 返回一致
- [ ] 一个张量创建 -> 计算 -> 释放的循环跑 100 万次,RSS 不增长
- [ ] `collectgarbage("collect")` 后显存确实下降(用 `core.memory_allocated()` 验证)
- [ ] 故意传错类型(如 `core.tensor("abc")`),报错信息可读且不崩
- [ ] `grep -c "#if LUA_VERSION" csrc/lua/binding.cpp` 结果为 **0**

---

## 6. 未解问题

- Lua 5.4 的 `__close`(to-be-closed 变量)对 userdata 是否按预期触发?
  影响 `paddle.scope` 能否在 5.4 上做得更优雅。见 `research/gc.md` §5,M0 第 9 项
