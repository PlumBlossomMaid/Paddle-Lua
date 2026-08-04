# P1 · C ABI 中间层

| | |
|---|---|
| 阶段 | P1 |
| 类别 | 手写 C++(输出纯 C 接口) |
| 开工条件 | P0 完工 |
| 预估 | 2 周 |

---

## 1. 做什么 / 不做什么

**做:** 在 Paddle 的 C++ 接口之上包一层**纯 C 接口**,`csrc/c_api/paddle_c.h` 里
不出现任何 C++ 语法(无 `class`、无模板、无异常、无 `std::`)。

**不做:** 不涉及 Lua。这一层对 Lua 一无所知。

**为什么不能省这一层**(`research/architecture.md` §B 的核心结论):

我们要支持五个 Lua 版本。sol2 是重模板库,每个翻译单元的编译开销很大。
如果 sol2 直接吃 Paddle 的 C++ 头,那么**每个 Lua 版本都要把整套 Paddle 头重编一遍**,
编译时间乘以 5。有了 C ABI 中间层之后:

```
paddle_c.cc  (重,含 Paddle C++ 头)      编 1 次
    ↓ 纯 C 接口
binding.cpp  (sol2 模板膨胀,但只吃 C 头)  编 5 次,但每次很轻
```

第二个理由更硬:**LuaJIT FFI 只能吃纯 C 声明**。即使当前决策是 LuaJIT 也走 sol2(D-R1),
留一条 C ABI 就等于给未来保留了 FFI 后路,而这条后路的成本是零。

---

## 2. 上游有什么可以用

| 出处 | 用途 |
|---|---|
| `paddle/phi/api/include/tensor.h:199` | `std::vector<int64_t> shape() const` |
| `paddle/phi/api/include/tensor.h:224` | `DataType dtype() const` |
| `paddle/phi/api/include/tensor.h:248` | `bool is_dense_tensor() const` |
| `paddle/phi/api/include/tensor.h:388,397` | `const void* data() const` / `void* data()` |
| `paddle/phi/api/include/tensor.h:492,501` | `Tensor copy_to(const Place&, bool blocking) const` |
| `paddle/phi/api/include/tensor.h:510` | `void copy_(const Tensor& src, const Place&, bool blocking)` |
| `paddle/phi/api/include/tensor.h:135` | `Tensor(const Place& place, const std::vector<int64_t>& shape)` |
| `paddle/fluid/eager/backward.h:25` | `egr::Backward(...)` |
| `paddle/fluid/eager/utils.h:123,149` | `EagerUtils::autograd_meta(Tensor*)` / `nullable_autograd_meta(const Tensor&)` |

---

## 3. 设计

### 3.1 句柄

```c
typedef struct pd_tensor_s*  pd_tensor;   /* 实为 new paddle::Tensor(...) */
typedef struct pd_place_s*   pd_place;
```

**不透明指针。** Lua 侧、FFI 侧都不知道里面是什么。

所有权规则**只有一条**:凡是名字里带 `_new_` / `_create_` / 返回新句柄的函数,
调用方负责 `pd_tensor_free`。没有例外,没有"有时候返回借用"。

### 3.2 错误处理

C 没有异常,而 Paddle 的 C++ 层**会抛异常**。约定:

```c
typedef enum { PD_OK = 0, PD_ERROR = 1, PD_OOM = 2, PD_INVALID_ARG = 3 } pd_status;

pd_status    pd_last_status(void);
const char*  pd_last_error(void);   /* 线程局部,下次调用前有效 */
```

**每一个 C 函数的实现都必须被 `try/catch(...)` 完整包裹。**
一个异常穿过 C ABI 边界就是未定义行为,在 Windows 上通常直接崩进程。
这条没有商量余地 —— 写一个宏,所有函数强制套用:

```cpp
#define PD_GUARD(body) \
  try { body } \
  catch (const std::exception& e) { pd_set_error(PD_ERROR, e.what()); } \
  catch (...) { pd_set_error(PD_ERROR, "unknown C++ exception"); }
```

错误信息缓冲区用 `__thread` / `thread_local` —— **Lanes 会真并行调用这些函数**(P13),
全局单缓冲区必然出错。

### 3.3 接口分组

| 组 | 函数数(估) | 内容 |
|---|---|---|
| 生命周期 | 6 | `pd_tensor_new_empty` / `_from_blob` / `_free` / `_clone` / `_retain` |
| 元信息 | 10 | shape / dtype / ndim / numel / place / name / is_contiguous |
| 数据搬运 | 8 | `pd_tensor_copy_to` / `_to_host_buffer` / `_from_host_buffer` |
| 自动微分 | 8 | `pd_backward` / `pd_grad` / `pd_set_stop_gradient` / `pd_tensor_grad` |
| 设备与流 | 6 | 设备数、当前设备、同步、显存统计 |
| 算子 | ~4000 | **全部由 P3 生成,不手写** |

**手写的只有前五组,约 40 个函数。** 算子那 4000 个是生成的,见 `03-codegen.md`。

### 3.4 dtype 与 place 的跨语言表示

用整数枚举,值**必须与 `phi::DataType` 一致**,不要自创编号:

```c
/* 与 paddle/phi/common/data_type.h 的 phi::DataType 逐一对应 */
typedef enum { PD_DTYPE_UNDEFINED = 0, PD_DTYPE_BOOL, ... } pd_dtype;
```

在 `paddle_c.cc` 里写一个 `static_assert` 数组,编译期校验两边对齐。
**这样上游改了枚举值,我们编译期就炸,而不是运行时算错。**

---

## 4. 已知的坑

**① `paddle::Tensor` 是 `shared_ptr` 语义。** `pd_tensor` 持有的是一个堆上的
`paddle::Tensor` 对象,它内部再持有 `shared_ptr<TensorBase>`。
`pd_tensor_free` 删的是外层壳,底层显存由引用计数决定 —— 这正是我们要的,别自作聪明去改。

**② `data()` 返回的指针不能长期持有。** 任何一次 in-place 算子或设备间拷贝都可能让它失效。
C ABI 里应该**只提供拷贝式访问**(`pd_tensor_to_host_buffer`),不暴露裸指针给 Lua。
唯一的例外是 P13 的跨 lane 传递,那里传的是 `pd_tensor` 句柄不是数据指针。

**③ 不要在 C 头里用 `bool`。** C99 的 `bool` 需要 `<stdbool.h>`,而 LuaJIT FFI 的
C 解析器对某些写法挑剔。统一用 `int`,`0`/`1`。

**④ 结构体不要出现在 ABI 里。** 传 shape 用 `(const int64_t* dims, int ndim)` 两个参数,
不要定义 `pd_shape` 结构体。结构体一旦定义就是 ABI 契约,加字段就破坏兼容。

---

## 5. 验收

- [ ] `gcc -std=c99 -c test_c_only.c`(**不是 g++**)编译通过,该文件只 include `paddle_c.h`
- [ ] 链接成可执行文件并跑通:创建张量 -> 填值 -> `multiply` -> `pd_backward` -> 读回梯度
- [ ] 故意触发一次异常(如 shape 不匹配),确认返回 `PD_ERROR` 而不是崩进程
- [ ] 两个线程同时调用并各自触发错误,确认 `pd_last_error()` 互不干扰
- [ ] `static_assert` 覆盖全部 dtype 枚举
- [ ] 头文件中 `grep -c "class\|template\|std::\|namespace"` 结果为 **0**

---

## 6. 未解问题

- C ABI 是否需要提供"批量算子调用"入口以摊薄跨界开销?
  **暂不做** —— 先测出真实开销再说,过早优化会污染接口设计
