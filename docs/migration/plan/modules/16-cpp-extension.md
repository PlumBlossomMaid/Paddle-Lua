# P16 · cpp_extension(自定义算子)

| | |
|---|---|
| 阶段 | P16 |
| 类别 | 部分直抄(**只有 1/3 能抄**) |
| 开工条件 | P4 完工 |
| 预估 | 2 周 |

---

## 1. 做什么 / 不做什么

**做:** 让用户能写一个 `.cc` / `.cu`,编译成动态库,在 Lua 里加载并当算子用。

**不做:** 不做 setuptools 那一套。见 §2 的翻案说明。

---

## 2. 上游有什么可以用 —— 以及一次翻案

> ⚠️ **翻案记录(`process/decisions.md` R10):**
> 曾判断"cpp_extension 似乎可以直接抄"。
> 实际读完 `python/paddle/utils/cpp_extension/` 全部 3712 行后,结论是**只有约 1/3 能抄**。

| 部分 | 行数(约) | 能否抄 |
|---|---|---|
| 工具链探测(找 nvcc / MSVC / gcc、版本匹配、ABI flag) | 800 | ✅ **语义直抄** |
| `cpp_extension.py` 主体 | 1400 | ❌ **整块作废** |
| 算子加载与包装生成 | 500 | ⚠️ 语义可抄,但产物要改成 Lua |
| 其它工具 | 1000 | 部分 |

**为什么那 1400 行整块作废:**

`python/paddle/utils/cpp_extension/cpp_extension.py:32-35` 从 setuptools/distutils
import 了 `easy_install`、`build_ext`、`build`、`install`,
并**继承它们**。这个文件本身就是一个 **setuptools 插件** ——
它的存在意义是让 `python setup.py install` 能编 CUDA。

我们没有 setuptools,也不需要它。**这部分不是"翻译困难",是"根本不适用"。**

**能用的 C++ 侧接口:**

| 出处 | 内容 |
|---|---|
| `paddle/fluid/framework/custom_operator.h:316` | `LoadOpMetaInfoAndRegisterOp(const std::string& dso_name)` |

**一个函数,把 dso 里注册的算子挂进 Paddle 的算子表。** 这是整条路径的核心。

---

## 3. 设计

### 3.1 用户体验

```lua
local ops = paddle.utils.cpp_extension.load{
  name    = "my_relu",
  sources = { "my_relu.cc", "my_relu.cu" },
  extra_cuda_cflags = { "-O3" },
}

local y = ops.my_relu(x)
```

**用户写的 `.cc` 与 Python 版完全一样** —— 用 `PD_BUILD_OP` 宏。
这一点很重要:**已有的 Paddle 自定义算子代码可以零改动在 Lua 下用**。

### 3.2 流程

```
load{...}
  ├─ 1. 探测工具链           ← 语义抄自 Python 版那 800 行
  ├─ 2. 算 sources 的哈希    ← 缓存,避免重复编译
  ├─ 3. 若缓存未命中,调 nvcc/cc 编译成 .so/.dll
  ├─ 4. pd_load_op_meta_info(dso_path)
  │        └─ LoadOpMetaInfoAndRegisterOp
  ├─ 5. 查询新注册的算子名与签名
  └─ 6. 生成 Lua wrapper 表并返回
```

**第 5 步依赖 `OpProtoHolder` 链** —— 这是 Q-07,M0 第 15 项。
若该链在 `WITH_PYTHON=OFF` 下不可用,退路是**要求用户在 `load` 里显式声明签名**:

```lua
paddle.utils.cpp_extension.load{
  name = "my_relu", sources = {...},
  ops = { my_relu = { inputs = {"X"}, outputs = {"Out"} } },  -- 退路:手工声明
}
```

### 3.3 缓存

编译一次 CUDA 算子可能要几十秒。缓存键 = 源文件哈希 + 编译参数 + Paddle 版本 + 编译器版本。
**Paddle 版本必须在键里** —— 换了 Paddle 版本,旧的 .so 可能 ABI 不兼容,
加载时不是报错而是崩溃。

### 3.4 JIT 编译还是预编译

Python 版提供两条路:`setup()`(预编译)和 `load()`(JIT 编译)。
**我们只做 `load()`。** 预编译那条路的价值全部在于 setuptools 生态,
而我们没有那个生态。用户要预编译就自己写 CMake,我们提供
`paddle.utils.cpp_extension.get_include_paths()` / `get_link_flags()` 供他调用。

---

## 4. 已知的坑

**① 编译器必须与 `libpaddle` 一致。** C++ ABI 不兼容会导致加载后崩溃,
而且崩得毫无线索。**`load()` 应该主动检查并拒绝**,而不是让用户去调试段错误。

**② Windows 上找 MSVC 比 Linux 上找 gcc 麻烦得多。** Python 版靠 distutils 的
`msvc` 模块做这件事,我们没有。要自己读注册表或调 `vswhere.exe`。
**这是那 800 行"可抄"部分里最贵的一块。**

**③ `PD_BUILD_OP` 注册的算子如何参与自动微分?** 用户要同时写前向和反向,
用 `.SetBackwardOp(...)`。这部分是 C++ 侧的既有机制,我们不用管 ——
但**文档要写清楚,不然用户会以为 Lua 侧要做什么**。

**④ 别把生成的 wrapper 缓存到磁盘。** 它很小,每次生成即可。
缓存两层(so 和 wrapper)会出现版本不同步的诡异问题。

---

## 5. 验收

- [ ] 一个自定义 relu 算子(CPU),`load()` 后前向正确
- [ ] 同一个算子的反向正确,能参与 `backward()`
- [ ] CUDA 版本同上
- [ ] 第二次 `load()` 命中缓存,耗时 < 100ms
- [ ] 改一行源码后 `load()`,缓存正确失效
- [ ] 用不匹配的编译器构建的 .so,`load()` 时**报错而不是崩溃**
- [ ] Python 版 Paddle 的自定义算子示例代码零改动可用
- [ ] Linux 与 Windows 各验一遍

---

## 6. 未解问题

- Q-07 `WITH_PYTHON=OFF` 下 `OpProtoHolder` 链是否存活(M0 第 15 项)——
  决定是自动查询签名还是要用户手工声明
