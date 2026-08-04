# Paddle-Lua 技术调研报告

> 目标:`local paddle = require "paddle"`,无 Python 运行时,支持 LuaJIT + Lua 5.1/5.2/5.3/5.4/5.5,API 基本对齐 Paddle。
> 调研基线:PaddlePaddle 3.5.0 (`E:/code/paddle`, branch `windows_dataloader_multiprocess`)
> 日期:2026-08-03 · 状态:**调研阶段,未动工**

---

## 0. 一句话结论

**技术上可行,而且比想象的容易 —— 因为 Paddle 的动态图训练引擎本身就是纯 C++ 的,Python 只是一层壳。**
真正的工作量不在"接引擎",而在"重写那 10 万行纯 Python 的 nn/optimizer/tensor 组合层"。

一个人全职:MVP(能训 MNIST/ResNet)约 **4-6 个月**;"API 基本对齐 Paddle" 约 **1.5-2 年**。

---

## 1. 关键发现:Paddle 的 Python 边界在哪里

### 1.1 训练引擎是纯 C++ 的(最重要的发现)

| 层 | 位置 | 规模 | 依赖 Python |
|---|---|---|---|
| Kernel | `paddle/phi/kernels/` | 2110 个 `.cc/.cu` | 否 |
| Tensor | `paddle/phi/api/include/tensor.h` | `class PADDLE_API Tensor` | 否 |
| 前向 API | `paddle/phi/api/include/api.h` (生成) | ops 529 + fused 86 + sparse 51 | 否 |
| **自动微分前向** | `.../eager_generated/forwards/dygraph_functions.h` | **786 个 `*_ad_func`** | 否 |
| **反向引擎** | `paddle/fluid/eager/backward.h` | `egr::Backward()` / `egr::Grad()` | **否** |

`egr::Backward()` 的签名完全是 C++ 类型,没有一个 `PyObject*`:

```cpp
// paddle/fluid/eager/backward.h
TEST_API void Backward(const std::vector<paddle::Tensor>& tensors,
                       const std::vector<paddle::Tensor>& grad_tensors,
                       bool retain_graph = false, bool create_graph = false, ...);
TEST_API std::vector<paddle::Tensor> Grad(...);
```

**铁证**:`test/cpp/eager/task_tests/backward_test.cc` 就是纯 C++ 跑完整 forward + backward + 校验梯度。
同目录还有 `forward_autograd_test / fwd_bwd_joint_test / grad_test / hook_test / cross_batch_accumulation_test`,
即**"C++ 里跑完整训练"这条路径 Paddle 官方已经在 CI 里持续验证**。

> 结论:Lua 绑定 **不需要重写任何引擎**,只需要重写"壳"。

### 1.2 Python 依赖已经被 CMake 隔离好了

```cmake
# cmake/configure.cmake:15
if(NOT WITH_PYTHON)
  add_definitions(-DPADDLE_NO_PYTHON)
endif()
```

整个 `paddle/fluid/eager/` 里只有 **5 个文件**碰 Python:

| 文件 | 用途 | 处理 |
|---|---|---|
| `tensor_wrapper.h` | 已用 `#ifndef PADDLE_NO_PYTHON` 包好(6 处) | 无需改 |
| `hooks.h` | Python hook 回调 | 需要 Lua 对应物 |
| `api/utils/global_utils.h` | 全局状态 | 需检查 |
| `pylayer/py_layer_node.{h,cc}` | `paddle.autograd.PyLayer` | **需写 LuaLayer 替代** |

且 eager 栈只在 `WITH_PYTHON=OFF AND ON_INFER=ON` 时才被裁掉:

```cmake
# paddle/fluid/eager/CMakeLists.txt
if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  add_subdirectory(accumulation)   # 梯度累加节点
  add_subdirectory(auto_code_generator)
```

**即 `WITH_PYTHON=OFF + ON_INFER=OFF` 理论上能得到"带完整训练能力、不含 Python"的库。**
但这个组合大概率**从来没人走过**(官方只有 WITH_PYTHON=ON 训练 / ON_INFER=ON 推理两条路),
**必须实测,这是 M0 的头号任务。**

> 注:构建期仍需 Python(`eager_gen.py` 等生成器)。这没关系 —— 要消灭的是**运行时** Python。

### 1.3 "壳"到底有多大

pybind 层总计 **273,909 行**,但绝大部分与我们无关:

| 文件 | 行数 | 作用 | Lua 版需要 |
|---|---|---|---|
| `eager_method.cc` | 4260 | **84 个 Tensor 方法** | 必需 |
| `eager_utils.cc` | 3686 | **42 个 `CastPyArg2*` 类型转换** | 必需(→ CastLuaArg2*) |
| `eager.cc` | 1572 | Tensor 类型对象/构造 | 必需 |
| `eager_properties.cc` | 1090 | `.shape/.grad/.stop_gradient` | 必需 |
| `arg_pre_process.cc` + `args_mapper.cc` | 1195 | 重载分派/参数规范化 | 必需 |
| `eager_op_function.cc` | 生成 | 786 算子绑定 | 必需,但**自动生成** |
| 其余 ~26 万行 | | distributed / pir / static / inference / cost model / auto_parallel | MVP 全砍 |

**核心必需手写壳 ≈ 12k-15k 行**,不是 27 万行。

### 1.4 Python 侧 57 万行,必须移植的是哪 10 万行

`python/paddle/` = 1228 个文件 / **574,104 行**:

| 目录 | 行数 | MVP 处置 |
|---|---|---|
| `distributed/` | 182,906 | 砍 |
| `incubate/` | 57,518 | 砍 |
| **`nn/`** | **55,876** | **必须移植**(Layer 体系 + 各种层) |
| `jit/` | 42,038 | 砍(动转静依赖 Python AST,Lua 侧无意义) |
| **`tensor/`** | **36,151** | **必须移植**(大量 API 是 Python 侧组合出来的) |
| `base/` | 31,892 | 部分 |
| `static/` | 25,445 | 砍 |
| **`optimizer/`** | **13,602** | 必须移植 |
| `io/` | 4,219 | 必需(DataLoader) |
| `autograd/` | 3,991 | 必需 |

**必须提供 Lua 等价物的 ≈ 110k 行 Python。MVP 只需其中 15-20%。**
**这是本项目最大的一块,占总工作量 60-70%。**

---

## 2. 好消息:API 对齐有现成的机器可读元数据

Paddle 自己就是靠 yaml + 代码生成器造 Python API 的,我们复用同一套元数据:

| 文件 | 内容 |
|---|---|
| `paddle/phi/ops/yaml/ops.yaml` | 529 个 op,完整签名/类型/默认值/inplace |
| `paddle/phi/ops/yaml/backward.yaml` | 396 个 backward op |
| `paddle/phi/ops/yaml/fused_ops.yaml` | 86 |
| `paddle/phi/ops/yaml/sparse_ops.yaml` | 51 |
| **`paddle/phi/ops/yaml/python_api_info.yaml`** | **138 条 op → Python API 名映射,含别名** |

`python_api_info.yaml` 尤其关键,直接给出 API 对齐所需的映射:

```yaml
- op : abs
  name : [paddle.abs, paddle.Tensor.abs, paddle.absolute, paddle.Tensor.absolute]
  args_alias :
    use_default_mapping : True
```

现成生成器(可照抄结构):

| 生成器 | 行数 | 产出 |
|---|---|---|
| `eager_gen.py` | 3918 | 786 个 `*_ad_func` + 反向节点 |
| **`python_c_gen.py`** | **1103** | **整个 Python 算子绑定层** |
| `codegen_utils.py` | 664 | 共用工具(可直接 import) |
| `monkey_patch_gen.py` | 172 | Tensor 方法挂载 |

> **`python_c_gen.py` 只有 1103 行就生成了 786 个算子的完整 Python 绑定。**
> 对应的 `lua_c_gen.py`(预计 1500-2500 行)就能生成整个 Lua 算子层。
> **这是整个项目性价比最高的一步。**

---

## 3. 架构评估:关于 "LuaJIT 走 FFI + 5.x 走 sol2"

方向是对的,但建议改成 **统一 C ABI 内核 + 两种前端**,而不是两套并行实现。

### 3.1 为什么不要两套并行实现

1. **类型系统会分裂**:sol2 绑的是 C++ 类型(`paddle::Tensor`),FFI 绑的是 C 句柄。
   两套 API 实现 → 维护量 x2,行为漂移(默认值、重载分派、错误信息)几乎必然发生。
2. **编译会爆炸**:sol2 是重模板库。若它直接 include Paddle C++ 头
   (`phi/api/include/tensor.h` 会拉进 Eigen/gflags/glog/CUDA 头),
   **每个 Lua 版本都要全量重编一遍重型 TU**。5 个版本 = 5 次全量编译,CI 撑不住。

### 3.2 建议的分层

```
       +---------------------------------------------+
       | Lua 侧纯 Lua 代码 (nn/optimizer/tensor API) |  <- 一份,5.1 兼容子集
       | 必须对所有引擎通用                          |
       +----------------------+----------------------+
                              |
        +---------------------+---------------------+
        |                                           |
  +-----v---------+                    +------------v-------------+
  | LuaJIT        |                    | paddle_lua51..55.dll     |  <- 薄 shim,每版一个
  | ffi.cdef      |                    | 裸 Lua C API 或 sol2      |     只 include paddle_c.h
  | 零 C 胶水     |                    | (~几百 KB, 秒级编译)      |     + lua.h
  +-----+---------+                    +------------+-------------+
        |                                           |
        +---------------------+---------------------+
                              |
        +---------------------v---------------------+
        | libpaddle_lua_c.dll  (纯 C ABI)           |  <- 编译一次,与 Lua 版本无关
        | 内部 C++,链接 phi.dll                     |
        | pd_tensor_* / pd_ops_* / pd_backward      |
        +---------------------+---------------------+
                              |
        +---------------------v---------------------+
        | Paddle (WITH_PYTHON=OFF)                  |
        | phi.dll + common.dll                      |
        +-------------------------------------------+
```

关键点:**薄 shim 只依赖 `paddle_c.h`(纯 C 头)+ `lua.h`,不碰任何 Paddle C++ 头。**
这样 5 个 Lua 版本的编译成本几乎为零,而且 shim 本身是**生成**出来的。

### 3.3 关于 sol2 的具体评估

sol2 的价值是自动抹平 5.1-5.4 的 C API 差异(`lua_objlen`/`lua_rawlen`、`LUA_GLOBALSINDEX`、
`luaL_register`/`luaL_setfuncs`、`lua_setuservalue` 等)。

但如果 shim 是**代码生成**出来的,这些差异用一个 `lua_compat.h`(约 200-300 行宏)就能完全覆盖,
不必引入 sol2 的模板重量和编译时间。

**建议:先做 `lua_compat.h` + 裸 Lua C API 的生成路径,sol2 作为可选后端。**
若后来发现手写 usertype/元表太痛苦再切 sol2 —— 因为 shim 是生成的,换后端成本很低。

**Lua 5.5 是必须实测的未知项。** Lua 5.5(2025 年发布)相对 5.4 有变更
(整数/浮点语义、`lua_resetthread` 签名、全局变量处理等)。
**sol2 是否已支持 5.5 我无法从本地代码确认,不要在计划里假定它开箱即用。**
这正是"不绑定 sol2"的另一个理由 —— 自己的 compat.h 可以立刻适配 5.5。

### 3.4 LuaJIT 的两个硬约束(现在就必须知道)

**1. LuaJIT 只有 Lua 5.1 语法**(+ 少量 5.2 扩展)。
而 1.4 节那 10 万行纯 Lua 代码要在所有引擎上跑 → **必须写成 Lua 5.1 兼容子集**。要避开:

- `goto` / `::label::`(5.2+)
- `//` 整除、`&|~<<>>` 位运算符(5.3+;LuaJIT 用 `bit` 库)
- **integer/float 子类型区分(5.3+)—— 这个对 Tensor shape/index 影响很大,必须统一约定**
- `__gc` on tables(5.1 只支持 userdata)
- `__len` on tables(5.1 不支持)
- `table.unpack` vs `unpack`、`table.pack`
- `_ENV`(5.2+)

→ 建议写一个 `compat.lua` + 用 luacheck 强制约束。

> ⚠️ **已被 D25 修正:** 不自写 `compat.lua`,用 vendored 的 `pl.compat`。
> 强制约束改为 `tools/lint_51.lua`(用 5.1 自身的 `loadstring` 去 load 每个源文件)。
> 见 `plan/foundations.md` §1.7 与 `process/conventions.md` §1.2。

**2. LuaJIT 的 FFI callback 很贵且有数量上限**(默认约 65536 个 slot,回收不佳)。
若计划让 Lua 函数作为 autograd hook / 自定义算子回调,这是真实的坑。
→ 对策:回调走"注册到 C 侧的 ID + 单一 trampoline",而不是每个 Lua 函数一个 FFI callback。

---

## 4. 最大的运行时风险:GC 与显存生命周期

**这是本项目我最担心的一点,建议在 M0 就设计好。**

- `paddle::Tensor` 是 `shared_ptr` 语义,Lua 侧持有句柄。
- LuaJIT 可 `ffi.gc(handle, C.pd_tensor_free)`,Lua 5.x 用 userdata `__gc`。
- **问题**:Lua 的 GC 是纯 mark-and-sweep,**没有引用计数**。
  一个 Tensor 在 Lua 侧只占 16 字节句柄,背后却是 2GB 显存。
  Lua GC 完全感知不到显存压力 → **训练时会 CUDA OOM,而 Lua 觉得内存很宽裕。**
- Python 有引用计数,`del x` 立即释放,**这个问题在 Python 里被大幅缓解了。Lua 没有这个缓解。**

必须提供的对策(建议全做):

1. 显式 `t:free()` / `t:release()`
2. scope 机制:除 `paddle.no_grad(fn)` 外再加 `paddle.scope(fn)`,自动回收作用域内张量
3. **C 侧分配失败时回调 Lua 做 `collectgarbage("collect")` 后重试**(关键兜底)
4. C 侧维护"已分配字节数",超阈值主动触发 Lua GC(`lua_gc(L, LUA_GCSTEP, n)`),
   把显存压力翻译成 Lua GC 压力

---

## 5. 错误处理:C++ 异常绝不能穿过 FFI 边界

- Paddle 用 C++ 异常:`PADDLE_ENFORCE_*` → `phi::enforce::EnforceNotMet`
- **C ABI 边界必须 `catch(...)` 全捕获**,转成 error code + 线程局部 message buffer
- 让 C++ 异常穿过 LuaJIT FFI 边界是**未定义行为**(LuaJIT 的 unwinding 与 MSVC SEH 混合会直接崩)
- Lua 侧统一在 wrapper 里检查返回码并 `error()`

约定:

```c
int pd_add(pd_tensor x, pd_tensor y, pd_tensor* out);  /* 0 = OK */
const char* pd_last_error(void);                       /* 线程局部 */
```

---

## 6. 数据交换

### 6.1 DLPack —— 需要自己补,但很便宜

**现状**:DLPack 只在 pybind 层实现(`paddle/fluid/pybind/tensor.cc`、`place.cc`),
**`phi/api` 没有 C++ 级的 DLPack 导出**。

→ 需要自己写,但工作量小:DLPack 是纯 C 结构体标准,pybind 里已有完整转换代码可直接搬。
有了它,LuaJIT FFI 就能零拷贝对接任何支持 DLPack 的库。
**建议 M0 就做,并考虑 PR 回上游(对 Paddle 自身也有价值)。**

### 6.2 Lua 没有 numpy

需要三层:

1. `Tensor.from_table(t, shape)` / `t:to_table()` —— 通用但慢
2. `t:data_ptr()` + `ffi.cast` —— 零拷贝,**仅 LuaJIT**
3. 轻量 strided buffer 类型(暂称 `paddle.array`)—— 给 5.1-5.5 用,
   底层是 C 分配的连续内存 + userdata,配 `__index/__newindex`

---

## 7. 模型互操作(决定这个项目有没有人用)

### 7.1 `paddle.save` 用的是 Python pickle —— Lua 读不了

`python/paddle/framework/io.py:21` `import pickle`。写 pickle 解析器不现实。

### 7.2 好消息:Paddle 已支持 safetensors

```python
# python/paddle/framework/io.py:375, 968, 1002
supported_configs = ['use_binary_format', 'pickle_protocol', 'safetensors']
from safetensors.paddle import save_file
```

**safetensors 格式极简:8 字节 header 长度 + JSON header + 裸 bytes。**
**Lua 实现只要约 200 行。这就是权重互操作的正确路径。**

用户流程:Python 侧 `paddle.save(sd, path, safetensors=True)` → Lua 侧直接加载。

### 7.3 推理场景几乎零成本

`paddle/fluid/jit/` 提供了 C++ 级模型加载:

```cpp
// paddle/fluid/jit/serializer.h:78
Layer Load(const std::string& path, const phi::Place& place);
// paddle/fluid/jit/layer.h
std::vector<Tensor> Layer::forward(const std::vector<Tensor>& inputs);
```

即 `paddle.jit.save` 产出的模型**可被纯 C++ 直接加载运行**。
包一层就有完整的 Lua 推理能力 —— **可作为独立的、极早期就能发布的产品**。

---

## 8. 构建与分发:Windows 上的现实

实测当前构建产物(`build/paddle_inference_install_dir/paddle/lib/`,CPU 构建):

| 文件 | 大小 |
|---|---|
| `phi.dll` | **363,866,112 B (347 MB)** |
| `paddle_inference.dll` | 95,339,520 B (91 MB) |
| `common.dll` | 807,936 B (0.8 MB) |
| `phi.exp` | 3,722,304 B |

**`local paddle = require "paddle"` 背后是几百 MB 的 DLL,CUDA 版会到 GB 级。**

影响:

- **LuaRocks 直接分发不现实**,必须自建 binary release + 下载器
  (`luarocks install paddle` 触发下载预编译包)
- 需要裁剪版:Paddle 支持按 op 裁剪推理库,可做 "paddle-lua-lite"

**另一个 Windows 特有坑:kernel 注册的静态链接丢失。**
测试里到处是 `PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT)`,
因为静态链接时未被引用的 kernel 注册全局对象会被 linker 丢掉。
走 `phi.dll` 动态链接可规避,但 **Windows DLL 导出符号上限 65535**,
而 `phi.exp` 已经 3.7 MB —— 加符号时要小心。

---

## 9. 工作量估算

### M0 · 可行性验证(1-2 周)—— **信息量最大的一步**

1. 验证 `WITH_PYTHON=OFF + ON_INFER=OFF` 能否构建出带 eager 训练能力的库
   **(头号任务,这条路径大概率没人走过,预期会踩编译错误)**
2. 手写最小 C ABI:`pd_tensor_create/free/full/add/matmul/backward/grad_of/data_ptr`
3. LuaJIT `ffi.cdef` 跑通:
   ```lua
   local x = paddle.full({2,2}, 1.0); x.stop_gradient = false
   local y = paddle.matmul(x, x):sum()
   y:backward()
   print(x.grad)
   ```
4. 验证 GC / 异常 / 显存回收三个设计

> **这一步能回答 80% 的技术风险。做完 M0 之前不要写任何生成器。**

### M1 · 算子层自动生成(1-2 个月)

- `lua_c_gen.py`(1.5k-2.5k 行):从 `ops.yaml` + `python_api_info.yaml`
  生成 786 算子的 C ABI + Lua wrapper
- 类型转换层(对应 42 个 `CastPyArg2*`)
- Tensor 84 个方法 + 属性
- 参数解析:Lua table 当 kwargs、默认值、重载分派(参考 `arg_pre_process.cc` 738 行)
- `lua_compat.h` + 5.1/5.2/5.3/5.4/5.5 shim 生成
- 产出:`paddle.add / paddle.matmul / t:reshape()` 全部可用

### M2 · Python 层等价物移植(3-6 个月+,最大的一块)

| 模块 | Python 行数 | Lua MVP 估计 |
|---|---|---|
| `nn.Layer` 体系 + 常用层 | 55,876 | 10k-15k |
| `tensor/` 组合式 API | 36,151 | 10k-15k |
| `optimizer/` | 13,602 | 3k-5k |
| `io/` DataLoader | 4,219 | 2k-4k |
| `autograd/` | 3,991 | 1k-2k |

**DataLoader 是特殊难点**:Lua 没有 multiprocessing,没有标准线程模型。
需重新设计(LuaJIT + pthread / 独立进程 + IPC / C 侧线程池 + 回调)。
巧的是当前 Paddle 分支正在做 Windows DataLoader 多进程 —— 那块经验能直接复用设计思路。

### 总量粗估

| 项 | 量 |
|---|---|
| C/C++ 手写 | 8k-12k 行 |
| 代码生成器 | 1.5k-2.5k 行 |
| 生成代码 | ~100k 行(不计人力) |
| Lua 手写 | 30k-50k 行(MVP 约 15k) |
| **MVP(能训 MNIST/ResNet)** | **4-6 人月** |
| **"API 基本对齐 Paddle"** | **1.5-2 人年** |

---

## 10. 建议的取舍

1. **不要一开始就追"API 基本对齐"。**
   先对齐最小闭环:`paddle.Tensor` + 算子 + autograd + `nn.Layer` + `optimizer`。
   `distributed/jit/static/incubate` 全砍,这四块占 Python 侧 308k 行(54%)。

2. **MVP 只做 LuaJIT。**
   LuaJIT 走 FFI 几乎零 C 胶水,迭代最快,能最快验证设计。
   C ABI 稳定后,5.1-5.5 的 shim 是纯生成品,补上很便宜。
   **反过来先做 sol2 会过早锁死类型系统设计。**

3. **上游改动尽量零侵入。** 目前看只需两处:
   - 补 C++ 级 DLPack 导出(可 PR 回上游)
   - 修 `WITH_PYTHON=OFF + ON_INFER=OFF` 的构建问题(同样可 PR)

   其余全在 `paddle-lua` 自己的仓库,以 submodule + patch 方式引用 Paddle。

4. **考虑先发布"Lua 推理库"作为早期产品。**
   基于 `jit::Layer`(7.3 节),工作量只有完整方案的 5%,但立刻可用、能收集反馈。

---

## 11. 下一步

**最有价值的单个动作,优先级远超其他所有事:**

```bash
cmake -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=OFF -DWITH_TESTING=ON ..
# 看 eager 栈能否编出来,test/cpp/eager 的测试能否通过
```

**这一个实验的信息量超过本报告其余所有调研。**
若通过,项目风险从"未知"降到"纯工程量";若不通过,先解决它再谈别的。

---

## 附:本报告的事实依据(均来自 `E:/code/paddle` 实际代码)

| 事实 | 出处 |
|---|---|
| `egr::Backward/Grad` 无 Python 依赖 | `paddle/fluid/eager/backward.h` |
| C++ 可跑完整训练 | `test/cpp/eager/task_tests/backward_test.cc` |
| `PADDLE_NO_PYTHON` 宏 | `cmake/configure.cmake:15` |
| eager 在 WITH_PYTHON=OFF 时仍构建 | `paddle/fluid/eager/CMakeLists.txt` |
| 786 个 `*_ad_func` | `.../forwards/dygraph_functions.h`(1459 行) |
| 529+396+86+51 个 op | `paddle/phi/ops/yaml/*.yaml` |
| `python_c_gen.py` 1103 行生成整个绑定层 | `paddle/fluid/eager/auto_code_generator/generator/` |
| 84 个 Tensor 方法 / 42 个类型转换 | `pybind/eager_method.cc` / `pybind/eager_utils.h` |
| Python 侧 574,104 行 / 1228 文件 | `python/paddle/` |
| DLPack 仅在 pybind 层 | `pybind/tensor.cc`;`phi/api` 无 |
| safetensors 已支持 | `python/paddle/framework/io.py:375,968,1002` |
| `jit::Layer` C++ 加载模型 | `paddle/fluid/jit/layer.h`,`serializer.h:78` |
| phi.dll = 347 MB | `build/paddle_inference_install_dir/paddle/lib/` |
| 静态链接 kernel 丢失问题 | `PD_DECLARE_KERNEL` 在测试中的广泛使用 |
