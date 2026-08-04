# 可复用资产清单 · "Paddle 的 C++ 执行器到底是干啥的"

> 起因:「我记得 Paddle 有个 C++ 执行器,那个是干啥的?我们要尽可能复用一切能利用的东西。」
>
> **简短答案:Paddle 里不止一个"C++ 执行器",而是五套,职责完全不同。**
> **其中三套对我们有用,一套决定了 §2.1.3「自研 Lua 静态图」的工作量能砍掉一半以上。**

---

## 1. 五套"执行器",先分清楚

| # | 名字 | 头文件 | 干什么 | 对我们 |
|---|---|---|---|---|
| 1 | `framework::Executor` | `framework/executor.h` | 老 fluid 执行器,顺序跑 `ProgramDesc` | ❌ 遗留,不碰 |
| 2 | `framework::NaiveExecutor` | `framework/naive_executor.h` | 精简版,不建 scope、不做 GC,推理内部用 | ⬜ 间接用到 |
| 3 | **`StandaloneExecutor` + `InterpreterCore`** | `framework/new_executor/` | **现代执行器**:依赖分析、多流、异步调度、**自带显存 GC** | ✅ **核心复用目标** |
| 4 | **`inference::Predictor`** | `inference/api/paddle_inference_api.h` | 完整推理栈:IR pass、TensorRT/oneDNN、内存复用 | ✅ 推理场景直接用 |
| 5 | **`jit::Layer` + `BaseEngine`** | `jit/layer.h`、`jit/engine/` | `paddle.jit.save` 产物的 C++ 加载/执行门面 | ✅ **最省事的入口** |

**用户口中的"C++ 执行器"多半指 4(推理 Predictor)或 3(新执行器)。
但对我们价值最大的其实是 3,而最容易上手的是 5。**

---

## 2. 第 3 套:新执行器(`InterpreterCore`)—— 最大的一块可复用资产

```
paddle/fluid/framework/new_executor/
├── standalone_executor.{h,cc}        对外门面
├── interpretercore.{h,cc}            调度核心
├── program_interpreter.{h,cc}        后端 A:老 ProgramDesc
├── pir_interpreter.{h,cc}            后端 B:PIR(新 IR)
├── garbage_collector/                ← 5 种显存回收策略
│   ├── fast_garbage_collector
│   ├── event_garbage_collector       CUDA event 驱动
│   ├── async_fast_garbage_collector
│   └── no_event_garbage_collector
├── workqueue/                        ← 无锁线程池
│   ├── nonblocking_threadpool.h
│   ├── run_queue.h
│   └── events_waiter.{h,cc}
└── instruction/  interpreter/  pir_adaptor/
```

它做的事:

1. 把 Program 编译成 `Instruction` 序列
2. **静态依赖分析** → 能并行的 op 并行发,不能的排队
3. **多 stream 调度 + CUDA event 同步**
4. **中间变量的生命周期分析与即时回收**(`garbage_collector/`)
5. 自带**无锁线程池**(`workqueue/`)

对外接口很干净:

```cpp
// framework/new_executor/standalone_executor.h:35
StandaloneExecutor(const Place& place, const interpreter::Plan& plan, Scope* scope);
FetchList Run(const std::vector<std::string>& feed_names, bool profile = false);
```

**已核对:`new_executor/CMakeLists.txt` 和 `jit/CMakeLists.txt` 里都没有
`WITH_PYTHON` / `ON_INFER` 守卫 —— 它们无条件参与构建。**

### 2.1 ⚠️ 这直接改写了 §2.1.3 的结论

`plan/overview.md` §2.1.3 我把"自研 Lua 静态图"评为 **A2 里最重的一档**,理由是要自己写
tracer、IR、拓扑排序、常量折叠、序列化、控制流。

**其中有一半根本不用写 —— 因为执行引擎已经在这儿了。**

| 静态图需要的能力 | 原评估 | 实际 |
|---|---|---|
| 算子调用录制(tracing) | ❌ 自己写 | ❌ 确实要自己写 |
| 图 IR 表示 | ❌ 自己定 | ✅ **用 `pir::Program`** |
| 拓扑排序 / 依赖分析 | ⚠️ 自己实现 | ✅ **InterpreterCore 内建** |
| 调度 / 多流 / 并行 | 没敢想 | ✅ **InterpreterCore 内建** |
| 中间变量显存回收 | 没敢想 | ✅ **`garbage_collector/` 内建** |
| 图优化 pass | ⚠️ 自己写 | ✅ **PIR pass 体系现成** |
| 序列化 | ❌ 自定义格式 | ✅ **PIR 有自己的序列化** |
| 控制流捕获 | ❌ 最难 | ❌ 仍然最难,且仍需 Lua 源码级变换 |

**修正:工作量从"独立项目体量"降到"写一个 tracer + 组装 pir::Program"。**
但**控制流那条没变** —— 它难在前端(要分析 Lua 源码),不在后端。

### 2.2 更关键的:静态图 = GC 风险的解药

这一条 `research/gc.md` 里没有,但它可能是本次调研最有价值的一个连带发现。

`research/gc.md` 全篇的核心风险是:**Lua 的 mark-sweep GC 管不住 off-heap 显存,
中间张量要等到不确定的时刻才释放。**

**但在 InterpreterCore 里跑的图,中间变量从来不是 Lua 对象。**

```
动态图:  每个中间张量 → 一个 Lua userdata → 生死由 Lua GC 决定  ← GC.md 的风险
静态图:  中间变量存在 framework::Scope 里
         → 由 InterpreterCoreGarbageCollector 按依赖分析即时回收
         → Lua 完全看不见它们,GC 压力为零
```

`garbage_collector.h:37` 的接口就是干这个的:

```cpp
virtual void Add(Variable* var, const InstructionBase* instruction) = 0;
```

**意味着:一旦静态图路径打通,它同时是"性能特性"和"GC 风险的兜底方案"。**
这把 §2.1.3 的优先级论证改变了 —— 它不再只是"锦上添花的加速",
而是 `research/gc.md` 那套九层缓解机制之外的**第十层,而且是最彻底的一层**。

> 但要诚实:这不改变 M3 的排期。GC 九层机制成本 ~130 行,静态图是人月级。
> **先做九层,静态图作为长期解。**

---

## 3. 第 5 套:`jit::Layer` —— 投入产出比最高的一块

```cpp
// paddle/fluid/jit/layer.h:43
std::vector<Tensor> forward(const std::vector<Tensor>& inputs);
void to(const phi::Place& place);
std::vector<std::string> FunctionNames() const;
std::shared_ptr<Layer> Clone(void* stream = nullptr);

// paddle/fluid/jit/serializer.h:78
Layer Load(const std::string& path, const phi::Place& place);
```

底下的引擎接口更干净:

```cpp
// paddle/fluid/jit/engine/base_engine.h
class BaseEngine {
  virtual std::vector<Tensor> operator()(const std::vector<Tensor>&) = 0;
  virtual std::unique_ptr<BaseEngine> Clone(void* stream) = 0;
};
```

三个具体引擎:`InterpreterEngine`(走新执行器)、`PirInterpreterEngine`、`PredictorEngine`(走推理栈)。

**为什么这是投入产出比最高的:**

| | |
|---|---|
| **接口货币就是 `paddle::Tensor`** | 和我们绑定层的通用类型完全一致,**零转换** |
| **绑定量极小** | `Load` + `forward` + `to` + `FunctionNames`,**几十行 sol2** |
| **立刻可用的能力** | 用户在 Python 侧 `paddle.jit.save`,Lua 侧直接加载推理。**不需要我们实现任何 nn 层** |
| **`Clone(stream)` 是多线程友好的** | 每个 lane 一份 Layer 克隆,天然配合 Lanes |

**建议:把它从 M2 提到 M1 的可选项。**
理由:它能在 M1 阶段就产出一个**完整可用的产品形态**——
"Python 训练 / Lua 推理"。这条路径不依赖 `paddle.nn` 移植完成,
是整个项目里**最早能交付真实价值**的一块。

---

## 4. 被忽略的桥梁:`run_program_ad_func`

这是我这次翻代码最意外的发现。

```cpp
// paddle/fluid/eager/to_static/run_program_func.h:23
namespace egr::to_static {
std::vector<paddle::Tensor> run_program_ad_func(
    const std::vector<paddle::Tensor>& x,
    const std::vector<paddle::Tensor>& params,
    std::vector<paddle::framework::Scope*>& step_scope,
    const paddle::framework::AttributeMap& prog_attrs,
    const paddle::framework::AttributeMap& cuda_graph_attrs);
}
```

**注意函数名后缀 `_ad_func` —— 它是一个参与 eager 自动微分的算子。**

它的作用:**把一整段静态图当作动态图里的一个"算子"来执行,
前向走 InterpreterCore,反向自动挂上 GradNode 跑反向 Program。**
这正是 `paddle.jit.to_static` 在动态图里的落地方式。

而 pybind 只是它的薄封装:

```
paddle/fluid/pybind/eager_custom_python_api.h:136
    auto out = egr::to_static::run_program_ad_func(...)
```

**对我们的意义:**

如果 Lua 侧的静态图能产出 `pir::Program`,那么
**它可以直接嵌进动态图训练循环,并且反向传播是免费的** ——
我们一行反向代码都不用写。

```
Lua tracer → pir::Program → run_program_ad_func → 前向由 InterpreterCore 跑
                                                → 反向由 Paddle 自动挂 GradNode
```

这把 §2.1.3 里"自研静态图"最令人生畏的部分(反向图构建)整个消掉了。

> ⚠️ 但要验证:`AttributeMap` 的构造在 C++ 侧是否好组装。
> Python 侧是通过 pybind 的 dict→AttributeMap 转换喂进去的,
> 我们要在 C ABI 中间层里手工构造。**列为 M0 验证项。**

---

## 5. 其他可复用件(逐条核对过)

| 资产 | 位置 | Python-free | 用途 | 优先级 |
|---|---|:-:|---|---|
| `egr::Backward` / `egr::Grad` | `eager/backward.h` | ✅ | 自动微分 | **M1 必须** |
| `paddle::Tensor` | `phi/api/include/tensor.h` | ✅ | 通用货币类型 | **M1 必须** |
| `RegisterOOMCallback` | `phi/core/memory/allocation/retry_allocator.h:19` | ✅ | GC-on-OOM(`research/gc.md` §3.1) | **M1** |
| 分配器全家桶 | `phi/core/memory/allocation/` | ✅ | auto_growth / VMM / 缓存 | 免费获得 |
| **`SaveTensor` / `LoadTensor`** | `framework/io/save_load_tensor.h` | ✅ | **原生张量存取,不经 pickle** | **M1,见 5.1** |
| `jit::Layer::Load` | `jit/serializer.h:78` | ✅ | 加载 `paddle.jit.save` 产物 | **M1 可选项** |
| `InterpreterCore` | `framework/new_executor/` | ✅ | 静态图执行 | M3 |
| `run_program_ad_func` | `eager/to_static/run_program_func.h` | ✅ | 静态图嵌入动态图 + 免费反向 | M3 |
| `inference::Predictor` | `inference/api/paddle_inference_api.h` | ✅ | 纯推理部署 | M2/M3 |
| `LoadOpMetaInfoAndRegisterOp` | `framework/custom_operator.h:316` | ✅ | 自定义算子加载(`plan/overview.md` §2.1.2) | M3 |
| `framework::Scope` | `framework/scope.h` | ✅ | 静态图变量容器 | 随 3/4 一起 |
| 无锁线程池 | `new_executor/workqueue/` | ✅ | 见 5.2 |  ⬜ |
| `strided_slice` + stride kernels | `phi/kernels/stride/` | ✅ | 字符串索引后端(`plan/overview.md` §6.1.4) | **M1** |
| `FLAGS_*` 全局开关 | `common/flags.h` | ✅ | 运行期调参 | M1 |

### 5.1 `SaveTensor`/`LoadTensor` —— 一条被漏掉的捷径

```cpp
// paddle/fluid/framework/io/save_load_tensor.h
void SaveTensor(const DenseTensor& x, const std::string& file_path, bool overwrite);
void LoadTensor(const std::string& file_path, DenseTensor* out);
```

`research/architecture.md` §C 花了很大篇幅论证要写**纯 Lua pickle 解析器**(600-1000 行)。
那个结论不变 —— 它解决的是**读 Python 存的 `.pdparams`**,是互操作刚需。

**但"Lua 自己存、Lua 自己读"这个场景,不需要 pickle。**
直接用这两个 C++ 函数,**绑定量约 20 行,零解析代码。**

建议的分工:

```
paddle.load(path)          → 嗅探格式:pickle → 纯 Lua 解析器
                                        原生 → LoadTensor
paddle.save(obj, path)     → 默认写原生格式(快、简单、无 pickle 依赖)
paddle.save{obj, path, format="pickle"}   → 写 Python 可读格式(混合表,§argrule 2.5⑧)
```

**净收益:M1 的存档/续训功能可以在 pickle 解析器写完之前就跑起来。**
这解耦了 M1 验收("训练→保存→加载→推理"闭环)对 pickle 的依赖 ——
pickle 从 **M1 阻塞项** 降级为 **M1 加分项**。这是个不小的排期改善。

### 5.2 `workqueue/` 线程池 —— 看着诱人,但别用

`new_executor/workqueue/nonblocking_threadpool.h` 是个成熟的无锁线程池,
第一反应是"DataLoader 可以用它"。**不行,而且理由值得记下来:**

| | |
|---|---|
| 它是**算子调度**用的 | 任务是 `std::function<void()>`,不涉及 `lua_State` |
| DataLoader worker 需要的是**独立 Lua 解释器** | 这是 Lanes deep userdata 要解决的问题,不是线程池问题 |
| 混用两套线程模型会打架 | `research/dataloader.md` §9 已定 Lanes;再引入一套是自找麻烦 |

**结论:`research/dataloader.md` §9 的 Lanes 方案不变。** 记在这里是为了防止将来有人
(包括我自己)看到这个线程池又想改方案。

---

## 6. 复用带来的净变化

| 项 | 原结论 | 现结论 |
|---|---|---|
| §2.1.3 自研静态图 | A2 最重档,"独立项目体量" | **只需 tracer + 组装 `pir::Program`**;IR/调度/GC/pass/序列化全部复用 |
| 静态图的反向 | 要自己建反向图 | **`run_program_ad_func` 免费提供** |
| 静态图的定位 | 性能特性,M3 | **兼任 GC 风险的第十层解法**(最彻底那层) |
| `jit::Layer` | M2 顺手绑一下 | **提到 M1 可选项**,几十行换一个完整产品形态 |
| `paddle.save/load` | 阻塞于 pickle 解析器 | **原生格式 20 行先行**,pickle 降级为加分项 |
| DataLoader 线程 | Lanes | **不变**(明确排除 `workqueue/`) |
| 控制流捕获 | 最难 | **不变,仍然最难** —— 难点在 Lua 前端,复用救不了 |

---

## 7. 新增 M0 验证项

| # | 验证 | 影响 | 预估 |
|---|---|---|---|
| 16 | `WITH_PYTHON=OFF` 下 `new_executor` / `jit` 是否真的编出来并可链接 | 决定第 3/5 套能否用 | 随验证项 1 一起 |
| 17 | `jit::Layer Load()` 能否加载一个 Python 侧 `paddle.jit.save` 的产物并 `forward` | 决定"Python 训练 / Lua 推理"能否提前交付 | 1 天 |
| 18 | C++ 侧手工构造 `AttributeMap` 喂 `run_program_ad_func` 的难度 | 决定静态图嵌动态图路径 | 1 天 |
| 19 | `SaveTensor`/`LoadTensor` 的文件格式是否稳定、跨版本可读 | 决定能否作为默认存档格式 | 0.5 天 |

---

## 8. 一句话总结

**"C++ 执行器"不是一个东西,是五个。**
对我们最有用的排序是:

```
jit::Layer(最省事,几十行换一个产品形态)
  > InterpreterCore(把自研静态图的工作量砍掉一半以上)
  > run_program_ad_func(把静态图的反向变成免费)
  > inference::Predictor(纯推理部署)
  > NaiveExecutor / 老 Executor(不碰)
```

**但复用救不了两件事:控制流捕获(难在 Lua 前端)、
以及 `paddle.nn` 那 6k 行手写层(难在没有可复用的等价物)。
这两块仍然是项目的真实成本重心。**
