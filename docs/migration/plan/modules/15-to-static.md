# P15 · 静态图 trace 模式

| | |
|---|---|
| 阶段 | P15 |
| 类别 | 复用 + 手写 |
| 开工条件 | P9 + P10 完工 |
| 预估 | 4 周(**复用之后已砍半**) |

---

## 1. 做什么 / 不做什么

**做:** `paddle.jit.trace(net, example_input)` —— 跑一遍前向,录下算子序列,
组装成 `pir::Program`,交给 Paddle 的执行器跑。

**不做:** 不处理控制流(`if` / `while`)。那是 P18 的 script 模式。
trace 模式对控制流的处理是"记录本次走过的分支",这是它的**已知且可接受的局限** ——
PyTorch 的 `torch.jit.trace` 也是这样。

---

## 2. 上游有什么可以用

**这一阶段是全项目复用率最高的地方,原先"自研静态图 = 独立项目体量"的评估已被推翻**
(`process/decisions.md` R11)。

| 出处 | 内容 | 省掉什么 |
|---|---|---|
| `paddle/fluid/framework/new_executor/standalone_executor.h:35` | `StandaloneExecutor(const Place&, const interpreter::Plan&, Scope*)` | 整个执行引擎 |
| 同上 | `FetchList Run(const std::vector<std::string>& feed_names, bool enable_job_schedule_profiler = false)` | 调度 |
| `.../garbage_collector/garbage_collector.h:37` | `virtual void Add(Variable*, const InstructionBase*) = 0`,共 5 种 GC 策略 | 中间变量回收 |
| `paddle/fluid/eager/to_static/run_program_func.h:23` | `egr::to_static::run_program_ad_func(x, params, step_scope, prog_attrs, cuda_graph_attrs)` | **反向图,免费** |
| `pir::Program` | 图 IR + 序列化 + pass | IR 设计与优化 |

**`run_program_ad_func` 是关键。** 它让静态图**参与 eager 自动微分** ——
我们不需要为静态图单独写一套反向。
上游唯一的调用点是 `paddle/fluid/pybind/eager_custom_python_api.h:136`,是个薄包装,
**说明它就是设计给前端绑定用的**。

### 2.1 附带的好处:静态图是 GC 风险的解药

`interpreter_engine.h:56` 显示中间变量活在 `framework::Scope` 里,
由 `InterpreterCoreGarbageCollector` 管理。

**这意味着走静态图路径时,中间张量从来不会变成 Lua 对象**,
P14 那套堆追踪机制在这条路径上根本用不上。
对显存敏感的用户,"用 `jit.trace` 包一下"本身就是一个可推荐的优化手段。

---

## 3. 设计

### 3.1 trace 怎么录

不需要 AST。**在算子调用层挂钩子:**

```
paddle.jit.trace(net, x)
  ├─ 打开 trace 模式
  ├─ 把 x 换成"符号张量"(有 shape/dtype,无数据)
  ├─ 跑 net:forward(x)
  │     每个算子调用被记录:(op_name, inputs, attrs, outputs)
  ├─ 关闭 trace 模式
  └─ 把记录序列组装成 pir::Program
```

**钩子挂在 P3 生成的 wrapper 里**,不是每个算子手写。
生成器多输出一个分支即可 —— 这是 P3 那 3 周投资的又一次回报。

### 3.2 组装 `pir::Program`

这是本阶段的主要工作量,也是唯一需要深入上游内部结构的地方。
⚠️ **具体 API 未核实**,P15 开工前必须先做一次专门的调研,
把 `pir::Program` 的构造接口摸清楚并写进本文件 §2。

### 3.3 训练路径

```
lua: y = traced_net(x)
  └─ pd_run_program(x, params, scope, prog_attrs, cuda_graph_attrs)
       └─ egr::to_static::run_program_ad_func(...)
            ├─ 前向:InterpreterCore 执行
            └─ 反向:自动挂进 eager 图,loss:backward() 时被调用
```

**Lua 侧看不出静态图和动态图的区别** —— 这正是目标。

---

## 4. 已知的坑

**① `AttributeMap` 要手工构造。** `run_program_ad_func` 的第 4、5 个参数是
`paddle::framework::AttributeMap`。Python 侧由 pybind 自动转换,
我们要在 C++ 里手工拼。难度未知,M0 第 18 项验证。
**若难度过高,静态图退回自建反向**,工期从 4 周涨到 8 周以上。

**② trace 的固有局限必须写进文档第一段。**
- 控制流被固化成本次走过的分支
- shape 被固化(除非用 dynamic shape 标注)
- Python 侧的 `print` / 随机数在 trace 后不再执行

**这些不是 bug。** 但用户会当成 bug 报,所以文档要写在最显眼处,
而且 `trace` 时应该**主动检测并警告**(比如发现前向里有 `if` 依赖张量值)。

**③ 别在 trace 模式下让符号张量参与 Lua 的算术判断。**
`if x:item() > 0` 在 trace 时会真的取值,导致分支固化且用户不自知。
符号张量的 `item()` 应该**报错**,而不是返回一个假值。

**④ 产物要能被 Python 读回。** 这是互通性的关键验收项。
如果我们组装的 `pir::Program` 与 Python 端产出的结构有差异,
`paddle.jit.load` 会失败或行为不一致。

---

## 5. 验收

- [ ] `trace` 一个 ResNet18,前向输出与动态图逐位一致
- [ ] traced 模型能训练,loss 曲线与动态图一致到 1e-5
- [ ] 产物能被 Python 端 `paddle.jit.load` 读回并推理一致
- [ ] 前向里有 `if x:item() > 0` 时,trace 报错而不是静默固化
- [ ] traced 模型的显存峰值 <= 动态图(验证 §2.1 的论断)
- [ ] trace 一次的耗时 < 前向 10 次

---

## 6. 未解问题

- Q-04 手工构造 `AttributeMap` 的难度(M0 第 18 项)—— **决定本阶段是 4 周还是 8 周**
- `pir::Program` 的构造 API(§3.2)—— 开工前必须专门调研
- Q-01 相关:`WITH_PYTHON=OFF` 下 `new_executor` 是否可用(M0 第 16 项)
