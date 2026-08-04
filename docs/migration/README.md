# Paddle-Lua 迁移计划

本目录是 **Paddle → Lua 前端** 的完整构建计划,共 34 份文档、约 8000 行。
除本目录外,`docs/` 下不放其它内容;未来的 API 使用说明、教程等另开目录。

⚠️ **当前处于论证阶段,尚无代码。**

---

## 目录组织

```
docs/migration/
├── WORKPLAN.md        ★★ 总工程树 —— 智能体按它深度优先遍历,遍历完 = 工程完成
├── README.md          ← 你在这里(总索引)
├── plan/              做什么、什么顺序、怎么做
│   ├── overview.md        范围、分层、工作量、风险
│   ├── roadmap.md         ★ 阶段顺序:先干什么再干什么(P0–P18)
│   ├── layout.md          ★ 代码目录与模块清单 + 文件级落地顺序
│   ├── ci.md              ★ CI 计划:四层、阶段解锁表、红线
│   ├── foundations.md     ★ 生态基座:Penlight 与 Insight7(跨阶段)
│   └── modules/           ★ 19 份模块详细设计,每个阶段一份
├── process/           作业流程:状态、任务板、约定、决策、未解问题
└── research/          技术论证:可行性、架构、GC、DataLoader、复用、动转静
    └── _ref/          第三方参考源码片段
```

**四份「顺序」文档的分工**(容易混,列清楚):

| 文档 | 粒度 | 回答 |
|---|---|---|
| `WORKPLAN.md` | 节点 | **执行**:现在做哪个,做完做哪个 |
| `plan/roadmap.md` | 阶段 | 依赖:P5 为什么必须在 P2 之后 |
| `plan/layout.md` | 文件 | 落地:P1 开工敲的第一个文件是哪个 |
| `process/tasks.md` | 原子任务 | M0/M1 叶节点的具体内容 |

三个子目录的分工:

| 子目录 | 回答的问题 | 变更频率 |
|---|---|---|
| `research/` | **能不能做?怎么做才对?** | 低 —— 结论一旦立住就冻结 |
| `plan/` | **做什么?按什么顺序?多久?** | 中 —— 里程碑推进时更新 |
| `process/` | **现在做到哪了?下一步做啥?** | 高 —— 每个任务都要动 |

---

## 一分钟了解

| | |
|---|---|
| 目标 | Lua 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT 上使用 Paddle,API 与 Paddle 对齐 |
| 不需要 | Python 运行时 |
| 绑定方式 | sol2 + 纯 C ABI 中间层 |
| 上游改动 | **零** |
| 多 worker | Lua Lanes(强制依赖) |
| 索引 | **统统 1-based**,切片用 `x["1:3, :"]` |
| 生态基座 | 类/集合/兼容层 = **Penlight**(vendored 纯 Lua);numpy 的位置 = **Insight7** |
| 工期 | MVP 4-6 人月;三仓库全可用 ~13.5 人月 |
| 生死判定 | `WITH_PYTHON=OFF` + `ON_INFER=OFF` 能否编出可用 libpaddle |

---

## 我该读哪个文件

### 如果你是**智能体**

```
/CLAUDE.md  →  WORKPLAN.md  →  process/status.md  →  <当前叶节点的文档>
(约束/决策)     (找到那个 🔵)    (上次卡在哪)          (这个节点怎么做)
```

**`WORKPLAN.md` 是入口。** 它是一棵树,深度优先遍历,遍历完就是工程完成。
全树同一时刻只有一个 🔵,那就是你的位置。

**不要一上来通读全部** —— 总量约 8000 行,会吃掉大量上下文。
`plan/modules/` 只读当前阶段那一份。

### 如果你是**人**

| 想知道 | 读 |
|---|---|
| 这事儿能不能干成 | `research/feasibility.md` |
| 先干什么、再干什么 | `plan/roadmap.md` |
| 某个模块具体怎么做 | `plan/modules/<阶段>.md` |
| 范围有多大、要多久 | `plan/overview.md` |
| 为什么是这个方案而不是那个 | `process/decisions.md` |
| 哪些还没想清楚 | `process/open-questions.md` |

---

## 文档索引

### `plan/` —— 实施计划

| 文件 | 内容 |
|---|---|
| `plan/overview.md` | 范围与总账:仓库划分、A/B 四象限分类、排除清单、分层架构、工作量、API 设计、M0 验证清单、风险登记 |
| **`plan/roadmap.md`** | **总计划。** P0–P18 的依赖图、每阶段的开工条件与完工判据、三条排期原则、允许打乱顺序的边界 |
| **`plan/layout.md`** | **代码目录与模块清单。** 权威目录树、每个模块的阶段/体量/性质、**文件级落地顺序**(每个阶段第一个敲哪个文件)、生成代码进不进版本库 |
| **`plan/ci.md`** | **CI 计划。** 四层 CI(为什么不能每次 push 都编 Paddle)、阶段解锁表、必须机器挡住的五条红线、golden 数据 |
| **`plan/foundations.md`** | **生态基座。** 为什么类系统用 `pl.class`、集合用 `pl.List`、numpy 的位置给 Insight7;Penlight 的 11 文件闭包与四个坑;`LayerList` 为什么继承 `Layer`(以及 `ipairs` 在 5.1 上的坑);Insight7 互操作与 `axis` 改 1-based 的落地方式 |
| **`plan/modules/`** | **19 份模块详细设计。** 每份回答:做什么 / 上游有什么可用(带 `file:line`)/ 怎么做 / 坑在哪 / 怎么验收 |

`plan/modules/` 清单:

| | | | | |
|---|---|---|---|---|
| `00-build.md` | `01-c-abi.md` | `02-binding.md` | `03-codegen.md` | `04-packaging.md` |
| `05-tensor.md` | `06-autograd.md` | `07-serialization.md` | `08-jit-infer.md` | `09-nn.md` |
| `10-optimizer.md` | `11-io.md` | `12-dataset-vision.md` | `13-lanes.md` | `14-gc.md` |
| `15-to-static.md` | `16-cpp-extension.md` | `17-distributed.md` | `18-script.md` | |

### `process/` —— 作业流程(智能体每次开工必读)

| 文件 | 内容 |
|---|---|
| **`process/status.md`** | 当前闸门 / 当前任务 / 环境记录 / M0 进度 |
| **`process/tasks.md`** | 原子任务板,含前置条件与验收标准 |
| `process/conventions.md` | Lua 5.1 子集规则、命名、索引转换、错误处理、测试、提交 |
| `process/decisions.md` | 决策理由 + **翻案记录** + 待人拍板事项 |
| `process/open-questions.md` | 未解问题 Q-01…Q-16 + 已关闭/已决问题 |

### `research/` —— 技术论证

| 文件 | 内容 | 关键结论 |
|---|---|---|
| `research/feasibility.md` | 可行性主报告 | 训练引擎是纯 C++,只需重写外壳 |
| `research/architecture.md` | 路线修正 | 全 sol2 + C ABI 中间层;pickle 可解析 |
| `research/gc.md` | 显存生命周期 | Torch7 九层方案;`RegisterOOMCallback` 已是公共 API |
| `research/dataloader.md` | 多 worker | 线程胜过进程;Lanes deep userdata |
| `research/reuse.md` | 可复用 C++ 资产 | 五套执行器辨析;`InterpreterCore` + `run_program_ad_func` |
| `research/to-static.md` | 动转静 | trace 先于 script;luacheck parser;源码契约可验证 |

---

## 三条最重要的结论

**1. 训练引擎不用重写。**
`egr::Backward` / `paddle::Tensor` / phi kernels 全是纯 C++,与 Python 无关。
真正要写的是外壳:约 15k 行手写 + 60k 行生成。

**2. 唯一的项目级风险是能不能编出来。**
`cmake/configure.cmake:15` 的守卫是 `(NOT WITH_PYTHON) AND ON_INFER` ——
上游预期的无 Python 场景**是推理**。我们要的训练态无 Python 组合可能从未被编译过。
**这一项挂了,整个项目重估。**

**3. 复用救不了两件事。**
控制流捕获(难在 Lua 前端)和 `paddle.nn` 那 6k 行手写层(没有可复用等价物)。
这两块是真实的成本重心。

---

## 相关仓库

| 仓库 | 关系 |
|---|---|
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | 上游,**零改动** |
| [PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean) | 高层 Trainer 的设计蓝本(替代 `paddle.Model`) |
| [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics) | 指标系统的设计蓝本(替代 `paddle.metric.*`) |
| [Penlight](https://github.com/lunarmodules/Penlight) | **一等公民**:`pl.class` / `pl.List` / `pl.compat` / `pl.pretty`。vendored 受限子集,MIT |
| Insight7 | **一等公民**:numpy 的位置。也是字符串索引 / `_wrap` 三模式调用的参考实现 |

未来切分为三个仓库:

```
paddle-lua      C++ 绑定 + 核心 Lua 层        ← 现在做这个
  ├─ metrics-lua   纯 Lua,torchmetrics 风格   ← M3
  └─ ocean-lua     纯 Lua,Lightning/Keras 风格 ← M3+
```
