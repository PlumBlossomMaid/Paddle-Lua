# Paddle Lua 前端 · 实施计划

> 前置文档:`research/feasibility.md`(可行性主报告) / `research/architecture.md`(路线修正) /
> `research/gc.md`(显存生命周期) / `research/dataloader.md`(多 worker) / `research/reuse.md`(可复用资产清单) /
> **`research/to-static.md`(动转静:源码契约与 AST 选型)**
> 本文是**计划**,不是调研 —— 前四篇已把技术问题问完,这里回答"做什么、按什么顺序做、不做什么"。

> ⚠️ **当前阶段:论证。不开工、不写代码。**
> 本文中"立即可做"只表示*技术上已就绪、无阻塞*,不表示*现在开始*。

---

## 0. 一页纸结论

| 维度 | 决定 |
|---|---|
| 宿主语言版本 | **Lua 5.1 / 5.2 / 5.3 / 5.4 + LuaJIT**(5.5 移出范围,见 `research/gc.md` §6.5) |
| 库代码语法 | **严格 Lua 5.1 语法子集**,5.4 特性只通过 C 侧和运行时检测提供 |
| 绑定层 | **全部 sol2**,底下垫一层纯 C ABI 中间层(`research/architecture.md` §B) |
| Python 运行时 | **不需要**。`WITH_PYTHON=OFF` |
| 上游 Paddle 改动 | **零**(`research/architecture.md` §A 已证明可行) |
| 多 worker | **Lua Lanes,强制依赖**。不装 Lanes = 没有多 worker |
| `paddle.Model` | ❌ **不移植** |
| `paddle.metric.*` | ❌ **不移植** |
| 替代方案 | 学 **PaddleOcean** / **PaddleMetrics**,**独立仓库**,但 **M3 才做** |
| 排期原则 | **先主框架,Ocean/Metrics 往后放** |
| 优先级排序 | **① 能直接抄语义的 > ② 需要自己重写的**;**① 单卡可测的 > ② 当前测不了的** |
| 索引 base | **统统 1-based**(含 `axis`/`dim`/返回的 index tensor)。见 §6.1 |
| 仓库数 | 3 个:`paddle-lua`(现在) / `ocean-lua`、`metrics-lua`(M3) |
| MVP 工期 | 4-6 人月(M1),核心 API 对齐 1.5-2 人年(M2) |

---

## 1. 仓库划分

用户的 Python 侧已经把这件事想清楚了:Paddle 是框架,Ocean 是训练器,Metrics 是指标,
**三个独立仓库**。Lua 侧照抄这个切分,理由不只是"对齐",而是三条硬理由:

| 理由 | 说明 |
|---|---|
| **依赖方向** | Ocean/Metrics 依赖 paddle,反过来不成立。塞进一个仓库会制造伪循环 |
| **发布节奏** | 核心绑定跟 Paddle C++ 版本走;Trainer 是纯 Lua,可以周更 |
| **构建成本** | 核心仓要编 ×5 个 Lua 版本的 C++;Ocean/Metrics 是**纯 Lua,零编译** |
| **可替换性** | 用户想自己写训练循环时,不该被迫装 Trainer |

```
paddle-lua      C++ 绑定 + 核心 Lua 层    有编译产物,×5 Lua 版本
   ├─ ocean-lua     纯 Lua,Lightning/Keras 风格 Trainer
   └─ metrics-lua   纯 Lua,torchmetrics 风格指标
```

三者都用 LuaRocks 分发;`ocean-lua` 和 `metrics-lua` 的 rockspec 里
`dependencies = { "paddle >= x.y" }`,LuaRocks 自动处理。

**关键约束:`ocean-lua` / `metrics-lua` 只能调用 `paddle-lua` 的公开 API。**
一旦它们需要 C 侧内部符号,说明核心层 API 有缺口 —— 这是一条很好的自检信号。

### 1.1 排期:先主框架,Ocean/Metrics 往后

> 决定:**先搞定主框架,Ocean 和指标系统以后做。**

三仓库的**切分**现在就定下来(影响 API 设计),但**开发**串行:

```
M0/M1/M2  ████████████████  paddle-lua        ← 全部精力
M3              ░░░░░░░░░░  metrics-lua
M3+                 ░░░░░░  ocean-lua
```

这个顺序不只是"先后",它有个实际好处:
**Ocean 和 Metrics 是核心层 API 质量的验收器。**
先把它们写出来,只会用一个还没定型的底座去试错;
等核心稳定后再写,它们能诚实地暴露 API 缺口 —— 而不是反过来让核心层去迁就它们。

`ocean-lua` 排在 `metrics-lua` 之后,因为 Ocean 的 `log_dict` / `EarlyStopping(monitor=...)`
都要对接 Metrics,反向不成立。

---

## 2. 取其精华去其糟粕:范围表

### 2.0 两个正交的分层维度

在给出范围表之前先建立坐标系。**"要不要做"和"什么时候做"是两个问题**,
后者由两个**互相独立**的维度决定:

**维度 A · 移植难度**

| 级别 | 含义 | 成本特征 |
|---|---|---|
| **A1 直接抄语义** | Python 侧是纯逻辑,翻译成 Lua 即可,行为可逐行对照 | 线性、可预测、可并行、错了容易发现 |
| **A2 需要自己重写** | 依赖 Python 特有设施(setuptools / import / 元类 / 协程),或需要另起炉灶的设计 | 非线性、要做设计决策、需要自己造验收标准 |

**维度 B · 可验证性**

| 级别 | 含义 | 风险特征 |
|---|---|---|
| **B1 当前环境可测** | 单卡 / CPU,写完就能验 | 写完即闭环 |
| **B2 当前环境测不了** | 多卡、多机 | **可以写,但只能"看起来对"** |

**排序规则:A1 优先于 A2;B1 优先于 B2。两条规则冲突时,B 优先。**

> 为什么 B 优先于 A?因为**测不了的代码即使"直接能抄"也是负债**——
> 它会以"已完成"的姿态进入代码库,然后在第一个真实用户手里炸掉。
> A2 的成本是**看得见的时间**,B2 的成本是**看不见的债**。宁可先付看得见的。

**四象限:**

```
              A1 直接抄            A2 需要重写
          ┌────────────────────┬────────────────────┐
 B1 可测  │  ★ 第一优先        │  ② 第二优先        │
          │  nn / optimizer    │  Layer 类系统      │
          │  vision / dataset  │  DataLoader+Lanes  │
          │  linalg / fft      │  jit 静态图(自研) │
          ├────────────────────┼────────────────────┤
 B2 测不了│  ③ 可写不可信      │  ④ 最后            │
          │  distributed 的    │  distributed 启动器│
          │  算子封装          │  / 通信策略        │
          └────────────────────┴────────────────────┘
```

### 2.1 做:按 A/B 分类

| Paddle 模块 | A | B | 里程碑 | 备注 |
|---|:-:|:-:|---|---|
| `paddle.Tensor` + 算子 | A1 | B1 | M1 | 从 yaml 生成,量最大但机械 |
| autograd(`backward`/`grad`) | A1 | B1 | M1 | `egr::Backward` 已 Python-free |
| `paddle.nn.functional` | A1 | B1 | M1 | 大部分直通生成的算子 |
| `paddle.nn.*` 层 | A1 | B1 | M1/M2 | ~200 个,多是 functional 薄封装 |
| `paddle.optimizer` | A1 | B1 | M1 | SGD/Momentum/Adam/AdamW + LRScheduler |
| `paddle.device` | A1 | B1 | M1 | |
| `paddle.save/load` | A1 | B1 | M1 | 含纯 Lua pickle 解析器 |
| `paddle.vision.transforms` | A1 | B1 | M2 | 纯计算,逐行翻译 |
| `paddle.linalg` / `paddle.fft` | A1 | B1 | M2 | 生成即可 |
| **`paddle.dataset` / `paddle.text`** | **A1** | B1 | **M2** | **见 2.1.1,判断已修正** |
| `paddle.amp` | A1 | B1 | M2 | thread_local,需按 lane 处理 |
| **`paddle.nn.Layer` 类系统** | **A2** | B1 | M1 | Python 元类/`__setattr__` 拦截 → Lua 元表重设计 |
| **`paddle.io` DataLoader** | **A2** | B1 | M1(单)/M2(多) | Lanes,非 Python 多进程模型 |
| **`paddle.utils.cpp_extension`** | **A1+A2** | B1 | **M2/M3** | **见 2.1.2,需拆开看** |
| **`jit` / 静态图(自研 Lua 版)** | **A2** | B1 | **M3** | **见 2.1.3** |
| **`paddle.distributed.*`** | A1+A2 | **B2** | **M3+** | **见 2.1.4,可写但测不了** |
| `paddle.hub` | A2 | B1 | **M3+** | 见 2.1.5 |

#### 2.1.1 `dataset` —— 我上一版判断偏了

上一版把 `paddle.dataset` / `paddle.text` 归到"与框架无关的下载脚本"排除掉。
**这个理由站不住:难度低不是排除的理由,只是排期靠后的理由。**

实际是标准的 **A1/B1**:下载 → 校验 md5 → 解压 → 按格式解析 → 返回 `Dataset`。
唯一的 Lua 侧缺口是 HTTP 下载和解压,`luasocket` + `lua-zlib` 即可,或直接 shell out 到 curl。
(「不引入新的强制 C 依赖」已于 2026-08-03 取消,R23 —— 这条缺口现在是**选路问题**,不是**能力问题**。)

**而且它有个被低估的价值:它是 `paddle.io.Dataset` 抽象的第一个真实使用者。**
只有 MNIST 这种现成数据集能让 M1 的验收("跑通 MNIST 并收敛")真正闭环,
否则连测试数据从哪来都是问题。**建议至少 MNIST/CIFAR 提到 M1。**

#### 2.1.2 `cpp_extension` —— "能直接抄"只对了 1/3

已核对源码(`python/paddle/utils/cpp_extension/`,3712 行),它是**三块拼起来的**,
三块的 A 级别完全不同:

| 块 | 行数 | A | 说明 |
|---|---|:-:|---|
| **工具链探测 + 编译参数计算** | ~800 | **A1** | `find_cuda_home` / `_get_cuda_arch_flags` / `prepare_unix_cudaflags` / `find_paddle_includes` / `find_paddle_libraries`。**纯路径与字符串逻辑,可逐行抄** |
| **setuptools 集成** | ~1400 | **A2 且不可抄** | `cpp_extension.py` 里 `BuildExtension(build_ext)`、`EasyInstallCommand(easy_install)`、`InstallCommand(install)` —— **它不是"用了 setuptools",它是一个 setuptools 插件**。整块作废 |
| **算子加载 + API 自动生成** | ~500 | **A2(语义可抄,产物要换)** | `load_op_meta_info_and_register_op` → `parse_op_info` → `_custom_api_content` 生成 Python 包装。语义照抄,但生成的是 **Lua** 包装 |

好消息:底层是 C++ 的,不依赖 Python ——

```cpp
// paddle/fluid/framework/custom_operator.h:316
LoadOpMetaInfoAndRegisterOp(const std::string& dso_name);
```

坏消息:`parse_op_info` 走的是 `OpProtoHolder`(旧 fluid 静态图算子表)。
**这条链在 `WITH_PYTHON=OFF` 下是否还活着,是 M0 要额外确认的一项。**

结论:**不能笼统说"直接抄"。** Lua 侧的正确形态是
"探测工具链(抄)+ 生成 CMakeLists 或直接调编译器(重写)+ 生成 Lua 包装(语义抄)",
比 Python 版**简单**(不必迎合 setuptools),但**不是复制粘贴**。约 1-1.5 人月,排 M2/M3。

#### 2.1.3 `jit` / 静态图 —— 自研 Lua 版,但排在最后

> 想法:自己实现一个 Lua 版本的静态图。

**这个想法本身是成立的,而且比"绑定 Paddle 的 Program/Executor"更合理** ——
后者要把整套 fluid IR、Program、Executor、Pass 都暴露给 Lua,是个死胡同;
前者只需要"录制算子调用 → 建图 → 重放/优化",在动态图之上做 tracing 即可。

但按 §2.0 的规则它是 **A2**,而且是 A2 里最重的那种:

| 它需要什么 | 有没有现成的可抄 |
|---|---|
| 算子调用录制(tracer) | ❌ 要自己设计拦截点 |
| 图 IR 表示 | ❌ Lua 里没有,要自己定 |
| 拓扑排序 / 死代码消除 / 常量折叠 | ⚠️ 算法通用,但实现全新 |
| 序列化格式 | ❌ 自定义 |
| 控制流(if/while)捕获 | ❌ 最难的一块。**但已有方案,见 `research/to-static.md`**:vendored luacheck parser(5.1-5.4+LJ,MIT,~59KB)+ 自写 printer(~500 行)|

**关键判断:控制流捕获决定了它是"玩具"还是"能用"。**
只做无控制流的直线图,一两周能出原型。

> ⚠️ **"Lua 没有 AST 库、要自己写 parser"这句话是错的,已修正(`research/to-static.md` §5)。**
> luacheck 的 `parser.lua` 覆盖 **Lua 5.1-5.4 + LuaJIT 语法**,自身就跑在 5.1 上,
> MIT,3 个文件 ~59 KB 可直接 vendor。**要自己写的只是 AST→源码 的 printer(~500 行)。**
>
> 更重要的是分层:**script 模式(需 AST)是 M4,trace 模式(零 AST)才是 M3。**
> 所以这条不影响 M3 排期,也不影响加密分发的用户使用基线功能。

> ⚠️ **上表已被 `research/reuse.md` §2.1 部分推翻。** 复用 `InterpreterCore` 后:
> 图 IR(`pir::Program`)、拓扑排序、调度、中间变量显存回收、优化 pass、序列化
> **全部现成**;`run_program_ad_func` 还让**反向图免费**(`research/reuse.md` §4)。
> **实际只需写 tracer + 组装 `pir::Program`。**
> 但**控制流那一行没变** —— 它难在 Lua 前端,复用救不了。

**建议:仍排 M3,且第一版明确只做直线图 tracing(推理/导出场景够用),
控制流留白。** 在此之前,推理场景先用只读加载已有产物:

```cpp
// paddle/fluid/jit/serializer.h:78
Layer Load(const std::string& path, const phi::Place& place);
```

这条是纯 C++、A1、几十行绑定就能用 —— **先要这个,自研静态图往后放。**

**补充(`research/reuse.md` §3):`jit::Layer` 的价值被我上一版低估了。**
它的接口货币就是 `paddle::Tensor`,零转换;`Clone(stream)` 天然配合 Lanes;
最重要的是**它不依赖 `paddle.nn` 移植完成** ——
"Python 训练 / Lua 推理"是整个项目里**最早能交付真实价值**的形态。
**已提为 M1 可选项。**

#### 2.1.4 `distributed` —— 可以写,但测不了,所以往后

> 判断:可以写但是测试不了,往后放放。

**完全同意,而且这正是 §2.0 里 B 优先于 A 的教科书案例。**

拆开看,难度其实不高:

| 部分 | A | B | 说明 |
|---|:-:|:-:|---|
| collective 算子封装(`all_reduce` 等) | **A1** | **B2** | 底层是 C++ 算子,封装是薄的。**但单卡验不了正确性** |
| `ProcessGroup` 初始化 | A1 | B2 | C++ 侧有 |
| **启动器**(`paddle.distributed.launch`) | **A2** | B2 | Python 版靠 `subprocess` + 环境变量拉进程组。Lua 侧要重写,且与 Lanes 是**两套并发模型** |
| 策略(DDP/FSDP/流水线) | A2 | B2 | 设计量大 |

**"能写"的部分恰恰是最危险的部分** —— `all_reduce` 封装二十分钟就能写完、能编译、
能在单卡上"跑通"(单卡 all_reduce 等于恒等映射,永远看起来是对的),
然后在真正的 8 卡机器上暴露出 bug。这就是 §2.0 说的"看不见的债"。

**建议:M3+,且开工前先解决测试环境**(哪怕是单机多进程模拟 / CI 上租 2 卡)。
**没有测试环境就一行都不要写。**

#### 2.1.5 `hub` —— 往后放,且要重新设计

> 判断:往后放放。同意。

它的核心是"从远端拉一个模型定义并执行它"。Python 版靠 `importlib` 动态 import,
Lua 侧对应 `loadfile`/`require` —— **技术上更简单**,但引入了一个 Python 版没认真处理的问题:
**执行远端下载的代码**。Lua 有 sandbox 传统(`setfenv`/5.2+ 的 `load` env 参数),
做得比 Python 版更干净是有可能的,但那是设计工作,不是移植工作 → **A2,M3+。**

### 2.2 明确排除(不做)

| 排除项 | 理由 | 替代 |
|---|---|---|
| **`paddle.Model`** | 过时。API 冻结、扩展性差 | **`ocean-lua`(M3)** |
| **`paddle.metric.*`** | 过时。5 个指标,无分布式同步、无组合 | **`metrics-lua`(M3)** |
| `paddle.static.*`(Program/Executor 绑定) | 要把整套 fluid IR 暴露给 Lua,死胡同 | **自研 Lua 静态图(2.1.3)** |
| `paddle.incubate.*` | 顾名思义,上游自己都不保证稳定 | — |
| `paddle.onnx` | 依赖 paddle2onnx(纯 Python 项目) | 导出走 Python 侧 |

**注意与上一版的差别:排除列表从 9 项缩到 5 项。**
`dataset` / `cpp_extension` / `jit` / `distributed` / `hub` 全部从"排除"改判为"延后"。
上一版把"难"和"该不该做"混为一谈了 —— **这两者应该分开,前者决定排期,后者决定范围。**

### 2.3 重做(照抄 Ocean/Metrics,M3)

| 能力 | Paddle 的做法 | 我们的做法 |
|---|---|---|
| 训练循环 | `paddle.Model.fit()` 单一模式 | Ocean 的双模式:Lightning hooks + Keras `prepare/fit` |
| 回调 | `paddle.callbacks` 少数几个 | Ocean 的 Callback 体系(先做 8 个) |
| 日志 | `VisualDL` 硬编码 | Ocean 的 Logger 抽象(先做 3 个) |
| 指标 | `paddle.metric.Accuracy` 等 5 个 | Metrics 的 `add_state` 声明式状态机 |
| 设备/精度策略 | 混在 `Model` 里 | Ocean 的 Accelerator / Strategy / Precision 三层分离 |

**全部排 M3。** 详见 §6.2 / §6.3(设计已定,只是不现在做)。

---

## 3. 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  用户代码       local paddle = require "paddle"          │
├─────────────────────────────────────────────────────────┤
│  ocean-lua          metrics-lua        (纯 Lua 5.1)      │  独立仓库
├─────────────────────────────────────────────────────────┤
│  paddle.nn / optimizer / io / vision   (纯 Lua 5.1)      │  ~12-15k 行手写
│  paddle.Tensor 方法糖 / __index / 广播规则               │
├─────────────────────────────────────────────────────────┤
│  sol2 绑定层                            (C++)            │  ×5 编译
│  usertype<Tensor>, 生成的算子表, __lanesclone             │
├─────────────────────────────────────────────────────────┤
│  纯 C ABI 中间层  paddle_capi.h          (C)             │  ×1 编译 ← 关键
│  paddle_tensor_t / paddle_status_t / 无异常穿越          │
├─────────────────────────────────────────────────────────┤
│  libpaddle.so  (WITH_PYTHON=OFF)                         │  上游,零改动
│  paddle::Tensor / egr::Backward / phi kernels            │
└─────────────────────────────────────────────────────────┘
```

**为什么中间层不能省(`research/architecture.md` §B 的核心结论):**

- sol2 是重模板库。绑定 2000+ 算子的 TU **×5 个 Lua 版本**重编,是编译时间灾难
- 中间层是纯 C,**只编一次**;sol2 层薄到只做类型转换
- C++ 异常不能穿过 Lua 的 `longjmp`(5.1/LuaJIT 尤其)→ 中间层强制转 `status_t`
- 将来若某个 Lua 版本 sol2 不支持(如 5.5),可以绕过 sol2 直接用裸 C API 接中间层

### 3.1 算子绑定的生成策略

**不手写。** 从 Paddle 的 `ops.yaml` / `backward.yaml` / `python_api_info.yaml`
生成三份产物(生成器用 Python 写,但**只在开发期跑**,发布产物里没有 Python):

```
ops.yaml ──► gen_capi.py    ──► paddle_capi_ops.{h,c}   纯 C 声明+实现
         ──► gen_sol.py     ──► paddle_sol_ops.cpp      sol2 注册
         ──► gen_luadoc.py  ──► paddle/_ops.lua         参数名/默认值/文档
```

第三份很重要:Lua 没有关键字参数,**默认值和参数名必须在 Lua 侧有元数据**,
否则 `paddle.sum(x, {axis=1, keepdim=true})` 这种 table 传参没法实现。

---

## 4. 工作量估算

| 模块 | 语言 | 行数(估) | 人月 | 说明 |
|---|---|---|---|---|
| C ABI 中间层(手写部分) | C | 1.5k | 0.5 | 生命周期/错误/dtype/place |
| C ABI 算子层 | C(生成) | 60k+ | 0.5 | 生成器本身 ~1.5k 行 |
| sol2 绑定层 | C++ | 3k + 生成 | 1.0 | usertype + 元表 + 运算符重载 |
| Lanes 跨线程传递(`__lanesclone` + 句柄表退路) | C++ | 100-150 | 0.2 | `plan/modules/13-lanes.md` §4.1(**已由 R18 从 DeepFactory 改为 `__lanesclone`,工作量下调**) |
| GC 机制(9 层) | C++/Lua | 130 | 0.2 | `research/gc.md` §4 |
| **pickle 解析器** | **纯 Lua** | **600-1000** | **0.5** | 顺带解锁 `.pt` 读取 |
| `paddle.Tensor` Lua 层 | Lua | 1.5k | 0.5 | 索引/切片/广播/打印 |
| `paddle.nn` | Lua | 6k | 2.0 | 最大的手写块 |
| `paddle.optimizer` | Lua | 2k | 0.8 | |
| `paddle.io` | Lua | 1.5k | 0.8 | Dataset/Sampler/DataLoader/Lanes |
| `paddle.vision` | Lua | 1.5k | 0.5 | |
| 构建/CI(×5 Lua × 3 OS) | CMake/CI | — | 1.0 | **别低估** |
| 测试 | Lua | 5k | 1.5 | |
| **小计(paddle-lua)** | | **~15k 手写** | **~10** | |
| `metrics-lua`(先做 20 个常用指标) | Lua | 3k | 1.5 | |
| `ocean-lua`(Trainer + 8 callback + 3 logger) | Lua | 4k | 2.0 | |
| **合计** | | | **~13.5 人月** | |

> 对照 `research/feasibility.md` §9:MVP(M1)4-6 人月的口径是"能跑通 MNIST 训练",
> 上表 13.5 人月是"三个仓库都到可用状态(M2)"。两者不冲突。

**关键认知:60k+ 行的算子绑定是生成的,不计入实际工作量;
真正的成本在 `paddle.nn`(2 人月)和构建矩阵(1 人月)。**

---

## 5. 强制依赖清单

| 依赖 | 版本 | 强制? | 理由 |
|---|---|---|---|
| Paddle C++ | 3.5+,`WITH_PYTHON=OFF` | ✅ | 本体 |
| sol2 | 3.5.0 | ✅ | 绑定层(LuaJIT 也走 sol2,见 `research/architecture.md` §B) |
| **Lua Lanes** | **v3.17.x** | ✅ **强制** | **不装 = 没有多 worker。**Tensor 是普通 sol2 usertype,跨 lane 走 `__lanesclone`(R18) |
| Lua | 5.1/5.2/5.3/5.4 或 LuaJIT 2.1(GC64) | ✅ | 宿主 |
| **Penlight** | `>= 1.13, < 2.0` | ✅ **强制,rock 依赖**(R30) | 类系统 / `List` / 兼容层 / 安全 table 读取(C11、D25)。**生态地基级** —— paddle-lua / Insight7 / `argsig` 共用同一份,`is_a` 处处成立 |
| **`argsig`**(参数签名层) | `>= 0.1, < 0.2` | ✅ **强制,rock 依赖**(R27) | 所有公开 API 的签名(C11)。暂定名,待拍板 P10。见 `plan/argsig.md` |
| LuaFileSystem(传递) | — | 🔶 **被 Penlight 拖入** | 我们自己不 `require "lfs"`;文件系统走 `paddle.utils.fs`。它只是 Penlight 的 rockspec 声明的 |
| **Insight7** | — | 🔶 **软强制** | numpy 的位置(C11、D26)。核心路径不 `require` 它,`paddle.np` / `from_insight` / `vision.transforms` 惰性加载 |
| ~~LuaFileSystem(直接依赖)~~ | — | ❌ **我们不直接用** | 原打算给 `paddle.io` 扫目录。改为 `paddle.utils.fs`(C++17 `std::filesystem`)。~~理由:少一个 C 依赖~~ -> **理由:我们的 `.so` 顺手就能做,边际成本 ≈ 0,比多 15 个构建组合划算**(R23 / `CLAUDE.md` §9.1) |
| HTTP+TLS(`luasocket`+`luasec`) | — | 🔶 **P12 待定** | R23 之后**允许引入**。先查 Paddle 有无可绑的 HTTP 能力(Q-08),没有就直接上 rock。见 `12-dataset-vision.md` §3.1 |
| 图像解码(JPEG/PNG) | — | 🔶 **P12 待定** | 同上。R23 之前这块被逼成"v1 只支持二进制原始格式",现在**可以在 v1 做** |

**Lanes 强制化的决策已回写 `research/dataloader.md` §9.5(a)。**
净收益:Tensor 单一表示、单一代码路径、无运行时检测、少一个 M0 验证项。
代价:发行包体积。Lanes 是纯 C++ 无外部依赖的小库,可接受。

~~**Penlight 为什么 vendor 而不是声明依赖**~~ 🔄 **已推翻(R30,2026-08-03)。**
人的原话:「pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西。」
版本锁定改为 rockspec 锁 minor + CI 语义测试;`luafilesystem` 作为传递依赖接受。
决定性理由:vendor 会让"系统 Penlight 与我们那份互不 `is_a`"在**每两个生态库之间**各出现一次,
而生态里将有 paddle-lua / Insight7 / `argsig` / metrics / ocean 五个消费者。
完整论证见 `plan/foundations.md` §1.3。

**Python 不在此表中。** 生成器脚本是开发期工具,不是运行期依赖。

---

## 6. 高层 API 设计

### 6.1 核心层:`paddle`

```lua
local paddle = require "paddle"
local class  = require "pl.class"             -- D25:基座是 Penlight
local nn = paddle.nn
local F  = paddle.nn.functional

local Net = class(nn.Layer)
Net._name = "Net"

function Net:_init()
  self:super()
  self.fc1 = nn.Linear(784, 128)              -- 自动注册(D23)
  self.fc2 = nn.Linear(128, 10)
end

function Net:forward(x)
  return self.fc2(F.relu(self.fc1(x)))
end

local net = Net()
local opt = paddle.optimizer.Adam{ learning_rate = 1e-3,
                                   parameters = net:parameters() }

for _, batch in paddle.io.DataLoader(ds, { batch_size = 64, num_workers = 4 }) do
  local loss = F.cross_entropy(net(batch[1]), batch[2])
  loss:backward()
  opt:step()
  opt:clear_grad()
end
```

**三个 Lua 化的设计决定(不是照抄 Python,是翻译):**

| Python | Lua | 理由 |
|---|---|---|
| `nn.Linear(784, 128)` 关键字参数 | **`_wrap` 三模式**(位置 / 具名 table / table 内位置) | 沿用 Insight7 `_wrap.lua`,见 6.1.2 |
| `for batch in loader:` | `for _, batch in loader` (`__call` 返回迭代器) | Lua 泛型 for 惯用法 |
| `class Net(nn.Layer)` | **`class(nn.Layer)`(`pl.class`)** + `self:super()` | Lua 无 class 语法;基座统一用 Penlight(D25) |
| `self.fc = nn.Linear(...)` 自动注册 | **一模一样**,`self.fc = nn.Linear(...)` | 实例 raw 表保持空 + 私有 `FIELDS` 键(D23) |
| `x[1:3, :]` | **`x["1:3, :"]`** 字符串索引 | **沿用 Insight7 的既有方案**,见 6.1.2 |

#### 6.1.1 索引 base:统统 1-based(已定案)

> 决定:**统统 1-based**。上一版我建议的"索引 1-based + `axis` 0-based"混合方案**撤回**。

撑住这个决定的三条:

1. **混合方案的认知负担是持续的**。纯 1-based 只要学一次;
   混合方案要在**每一个调用点**判断"这个参数算索引还是算 axis"。
2. **Torch7 先例**。Torch7 是彻底 1-based 的 —— 切片、`dim` 参数、返回的 index tensor、
   `ClassNLLCriterion` 的 target 全都是。它活了很多年,这条路已被验证可行。
3. **边界在库里,不在用户代码里**。转换集中在绑定层一处,用户侧零认知。

**转换规则(必须写死在生成器里):**

| 类别 | 规则 |
|---|---|
| 正数 `axis`/`dim` | Lua `n` → C++ `n-1` |
| 负数 `axis`/`dim` | **不变**(`-1` 仍表示最后一维) |
| 切片区间 | Lua `{i,j}` 闭区间 → C++ `[i-1, j)` |
| 吃 index 的 tensor 参数 | `index_select`/`gather`/`scatter`/`take_along_axis`/`embedding` → 入参 `-1` |
| 吐 index 的 tensor 返回值 | `argmax`/`argmin`/`argsort`/`topk`/`nonzero`/`max(dim)` → 出参 `+1` |
| 分类 label / token id | **也是 1-based**(`cross_entropy` 的 label 取值 1..C) |

**三个必须提前知道的代价 —— 这不是免费的:**

**(a) 生成器需要一张手工标注表。**
`ops.yaml` **没有**"哪个参数是 index 语义"这个信息。
2000+ 算子里约 50-100 个涉及 index,**这张表必须人工维护**,
而且 Paddle 每次新增算子都可能需要补。
这是一笔**长期维护成本**,不是一次性投入。必须配套：
一个 CI 检查,发现 yaml 里出现未标注的新算子就**报错而不是默认放行** ——
默认放行会产生静默的 off-by-one。

**(b) 跨语言数据互操作会碰壁。**
从 Python 存的数据集 label 是 0-based。用 Lua 读进来直接喂给 `cross_entropy`,
**会全错一位**。对策:

- `paddle.load` **不做任何隐式转换**(数据就是数据)
- 提供显式的 `paddle.index_from_zero(t)` / `index_to_zero(t)`
- 文档在数据加载章节置顶警告

**(c) 但 1-based 自带一个意外的好处:off-by-one 变得可检测。**
1-based 语义下 `label == 0` 本身就是**非法值**。
所以 `cross_entropy`/`embedding` 可以直接做范围检查:

```
label 出现 0        → 报错:"看起来是 0-based 数据,请用 index_from_zero()"
label 出现 C+1      → 报错:越界
```

**两个方向都有天然护栏。**
这把 (b) 的风险从"静默地少几个点准确率"降级为"启动时直接报错并告诉你怎么改"。
**这条应当作为必做项写进 M1**,而不是"有空再加" ——
它是 1-based 方案能不能安全落地的关键。

#### 6.1.2 字符串索引:直接沿用 Insight7 的既有方案

> 补充:Insight7(`E:\code\Insight7`)已经在**任意语言**中用**字符串**做 Python 风格索引,
> 字符串内容的语法与 Python 对齐。

**已核对源码。这个方案是对的,而且已经在生产代码里跑通了 —— 上面 §6.1 表里我原先写的
`x:slice{{1,3},{}}` 作废,改用字符串索引。**

核对到的实现:

```cpp
// Insight7 include/insight/core/slice.h:67
Slice parse_slice(const std::string &spec);       // "::2" / "1:10:2" / "::-1"

// Insight7 src/core/array.cpp:628
Array Array::operator[](const std::string &spec) const;   // 逗号分维,每维 parse_dim_spec
```

```cpp
// Insight7 bindings/lua/insight_lua.cpp:218
static std::string lua_spec_to_cpp(const std::string &spec);
// "1:3" → "0:2"   "1:5:2" → "0:4:2"   "2,1:-1" → "1,0:-1"
// "::-1" → "::-1" (不变)   "::2" → "::2" (不变)
```

```lua
-- Insight7 bindings/lua/README.md:360
arr["1:3, :"]                 -- string-based slicing
```

**它为什么是对的(三条,不是"因为已经有了"):**

| | |
|---|---|
| **语法零学习成本** | 用户脑子里那份 NumPy/Paddle 切片知识 100% 复用,包括 `::-1`、`:5`、`2:`、负索引 |
| **绕开了 Lua 的语法天花板** | Lua 没有 `:` 切片语法、没有 `...`(Ellipsis)、`__index` 只收一个参数。字符串是**唯一**能表达完整多维切片的载体 |
| **跨语言一份实现** | 解析逻辑在 C++ 里一份,Python/Lua/Julia 三个绑定共用。**这是"语言无关"的真正含金量所在** |

**转换规则已被验证。** `lua_spec_to_cpp` 的规则和我在 6.1.1 表里推导的完全一致:

```
只转 start 和 stop 两个字段;正数 -1;负数原样透传(两种约定下都是从末尾数);step 不动
```

**这不是巧合 —— 它是这个问题唯一的正确解。** 我的推导和你的实现独立收敛到同一处,
可以把这条当成已验证结论,不必在 M0 再花时间。

**可直接复用的资产(同一作者,无授权问题):**

| 资产 | 位置 | 规模 | 用途 |
|---|---|---|---|
| `lua_spec_to_cpp` | `bindings/lua/insight_lua.cpp:212-266` | ~45 行 | **几乎逐字复用** |
| `parse_slice` / `parse_dim_spec` | `src/core/slice.cpp:178` / `array.cpp:590` | ~120 行 | 语义复用,后端换成 Paddle 算子 |
| `_wrap.lua` 双模式调用 | `bindings/lua/insight/_wrap.lua` | ~30 行 | **见下** |

**额外收获:`_wrap.lua` 比我在 §6.1 提的"末位 table"方案好。**
它支持三种写法同时成立:

```lua
ins.sum(x, 1, true)                    -- 纯位置
ins.sum{ x = x, axis = 1, keepdims = true }   -- 纯具名
ins.sum{ x, 1, true }                  -- table 里的位置
```

这比"位置参数 + 末位 table" 更接近 Python kwargs 的体验,而且**是纯 Lua 5.1 语法**
(`local unpack = table.unpack or unpack` 就是标准的 5.1 兼容垫片)。
**决定:Paddle-Lua 直接采用 `_wrap` 模式,§6.1 表里"末位 table"那条改掉。**

#### 6.1.3 ✅ 一处已定要修的不一致:Insight7 的 Lua 绑定里 `axis` 是 0-based

核对三个绑定后发现的:

| 绑定 | 元素索引 | `axis` 参数 | 一致? |
|---|---|---|---|
| Python | 0-based | 0-based | ✅ |
| **Julia** | 1-based | **1-based**(且因列主序做了维度翻转) | ✅ |
| **Lua** | **1-based** | **0-based(未做转换)** | ❌ |

证据 —— Julia 侧**明确处理了**:

```julia
# bindings/julia/Insight.jl:726-729
# Helper: convert Julia 1-based axis to Insight 0-based axis.
_julia_axis(arr, axis) = axis <= 0 ? axis : ndim(arr) - axis
```

Lua 侧**直接透传**:

```lua
-- bindings/lua/insight/indexing.lua:40
M.argsort = _wrap({ "x", "axis" }, function(x, axis)
  return native.argsort(x, axis or -1)      -- 没有 -1 转换
end)
-- reduction.lua:18  M.sum 同样直接透传 axis
```

**所以 Insight7 的 Lua 绑定里,`x["1:3"]` 是 1-based 而 `sum(x, axis)` 是 0-based。**
Julia 侧显然认真想过这件事,Lua 侧看起来是漏了 —— 这更像疏忽,不像刻意设计。

**这直接关系到 Paddle-Lua 的定案,有两条路:**

| 路线 | 内部一致性 | 与 Insight7 一致性 | 评价 |
|---|:-:|:-:|---|
| **甲:统统 1-based** | ✅ | ❌ 与 Insight7 **Lua** 绑定冲突;✅ 与 **Julia** 绑定同philosophy | 符合你"不然好混乱啊"的原意 |
| 乙:索引 1-based + axis 0-based | ❌ | ✅ 与 Insight7 Lua 现状一致 | 抄现状,但把疏忽固化 |

**建议走甲,并顺手修 Insight7 的 Lua 绑定。**
理由:同一个作者的 Lua 生态里,`axis` 出现两种约定比"和自己三个月前的实现不一致"糟糕得多;
而且 Julia 绑定已经证明了你原本的设计意图就是"跟随宿主语言"。

> ✅ **2026-08-03 已拍板:走甲,修 Insight7(R24 / D29)。**
> 人的原话:"Insight7 的 bug 以后是要修的,以后会顺手修复的,**成员索引和维度索引都要 1-base**"。
>
> 定性:**按 bug 修,不是 breaking change** —— 同一个 Lua 绑定里切片已经是 1-based
> (`insight_lua.cpp:212-266` 的 `lua_spec_to_cpp`),Julia 绑定的 axis 也已转换,
> 所以 Lua 侧 axis 属于**漏做**,库内部本就不自洽。
> 节奏:**顺手做,不专门排期,但要在 P12 之前完成。**
> 落地细节(负数轴保持、`axis=0` 报错、多轴逐元素转换)见 `plan/foundations.md` §3.4。

#### 6.1.4 后端映射:Paddle 不是 Insight7,有一处语义差

Insight7 的 `Array` 是**真正的 stride view**:

```cpp
// src/core/array.cpp:568-575
new_strides_vec[i] = strides_[i] * slices[i].step;    // step 为负 → 负 stride
return Array(*this, new_shape, new_strides, new_offset);
```

Paddle 侧的对应物已核对:

| | Insight7 | Paddle |
|---|---|---|
| 算子 | `Array::slice()` 直接改 strides | `strided_slice`(`ops.yaml:5443`) |
| 负 step | ✅ 负 stride,零拷贝 view | ✅ 语义支持,但**默认可能落成拷贝** |
| view 开关 | 无(总是 view) | **`FLAGS_use_stride_kernel`**(运行时开关) |
| stride kernel | — | `paddle/phi/kernels/stride/strided_slice_kernel.cc` 等 6 个 |

**结论:解析层可以逐字复用,后端映射要重写 —— 但重写量很小(映射到 `strided_slice` 而已)。**

**新增 M0 验证项**:`FLAGS_use_stride_kernel=true` 下,
`x["::-1"]` 走 `strided_slice` 是否真的返回 view、是否与 autograd 兼容。
这条不通不致命(退回拷贝语义,`x["::-1"]` 变成 `flip`),但会影响性能承诺,应当早知道。


### 6.2 `metrics-lua`(学 PaddleMetrics / torchmetrics)· **M3,设计先存档**

核心是 `add_state` 声明式状态机:`reset` / 设备迁移 / 分布式同步全自动。

```lua
local metrics = require "metrics"

local Acc = class(metrics.Metric)   -- pl.class,三仓库共用同一基座

function Acc:init(cfg)
  Acc.super.init(self)
  self.threshold = (cfg and cfg.threshold) or 0.5
  self:add_state("correct", paddle.zeros{1}, { dist_reduce_fx = "sum" })
  self:add_state("total",   paddle.zeros{1}, { dist_reduce_fx = "sum" })
end

function Acc:update(preds, target)
  local p = paddle.cast(paddle.greater_than(preds, self.threshold), "int64")
  self.correct = self.correct + paddle.sum(paddle.cast(paddle.equal(p, target), "float32"))
  self.total   = self.total + preds:numel()
end

function Acc:compute() return self.correct / self.total end
```

移植策略:**不移植 100+ 指标。** 先做 20 个覆盖 90% 场景的:

```
classification  Accuracy Precision Recall F1 AUROC AveragePrecision ConfusionMatrix
regression      MAE MSE RMSE R2 PearsonCorrCoef
image           PSNR SSIM
retrieval       MRR NDCG
text            BLEU
其他            MetricCollection MeanMetric SumMetric CatMetric
```

`MetricCollection` 必做 —— 它是 Ocean 侧 `self:log_dict()` 的对接点。

### 6.3 `ocean-lua`(学 PaddleOcean / Lightning)· **M3+,设计先存档**

**双模式,和 Ocean 一致:**

```lua
-- ① Lightning 风格:定义 hooks,Trainer 负责循环
local ocean = require "ocean"
local MyModel = class(ocean.Model)  -- pl.class,三仓库共用同一基座

function MyModel:init()
  MyModel.super.init(self)
  self.net = paddle.nn.Linear(28, 10)
end

function MyModel:training_step(batch, batch_idx)
  local loss = paddle.nn.functional.cross_entropy(self.net(batch[1]), batch[2])
  self:log("train_loss", loss)
  return loss
end

function MyModel:configure_optimizers()
  return paddle.optimizer.Adam{ learning_rate = 1e-3, parameters = self:parameters() }
end

local trainer = ocean.Trainer{ max_epochs = 10, accelerator = "gpu",
                               callbacks = { ocean.callbacks.ModelCheckpoint{ monitor = "val_loss" } } }
trainer:fit(model, train_loader, val_loader)

-- ② Keras 风格:快速原型
local model = ocean.Model{ net = paddle.nn.Sequential(...) }
model:prepare{ optimizer = opt, loss = loss_fn, metrics = { acc } }
model:fit(train_loader, { epochs = 10 })
```

**M3 范围内的 Ocean 子集(Ocean 有 ~140 个源文件,我们不全做):**

| Ocean 组件 | Lua 侧 M3 范围 | 说明 |
|---|---|---|
| `Model`(双模式) | ✅ 全做 | 核心价值 |
| `Trainer` fit/validate/test/predict | ✅ 全做 | |
| `DataModule` | ✅ | 薄,但很有用 |
| Callbacks(18) | ⚠️ **先做 8 个** | Checkpoint / EarlyStopping / LRMonitor / ProgressBar / Timer / GradAccum / ModelSummary / RichLog |
| Loggers(9) | ⚠️ **先做 3 个** | CSV / VisualDL / 内存 |
| Strategies(6) | ⚠️ **只做 SingleDevice** | 分布式测不了,已延后(§2.1.4) |
| Accelerators(7) | ⚠️ **CPU / CUDA** | 其余靠 `paddle.device` 泛化 |
| Precision(4) | ⚠️ **fp32 + AMP O1** | |
| `Gear`(Fabric 等价) | ⬜ M3 | 手动训练 API,优先级低于 Trainer |
| `cli` / `tuner` / `profilers` | ⬜ M3+ | |
| `distributed`(70+ API) | ⬜ 延后 | 见 §2.1.4:测不了就不写 |

**取舍原则:凡是 Ocean 里依赖"Python 生态集成"的(wandb/mlflow/comet/deepspeed),
一律不做 —— 那些库本身就没有 Lua 客户端。**

---

## 7. 里程碑

### M0 · 可行性钉死(2-3 周,1 人)

**目标:把所有"我不确定"变成"我知道"。一个功能都不做,只做验证。**
只要 M0 的 (1) 挂了,整个项目重新评估。

产出:`process/m0-report.md` + 一个能跑的最小 demo。

### M1 · MVP(4-6 人月)· 只做 A1/B1

**验收标准:纯 Lua 5.1 语法,无 Python 运行时,跑通 MNIST 全流程并收敛。**

范围(全部落在 §2.0 四象限的 ★ 第一优先格,唯二例外已标注):

- Tensor + ~200 个常用算子(生成)
- **字符串索引 `x["1:3, :"]`**(复用 Insight7 解析层 → 映射 `strided_slice`)
- **1-based 护栏**:`cross_entropy`/`embedding` 的 label 范围检查(§6.1.1 (c),必做)
- autograd backward
- `nn.Layer` 类系统(**A2**,但绕不开,必须先有)
- Linear / Conv2D / BN / ReLU / Dropout / Sequential
- `optimizer.SGD` / `Adam`
- `io.Dataset` / `DataLoader`(**单 worker**,先不上 Lanes)
- **`paddle.dataset` 的 MNIST / CIFAR**(§2.1.1:没有它 M1 验收无法闭环)
- `paddle.save` / `paddle.load`:**先接 C++ `SaveTensor`/`LoadTensor`(~20 行)**,
  纯 Lua pickle 解析器**降级为加分项**(`research/reuse.md` §5.1 —— 它不再阻塞 M1 验收)
- **[可选但强烈建议] `jit::Layer::Load` + `forward` 绑定(~几十行)**
  → 立刻拿到"Python 训练 / Lua 推理"这一完整产品形态(`research/reuse.md` §3)
- CPU only,**只支持 Lua 5.1 + LuaJIT**

**M1 刻意砍掉:GPU、多 worker、AMP、Lua 5.2/5.3/5.4。**
这些都是"再来一遍"型工作,不是"能不能做成"型工作。

### M2 · 核心 API 对齐(+8-12 人月)· 补齐 A1/B1,吃掉 A2/B1

- 算子扩到 yaml 全量
- GPU + AMP
- **Lanes 多 worker DataLoader**(A2/B1)
- Lua 5.2/5.3/5.4 全支持,3 OS × 5 Lua 版本 CI 矩阵
- `paddle.vision` / `linalg` / `fft` / `dataset` 全量(A1/B1)
- **index 语义标注表 + CI 未标注即报错**(§6.1.1 (a),长期维护机制)
- 完整 GC 机制(`research/gc.md` §4 的九层)
- `jit.Layer Load()` 只读推理绑定(A1,几十行)
- 文档站

### M3 · 生态与重活

按 §2.0 的排序,**A2/B1 先于 B2**:

| 顺序 | 内容 | 象限 | 前置条件 |
|---|---|---|---|
| 1 | **`metrics-lua`**(20 个指标) | A1/B1 | 核心 API 稳定 |
| 2 | **`ocean-lua`**(Trainer + 8 callback + 3 logger) | A1+A2/B1 | metrics-lua |
| 3 | `cpp_extension`(探测抄 + 构建重写 + Lua 包装生成) | A1+A2/B1 | §2.1.2 的 `OpProtoHolder` 验证通过 |
| 4 | **自研静态图 trace 模式**(直线图,**零 AST 依赖**) | A2/B1 | §2.1.3;复用 `InterpreterCore` |
| 4b | **script 模式(带控制流)** → **M4** | A2/B1 | `research/to-static.md`:vendored luacheck parser + 自写 printer + 源码契约 |
| 5 | `hub`(带 sandbox 设计) | A2/B1 | — |
| 6 | `distributed` | A1+A2/**B2** | **⚠️ 先有多卡测试环境,否则一行都不写** |

---

## 8. M0 验证清单(四篇文档汇总)

按"挂了会死"排序。

| # | 验证项 | 来源 | 挂了的后果 | 预估 |
|---|---|---|---|---|
| **1** | **`WITH_PYTHON=OFF` + `ON_INFER=OFF` 能否编出可用的 libpaddle** | `research/feasibility.md` §8 | 🔴 **项目不成立** | 3-5 天 |
| **2** | 该配置下 `egr::Backward` 能否真的跑通反向 | `research/feasibility.md` §1 | 🔴 项目不成立 | 1 天 |
| 3 | phi kernel / DeviceContext 多线程共享是否安全 | `research/dataloader.md` §3 | 🟠 多 worker 方案要改 | 2 天 |
| 4 | Lanes v3.17.x 的 deep API 与 master(4.x) 差异 | `research/dataloader.md` §9.5(b) | 🟠 换版本或自建线程池 | 1 天 |
| 5 | Lanes 与我们的绑定 C++ ABI 一致性(MSVC 尤其) | §9.5(c) | 🟠 构建矩阵复杂化 | 1 天 |
| 6 | Lanes 有无 `on_state_create` 类钩子(每 lane 初始化 tracer) | §9.5(d) | 🟡 需另找注入点 | 0.5 天 |
| 7 | `RegisterOOMCallback` + `lua_gc` 能否真的救回一次 OOM | `research/gc.md` §5 | 🟡 GC 方案降级 | 1 天 |
| 8 | LuaJIT 构建是否开启 GC64 | `research/gc.md` §2.1 | 🟡 大模型撞 2GB 墙 | 0.5 天 |
| 9 | `__close` 在 5.4 上对 userdata 是否按预期触发 | `research/gc.md` §5 | 🟢 少一个特性 | 0.5 天 |
| 10 | 堆追踪与 Lanes 自定义分配器是否冲突 | `research/dataloader.md` §9.5(e) | 🟢 调参 | 1 天 |
| 11 | 纯 Lua pickle 能否读通真实 `.pdparams`(protocol 4) | `research/architecture.md` §C | 🟢 退回 safetensors | 1 天 |
| ~~12~~ | ~~sol2 能否编过 Lua 5.5~~ | `research/gc.md` §6.4 | ✅ **已完成:不能,5.5 已出范围** | — |
| ~~13~~ | ~~Lanes 能否做成可选依赖~~ | `research/dataloader.md` §9.5(a) | ✅ **已决策:强制依赖,不需验证** | — |
| **14** | **`FLAGS_use_stride_kernel=true` 下 `x["::-1"]` 是否真 view、是否兼容 autograd** | §6.1.4 | 🟢 退回拷贝语义 | 0.5 天 |
| **15** | **`WITH_PYTHON=OFF` 下 `OpProtoHolder` 链是否还活着**(决定 `cpp_extension` 可行性) | §2.1.2 | 🟢 `cpp_extension` 延到 M3+ | 0.5 天 |

| **16** | **`WITH_PYTHON=OFF` 下 `new_executor` / `jit` 是否编出并可链接** | `research/reuse.md` §7 | 🟠 第 3/5 套执行器全部作废 | 随 #1 |
| **17** | **`jit::Layer Load()` 能否加载 Python 侧 `paddle.jit.save` 产物并 forward** | `research/reuse.md` §3 | 🟡 "Python训练/Lua推理"提前交付路径没了 | 1 天 |
| **18** | **C++ 侧手工构造 `AttributeMap` 喂 `run_program_ad_func` 的难度** | `research/reuse.md` §4 | 🟢 静态图退回自建反向 | 1 天 |
| **19** | **`SaveTensor`/`LoadTensor` 格式是否稳定、跨版本可读** | `research/reuse.md` §5.1 | 🟢 退回 pickle,M1 重新被阻塞 | 0.5 天 |

**17 项 × 约 17 天 → 3 周。**

> ⚠️ 顺序不是随意的:**第 1 项是唯一的项目级生死判定,必须第一个做。**
> 它挂了,后面 16 项全部作废 —— 别并行,别先做"看起来更有意思"的那几项。
> **第 16 项应该和第 1 项合并做**(同一次编译里一起看),不额外花时间。


---

## 9. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| `WITH_PYTHON=OFF` 是无人走过的路,可能一堆编译错误 | 🔴 高 | M0 第一件事;真挂了就退回"最小 patch 上游"(违背零改动,但不致命) |
| Paddle 上游 API 变动打断绑定 | 🟠 中 | 锁定 release tag;中间层隔离;生成器让重生成成本≈0 |
| GC 与显存的语言级不匹配 | 🟠 中 | `research/gc.md` §4 九层机制,`paddle.scope` 从第一天就作为文档默认写法 |
| 构建矩阵爆炸(3 OS × 5 Lua × CPU/GPU) | 🟠 中 | 中间层只编 1 次;M1 只支持 5.1+LuaJIT+CPU |
| **单人项目,13.5 人月是硬时长** | 🟠 中 | 分三仓库且**串行**;`metrics-lua`/`ocean-lua` 是纯 Lua,可外部贡献 |
| **index 语义标注表的长期维护** | 🟠 中 | §6.1.1 (a):Paddle 每加算子都可能要补。**必须配 CI 强制,未标注即报错,不能默认放行** |
| 与 Insight7 的 `axis` 约定分裂 | 🟡 | §6.1.3:两库必须同一个答案,待拍板 |
| 无人使用 | 🟡 | 先服务明确场景(嵌入式/游戏引擎/OpenResty 里做推理),别一上来对标 PyTorch。**`jit::Layer` 让这条在 M1 就能兑现**(`research/reuse.md` §3) |
| Lua 生态缺 numpy 等价物 | 🟡 | Tensor 本身就是;`paddle.io` 补 CSV/图像读取 |

---

## 10. 目录结构

```
paddle-lua/
├── csrc/
│   ├── capi/            paddle_capi.h/.c        纯 C,编 1 次
│   ├── capi_gen/        生成的算子 C ABI
│   ├── sol/             sol2 绑定,编 5 次
│   ├── lanes/           __lanesclone + worker 池
│   └── gc/              堆追踪 + OOM 回调
├── lua/paddle/
│   ├── init.lua         require "paddle" 入口
│   ├── slice.lua        字符串索引 1-based→0-based(移植自 Insight7)
│   ├── tensor.lua  autograd.lua  device.lua
│   ├── nn/  optimizer/  io/  vision/  amp/
│   └── serialize/       pickle.lua  npy.lua  safetensors.lua
├── tools/gen/           生成器(开发期 Python,不进发布包)
├── tests/               busted
├── rockspecs/
└── docs/

ocean-lua/    lua/ocean/{init,model,trainer,datamodule,callbacks,loggers}.lua
metrics-lua/  lua/metrics/{init,metric,collections,classification,regression}.lua
```

---

## 11. 当前状态与下一步

> ⚠️ **现在处于论证阶段,不开工。** 本节列的是"论证收敛后第一个动作是什么",不是排期。

### 11.1 论证阶段已收敛的决定

| # | 决定 | 状态 |
|---|---|---|
| 1 | Lua 5.1/5.2/5.3/5.4 + LuaJIT,5.5 出局 | ✅ 已证实(sol2 不支持) |
| 2 | 全 sol2 + 纯 C ABI 中间层 | ✅ 已定 |
| 3 | 零上游 Paddle 改动 | ✅ 已论证 |
| 4 | Lanes 强制依赖,Tensor 单一 sol2 usertype 表示(跨 lane 走 `__lanesclone`) | ✅ 已定 |
| 5 | 三仓库切分,**先主框架,Ocean/Metrics 排 M3** | ✅ 已定 |
| 6 | 优先级:A1 优于 A2;B1 优于 B2;冲突时 B 优先 | ✅ 已定 |
| 7 | **统统 1-based** | ✅ 已定(§6.1.3 的 Insight7 `axis` 分歧待拍板) |
| 8 | **字符串索引 `x["1:3, :"]`,复用 Insight7 解析层** | ✅ 已定 |
| 9 | `_wrap` 三模式调用 | ✅ 已定 |
| 10 | 排除清单从 9 项缩到 5 项(其余改判"延后") | ✅ 已修正 |
| 11 | **静态图复用 `InterpreterCore` + `run_program_ad_func`,不自己写引擎/反向** | ✅ 已定(`research/reuse.md`) |
| 12 | **`jit::Layer` 提到 M1 可选项**("Python训练/Lua推理"最早交付形态) | ✅ 已定 |
| 13 | **`paddle.save/load` 先走原生 `SaveTensor/LoadTensor`**,pickle 降为加分项 | ✅ 已定 |
| 14 | **DataLoader 不用 `new_executor/workqueue/`**,维持 Lanes | ✅ 已定(`research/reuse.md` §5.2) |

### 11.2 仍未收敛的

| # | 待决 | 谁能定 |
|---|---|---|
| 1 | **Insight7 Lua 绑定的 `axis` 要不要改成 1-based** | 只能你定,涉及已发布库的 breaking change |
| 2 | 多卡测试环境何时具备(决定 `distributed` 能否进日程) | 你 |
| 3 | 自研静态图第一版要不要碰控制流 | 建议不碰(trace 先行,script 排 M4),但你定 |
| 4 | 源码契约走"纯君子之约"还是"约定 + `string.dump` 往返验证" | 建议后者(`research/to-static.md` §4),待 M0 验证第 20 项 |

### 11.3 论证收敛后的第一个动作(唯一一个)

> 你说的对:开工时应该先 `WITH_PYTHON=OFF + ON_INFER=OFF` 编译一顿,因为这是基础。

**第一个动作就是且只是它。**

```
cmake -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=OFF ...
```

理由(不只是"它是基础"):

| | |
|---|---|
| **它是唯一的项目级生死判定** | M0 其余 12 项挂了都只是降级,这一项挂了整个项目重估 |
| **它大概率是没人走过的路** | `cmake/configure.cmake:15` 有 `-DPADDLE_NO_PYTHON`,但 `(NOT WITH_PYTHON) AND ON_INFER` 的组合守卫说明**上游预期的无 Python 场景是推理**。我们要的是 `ON_INFER=OFF` 的训练态无 Python —— 这个组合很可能从未被编译过 |
| **它的失败模式是"一堆编译错误"** | 这类问题只能靠实际编译暴露,再多论证也问不出来 |

**在它通过之前,其他一切(包括那些"独立可测、不亏"的纯 Lua 组件)都不应该开始。**
不是因为技术上有依赖,而是因为**它决定了这些组件有没有宿主。**

### 11.4 一旦第 1 项通过,才谈得上并行

那时才可以把这三件独立可测的事铺开:

1. 纯 Lua pickle 解析器(自身即是有价值的独立库)
2. `argsig` 参数签名层 + `slice.lua`(生态共用地基;Penlight 直接 rock 依赖,不再 vendor)
3. 从 yaml 生成算子绑定的生成器原型
