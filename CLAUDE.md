# CLAUDE.md · Paddle-Lua 智能体作业手册

> **你(智能体)在开始任何工作前必须完整读完本文件。**
> 本文件是**约束与流程**的唯一来源。技术论证在其他文档里,本文件不重复论证,只给结论和规则。

---

## 0. 三十秒定位

**项目**:给 PaddlePaddle 做一个 Lua 前端,让用户 `local paddle = require "paddle"`,
**不需要 Python 运行时**。

**当前阶段**:⚠️ **论证阶段。M0 第 1 项未通过之前,不写任何产品代码。**

**你最可能需要做的事,按概率排序**:
1. 执行 `process/tasks.md` 里下一个未完成的任务
2. 回答关于某个已定决策的问题 → 查 §2 决策台账,**不要重新论证**
3. 补充/修正文档 → 遵守 §7 的文档规则

---

## 1. 硬约束(不可协商,违反即回滚)

这 11 条是项目的地基。**任何代码、任何提案,违反其一即无效。**
如果你认为某条应该改,**停下来问人,不要自行变通。**

| # | 约束 | 检验方法 |
|---|---|---|
| **C1** | **不需要 Python 运行时**。发布产物中不得有任何 Python 依赖 | 发布包里搜不到 `.py`;`ldd`/`dumpbin` 无 `libpython` |
| **C2** | **零上游 Paddle 改动**。不提交任何 patch 到 Paddle 仓库 | Paddle 仓库 `git status` 必须干净 |
| **C3** | **我们写的 Lua 代码必须是 Lua 5.1 语法子集** | 见 `process/conventions.md` §1 + `tools/lint_51.lua` |
| **C4** | **面向用户的一切索引是 1-based**,含 `axis`/`dim`/返回的 index tensor/label | 见 `process/conventions.md` §5 |
| **C5** | **目标 Lua 版本 = 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT**。**不支持 5.5** | sol2 不支持 5.5,已证实 |
| **C6** | **Lanes 是强制依赖**,Tensor 跨线程走 `__lanesclone`,**不做双表示** | 无运行时"有无 Lanes"分支 |
| **C7** | **绑定层用 sol2,底下必须垫纯 C ABI 中间层**。C++ 异常不得穿过 Lua | 中间层头文件必须 `extern "C"` 且无 C++ 类型 |
| **C8** | **不手写算子绑定**。算子从 Paddle 的 yaml 生成 | `csrc/capi_gen/` 与 `lua/paddle/_ops.lua` 只能由生成器写 |
| **C9** | **生成器是开发期工具**,可以用 Python;但产物不得依赖 Python | 生成器在 `tools/gen/`,不进发布包 |
| **C10** | **不发明 Paddle C++ API**。任何 Paddle 符号必须先在源码里读到才能用 | 见 §4 |
| **C11** | **生态基座只有一套**:类系统 = `pl.class`,集合 = `pl.List`,跨版本兼容 = `pl.compat`,数组(numpy 的位置)= Insight7,**参数检查 = `_args`**。**不得引入第二套** | 见 `process/conventions.md` §2 + `plan/foundations.md` |

---

## 2. 决策台账(已定,不要重新论证)

**下面每一条都经过论证并已定案。你看到它们时的正确反应是"照做",不是"评估"。**
若确有新证据推翻某条,写进 `process/decisions.md` 的"翻案记录",并**先问人**。

| # | 决策 | 依据文档 |
|---|---|---|
| D1 | 全部用 sol2 绑定(含 LuaJIT,不走 FFI) | `research/architecture.md` §B |
| D2 | 纯 C ABI 中间层,只编一次;sol2 层薄,编 5 次 | `research/architecture.md` §B |
| D3 | 零上游改动可行 | `research/architecture.md` §A |
| D4 | 5.5 出局(sol2 三个 issue 全 open,且报错形态是一堆编译错误) | `research/gc.md` §6.4 |
| D5 | GC 用 Torch7 九层方案;`RegisterOOMCallback` 已是公共 API | `research/gc.md` §4 |
| D6 | 堆追踪**默认开启**(Torch7 就是默认开的) | `research/gc.md` §6.3 |
| D7 | DataLoader 用线程不用进程;用 Lanes deep userdata + Linda | `research/dataloader.md` §9 |
| D8 | Lanes 强制依赖 | `research/dataloader.md` §9.5(a) |
| D9 | **不**用 `new_executor/workqueue/` 做 DataLoader | `research/reuse.md` §5.2 |
| D10 | `paddle.Model` / `paddle.metric.*` 不移植 | `plan/overview.md` §2.2 |
| D11 | 三仓库切分;**先主框架**,Ocean/Metrics 排 M3 | `plan/overview.md` §1.1 |
| D12 | 优先级:A1>A2,B1>B2,**冲突时 B 优先** | `plan/overview.md` §2.0 |
| D13 | 统统 1-based | `plan/overview.md` §6.1.1 |
| D14 | 字符串索引 `x["1:3, :"]`,移植 Insight7 的 `lua_spec_to_cpp` | `plan/overview.md` §6.1.2 |
| D15 | `_wrap` 三模式调用(位置/具名 table/table 内位置) | `plan/overview.md` §6.1.2 |
| D16 | 静态图复用 `InterpreterCore`,**不自己写执行引擎** | `research/reuse.md` §2.1 |
| D17 | 静态图反向用 `run_program_ad_func`,**不自己建反向图** | `research/reuse.md` §4 |
| D18 | `jit::Layer` 提到 M1 可选项("Python训练/Lua推理") | `research/reuse.md` §3 |
| D19 | `paddle.save/load` 先走 `SaveTensor`/`LoadTensor`;pickle 降为加分项 | `research/reuse.md` §5.1 |
| D20 | 动转静:**trace 模式(M3,零 AST)先于 script 模式(M4)** | `research/to-static.md` §0 |
| D21 | AST 用 vendored luacheck `parser/lexer/utils`;printer 自己写 | `research/to-static.md` §5 |
| D22 | 源码契约 = 君子之约 + `string.dump` 往返验证 | `research/to-static.md` §4 |
| D23 | **`nn.Layer` 自动注册**(实例 raw 表保持空 + 私有 `FIELDS` 键) | `plan/modules/09-nn.md` §3.2 |
| D24 | **Tensor 跨 lane 用 sol2 usertype + `__lanesclone`**,不用 deep userdata | `plan/modules/13-lanes.md` §4.1 |
| D25 | **类系统用 `pl.class`**(vendored 受限 Penlight,11 文件纯 Lua);集合返回 `pl.List` | `plan/foundations.md` §1 §2 |
| D26 | **Insight7 顶替 numpy 的位置**,一等公民、软强制依赖 | `plan/foundations.md` §3 |
| D27 | **`nn.LayerList` 继承 `Layer`**(Layer 优先)。`is_a(nn.Layer)` 必须为真;因此 `ipairs(ml)` 在 5.1 上跑空,**一律用 `ml:iter()` / `ml:len()`** | `plan/foundations.md` §2.4 |
| D28 | **「不引入新的强制 C 依赖」已取消。** 判据改为**边际成本**:先看我们自己的 `.so` 能不能顺手做,做不了就引入 | 本文件 §9.1 |
| D29 | **Insight7 的 `axis` 按 bug 修成 1-based**(成员索引与维度索引都要 1-based)。顺手做,P12 之前完成 | `plan/foundations.md` §3.4 |
| D30 | **参数检查:取 argcheck 的形,不取它的实。** 规则表 schema + `usage.lua` 照搬,求解器自写 `_args.lua`(~150 行,O(N))。**不 vendor argcheck 本体** —— 它 3^N 枚举,编不出 `Conv2D`(11 可选)/`DataLoader`(16 可选) | `plan/foundations.md` §4.5 |

---

## 3. 阶段闸门(Gate)

**闸门未开时,做闸门后的事 = 无效工作。**

```
G0 ──▶ M0 验证第 1 项:WITH_PYTHON=OFF + ON_INFER=OFF 能编出可用 libpaddle
        ⛔ 未通过 → 除文档外,不写任何代码
                    不建目录结构,不写 rockspec,不写 CMakeLists

G1 ──▶ M0 全部 17 项必做项完成,产出 M0_REPORT.md
        ⛔ 未通过 → 不进入 M1

G2 ──▶ M1 验收:纯 Lua 5.1,无 Python,MNIST 训练收敛
        ⛔ 未通过 → 不进入 M2
```

**当前闸门状态**:见 `process/status.md`。**每次开工前先读它。**

> 为什么 G0 这么严:`cmake/configure.cmake:15` 有 `-DPADDLE_NO_PYTHON`,
> 但守卫写法是 `(NOT WITH_PYTHON) AND ON_INFER` —— 说明上游预期的"无 Python"场景**是推理**。
> 我们要的是 `ON_INFER=OFF` 的**训练态**无 Python,这个组合很可能从未被编译过。
> 它挂了,整个项目重估。**没有任何理由先做别的。**

---

## 4. 反幻觉规程(最重要的一节)

**这个项目最大的失败模式不是写错代码,是编造 Paddle API。**
生成的绑定编不过还能发现;编造的函数签名可能编过、跑通、结果错。

### 4.1 铁律

> **任何 Paddle C++ 符号,你必须先在 Paddle 源码里亲眼读到,才能写进代码。**
> **"我记得 Paddle 有个 XXX" 一律不成立。**

### 4.2 强制流程

引用任何 Paddle 符号前:

```bash
# 1. 定位
grep -rn "SymbolName" $PADDLE_ROOT/paddle --include=*.h | head

# 2. 读到完整签名
sed -n 'N,Mp' $PADDLE_ROOT/paddle/path/to/header.h

# 3. 在代码注释里写下出处
//  paddle/phi/api/include/tensor.h:388
```

**每一处 Paddle API 调用,注释里必须有 `文件路径:行号`。**
这不是繁文缛节 —— 它让下一个智能体能在 30 秒内验证你没编。

### 4.3 `$PADDLE_ROOT` 的发现顺序

```
1. 环境变量 PADDLE_ROOT
2. STATUS.md 里记录的路径
3. 向上级目录找含 paddle/phi/api/include/tensor.h 的目录
4. 都找不到 → 停下来问人,不要猜
```

### 4.4 不确定时的正确行为

| 情况 | 错误做法 | 正确做法 |
|---|---|---|
| 不确定某函数是否存在 | 先写着,编译时再说 | `grep` 确认;找不到就在 `process/open-questions.md` 记一条并**停** |
| 签名记不清 | 按印象写 | 读头文件 |
| 某个行为不确定 | 猜一个合理的 | 写最小 C++ 测试跑一下;跑不了就记 OPEN_QUESTION |
| 文档和源码冲突 | 信文档 | **信源码**,并修文档 |

---

## 5. 工作流程

### 5.1 每次开工

```
1. 读 STATUS.md          → 当前闸门、当前任务、上次卡在哪
2. 读 TASKS.md           → 找到下一个 status=TODO 且前置已满足的任务
3. 确认该任务在当前闸门允许范围内(§3)
4. 干活
5. 更新 STATUS.md 和 TASKS.md
```

### 5.2 每个任务的完成定义(DoD)

一个任务只有满足**全部**下列条件才算完成:

- [ ] 代码通过 `tools/lint_51.lua`(如果是 Lua 代码)
- [ ] 所有 Paddle API 调用带 `文件:行号` 出处注释
- [ ] 有对应测试,且在**目标 Lua 版本上**跑过
- [ ] `process/tasks.md` 里状态更新为 DONE,并写下实际耗时
- [ ] 如果推翻了任何已有结论 → 已写进 `process/decisions.md` 翻案记录

### 5.3 遇到阻塞

**不要绕过,不要降级实现,不要"先 TODO 着"。**

```
1. 在 OPEN_QUESTIONS.md 记一条:现象 / 已排查的 / 需要什么才能继续
2. STATUS.md 标记该任务为 BLOCKED
3. 换到下一个无依赖的任务;若没有,停下来报告
```

> 理由:这个项目里"先 TODO 着"的成本极高。
> 一个被跳过的 API 语义差异,会在 nn 层堆到几十处才暴露。

---

## 6. 文档地图

**仓库结构**

```
CLAUDE.md            ← 本文件,唯一在根目录的作业文档
README.md            英文(默认)
README.zh-CN.md      简体中文
README.zh-TW.md      繁體中文
docs/
└── migration/       ← 目前 docs/ 下唯一的子目录:完整的 Paddle-Lua 构建计划
    ├── WORKPLAN.md        ★★ 总工程树:DFS 遍历它,遍历完 = 工程完成。**执行入口**
    ├── README.md          计划总索引
    ├── plan/              做什么、什么顺序、怎么做
    │   ├── overview.md        范围与总账
    │   ├── roadmap.md         ★ 阶段顺序 P0–P18:依赖图、开工条件与完工判据
    │   ├── layout.md          ★ 代码目录与模块清单 + 文件级落地顺序
    │   ├── ci.md              ★ CI 计划:四层、阶段解锁表、红线
    │   ├── foundations.md     ★ 生态基座:Penlight 与 Insight7(跨阶段,P0 前就该读)
    │   ├── api/               ★ 按 Lua 模块的接口设计(先接口后实现)
    │   │   ├── README.md          模块总清单 + 跨模块硬规则 + 骨架
    │   │   └── io.md              paddle.io —— 样板文档
    │   └── modules/           ★ 19 份实现设计,每阶段一份
    │       ├── README.md          模块索引 + 固定写作骨架
    │       ├── 00-build.md        无 Python 构建
    │       ├── ...                (完整清单见 modules/README.md)
    │       └── 18-script.md       script 模式
    ├── process/           现在做到哪了、下一步做啥(高频变更)
    │   ├── status.md          当前闸门 / 任务 / 环境记录
    │   ├── tasks.md           原子任务板
    │   ├── conventions.md     代码约定
    │   ├── decisions.md       决策理由 + 翻案记录
    │   └── open-questions.md  未解问题
    └── research/          能不能做、怎么做才对(结论冻结)
        ├── feasibility.md     可行性主报告
        ├── architecture.md    绑定层路线
        ├── gc.md              显存生命周期
        ├── dataloader.md      多 worker
        ├── reuse.md           可复用 C++ 资产
        ├── to-static.md       动转静
        └── _ref/              外部参考源码出处
```

> **路径约定:** 本文件及 `docs/migration/` 内所有文档中,形如 `process/status.md`、
> `research/reuse.md` 的路径**一律相对 `docs/migration/`**。
> 只有 `CLAUDE.md` 和三个 README 在仓库根目录。
>
> **`docs/` 下不要再新建平级目录**,除非是与"迁移计划"无关的新用途
> (例如未来的 `docs/api/` 放 API 使用说明、`docs/tutorials/` 放教程)。
> 迁移计划的任何新文档,都放进 `docs/migration/` 的 `plan/` `process/` `research/` 之一。

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| **`/CLAUDE.md`** | 本文件:约束、决策、流程 | **每次开工必读** |
| **`docs/migration/WORKPLAN.md`** | **总工程树。全树唯一的 🔵 就是你的位置** | **每次开工必读,第二个读** |
| **`process/status.md`** | 当前闸门/任务/环境路径 | **每次开工必读** |
| **`process/tasks.md`** | 任务板,原子任务 + 验收标准 | 每次开工必读 |
| `process/conventions.md` | Lua 5.1 子集规则、命名、API 映射规则 | 写代码前 |
| **`plan/roadmap.md`** | P0–P18 的顺序与门槛 | **每次开工必读** |
| **`plan/api/README.md`** | 模块清单 + 命名映射 + 返回类型 + 惰性加载规则 | **设计任何模块接口前** |
| **`plan/api/<模块>.md`** | 该模块的导出契约 | **实现该模块前先写完它** |
| **`plan/layout.md`** | 目录树、模块清单、文件级落地顺序 | **建目录 / 决定先写哪个文件时** |
| **`plan/ci.md`** | 四层 CI、阶段解锁表、红线 | **每个阶段完工前**(没 CI 不算完工) |
| **`plan/foundations.md`** | 生态基座:Penlight / Insight7 的选型、闭包、坑 | **写任何 Lua 代码前必读一次** |
| **`plan/modules/<阶段>.md`** | 当前阶段的详细设计 | **开工前读当前这一份,不要读全部** |
| `plan/overview.md` | 范围表、工作量、风险登记 | 需要全局视野时 |
| `research/feasibility.md` | 可行性主报告 | 想知道"为什么可行" |
| `research/architecture.md` | 路线修正(sol2/C ABI/pickle) | 涉及绑定层设计 |
| `research/gc.md` | 显存生命周期、Torch7 先例 | 涉及内存/GC |
| `research/dataloader.md` | 多 worker、Lanes | 涉及 `paddle.io` |
| `research/reuse.md` | Paddle 可复用 C++ 资产清单 | **动手前必读**,避免重造轮子 |
| `research/to-static.md` | 动转静、源码契约、AST 选型 | 涉及 jit |
| `process/decisions.md` | 决策记录 + 翻案记录 | 想改已定决策时 |
| `process/open-questions.md` | 未解问题 | 卡住时写、开工时扫一眼 |
| `docs/migration/README.md` | 计划总索引 | 找不到东西时 |

**阅读优先级:**

```
/CLAUDE.md  →  process/status.md  →  plan/roadmap.md  →  plan/modules/<当前阶段>.md
(约束/决策)      (做到哪了)          (下一步是哪个阶段)     (这个阶段怎么做)
```

不要一上来通读全部 —— 总量约 8800 行,会吃掉大量上下文。
**`plan/modules/` 下只读当前阶段那一份**,其余 18 份与你当前的任务无关。

**语言约定**
- 三个 README:**英文为默认**,另有 zh-CN / zh-TW,三份内容必须同步
- `docs/` 下的技术文档:**中文**(内部工作文档,不做多语)
- 代码注释与 commit message:**英文**

---

## 7. 文档规则

### 7.1 写文档时

- **结论在前,论证在后。** 智能体和人都是先看结论
- **每个事实性断言必须有出处**:`文件:行号` 或 URL
- **区分"已核实"和"推测"**,用 ✅ / ⚠️ / ❌ 标记
- **推翻旧结论时,保留旧结论并标注"已被 X 推翻"**,不要静默删除
  > 理由:后来者会疑惑"这条为什么没考虑",保留翻案痕迹能省掉重复论证

### 7.2 禁止

- ❌ 在多个文档里重复同一份论证 → 一处论证,他处引用
- ❌ 无出处的性能数字 / 行数估算 → 标"估"
- ❌ 把未验证的东西写成已验证
- ❌ 在仓库根目录新建除 `CLAUDE.md` / 三个 README / `LICENSE` 以外的文件
- ❌ 在 `docs/` 下直接放松散文件 —— 必须落进某个子目录

### 7.3 新文档放哪儿

先问一句:**这份文档是"怎么把 Paddle 迁到 Lua"的一部分吗?**

- 是 → 进 `docs/migration/`,再按下表选子目录
- 否(例如给最终用户看的 API 说明、教程、benchmark 报告)→ 新开 `docs/<用途>/`,
  并在本文件 §6 的树里登记

| 这份文档回答的问题 | 放进 |
|---|---|
| 能不能做?怎么做才对?(一次性论证,结论冻结) | `docs/migration/research/` |
| 范围有多大?什么顺序? | `docs/migration/plan/`(`overview.md` / `roadmap.md`) |
| **某个阶段具体怎么做?** | `docs/migration/plan/modules/<NN>-<名字>.md` |
| 现在做到哪了?下一步做啥?怎么写代码? | `docs/migration/process/` |

**新增模块文档必须照抄 `plan/modules/README.md` 里的六节骨架**,不要自创结构。
六节是:做什么 / 上游有什么可用 / 设计 / 已知的坑 / 验收 / 未解问题。

新增文档后,**必须**同时更新 `docs/migration/README.md` 的文档索引和本文件 §6 的表格。
两处都不更新的文档等于不存在。

### 7.4 三个 README 的同步

`README.md`(英文,默认)/ `README.zh-CN.md` / `README.zh-TW.md` **内容必须等价**。
改其中一个就要改另外两个,不允许出现"英文版有中文版没有"的段落。

README 是**对外**的门面,不是工作文档:

- 只写**已经定下来**的事,不写待决策项、不写任务进度
- 状态徽章要跟 `process/status.md` 的当前闸门一致
- 里程碑表只写交付物,不写人月分解的细节(细节在 `plan/overview.md`)

---

## 8. 代码约定速查(详见 `process/conventions.md`)

### 8.1 Lua 5.1 子集 —— 最常踩的 8 个坑

| ❌ 不能用 | ✅ 用这个 | 来自 |
|---|---|---|
| `goto` / `::label::` | 重构控制流 | 5.2 |
| `//`(整除) | `math.floor(a/b)` | 5.3 |
| `&` `|` `~` `<<` `>>` | 自己写位运算函数 | 5.3 |
| `table.unpack` | `local unpack = table.unpack or unpack` | 5.2 |
| `<close>` / `<const>` | 普通 local | 5.4 |
| `math.type` / 整数子类型 | 只假设 double | 5.3 |
| `\z` `\x41` `\u{}` 转义 | 显式拼接 / `string.char` | 5.2/5.3 |
| `#` 用于有洞的表 | 显式存长度 | 全版本未定义行为 |

**但注意区分语法与运行时特性**(`research/gc.md` §3.3):
`collectgarbage("generational")` 是**运行时字符串参数**,不是语法 →
可以用 `pcall` 保护后使用。C 侧挂 `__close` 元方法也不受此约束。

### 8.2 索引转换规则(C4)

| 类别 | 规则 |
|---|---|
| 正数 `axis`/`dim` | Lua `n` → C++ `n-1` |
| **负数 `axis`/`dim`** | **不变**(`-1` 仍是最后一维) |
| 切片 | Lua `{i,j}` 闭区间 → C++ `[i-1, j)` |
| 吃 index 的 tensor | `-1` |
| 吐 index 的 tensor | `+1` |
| label / token id | **也是 1-based** |

⚠️ **`ops.yaml` 里没有"哪个参数是 index 语义"的信息。**
这张标注表在 `tools/gen/index_semantics.lua`,**人工维护**。
生成器遇到未标注的新算子必须**报错**,不得默认放行 ——
默认放行会产生静默的 off-by-one,这是本项目最难查的一类 bug。

### 8.3 生成 vs 手写

```
csrc/capi_gen/          ← 生成,禁止手改
lua/paddle/_ops.lua     ← 生成,禁止手改
csrc/sol/gen_*.cpp      ← 生成,禁止手改
其余                     ← 手写
```

每个生成文件头部必须有:

```
-- GENERATED BY tools/gen/xxx.py FROM <paddle yaml path> @ <paddle git sha>
-- DO NOT EDIT
```

---

## 9. 明确不要做的事

| ❌ | 为什么 |
|---|---|
| 改 Paddle 仓库任何文件 | C2 |
| 为了绕过困难而放宽 Lua 5.1 约束 | C3。5.1 语法在任何项目里都能用,这是核心卖点 |
| ~~引入新的强制 C 依赖(除 sol2/Lanes)~~ | ❌ **本条已于 2026-08-03 由人取消。** 需要 C 库就引入。**但成本没有消失**,见 §9.1 的取舍程序 |
| ~~用 `pl.path` / `pl.dir` / `pl.file` / `pl.app`~~ | ❌ **禁令随上条一并取消。** 需要就用,`lfs` 直接进依赖 |
| 用 `class.cast` | 它绕过 `_create`,造出没有 `FIELDS` 的破 Layer 实例(`plan/foundations.md` §1.5) |
| 自己再写一套 class / list / 参数检查 | C11 |
| 直接依赖 `argcheck` 这个 rock | D30。它编不出 `Conv2D` / `DataLoader` 的签名(`plan/foundations.md` §4.5) |
| 在参数检查里枚举「哪些参数被省略」的组合 | 那是 3^N。9 个可选参数 = 1.37 MB 生成代码,10 个直接编不出来 |
| 支持「位置参数中间省略、靠类型猜」 | Python 不允许这么写,而支持它要付上一行的代价。跳过参数就用具名表 |
| 手写算子绑定"就这一个特例" | C8。特例会繁殖 |
| 移植 `paddle.Model` / `paddle.metric.*` | D10 |
| 在 M0 通过前写产品代码 | G0 |
| 写 `distributed` 相关代码 | 没有多卡测试环境前一行都不写(`plan/overview.md` §2.1.4) |
| 用 `new_executor/workqueue/` 做 DataLoader | D9 |
| 实现 Lua 侧的静态图执行引擎 | D16,用 `InterpreterCore` |
| 自己建静态图的反向图 | D17,用 `run_program_ad_func` |
| 假设某个 Paddle API 存在 | §4 |
| 静默跳过失败的测试 | §5.3 |

### 9.1 引入 C 依赖的取舍程序(替代原来的禁令)

**禁令取消了,成本没取消。** 每个 C 依赖仍然要 ×5 Lua 版本 ×3 OS 构建、
仍然会出现在用户的安装失败报告里。所以不是随便加,而是走这三步:

```
1. 先问:我们自己的 C++ 层能不能顺手做?
   我们已经有一个链着 libpaddle 的 .so,加一个函数的边际成本 ≈ 0,
   而新 rock 的边际成本 = 15 个构建组合。
   典型:文件系统 -> std::filesystem(C++17,Paddle 本来就要求)
2. 做不了 / 明显不划算,就直接引入,不要绕。
   典型:HTTP+TLS(自己写 = 重造 luasec)、JPEG/PNG 解码、zlib
3. 引入时必须同时给出:
   - 它在 5.1/5.2/5.3/5.4/LuaJIT × Win/Linux/macOS 上都有 rock 的证据
   - 装不上时的降级路径(硬失败还是软失败)
   - 写进 `plan/overview.md` §5 的依赖表
```

**判据是"边际成本",不是"数量"。** 以前那条禁令的问题在于它把
「我们自己的 .so 里加个函数」和「新增一个 rock」当成同一件事来禁,
结果是把 §3.1 的下载能力和 §3.3 的图像解码逼进了死胡同。

---

## 10. 环境无关性

**你可能在任何机器上启动。不要假设路径、编译器、是否有 GPU。**

开工时按此顺序确定环境:

```
1. 读 STATUS.md 的"环境记录"节 —— 若与当前机器不符,重新探测并更新
2. 探测:
   - $PADDLE_ROOT           (§4.3)
   - 可用的 Lua 版本         lua -v / luajit -v
   - 编译器                  cc --version / cl
   - 有无 GPU               nvidia-smi
3. 把探测结果写回 STATUS.md 的"环境记录",附机器标识
```

**若当前环境不满足任务前置条件(如没有 GPU 却要做 GPU 任务):
不要模拟、不要假装通过,直接在 TASKS.md 标 `BLOCKED-ENV` 并换任务。**

---

## 11. 与人交互的准则

- **不确定就问,不要猜。** 本项目的错误成本远高于一次提问的成本
- **报告时先说结论,再说证据。** 不要复述过程
- **发现自己之前搞错了,直接说"我错了"并给出修正**,不要淡化
- **不要为了让报告好看而夸大进展。** 进度虚报会让闸门失效
- 涉及 §1 硬约束或 §2 决策台账的改动,**必须问人**

---

## 12. 本文件的维护

- 新增硬约束 → §1,并同步 `process/decisions.md`
- 新增已定决策 → §2
- 闸门状态变化 → 改 `process/status.md`,**不改本文件**
- 本文件应保持在 300 行以内。细节放专题文档,这里只留索引
