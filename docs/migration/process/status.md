# STATUS.md · 当前状态

> **每次开工第一个读的文件。每次收工最后一个更新的文件。**
> 本文件记录**易变**信息;稳定的约束和决策在 `CLAUDE.md`。

---

## 1. 当前闸门

```
G0  ⬜ 未通过   WITH_PYTHON=OFF + ON_INFER=OFF 能编出可用 libpaddle
G1  ⬜ 未开始   M0 全部 17 项必做验证完成
G2  ⬜ 未开始   M1 验收:MNIST 训练收敛
```

**当前允许做的事:仅文档工作 + M0 验证。**
**⛔ 不得创建 `csrc/` `lua/` 目录,不得写 rockspec/CMakeLists。**

---

## 2. 当前任务

> **位置以 `docs/migration/WORKPLAN.md` 里唯一的 🔵 为准。** 本节只记现场笔记。

| | |
|---|---|
| 阶段 | **论证 / 文档** |
| 工程树位置 | `WORKPLAN.md` 节点 **0.14**(参数检查选型 argcheck)-> 下一个 **0.15 L0 CI 落地** |
| 下一个动作 | **0.15** L0 CI(不需要 libpaddle,现在就能建,见 `plan/ci.md` §2),之后才是 **1.1 无 Python 构建**。**并行**:M0 #22/#23(argcheck 的 Q-17,纯 Lua,不阻塞于 G0,但必须在 P5 前闭掉) |
| 全部 ⛔ 阻塞节点 | `WORKPLAN.md` 4.3 distributed —— 无多卡环境(待拍板 P2) |
| 待人拍板 | ~~P1(Insight7 `axis`)~~ ✅ **2026-08-03 已拍板:改**(R24)。~~P9(还 vendor Penlight 吗)~~ ✅ **2026-08-03 已拍板:不 vendor,全生态 rock 依赖**(R30)。剩余待拍板:P3 / P4 / P6 / P7(改写)/ **P10(签名层叫什么)** —— 见 `process/decisions.md` §3 §4 |
| 上次卡在哪 | — |

---

## 3. 环境记录

> 智能体开工时若发现与当前机器不符,**重新探测并覆盖本节**,附机器标识。

| 项 | 值 | 探测时间 |
|---|---|---|
| 机器标识 | `E:\code` (Windows) | 2026-08-03 |
| `$PADDLE_ROOT` | `E:\code\paddle` | 2026-08-03 |
| Paddle 分支 | `windows_dataloader_multiprocess` | 2026-08-03 |
| Paddle 版本 | 3.5.0 | 2026-08-03 |
| 本仓库 | `E:\code\paddle-lua` | 2026-08-03 |
| Insight7(参考实现) | `E:\code\Insight7` | 2026-08-03 |
| vcpkg | `E:\code\vcpkg`(有 lua 5.5.0 / sol2 3.5.0 / luajit) | 2026-08-03 |
| Lua 版本 | 未探测 | — |
| 编译器 | 未探测 | — |
| GPU | 未探测 | — |
| 代理 | `http://127.0.0.1:7890`(http/https 均可) | 2026-08-03 |

---

## 4. M0 验证进度

| # | 项 | 状态 |
|---|---|---|
| 1 | `WITH_PYTHON=OFF`+`ON_INFER=OFF` 编译 | ⬜ **生死判定,先做这个** |
| 2 | 该配置下 `egr::Backward` 跑通反向 | ⬜ |
| 3 | phi kernel / DeviceContext 多线程安全 | ⬜ |
| 4 | Lanes v3.17.x deep API 与 4.x 差异 | ⬜ |
| 5 | Lanes 与绑定的 C++ ABI 一致性 | ⬜ |
| 6 | Lanes `on_state_create` 钩子 | ⬜ |
| 7 | `RegisterOOMCallback`+`lua_gc` 救回 OOM | ⬜ |
| 8 | LuaJIT GC64 | ⬜ |
| 9 | `__close` 对 userdata(5.4) | ⬜ |
| 10 | 堆追踪与 Lanes 分配器冲突 | ⬜ |
| 11 | 纯 Lua pickle 读 `.pdparams` | ⬜ |
| 12 | ~~sol2 + Lua 5.5~~ | ✅ 已完成:不支持,5.5 出局 |
| 13 | ~~Lanes 可选依赖~~ | ✅ 已决策:强制依赖 |
| 14 | `x["::-1"]` 走 `strided_slice` 是否 view | ⬜ |
| 15 | `WITH_PYTHON=OFF` 下 `OpProtoHolder` 链 | ⬜ |
| 16 | `WITH_PYTHON=OFF` 下 `new_executor`/`jit` 可链接 | ⬜ 与 #1 合并做 |
| 17 | `jit::Layer Load()` 加载 `jit.save` 产物 | ⬜ |
| 18 | 手工构造 `AttributeMap` 喂 `run_program_ad_func` | ⬜ |
| 19 | `SaveTensor`/`LoadTensor` 格式稳定性 | ⬜ |
| 20 | `string.dump` 往返比对(服务 M4) | ⬜ 可跳过 |
| 21 | luacheck parser 独立跑 5.1(服务 M4) | ⬜ 可跳过 |
| 22 | **LuaJIT 上 `debug.setupvalue` 注入 upvalue 是否可用**(`_args` 的命脉,Q-17) | ⬜ **纯 Lua,不依赖 libpaddle,现在就能做** |
| 23 | **`loadstring`/`load` 在 5 个 Lua × 3 OS 上行为一致**(chunkname 与错误行号) | ⬜ 同上,现在就能做 |
| 24 | **三道墙的实测值在 5 个 Lua 上各是多少**(upvalue 上限 / 局部变量上限 / 寄存器上限)。5.1 已测 = 60 / 200 / N=122 | ⬜ 同上。基准用 Paddle 最大签名 43 参数(`foundations.md` §5.4.6) |

**必做 20 项(#20/#21 可跳过),预估 3 周。**

> **#22/#23 不阻塞于 G0** —— 它们是纯 Lua 验证,`WITH_PYTHON=OFF` 编不编得出来与它们无关。
> 挂了的后果是明确的:`_args` 走**解释式降级路径**(`plan/foundations.md` §4.7)。
> 越早知道越好,因为 P5 之后再改就要重写所有已写的构造期签名。

---

## 5. 文档完成度

> 路径相对 `docs/migration/`;`CLAUDE.md` 与三个 README 在仓库根目录。

### 5.1 顶层

| 文档 | 行数 | 状态 |
|---|---|---|
| **`WORKPLAN.md`(总工程树)** | 248 | ✅ **新增** |
| `/CLAUDE.md` | 481 | ✅ |
| `/README.md`(英文,默认) | 192 | ✅ |
| `/README.zh-CN.md` | 186 | ✅ |
| `/README.zh-TW.md` | 186 | ✅ |
| `README.md`(计划总索引) | 148 | ✅ |

### 5.2 `plan/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `plan/overview.md` | 915 | ✅ |
| `plan/roadmap.md` | 344 | ✅ |
| `plan/foundations.md` | 1082 | ✅ **+参数检查(§4)+ 基座边界与解析器项目(§5)** |
| `plan/argsig.md` | 414 | ✅ **新增**(孵化说明书,建仓后迁走)|
| `plan/layout.md` | 261 | ✅ **新增** |
| `plan/ci.md` | 239 | ✅ **新增** |
| `plan/api/README.md` | 171 | ✅ **新增** |
| `plan/api/io.md` | 236 | ✅ **新增**(样板)|
| `plan/api/<其余 15 个模块>` | — | ⬜ 各模块开工时写 |
| `plan/modules/README.md` | 70 | ✅ |
| `plan/modules/00-build.md` | 115 | ✅ |
| `plan/modules/01-c-abi.md` | 150 | ✅ |
| `plan/modules/02-binding.md` | 173 | ✅ |
| `plan/modules/03-codegen.md` | 164 | ✅ |
| `plan/modules/04-packaging.md` | 96 | ✅ |
| `plan/modules/05-tensor.md` | 156 | ✅ |
| `plan/modules/06-autograd.md` | 147 | ✅ |
| `plan/modules/07-serialization.md` | 126 | ✅ |
| `plan/modules/08-jit-infer.md` | 115 | ✅ |
| `plan/modules/09-nn.md` | 360 | ✅ |
| `plan/modules/10-optimizer.md` | 135 | ✅ |
| `plan/modules/11-io.md` | 123 | ✅ |
| `plan/modules/12-dataset-vision.md` | 129 | ✅ |
| `plan/modules/13-lanes.md` | 163 | ✅ |
| `plan/modules/14-gc.md` | 145 | ✅ |
| `plan/modules/15-to-static.md` | 131 | ✅ |
| `plan/modules/16-cpp-extension.md` | 141 | ✅ |
| `plan/modules/17-distributed.md` | 98 | ✅ |
| `plan/modules/18-script.md` | 178 | ✅ |

### 5.3 `process/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `process/status.md` | 本文件 | ✅ |
| `process/tasks.md` | 137 | ✅ |
| `process/conventions.md` | 280 | ✅ |
| `process/decisions.md` | 240 | ✅ |
| `process/open-questions.md` | 160 | ✅ |

### 5.4 `research/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `research/feasibility.md` | 464 | ✅ |
| `research/architecture.md` | 251 | ✅ |
| `research/gc.md` | 417 | ✅ |
| `research/dataloader.md` | 380 | ✅ |
| `research/reuse.md` | 307 | ✅ |
| `research/to-static.md` | 239 | ✅ |

**论证阶段文档已齐(约 8800 行)。G0 未通过之前不新增论证文档,只修正已有的。**

---

## 5.5 阶段进度(P0–P18)

> 顺序、开工条件与完工判据见 `plan/roadmap.md`;每阶段怎么做见 `plan/modules/`。

| 阶段 | 模块 | 状态 |
|---|---|---|
| P0 | 无 Python 构建 | ⬜ **下一个动作,生死判定** |
| P1 | C ABI 中间层 | ⬜ 阻塞于 P0 |
| P2 | sol2 绑定层 | ⬜ |
| P3 | 算子代码生成 | ⬜ |
| P4 | 构建与打包 | ⬜ |
| P5 | `paddle.Tensor` | ⬜ |
| P6 | 自动微分与生命周期 | ⬜ |
| P7 | 序列化 | ⬜ |
| P8 | `jit::Layer` 推理 ★ | ⬜ |
| P9 | `paddle.nn` | ⬜ |
| P10 | `paddle.optimizer` | ⬜ |
| P11 | `paddle.io`(单 worker) | ⬜ |
| — | **M1 验收:MNIST** | ⬜ |
| P12 | dataset + vision | ⬜ |
| P13 | Lanes 多 worker | ⬜ |
| P14 | GC 九层机制 | ⬜ |
| P15 | 静态图 trace 模式 | ⬜ |
| P16 | cpp_extension | ⬜ |
| P17 | distributed | ⬜ ⚠️ 当前环境无法验收 |
| P18 | script 模式 | ⬜ |

---

## 6. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-08-03 | 建立 `CLAUDE.md` / `process/status.md` / `process/tasks.md` / `process/conventions.md` / `process/decisions.md` / `process/open-questions.md` / `README.md` |
| 2026-08-03 | `research/reuse.md`:执行器复用调研,推翻"自研静态图=独立项目"的评估 |
| 2026-08-03 | `research/to-static.md`:AST 选型 luacheck parser;源码契约可验证化 |
| 2026-08-03 | `plan/overview.md`:索引改全 1-based;排除清单 9→5 项;Ocean/Metrics 移至 M3 |
| 2026-08-03 | 目录重构:全部文档收进 `docs/migration/{plan,process,research}/`;`docs/` 下只留这一个子目录,未来的 API 说明另开目录 |
| 2026-08-03 | 建立根目录三语 README(英文默认 + zh-CN + zh-TW)与 `LICENSE`(Apache 2.0) |
| 2026-08-03 | `plan/roadmap.md`:新增总计划,P0–P18 依赖图 + 每阶段开工条件与完工判据 |
| 2026-08-03 | `plan/modules/`:新增 19 份模块详细设计(约 2700 行);`plan.md` 更名 `overview.md` |
| 2026-08-03 | `09-nn.md`:恢复**自动注册**(R17);`13-lanes.md`:Tensor 跨 lane 改用 `__lanesclone`,放弃 deep userdata(R18);Q-11 关闭 |
| 2026-08-03 | **新增 `plan/api/`(按 Lua 模块的接口设计)** —— 与 `plan/modules/`(按阶段的实现设计)分离:一个是「造出来长什么样」,一个是「怎么造」。立了跨模块硬规则(命名映射、index 语义必须逐模块标注、返回 `pl.List` vs 裸表的边界、`require "paddle"` 惰性加载)与固定骨架;`api/io.md` 作为样板写完。工程树里每个模块新增 `x.y.0 API 设计` 子节点 —— **先接口后实现** |
| 2026-08-03 | **新增 `WORKPLAN.md`(总工程树)** —— 智能体按它 DFS 遍历,遍历完 = 工程完成;新增 `plan/layout.md`(权威目录树 + 文件级落地顺序 + 生成代码进版本库的决策)与 `plan/ci.md`(四层 CI、阶段解锁表、五条机器红线)。关键设计:**兄弟节点排成拓扑序**,使朴素 DFS 自动满足 DAG 依赖,`前置` 字段退化为断言 |
| 2026-08-03 | 人的三条决定落盘:**① `nn.LayerList` 改为继承 `Layer`**(R22,Layer 优先;连带发现 5.1 的 `ipairs` 用 `lua_rawgeti`,`ipairs(ml)` 会静默跑空 -> 改用 `ml:iter()`,新增 Q-16);**② 「不引入新的强制 C 依赖」禁令取消**(R23,判据改为边际成本,`CLAUDE.md` §9.1;解禁 HTTP 下载与图像解码,Q-08 风险下降);**③ Insight7 的 `axis` 按 bug 修成 1-based**(R24,成员索引与维度索引都要 1-based,顺手做、P12 前完成;待拍板 P1 关闭、Q-12 转已决)。新增 D27–D29 |
| 2026-08-03 | **新增 `plan/foundations.md`(生态基座)**:Penlight 定为一等公民(R19/R21),Insight7 顶替 numpy 的位置(R20)。新增硬约束 C11、决策 D23–D26、未解问题 Q-12–Q-15;`conventions.md` 章节重编号(§2 生态基座、§3 与 Python 的已知差异) |
| 2026-08-03 | **argcheck 评估落盘,当天翻案两次(R25 -> R26)** —— ① 先推翻人的两个前提:「硬编码 Torch7」不成立(耦合共 32 行 2 处,`env.istype` 原注释就是 `-- user configurable function`);「过时」也不准,真实病灶是 `graph.lua:13` 拿 `tostring(t):match(0x…)` 当标识符,**MSVC 的 `%p` 不带 `0x` -> 带默认值的规则集在 Windows 上必崩**(改 2 行后上游全套测试通过)。② 据性能(2597 ns/call)给出「冷路径 vendored argcheck」= **R25,当天即被自己推翻**。③ 补测「能不能表达」:argcheck 对每条规则枚举三态共 **3^N** 路径,9 个可选参数 1.37 MB / 840 ms,**10 个就 `control structure too long`** —— 而 `Conv2D` 11 个、`Adam` 12 个、`DataLoader` 16 个(3^16,**挂死**),**R25 说的「冷路径」正好就是这张表**。④ 定案 **R26:不 vendor,取 schema + `usage.lua`,自写 ~150 行 O(N) 生成器 `_args.lua`** —— 原型实测 3 参 403 ns(快 6.4 倍)、17 参 3.6 KB / <1 ms。教训写进 `decisions.md` §2.11:**找到第一个可接受的缺点会让人停止寻找第二个;「多慢」是程度问题,「能不能表达」是有无问题,应该先问有无** |
| 2026-08-03 | **参数解析器独立成项目(R27/D31),并按人的两条追加约束定型** —— ① **不轻易造轮子**:普查了 luarocks 全部候选(`tableshape` / `typecheck` / `checks` / `ltypekit` / `geoffleyland/argcheck`),**没有一个提供「同一份签名同时吃位置调用与具名表调用 + 默认值 + usage」** —— 因为那是 **Python kwargs 的形状**,而 Lua 自己没有 kwargs。所以只自写「调用约定层」(~200 行),**类型判定层借现成的**:`type` 槽定为可调用契约,`tableshape` 的类型对象本身就是 callable,塞进去就能用(R28/D32)。② **50 个参数**不是假想需求:Paddle 真实最大签名 `ResNetBasicBlock.__init__` **43 参数 / 33 可选**(`resnet_block.py:434`),≥12 参数的有 156 个。实测 Lua 5.1 三道墙 = **60 upvalue / 200 局部变量 / N=122 寄存器**,而 argcheck 是**一规则若干 upvalue** -> 43 参数 ≈ 86 个 upvalue,**即使没有 3^N 也编不出 Paddle 最大的签名**(R29/D33)。改成单表 upvalue 后实测 50 参数 7.4 KB / ~1 ms / 3.75 µs,N>100 走表形态(1000 参数仍可编译)。③ 名字不能叫 `argcheck`(已被 `geoffleyland/argcheck` 占用),建议 `argsig`(待拍板 P10);④ 「基于 Penlight」与 §1.3 的 vendor 决定冲突(Penlight rockspec 依赖 `lfs`),**新增待拍板 P9**,拍板前按「两边都改声明依赖」写。CI 红线 ①b 从 2 条扩到 4 条,新增 43 参数基准 |
| 2026-08-03 | **人拍板 P9:Penlight 不再 vendor,改为全生态地基级 rock 依赖(R30/D34)** —— 原话「pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西」。vendor 的四条理由里第 1 条(避 `lfs`)在 R23 已作废、第 3/4 条是便利性,只剩版本锁定,而它靠 rockspec 锁 minor + CI 语义测试就能达成。决定性的是 vendor 自带的代价:**两份 `pl.class` 的实例互不 `is_a`** —— 生态里有 paddle-lua / Insight7 / `argsig` / metrics / ocean 五个消费者,这个坑会在每两个库之间各出现一次。连带:`_vendor/pl/`(5374 行)从 `layout.md` / `04-packaging.md` / `overview.md` 删除;`luafilesystem` 进入传递依赖(但我们自己不 `require "lfs"`);待拍板 P7 改写为「rock 版本区间怎么定」;CI 新增红线 `grep -rn "_vendor/pl" -> 失败`。同时新增 `plan/argsig.md`(176 行)——「新时代 argcheck」的孵化说明书,对着 argcheck 列出继承/丢弃/新增三列账,每行带上游 `file:line` |
| 2026-08-03 | **签名层的用户端表面定型(`plan/argsig.md` §2 §2.5,384 行)** —— 人给的形状:`local rule = require "argrule.rule"`;规则里 `name`/`type` 可位置写(`{"x","Tensor"}`),**规则表自己也遵守它自己的调用约定**;`help` 砍掉只留 `doc`(argcheck 两个都有还要 assert 二选一,`init.lua:80`);装饰器直接吃 `_C_ops.softmax`,不包 lambda。落地时补了四条:① **位置槽只到 2**(第三个位置是 `default` 还是 `doc` 说不清);② 类型短名 `"Tensor"` 走 `alias`,**全名是规范,短名重复注册直接报错**(一个进程里 paddle 和 Insight7 同时在);③ 公开 API 一律全具名(要拿来生成文档,且会被当范例抄)—— 写进 conventions,不进 CI;④ ★ **`paddle.zeros{2,3}` 的真歧义**:第一个参数就是 table 时,表形式调用无法与「一个 table 实参」区分,**必须显式 `nonamed = true`**(逃生舱从 argcheck 继承,`init.lua:53-57`),判据机械可查已进 CI 红线 ①b —— 不用启发式,因为猜错的那次是静默的错误结果 |
| 2026-08-03 | **新增跨模块硬规则「枚举参数一律不接受裸数字」(`api/README.md` §2.1.1)** —— 人的原话:「dtype 必须是 paddle.dtype 类型或者是 string,数字不应该代表类型」。三条理由里第二条是意外收获:**它把 `paddle.zeros{2,3}` 的调用歧义消掉了** —— `{2,3}` 当「表内位置」解释时 `dtype = 3` 过不了类型检查,于是确定性地落到「整个表是第一个实参」,**不需要写 `nonamed`**。因此 `argsig.md` §2.6 从「必须写 `nonamed`」改成**四步定死的解析顺序**,只有「表内位置」与「整表实参」两种解释**同时**成立时(如 `full{{2,3},1.0}`)才要求显式声明。两条规则相互依赖,**要改一起改**。同时定下短名原则:**类型短名 = 用户能 `local Tensor = paddle.Tensor` 拿到的那个导出名**(对应 Python 的 `from paddle import Tensor`),`Tensor`/`Place`/`DType` 归 paddle、`Array` 归 Insight7,短名重复注册直接报错。CI 红线 ①b 增至 6 条 |
