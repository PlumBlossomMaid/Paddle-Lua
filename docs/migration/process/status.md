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
| 待人拍板 | ~~P1(Insight7 `axis`)~~ ✅ **2026-08-03 已拍板:改**(R24)。~~P9(还 vendor Penlight 吗)~~ ✅ **2026-08-03 已拍板:不 vendor,全生态 rock 依赖**(R30)。~~P10(签名层叫什么)~~ ✅ **2026-08-03 已拍板:`argrule`**。剩余待拍板:P3 / P4 / P6 / P7(改写)—— 见 `process/decisions.md` §3 §4 |
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
| `/CLAUDE.md` | 491 | ✅ |
| `/README.md`(英文,默认) | 192 | ✅ |
| `/README.zh-CN.md` | 186 | ✅ |
| `/README.zh-TW.md` | 186 | ✅ |
| `README.md`(计划总索引) | 148 | ✅ |

### 5.2 `plan/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `plan/overview.md` | 915 | ✅ |
| `plan/roadmap.md` | 344 | ✅ |
| `plan/foundations.md` | 1105 | ✅ **+参数检查(§4)+ 基座边界与解析器项目(§5)** |
| `plan/argrule.md` | 650 | ✅ **新增**(孵化说明书,建仓后迁走;原 `argsig.md`)|
| `plan/layout.md` | 261 | ✅ **新增** |
| `plan/ci.md` | 246 | ✅ **新增** |
| `plan/api/README.md` | 304 | ✅ **新增** |
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
| `process/decisions.md` | 246 | ✅ |
| `process/open-questions.md` | 197 | ✅ |

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
| 2026-08-03 | **人拍板 P9:Penlight 不再 vendor,改为全生态地基级 rock 依赖(R30/D34)** —— 原话「pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西」。vendor 的四条理由里第 1 条(避 `lfs`)在 R23 已作废、第 3/4 条是便利性,只剩版本锁定,而它靠 rockspec 锁 minor + CI 语义测试就能达成。决定性的是 vendor 自带的代价:**两份 `pl.class` 的实例互不 `is_a`** —— 生态里有 paddle-lua / Insight7 / `argrule` / metrics / ocean 五个消费者,这个坑会在每两个库之间各出现一次。连带:`_vendor/pl/`(5374 行)从 `layout.md` / `04-packaging.md` / `overview.md` 删除;`luafilesystem` 进入传递依赖(但我们自己不 `require "lfs"`);待拍板 P7 改写为「rock 版本区间怎么定」;CI 新增红线 `grep -rn "_vendor/pl" -> 失败`。同时新增 `plan/argrule.md`(176 行)——「新时代 argcheck」的孵化说明书,对着 argcheck 列出继承/丢弃/新增三列账,每行带上游 `file:line` |
| 2026-08-03 | **签名层的用户端表面定型(`plan/argrule.md` §2 §2.5,384 行)** —— 人给的形状:`local rule = require "argrule.rule"`;规则里 `name`/`type` 可位置写(`{"x","Tensor"}`),**规则表自己也遵守它自己的调用约定**;`help` 砍掉只留 `doc`(argcheck 两个都有还要 assert 二选一,`init.lua:80`);装饰器直接吃 `_C_ops.softmax`,不包 lambda。落地时补了四条:① **位置槽只到 2**(第三个位置是 `default` 还是 `doc` 说不清);② 类型短名 `"Tensor"` 走 `alias`,**全名是规范,短名重复注册直接报错**(一个进程里 paddle 和 Insight7 同时在);③ 公开 API 一律全具名(要拿来生成文档,且会被当范例抄)—— 写进 conventions,不进 CI;④ ★ **`paddle.zeros{2,3}` 的真歧义**:第一个参数就是 table 时,表形式调用无法与「一个 table 实参」区分,**必须显式 `nonamed = true`**(逃生舱从 argcheck 继承,`init.lua:53-57`),判据机械可查已进 CI 红线 ①b —— 不用启发式,因为猜错的那次是静默的错误结果 |
| 2026-08-03 | **新增跨模块硬规则「枚举参数一律不接受裸数字」(`api/README.md` §2.1.1)** —— 人的原话:「dtype 必须是 paddle.dtype 类型或者是 string,数字不应该代表类型」。三条理由里第二条是意外收获:**它把 `paddle.zeros{2,3}` 的调用歧义消掉了** —— `{2,3}` 当「表内位置」解释时 `dtype = 3` 过不了类型检查,于是确定性地落到「整个表是第一个实参」,**不需要写 `nonamed`**。因此 `argrule.md` §2.6 从「必须写 `nonamed`」改成**四步定死的解析顺序**,只有「表内位置」与「整表实参」两种解释**同时**成立时(如 `full{{2,3},1.0}`)才要求显式声明。两条规则相互依赖,**要改一起改**。同时定下短名原则:**类型短名 = 用户能 `local Tensor = paddle.Tensor` 拿到的那个导出名**(对应 Python 的 `from paddle import Tensor`),`Tensor`/`Place`/`DType` 归 paddle、`Array` 归 Insight7,短名重复注册直接报错。CI 红线 ①b 增至 6 条 |
| 2026-08-03 | **消歧判据补上「必填参数」这一半,并新增 `extend` 处理共有类型** —— ① 人指出 `full{{2,3},1.0}` 其实不歧义:`fill_value` 必填无默认,所以「整表当 shape」这条路本身走不通。判据因此从「类型全过」改成**「类型全过 **且** 必填参数都有值」** —— 少了后半句会凭空造出一堆假歧义。真歧义的条件变得很苛刻:规则 #1 接受 table **且其余参数全可选**,典型是 `concat` / `stack` / `meshgrid` 这批**第一个参数是列表**的函数,它们一律写 `nonamed = true`(写了之后 `concat{a,b}` 反而天经地义)。② 人指出 **Insight7 也有 `Place` / `DType`** —— 因为 Insight7 的 API 本来就照着 Paddle 设计。所以这不是撞名是同一概念:新增 `argrule.extend(name, pred)`(谓词取或),`register` 撞名仍报错,两者区分开。顺带记一笔长期方向:`DType`/`Place` 最好在 C 层就是同一个值类型,零拷贝互操作已要求布局对齐,dtype 却还要在边界转换一次 |
| 2026-08-03 | **`shape` 一类参数改用 `IntList`,且判据是「是个容器 + 装的是整数」而不是类名单(R31)** —— 人的原话:「paddle 里面 shape 必须是 tuple or list or np.ndarray 不能是数字」「**反正需要是个容器**」「**而且装的是整数**」。我先写成 `{"table","pl.List","insight.Array"}`,**当天即被自己推翻**:那是把「容器」硬编码成框架名单,用户自己的容器类会被无理由挡住,正是 §4 零框架硬编码要挡的。改为组合子 `argrule.list_of(elem)`(探容器协议 + 逐元素套 `elem`),`IntList` / `TensorList` 只是宿主注册的别名。容器协议:长度 `#o` -> `:len()` -> `:size()`(⚠️ **5.1 的 `__len` 对 table 无效**,进 M0 #24 实测)、取元素 `o[i]` 1-based、自报 dtype 走 O(1) 快路否则逐元素 O(n)。**意外收获:元素逐个查把 `concat` 的歧义也消掉了** —— `concat{a,b}` 的②里 `x=a` 是 Tensor 不是 TensorList、`concat{{a,b},2}` 的③里元素 `2` 不是 Tensor,各自出局。**所以上一版「第一个参数是列表的函数必须写 `nonamed`」作废**,真歧义要求 `list_of(E)` 且 `E` 自己接受容器,现有 API 里一个都没有。教训与「必填参数是天然消歧器」同源:**判据越弱,假歧义越多** |
| 2026-08-03 | **`Tensor` 也满足容器协议(人的决定),并纠正我自己的两处断言** —— 人:「Tensor 也应该满足容器协议,毕竟只要是个 int 容器按说都应该行」。① 上游核实通过:`ShapeLike = Sequence[int|Tensor|None] | Tensor`(`_typing/shape.py:22-33`),"若 shape 是 Tensor,须是一维"(`creation.py:1832`),**且 list 的元素可以是 0-D Tensor**(`:1831`)。② 因此容器协议分 **plain / opaque** 两级 —— 区别不是"谁写的",是**逐元素访问贵不贵**:对 GPU 上的 Tensor 逐元素查类型 = n 次设备同步,**红线,禁止**。选路机械:`E` 是标量类型就用容器自报的 dtype+ndim 做 O(1) 证明,否则逐元素;没有 O(1) 证明又不许逐元素 -> 判否。**这条判据自己算出了上游的行为:一维 int Tensor 是合法 `shape`(收),二维 Tensor 不是 `TensorList`(`concat` 的 `x` 是 `Sequence[Tensor]`,`manipulation.py:1482,1507`,不收)—— 没写任何特例。** ③ ⚠️ **纠正**:我昨天写的「`paddle.zeros(5)` 本来就是错的」在当前 develop 上**不成立** —— 上游有 `@size_args_decorator`(`decorator_utils.py:406-437`),`ones(1,2,3)` / `ones(5)` / `ones(size=[...])` 全合法。但注意上游的做法:**没有把 `int` 加进 `ShapeLike`,而是用装饰器在签名外面归一化**。我们照抄这个分层 —— 表级选项 `sizeargs = "shape"`,`type` 仍是 `IntList`(`argrule.md` §2.4)。连带:第一个位置实参是整数时全部位置实参归 `shape`、`dtype` 只能具名(上游同此) |
| 2026-08-03 | **P10 拍板:签名层定名 `argrule`** —— `plan/argsig.md` -> `plan/argrule.md`,全仓库引用、rockspec 依赖项、WORKPLAN 节点 1.7.0 同步改名;`foundations.md` §5.4.7 与 `decisions.md` P10 记下"我建议的是 `argsig`,人选了 `argrule`"。**剩下的动作:去 luarocks 占位** |
| 2026-08-03 | **`sizeargs` 降级出签名层,`IntList` 的元素检查下放转换层(R32)** —— 人的两问:「`sizeargs` 之前有讨论过吗」**没有,是我上一轮临时加的**;「我想的是在函数里面有个 if 判断如果有小数直接 error」**对,而且比放类型层更好**。① `sizeargs` 全 Paddle 只用在 5 个函数(`ones`/`zeros`/`empty`/`randn`/`rand`,`creation.py:1644,1807,3081`、`random.py:961,2343`),而上游自己就是**装饰器在签名外面** —— 照抄分层比照抄行为更重要,做成 paddle-lua 侧 ~10 行,签名层零改动。② 元素检查下放的判据是机械的:**去掉它之后调用仍唯一解释 -> 可下放**。`IntList` 满足 -> 转换层 `if` + `error`(那步反正要遍历 n 次,类型层再扫一遍是查两遍;0-D Tensor 元素也只有那层处理得了;报错还能带下标 `shape[3] must be an integer, got 2.5`);`TensorList` 不满足(`concat{{a,b},2}` 靠元素类型出局)-> 留类型层。**红线:必须 error 不许静默取整、必须指向调用点、C++ 异常不得穿过 Lua(C7)** |
| 2026-08-03 | **`sizeargs` 整个砍掉,并升成一条通则:上游的兼容糖一律不移植(R33)** —— 人:「用得少的东西,为了 API 简洁似乎就该去掉」。决定性的不是"用得少",是**上游加糖的原因在 Lua 侧不存在**:Python 的 `zeros([2,3])` 又是括号又是方括号才需要 `zeros(2,3)`;Lua 的表调用本来就省括号,**`zeros{2,3}` 一样短** —— 收益 0,成本照付。同一条理由挡住整层兼容装饰器:**291 处参数别名**(`concat(tensors=,dim=)`)、5 处变长 size、`reshape(2,3)` 等。别名还额外带两个坑:`concat{x=a, tensors=b}` 要复制 291 次「不能同时给」的检查(上游 `decorator_utils.py:181-184` 抛 ValueError);`dim`/`axis` 两个名字 = index 语义标注表两条记录,漏标一条就是静默 off-by-one。**代替方案:教学式报错 `did you mean: paddle.zeros{2, 3}`**,只在错误路径上,零运行时成本。新增 `api/README.md` §2.1.3,例外判据「这个写法在 Lua 里比规范写法更短或更清楚吗」,**理由不能是"上游有"** |
| 2026-08-03 | **`decorator_utils.py` 整层不移植(R34/D35)** —— 人的原话:「这东西就是 Paddle 为了 PyTorch 用户用得惯才搞的,lua 里面不用考虑这些,**我们 Paddle 框架就应该用自己的语法和规范**」。范围从「逐个判断哪些糖要移植」升成「整层跳过」:1451 行 / 30+ 装饰器 / 68 个模块 import 它,全文 **51 处** `PyTorch`/`torch.`(docstring 直写 `PyTorch: torch.block_diag(x,y,z)` / `Paddle: paddle.block_diag([x,y,z])`,`:923-940`),甚至有专门劝返 torch 关键字的 `forbid_keywords`(`:517-542`)。**关键证据:35 处 `wrapper.__signature__ = inspect.signature(func)`** —— 上游自己把规范签名保留在内层,**所以我们的生成器读到的本来就是规范签名,忽略这层是默认行为而不是额外工作量**。⚠️ 唯一要逐个确认的:少数装饰器可能不只改参数(`legacy_reduction_decorator` / `view_decorator`),判据「壳只动参数、内层签名不变」,发现改了行为的记 OPEN_QUESTION |
| 2026-08-03 | **新增硬约束 C12:用户可见的 API 一律用 Paddle 自己的规范,不引入 PyTorch 的命名与惯例(R35/D35)** —— 人:「所以我们的 paddle-lua 也不要那些 PyTorch 的东西」。**光跳过 `decorator_utils.py` 挡不住** —— 一批 torch 风格参数**已经进了上游的规范签名本身**:`zeros(shape, dtype, name, *, out, device, requires_grad, pin_memory)`(`creation.py:1807`),而 Paddle 自己的名字在 `to_tensor` 里原样留着(`place` / `stop_gradient`,`:1124-1129`)。取 Paddle 那套:`stop_gradient` / `place` / `axis` / `x` / `shape` / `place=CUDAPinnedPlace()`。**`requires_grad` 必须挡住的真正理由:它与 `stop_gradient` 取反、默认值也反**(`False` vs `True`)—— 两个名字并存时搞混一次就是**静默地训不动**,没有报错只有 loss 不降。⚠️ 例外:`Tensor.size`(元素总数)是 Paddle 原生属性,不在禁用之列。**边界写死:约束用户看得见的表面,实现层不受限**(GC 抄 Torch7 九层 D5、schema 抄 argcheck D30 照做,因为用户不会写到它们)。CI 判据 `grep -rnE 'name *= *"(requires_grad|device|dim|input|tensors|pin_memory)"' lua/`。新增 Q-19(`out=` 是能力不是命名,倾向 v1 不做)、Q-20(`decorator_utils.py` 里是否有"壳改了语义"的例外,P3 前查完)|
