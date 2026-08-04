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
| 工程树位置 | `WORKPLAN.md` 节点 **0.13**(API 设计规范 + 样板)-> 下一个 **0.14 L0 CI 落地** |
| 下一个动作 | **0.14** L0 CI(不需要 libpaddle,现在就能建,见 `plan/ci.md` §2),之后才是 **1.1 无 Python 构建** |
| 全部 ⛔ 阻塞节点 | `WORKPLAN.md` 4.3 distributed —— 无多卡环境(待拍板 P2) |
| 待人拍板 | ~~P1(Insight7 `axis`)~~ ✅ **2026-08-03 已拍板:改**(R24)。剩余待拍板:P3 / P4 / P6 / P7 —— 见 `process/decisions.md` §3 §4 |
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

**必做 17 项(#20/#21 可跳过),预估 3 周。**

---

## 5. 文档完成度

> 路径相对 `docs/migration/`;`CLAUDE.md` 与三个 README 在仓库根目录。

### 5.1 顶层

| 文档 | 行数 | 状态 |
|---|---|---|
| **`WORKPLAN.md`(总工程树)** | 200 | ✅ **新增** |
| `/CLAUDE.md` | 432 | ✅ |
| `/README.md`(英文,默认) | 192 | ✅ |
| `/README.zh-CN.md` | 186 | ✅ |
| `/README.zh-TW.md` | 186 | ✅ |
| `README.md`(计划总索引) | 148 | ✅ |

### 5.2 `plan/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `plan/overview.md` | 915 | ✅ |
| `plan/roadmap.md` | 344 | ✅ |
| `plan/foundations.md` | 521 | ✅ |
| `plan/layout.md` | 261 | ✅ **新增** |
| `plan/ci.md` | 197 | ✅ **新增** |
| `plan/api/README.md` | 139 | ✅ **新增** |
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
| `process/decisions.md` | 167 | ✅ |
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
