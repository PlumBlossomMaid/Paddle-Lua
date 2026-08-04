# LAYOUT · 代码目录与模块清单

> **本文件是目录结构的权威版本。** `plan/overview.md` §10 与
> `plan/modules/04-packaging.md` §3.1 的树是摘要,以本文件为准。
>
> 回答三个问题:**有哪些目录和模块 / 每个属于哪个阶段 / 开工时文件级的顺序**。
> 阶段级顺序在 `plan/roadmap.md`,CI 在 `plan/ci.md`。

---

## 0. 三条布局原则

| # | 原则 | 为什么 |
|---|---|---|
| ① | **手写 / 生成 / vendored 三类代码物理隔离** | 60k 行生成代码和 15k 行手写代码混在一起,PR diff 就没法看了。隔离之后"这个目录不许手改"是一条可以被 CI 强制的规则 |
| ② | **编 1 次的和编 5 次的分开** | `csrc/capi/` 是纯 C,5 个 Lua 版本共用一份编译产物;`csrc/sol/` 是重模板,每个 Lua 版本各编一次。混在一个 target 里等于把编译时间 ×5(D2) |
| ③ | **目录边界 = 阶段边界** | 每个 P* 阶段应该只动少数几个目录。做不到就说明阶段划分错了,而不是目录划分错了 |

---

## 1. 目录树(权威)

```
paddle-lua/
├── CMakeLists.txt              顶层:找 libpaddle、找 Lua、定义 3 个 target
├── rockspecs/                  paddle-lua-scm-1.rockspec 等
├── cmake/
│   ├── FindLibPaddle.cmake     $PADDLE_ROOT 发现顺序(04-packaging §3.2)
│   └── FindLua5x.cmake         5 个版本 + LuaJIT 的探测
│
├── csrc/                       ── C/C++ 侧 ──────────────────────────
│   ├── capi/                   【P1】纯 C ABI 中间层,**编 1 次**
│   │   ├── paddle_c.h              唯一对外头文件,纯 C,无 C++ 类型
│   │   ├── status.cc               C++ 异常 -> status code(不许异常穿过 longjmp)
│   │   ├── tensor.cc               Tensor 创建/析构/属性/拷贝
│   │   ├── autograd.cc             egr::Backward 转发
│   │   ├── device.cc               Place / DeviceContext
│   │   └── jit.cc                  jit::Layer 加载与推理
│   ├── capi_gen/               【P3】**生成**的算子 C ABI —— 不许手改
│   │   └── ops_*.cc                按首字母或算子族分片,避免单文件过大
│   ├── sol/                    【P2】sol2 绑定,**编 5 次**
│   │   ├── module.cc               luaopen_paddle_core
│   │   ├── tensor_binding.cc       usertype<LuaTensor>
│   │   ├── slice.cc                字符串索引解析(移植自 Insight7)
│   │   └── error.cc                status code -> lua_error
│   ├── lanes/                  【P13】__lanesclone + worker 池
│   ├── gc/                     【P14】堆追踪 + RegisterOOMCallback
│   └── utils/
│       └── fs.cc                   C++17 <filesystem>,替代 lfs(foundations §1.2)
│
├── lua/paddle/                 ── Lua 侧 ────────────────────────────
│   ├── init.lua                    require "paddle" 入口,只做装配
│   ├── _ops/                   【P3】**生成**的 Lua wrapper —— 不许手改
│   │                            ⚠️ 参数解析器**不在这里** —— 独立 rock,外部依赖(R27)
│   │                               `_args.lua` / `_wrap.lua` 都已删除,见 foundations §5.4.4
│   ├── slice.lua               【P5】字符串索引的 Lua 侧
│   ├── tensor.lua              【P5】
│   ├── dtype.lua  device.lua   【P5】
│   ├── autograd.lua            【P6】
│   ├── scope.lua               【P6】显存作用域(gc.md 九层里的用户可见层)
│   ├── serialize/              【P7】save.lua load.lua pickle.lua npy.lua safetensors.lua
│   ├── jit/                    【P8】load.lua(jit::Layer)/【P15】trace.lua /【P18】script.lua
│   ├── nn/                     【P9】layer.lua + layers/*.lua + functional/*.lua + init/*.lua
│   ├── optimizer/              【P10】
│   ├── io/                     【P11】dataset.lua sampler.lua dataloader.lua collate.lua
│   ├── dataset/                【P12】mnist.lua cifar.lua ...
│   ├── vision/                 【P12】transforms/ models/
│   ├── amp/                    【延后】
│   ├── distributed/            【P17】
│   └── utils/                  fs.lua(csrc/utils/fs.cc 的 Lua 面)/ download.lua
│
├── tools/gen/                  【P3】生成器。**开发期 Python**,不进发布包
│   ├── parse_yaml.py               读 Paddle 的 ops.yaml
│   ├── emit_capi.py                -> csrc/capi_gen/
│   ├── emit_lua.py                 -> lua/paddle/_ops/
│   └── index_semantics.yaml        哪些参数是 index 语义(1-based 转换表,overview §6.1.1)
│
├── tests/                      busted
│   ├── unit/                       每个模块一份,跟着模块走
│   ├── parity/                     与 Python 对拍(需要 CI 侧有 Python)
│   ├── compat/                     5 个 Lua 版本的行为差异断言(见 §3 的 D27 案例)
│   └── fixtures/                   golden 数据的**生成脚本**,数据本身不进库(ci.md §7)
│
├── examples/                   mnist.lua 等,同时是 M1/M2 的验收载体
├── ci/                         CI 用的脚本,见 plan/ci.md
└── docs/
```

---

## 2. 模块清单:归属、体量、性质

| 目录 | 阶段 | 量级 | 性质 | 关键依赖 |
|---|---|---|---|---|
| `cmake/` `CMakeLists.txt` | P0/P4 | ~400 | 手写 | — |
| `csrc/capi/` | **P1** | ~2k | 手写,纯 C | libpaddle |
| `csrc/sol/` | **P2** | ~2k | 手写,C++ | capi + sol2 |
| `csrc/capi_gen/` | P3 | ~40k | **生成** | ops.yaml |
| `lua/paddle/_ops/` | P3 | ~20k | **生成** | ops.yaml |
| `csrc/utils/fs.cc` | P4 | ~200 | 手写 | C++17 |
| `lua/paddle/{tensor,slice,dtype,device}.lua` | **P5** | ~1.4k | 手写 | 基座 |
| *(Penlight)* | — | — | **rock 依赖**(R30),不 vendor | `lfs`(传递) |
| *(`argrule` 参数签名层)* | — | — | **rock 依赖**(R27),已定名(P10)| Penlight |
| `lua/paddle/{autograd,scope}.lua` | P6 | ~600 | 手写 | P5 |
| `lua/paddle/serialize/` | P7 | ~1.2k | 手写 | P5 |
| `lua/paddle/jit/load.lua` | P8 | ~300 | 手写 | P5 |
| `lua/paddle/nn/` | **P9** | **~6k** | 手写 | P5/P6 + 基座 |
| `lua/paddle/optimizer/` | P10 | ~1.5k | 手写 | P9 |
| `lua/paddle/io/` | P11 | ~1k | 手写 | P5 |
| `lua/paddle/{dataset,vision}/` | P12 | ~2k | 手写 | P11 + Insight7 |
| `csrc/lanes/` | P13 | ~800 | 手写 | Lanes |
| `csrc/gc/` | P14 | ~600 | 手写 | capi |
| `lua/paddle/jit/trace.lua` | P15 | ~1.5k | 手写 | P6 |
| `lua/paddle/jit/script.lua` | P18 | ~3k | 手写 | luacheck parser |

**手写合计约 15k 行,生成约 60k 行** —— 与 `overview.md` 的估算一致。
**`lua/paddle/nn/` 一个目录占了手写量的 40%**,它是成本重心,不是 `csrc/`。

---

## 3. 文件级落地顺序

`roadmap.md` 给的是阶段顺序。这里给**每个阶段内部第一个动的文件**,
因为"P1 开工"这句话不足以让人坐下来敲第一行。

规则:**每个阶段的第一个文件,必须是能让这个阶段的验收跑起来的最小闭环。**
不是"最基础的那个",是"最早能证明这条路通的那个"。

### P0 · 无 Python 构建(生死判定)

**这个阶段不写我们的代码**,只产出两样东西:

1. 一份**能复现的构建脚本**(记进 `ci/build-libpaddle.sh`)
2. 一个**最小 C++ 验证程序** —— 不进 `csrc/`,放 `ci/probe/`,
   内容是:造两个 Tensor -> 相加 -> `egr::Backward` -> 读梯度

第 2 项是关键。**能编过 ≠ 能用。** `WITH_PYTHON=OFF` 下可能链得上但
反向图的注册在初始化期就没跑(算子注册常走静态初始化)。
只有真的跑出一个正确梯度,G0 才算过。

### P1 · C ABI 中间层

```
1. paddle_c.h          ← 先定接口,不写实现
2. status.cc           ← 第二个就是它。异常穿透是会让整个进程崩的那类 bug,
                          晚一天引入,前面所有代码都要回头补 try/catch
3. tensor.cc           ← 创建/析构/shape/dtype
4. autograd.cc         ← backward 一条
5. (此时可以用 C 写一个 mini 训练循环验证,不需要 Lua)
6. device.cc  jit.cc
```

**第 5 步不要跳。** 在还没有 Lua 的时候,用纯 C 跑通一次"前向+反向+更新",
能把"是 C ABI 的问题还是绑定的问题"永久区分开。
后面每次出怪事,都可以回到这个 C 程序上二分。

### P2 · sol2 绑定层

```
1. module.cc + error.cc     ← luaopen 能被 require 到,且错误能变成 lua_error
2. tensor_binding.cc        ← usertype,先只绑 shape/dtype
3. (跑通 require "paddle_core"; print(t:shape()) )
4. slice.cc                 ← 字符串索引
```

**第 3 步是 P2 真正的里程碑** —— `require` 成功那一刻,后面全是加法。

### P3 · 代码生成

```
1. index_semantics.yaml     ← **先写标注表,再写生成器**
2. parse_yaml.py
3. emit_capi.py -> 先只生成 3 个算子(add / matmul / relu)
4. 手工 review 这 3 个的产物
5. 放开全量
```

**第 1 步的顺序不能反。** 标注表(哪些参数是 index 语义、要做 1-based 转换)
是 D13 的落地点。先写生成器再补标注表,等于先生成 2000 个错误的算子。
**第 4 步也不能跳** —— 生成器的 bug 会被复制 2000 遍。

### P5 · Tensor

```
1. dtype.lua + device.lua    ← 无依赖,最容易对齐 Paddle 命名
2. _wrap.lua                 ← 后面所有 API 都要用它
3. tensor.lua                ← __index/__newindex/元方法
4. slice.lua                 ← 1-based 转换的 Lua 侧
```

### P9 · nn(最大的一块,顺序最值得讲究)

```
1. nn/layer.lua              ← Layer 基类。_create/_class_init/FIELDS 三件套
2. tests/unit/nn_layer_spec  ← **和 layer.lua 同时写,不是之后写**
3. nn/layers/linear.lua      ← 第一个真实层,验证注册/参数/前向
4. (跑通:Linear 的 state_dict 键名与 Python 逐字符一致)
5. nn/functional/            ← 大部分是对 _ops 的一行转发
6. 其余 39 个层
7. nn/containers.lua         ← Sequential / LayerList / LayerDict
```

**第 2 步为什么特殊:** `Layer` 基类的错误是**静默**的 ——
参数漏注册不会报错,只会让优化器少更新一个张量,表现为"收敛慢一点"。
这类 bug 在第 40 个层写完之后再查,成本是现在的几十倍。

**第 7 步为什么排最后:** 容器依赖基类的遍历语义已经稳定。
而且 `LayerList` 有那个 `ipairs` 跨版本坑(D27),
它需要 `tests/compat/` 已经建起来 —— 那是 P4 的产物。

### 其余阶段

按 `roadmap.md` 的依赖顺序,内部一律遵循同一条规则:
**先建骨架 + 一个真实样本 + 它的测试,验证闭环,再铺量。**

---

## 4. 什么不进这个仓库

| 东西 | 去哪 | 为什么 |
|---|---|---|
| `libpaddle` 二进制 | 用户自备 / CI 缓存 | 体积以 GB 计(04-packaging §1) |
| golden 对拍数据 | CI 生成,`tests/fixtures/` 只放生成脚本 | 二进制进 git 会让仓库迅速膨胀且不可 diff |
| `ocean-lua` / `metrics-lua` | 独立仓库(M3) | overview §11 的三仓库切分 |
| Python 运行时 | **不存在于用户端** | 硬约束。`tools/gen/` 的 Python 只在开发期和 CI 期跑 |

---

## 5. 生成的代码进不进版本库:**进**

这是必须现在定的,因为它决定 `.gitignore` 和 CI 的形状。

**结论:`csrc/capi_gen/` 和 `lua/paddle/_ops/` 都提交进 git。**

| 理由 | 说明 |
|---|---|
| ① **用户端没有 Python** | 这是项目的第一硬约束。rock 的 tarball 里必须已经有生成产物 —— 用户装的时候不可能现场跑 `tools/gen/*.py` |
| ② 上游 API 变动会在 diff 里显形 | Paddle 升个版本,`git diff` 直接告诉你哪 37 个算子签名变了。不提交的话,这个变化是**隐形**的,直到运行时才炸 |
| ③ 可审计 1-based 转换 | D13 的转换发生在生成代码里。它进版本库,才能在 PR 里 review "这个 axis 到底转没转" |

**代价与对策:**

| 代价 | 对策 |
|---|---|
| 仓库变大(~60k 行文本) | 可接受。纯文本,压缩率高 |
| PR diff 噪音 | `.gitattributes` 给这两个目录标 `linguist-generated`,GitHub 默认折叠 |
| **有人手改生成代码** | ⚠️ **这是真正的风险。** 靠 CI 挡:重新生成 -> `git diff --exit-code`,非零即失败(`ci.md` §6) |

第三条是重点。60k 行里手改一处,当时能跑,下次重新生成就被静默覆盖 ——
**改动消失,而且没有任何报错**。CI 的 regen-diff job 是这个方案唯一的安全绳,
不是可选项。

---

## 6. 与其它文档的关系

| 文档 | 关系 |
|---|---|
| `plan/roadmap.md` | 阶段级顺序;本文件是它的**文件级展开** |
| `plan/ci.md` | 每个目录对应哪些 CI job |
| `plan/modules/04-packaging.md` | 发布产物的布局(用户看到的),本文件是源码布局(开发者看到的) |
| `plan/overview.md` §10 | 摘要版,**以本文件为准** |
