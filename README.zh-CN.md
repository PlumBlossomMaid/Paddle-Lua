[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Lua](https://img.shields.io/badge/Lua-5.1%20|%205.2%20|%205.3%20|%205.4%20|%20JIT-blue.svg)](https://www.lua.org/)
[![Upstream](https://img.shields.io/badge/Paddle-3.5.0%20零改动-green.svg)](https://github.com/PaddlePaddle/Paddle)
[![Status](https://img.shields.io/badge/status-论证阶段-orange.svg)](docs/migration/README.md)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Paddle-Lua

**给 PaddlePaddle 做一个 Lua 前端。`local paddle = require "paddle"`,不需要 Python 运行时。**

> ⚠️ **当前处于论证阶段,尚无代码。**
> 本仓库目前是一份完整的技术方案。
> 见 **[docs/migration/](docs/migration/README.md)** —— 13 份文档,约 3000 行。

---

## 为什么这件事做得成

Paddle 的训练引擎是**纯 C++** 的。自动微分(`egr::Backward`)、张量类型
(`paddle::Tensor`)、以及全部算子(phi)都和 Python 没有任何关系。Python 只是一层**外壳**:
参数解析、`nn.Layer` 的簿记、优化器循环、数据管线。

所以这个项目不是重新实现一个深度学习框架,而是**换掉那层外壳**。

```
             Python 外壳                     Lua 外壳          ← 我们写这个
  ─────────────────────────────   ─────────────────────────
   pybind11 绑定                   sol2 + 纯 C ABI 中间层     ← 我们写这个
  ═══════════════════════════════════════════════════════════
   生成的前向 API  ·  eager 自动微分  ·  phi 算子             ← 原样不动,纯 C++
```

**上游零改动。** Paddle 按原样使用。

---

## 长什么样

```lua
local paddle = require "paddle"
local class  = require "pl.class"
local nn, F  = paddle.nn, paddle.nn.functional

local Net = class(nn.Layer)
function Net:_init()
  self:super()
  self.fc = nn.Linear(784, 10)     -- 和 Python 一样,自动注册
end
function Net:forward(x) return self.fc(x) end

local net = Net()
local opt = paddle.optimizer.Adam{ learning_rate = 1e-3, parameters = net:parameters() }

for _, batch in paddle.io.DataLoader{ ds, batch_size = 64, num_workers = 4 } do
  local loss = F.cross_entropy(net(batch[1]), batch[2])
  loss:backward()
  opt:step()
  opt:clear_grad()
end
```

索引**统统 1-based** —— 包括 `axis` 参数。切片借用 Python 的语法,通过字符串传入,
这个做法在 [Insight7](https://github.com/PlumBlossomMaid) 里已经跨语言验证过:

```lua
x["1:3, :"]     -- 第 1、2 行,全部列
x["::-1"]       -- 反转
x[{1, 2}]       -- 取单个元素
```

---

## 设计概览

| | |
|---|---|
| 宿主语言 | Lua 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT |
| Python 运行时 | **不需要** |
| 绑定层 | sol2,底下垫一层纯 C ABI |
| 上游改动 | **零** |
| 库代码方言 | 只用 Lua 5.1 语法(一份源码同时服务五个虚拟机) |
| 多 worker 数据加载 | [Lua Lanes](https://github.com/LuaLanes/lanes),强制依赖 —— 用线程,不用进程 |
| 索引 | 全部 1-based |
| 算子绑定 | 从 Paddle 自己的 `ops.yaml` / `backward.yaml` 生成,不手写 |
| 类与集合 | [Penlight](https://github.com/lunarmodules/Penlight) —— `pl.class`、`pl.List`,vendored 纯 Lua |
| numpy 的位置 | [Insight7](https://github.com/PlumBlossomMaid) —— 无梯度的数组,用于数据准备 |

### 三个值得解释的决策

**LuaJIT 也走 sol2,不走 FFI。** 绑定开销是 50–200 ns,而一次 kernel 启动是 1–10 µs。
为了省这 1–2% 去维护两套绑定路径,不划算。

**`num_workers` 用线程而不是进程。** Python 的 DataLoader 之所以 fork,是因为 GIL。
Lua 没有 GIL —— 多个 `lua_State` 在同一进程里是真并行的,这意味着 worker 可以把张量
**按指针**交给主线程,完全不需要序列化。`paddle::Tensor` 本身就用 `shared_ptr` 对缓冲做
原子引用计数,跨线程的代价只是一次加一。

**Penlight 是一等公民,不是随手一提。** Lua 的标准库几乎什么都没有:没有类、没有 list、
没有跨版本兼容层。与其自己造一套(结果是只有我们自己会调试),不如让 `nn.Layer`
**就是**一个 `pl.class`,`parameters()` **就是**返回 `pl.List`,版本垫片**就是** `pl.compat`。
你已有的 Penlight 知识在这里直接可用。

**静态图复用 Paddle 的 C++ 执行器。** `StandaloneExecutor` / `InterpreterCore` 已经提供了
IR、调度、垃圾回收、pass 和序列化;`run_program_ad_func` 已经让 traced 图能参与 eager 自动微分。
Lua 侧只需要写一个 tracer。

---

## 唯一能决定项目生死的风险

`cmake/configure.cmake:15` 对无 Python 路径的守卫是 `(NOT WITH_PYTHON) AND ON_INFER`。
**上游预期的无 Python 场景是推理。** 我们要的这个组合 —— 无 Python 但要训练 ——
可能从来没有人编译过。

所以第一件事不是写代码,而是:

```bash
cmake .. -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=ON
```

如果这条路编不出一个 `egr::Backward` 真能跑的 `libpaddle`,整个项目需要重新评估。
计划里其它所有东西都在这一项的下游。

---

## 范围

**做:** `paddle.Tensor`、`paddle.nn`、`paddle.optimizer`、`paddle.io`、`paddle.vision`、
`paddle.jit`(先 trace 后 script)、`paddle.dataset`、`paddle.static`、
`paddle.distributed`(可以写,但目前测不了)。

**不做:** `paddle.Model` 和 `paddle.metric.*` —— 这是故意的。它们已被
[PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean)(Lightning + Keras 风格的
trainer)和 [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics)
(torchmetrics 风格)取代,我们照后两者的设计走。

同样不做:`paddle.incubate`、`paddle.fluid`,以及一切只为伺候 Python 打包生态而存在的东西。

---

## 路线图

| | 里程碑 | 交付物 |
|---|---|---|
| **M0** | 可行性闸门 | 无 Python 构建;反向传播跑通 |
| **M1** | MVP | MNIST 端到端训练;用 `jit::Layer` 推理 Python 训练的模型 |
| **M2** | 可用 | `paddle.vision`、Lanes 多 worker 的 DataLoader、完整 GC 机制 |
| **M3** | 完整 | trace 模式动转静;`metrics-lua` |
| **M4** | 更远 | script 模式(带控制流,vendored luacheck parser) |

粗估:MVP 4–6 人月。

---

## 相关项目

```
paddle-lua        C++ 绑定 + 核心 Lua 层            ← 本仓库
  ├─ metrics-lua     纯 Lua,torchmetrics 风格
  └─ ocean-lua       纯 Lua,Lightning/Keras 风格
```

| | |
|---|---|
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | 上游。原样使用,**永不改动** |
| [PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean) | 取代 `paddle.Model`,我们照它的设计走 |
| [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics) | 取代 `paddle.metric.*`,我们照它的设计走 |
| [Lua Lanes](https://github.com/LuaLanes/lanes) | 多线程,`num_workers > 0` 的前提 |
| [sol2](https://github.com/ThePhD/sol2) | C++/Lua 绑定库 |
| [Penlight](https://github.com/lunarmodules/Penlight) | 类、集合、兼容层。vendored 受限子集,MIT |
| [Insight7](https://github.com/PlumBlossomMaid) | 这套技术栈里 numpy 的位置。字符串切片与关键字参数约定也源自它 |

---

## 文档

完整技术方案在 **[`docs/`](docs/migration/README.md)**。

---

## License

Apache 2.0,与 PaddlePaddle 一致。
