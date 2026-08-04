# 模块详细设计索引

> 每个文件回答同一组问题:**做什么 / 上游有什么可用 / 怎么做 / 坑在哪 / 怎么验收。**
> 顺序与依赖在 `plan/roadmap.md`;范围与工作量在 `plan/overview.md`。

> 📌 **`plan/foundations.md` 不在这张表里,因为它不属于任何单一阶段。**
> 它定的是生态基座(类系统 = `pl.class`,集合 = `pl.List`,numpy 的位置 = Insight7),
> 从 P5 开始每个阶段都受它约束。**写第一行 Lua 之前读一次。**

---

## 文件清单

| 阶段 | 文件 | 模块 | 类别 | 预估 |
|---|---|---|---|---|
| P0 | `00-build.md` | 无 Python 构建 | 基础设施 | 3–5 天 |
| P1 | `01-c-abi.md` | C ABI 中间层 | 手写 C++ | 2 周 |
| P2 | `02-binding.md` | sol2 绑定层 | 手写 C++ | 2 周 |
| P3 | `03-codegen.md` | 算子代码生成 | 生成器 | 3 周 |
| P4 | `04-packaging.md` | 构建与打包 | 基础设施 | 1 周 |
| P5 | `05-tensor.md` | `paddle.Tensor` | 手写 C+Lua | 2 周 |
| P6 | `06-autograd.md` | 自动微分与生命周期 | 手写 C+Lua | 2 周 |
| P7 | `07-serialization.md` | `paddle.save/load` | 手写 Lua | 1 周 |
| P8 | `08-jit-infer.md` | `jit::Layer` 推理 ★ | 薄绑定 | 1 周 |
| P9 | `09-nn.md` | `paddle.nn` | 手写 Lua(最大) | 6 周 |
| P10 | `10-optimizer.md` | `paddle.optimizer` | 手写 Lua | 2 周 |
| P11 | `11-io.md` | `paddle.io`(单 worker) | 手写 Lua | 1.5 周 |
| P12 | `12-dataset-vision.md` | `dataset` + `vision` | 语义直抄 | 3 周 |
| P13 | `13-lanes.md` | Lanes 多 worker | 手写 C++ | 3 周 |
| P14 | `14-gc.md` | GC 九层机制 | 手写 C+Lua | 2 周 |
| P15 | `15-to-static.md` | 静态图 trace 模式 | 复用 + 手写 | 4 周 |
| P16 | `16-cpp-extension.md` | 自定义算子 | 部分直抄 | 2 周 |
| P17 | `17-distributed.md` | 分布式(测不了) | 薄绑定 | 3 周 |
| P18 | `18-script.md` | script 模式(AST) | 手写 Lua | 6 周 |

---

## 类别的含义

| 类别 | 含义 | 风险 |
|---|---|---|
| **语义直抄** | Python 侧逻辑与语言无关,逐行翻译即可 | 低 —— 主要成本是耐心 |
| **薄绑定** | 上游已有完备 C++ 接口,我们只包一层 | 低 —— 但受 M0 验证结果制约 |
| **生成器** | 一次写对,产出上万行 | 中 —— 生成器的 bug 会被放大 |
| **手写 Lua** | 无可复用等价物,必须重写 | 中 —— 量大,但每一块都简单 |
| **手写 C++** | 涉及 ABI / 线程 / 生命周期 | 高 —— 错了会崩进程,不是抛异常 |
| **复用 + 手写** | 拼接上游 C++ 组件 | 高 —— 依赖对上游内部结构的正确理解 |

---

## 每份模块文档的固定结构

写新模块文档时照抄这个骨架,**不要自创结构**:

```markdown
# P<n> · <模块名>

| | |
|---|---|
| 阶段 | P<n> |
| 类别 | ... |
| 开工条件 | ... |
| 预估 | ... |

## 1. 做什么 / 不做什么
## 2. 上游有什么可以用            ← 必须带 file:line
## 3. 设计
## 4. 已知的坑
## 5. 验收
## 6. 未解问题                    ← 同步到 process/open-questions.md
```

**§2 里每一条 Paddle API 都必须有 `file:line` 出处**,这是 `CLAUDE.md` §4 反幻觉规程的硬要求。
写不出出处的,标 `⚠️ 未核实`。
