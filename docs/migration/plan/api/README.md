# API 设计 · 按 Lua 模块组织

> **`plan/modules/` 回答「怎么造」(按阶段),本目录回答「造出来长什么样」(按模块)。**
> 一个是实现,一个是接口。分开放,因为它们的读者、变更频率、稳定性要求都不同。

| | `plan/modules/NN-*.md` | `plan/api/*.md`(本目录) |
|---|---|---|
| 组织方式 | 按**阶段** P0–P18 | 按 **Lua 模块** `paddle.io` / `paddle.nn` … |
| 内容 | 上游可抄什么、实现设计、坑、验收 | **导出清单、签名、参数名、返回类型、与 Python 的差异** |
| 读者 | 实现者 | 实现者 **+ 将来的用户文档** |
| 稳定性 | 做完就冻结 | **一旦发布就是契约**,改动要走废弃流程 |

**工作方式:一个模块一个模块来。** 每个模块开工的第一件事是把它的 api 文档写完 ——
**先定接口再写实现**,否则实现会反过来决定接口,最后 API 长成实现的形状。

---

## 1. 模块清单

| Lua 模块 | 阶段 | api 文档 | 说明 |
|---|---|---|---|
| `paddle`(顶层) | P5 | `top.md` | Tensor 构造、算子、`paddle.to_tensor`、懒加载装配 |
| `paddle.Tensor`(类型) | P5 | `tensor.md` | 方法、元方法、字符串索引 |
| `paddle.dtype` / `paddle.device` | P5 | `dtype-device.md` | 与 Insight7 命名对齐(foundations §3.2) |
| `paddle.autograd` | P6 | `autograd.md` | `backward` / `no_grad` / `grad` |
| `paddle.scope` | P6/P14 | `scope.md` | 显存作用域,gc.md 九层里唯一用户可见的一层 |
| `paddle.linalg` | P5 | `linalg.md` | 绝大部分是生成的算子的重导出 |
| `paddle.nn` | **P9** | `nn.md` | 40 个层 + 容器 + 损失 |
| `paddle.nn.functional` | P9 | `nn-functional.md` | 多数是对 `_ops` 的一行转发 |
| `paddle.nn.initializer` | P9 | `nn-initializer.md` | 随机数必须走 Paddle 的生成器 |
| `paddle.optimizer` | P10 | `optimizer.md` | 优化器 + `paddle.optimizer.lr` 调度器 |
| `paddle.io` | **P11** | `io.md` | ★ **本目录的样板文档** |
| `paddle.dataset` | P12 | `dataset.md` | MNIST / CIFAR … |
| `paddle.vision` | P12 | `vision.md` | `transforms` / `models` / `datasets` |
| `paddle.jit` | P8/P15/P18 | `jit.md` | `load`(P8)/ `trace`(P15)/ `script`(P18)分三批 |
| `paddle.amp` | 延后 | — | 未排期 |
| `paddle.distributed` | P17 | `distributed.md` | ⛔ 无环境,写得了测不了 |
| `paddle.utils` | P4 | `utils.md` | `fs` / `download`,我们自己的,Python 侧无对应 |
| `paddle.pl` | P4 | — | 暴露 vendored Penlight(foundations §1.3) |

**明确不做的模块**(理由见 `plan/overview.md` §2.2):

| | 替代 |
|---|---|
| `paddle.Model` | **永久排除** -> `ocean-lua` |
| `paddle.metric` | -> `metrics-lua` |
| `paddle.static` | 静态图入口收敛到 `paddle.jit`,不复刻 fluid 的 `Program`/`Executor` API |
| `paddle.fluid` | 上游自己都在删 |

---

## 2. 跨模块的硬规则

写任何一份 api 文档前先看这一节。**这些规则在每个模块里都成立,不要在模块文档里重复论证。**

### 2.1 命名映射

| Python | Lua | 备注 |
|---|---|---|
| `paddle.nn.Linear` | `paddle.nn.Linear` | **类名保持 CamelCase** |
| `paddle.add_n` | `paddle.add_n` | **函数名保持 snake_case**,不改成 camelCase |
| `__init__` | `_init` | Penlight 约定(D25) |
| `__len__` | `:len()` | Lua 的 `#` 对我们的实例不可靠(D27) |
| `__getitem__` | `:get(i)` 或 `t["1:3"]` | 数据访问用 `get`,张量切片用字符串索引(D14) |
| `__call__` | `__call` 元方法 | `layer(x)` 能用 |
| `super().__init__()` | `self:super()` | Penlight 约定 |
| 关键字参数 | `_wrap` 三模式(D-R8) | `f{a=1,b=2}` / `f(1,2)` / `f(1,{b=2})` |

**不做"Lua 风格化"改名。** 用户是冲着 Paddle API 来的,
`paddle.add_n` 改成 `paddle.addN` 只会让所有 Python 文档失效。

### 2.2 索引:全 1-based,且必须在文档里逐个标出来

每份 api 文档**必须有一节列出该模块所有吃 index 语义的参数**,
这是 `overview.md` §6.1.1 那张标注表在模块层面的落点,也是 `ci.md` §6 第 ④ 条的检查对象。

**"这个模块没有 index 参数"也要显式写出来**,不能留空 —— 留空分不清是"没有"还是"忘了看"。

### 2.3 返回类型

| 场景 | 返回 |
|---|---|
| 集合、要给用户遍历的 | **`pl.List`**(D-R21) |
| 与 Python 的 generator 对应 | **迭代器函数**(`named_parameters()` 一类) |
| **热路径 / 框架内部 / 生成代码** | **裸表**(`x:shape()`、`_wrap` 的解析结果) |

边界在 `foundations.md` §2.3。**这条最容易在模块文档里写反** ——
判据是"用户会不会拿它做集合操作",不是"它是不是数组"。

### 2.4 稳定性标记

每个导出项标一个:

| 标记 | 含义 |
|---|---|
| **稳定** | 发布后不改签名。改动要走废弃流程(保留一个大版本 + 警告) |
| **实验** | 可能改。文档里显式标注,名字**不加**前缀(加前缀会导致转正时全体改名) |
| **内部** | `_` 前缀。用户碰了出问题不管 |

### 2.5 惰性加载

`require "paddle"` **不得**递归加载全部子模块。理由三条:

1. `_ops/` 是 ~20k 行生成的 Lua,全量加载拖慢启动
2. `paddle.vision` / `paddle.dataset` 依赖 **Insight7,而它是软强制**(foundations §3.7)——
   没装 Insight7 的用户 `require "paddle"` 必须能成功
3. `paddle.distributed` 在无环境时应该在**用到时**报错,而不是在 `require` 时

实现:顶层 `paddle` 表挂 `__index`,首次访问 `paddle.nn` 时才 `require`。
**缺失的可选依赖必须在这一刻给出能照着做的错误信息**,而不是 `module not found`。

---

## 3. 每份 api 文档的固定骨架

照抄,不要自创:

```markdown
# paddle.<模块名>

| | |
|---|---|
| 阶段 | P<n> |
| 依赖模块 | ... |
| 可选依赖 | ...(缺失时的行为) |
| 稳定性 | 稳定 / 实验 |

## 1. 导出清单          ← 一张表,列全。这是契约
## 2. 详细签名          ← 每个导出项的参数名、类型、默认值、返回
## 3. index 语义参数     ← §2.2,没有也要写"无"
## 4. 与 Python 的差异   ← 用户迁移时会踩的,逐条
## 5. 用法示例          ← 至少一个完整可跑的
## 6. 本模块特有的坑
## 7. 未实现 / 延后      ← 明确说不做什么,比含糊留白好
```

**§4 和 §7 是最容易偷懒、又最值钱的两节。**
用户不会读我们的实现,他们会拿 Python 代码逐行翻译 ——
差异表就是他们唯一的地图;而 §7 决定他们会不会花两小时找一个根本不存在的函数。
