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
| 关键字参数 | **`argrule` 规则表**(R26/R27) | schema 抄 argcheck:`{name=,type=,default=,defaulta=,defaultf=,opt=,check=,doc=}`(**只留 `doc`,没有 `help`**);求解器是我们自己的 O(N) 生成器。独立 rock,见 `plan/argrule.md` |
| `from paddle import Tensor` | `local Tensor = paddle.Tensor` | **短名要能直接 local 化** —— Python 侧就是这么用的。因此 `argrule` 里的类型短名 `"Tensor"` 与这个导出名**必须是同一个标识符** |
| 第一个参数是一个列表 | 写 `type = "TensorList"`(= `list_of("Tensor")`),**不要写 `"table"`** | `concat` / `stack` / `meshgrid` / `broadcast_tensors`。元素类型一查,`concat{a,b}` 与 `concat{{a,b},2}` 都变确定,**不需要 `nonamed`**(`plan/argrule.md` ⑧)|
| 位置参数中间省略 | **不支持** | `f(1, nil, 3)` 要跳过就用具名表 `f{a=1,c=3}`。**与 Python 一致**,且支持它要付 3^N 的代价(`foundations.md` §4.5)|

**不做"Lua 风格化"改名。** 用户是冲着 Paddle API 来的,
`paddle.add_n` 改成 `paddle.addN` 只会让所有 Python 文档失效。

**第二行是硬规则,不是风格偏好。** 我们的调用形式**只有两种**:
全位置(可以尾部省略)、或者一个具名表。**不允许位置参数中间留空靠类型猜** ——
Python 不允许,而支持它要付 3^N 的生成代码代价,argcheck 就是死在这上面的
(`foundations.md` §4.5)。

**每份 api 文档在导出清单里给出规则表即可**,不必标「用哪个实现」——
只有一套(C11)。P3 生成的 2000+ 算子共用同一份 schema,但**构建期展开**,
运行时不调 `argrule`。

### 2.1.1 ★ 枚举参数一律不接受裸数字

`dtype` / `device` / `layout` / `reduction` 这类枚举参数,**只接受两种东西**:

| ✅ 接受 | ❌ 不接受 |
|---|---|
| 字符串:`"float32"` / `"cpu"` / `"NCHW"` / `"mean"` | **裸整数** `5`(某个 enum 的序号) |
| 类型化常量:`paddle.float32` / `paddle.CPUPlace()` | Python 侧偶尔能塞进去的 C++ enum 值 |

三个理由,按重要性:

1. **数字不表示类型。** `zeros({2,3}, 5)` 应该报错,而不是"dtype 取第 5 个"。
   enum 序号是 C++ 的实现细节,一旦上游插入一个新 dtype,所有写死数字的代码**静默错**
2. **它消掉调用歧义。** 见 `plan/argrule.md` §2.6:`paddle.zeros{2, 3}` 之所以能被
   确定性地解析成「一个 table 实参」,正是因为 `{2, 3}` 当作「表内位置」时
   `dtype = 3` 过不了类型检查。**枚举一旦收数字,这个判定就塌了**
3. 与 1-based 同源:我们不让用户去数序号

⚠️ 这条对**生成的算子**同样成立 —— 生成器给 `DataType` / `Place` / `DataLayout`
参数发的类型是 `{"string", "paddle.dtype"}`,不含 `"number"`。

### 2.1.2 ★ `shape` 一类的参数是 `IntList`,不接受数字

`shape` 的**类型**是容器,不是数字 —— 上游的 `ShapeLike` 里确实没有 `int`
(`python/paddle/_typing/shape.py:22-33`)。Lua 侧对等,**统一用一个注册类型 `IntList`**:

| ✅ | ❌ |
|---|---|
| `{2, 3}` / `List{2, 3}` / 整数一维 `insight.Array` / **一维 int `Tensor`** | 带小数的 table、二维 Array 或 Tensor、非整数 dtype |

适用于 `shape` / `axes` / `perm` / `strides` 等一切"整数列表"参数。

⛔ **`IntList` 的判据是「是个容器 + 装的是整数」,不是「是这三个类之一」。**
写成 `{"table", "pl.List", "insight.Array"}` 就把「容器」硬编码成了一张框架名单,
用户自己的容器类会被无理由挡住。**一维 int `Tensor` 也算**(上游收:`creation.py:1832`);
元素还可以是 0-D 整数 Tensor(`creation.py:1831`)。
容器协议、O(1)/O(n) 选路见 `plan/argrule.md` §2.3。

⚠️ **`paddle.zeros(2, 3)` / `zeros(5)` 在上游是合法的,我们仍然不支持** —— 见下面 §2.1.3。

**元素是不是整数,在转换层查(table -> `vector<int64_t>` 那步反正要遍历),
不在类型层查** —— 但必须 `error`、必须带下标、必须指向调用点,不许静默取整。

**永远不要把 `number` 塞进 `shape` 的 `type` 里** —— `{"number", "IntList"}`
会让「表内位置」解释重新成立,凭空造出调用歧义。上游也没这么干
(`ShapeLike` 里没有 `int`,它用的是签名外面的装饰器)。

### 2.1.3 ★ `decorator_utils.py` 整层不移植 —— 我们用 Paddle 自己的规范

> 人的话:「变长 size `zeros(2,3)` 这种纯粹是照搬 PyTorch,我们在 Lua 里面不打算引入
> PyTorch 的东西。`decorator_utils.py` 这玩意我觉得可以忽略了 —— 这东西就是 Paddle
> 为了 PyTorch 用户用得惯才搞的。**我们 Paddle 框架就应该用自己的语法和规范**。」

**结论:`python/paddle/utils/decorator_utils.py`(1451 行,30+ 个装饰器,68 个模块 import 它)
整层跳过。** 绑定的是**被装饰的那个函数**,不是外面那层壳。

**✅ 证据(不是印象,是读出来的):**

| 事实 | 出处 |
|---|---|
| 全文 **51 处**出现 `PyTorch` / `torch.`,多处 docstring 直接写 `PyTorch: torch.block_diag(x,y,z)` / `Paddle: paddle.block_diag([x,y,z])` | `decorator_utils.py:923-940` |
| **每个 wrapper 都写 `wrapper.__signature__ = inspect.signature(func)`(35 处)** —— 上游自己把**规范签名**保留在内层 | `:189, 435, 466, 540, 566` … |
| 291 处参数别名(`@param_one_alias` / `@param_two_alias`) | `:171-223` |
| 5 处变长 size(`@size_args_decorator`) | `:406-437` |
| 甚至有专门"劝返"的装饰器:收到 torch 才有的关键字就报错并指向 compat API | `:517-542` `forbid_keywords` |

**最后一行证据最关键:上游自己认为规范签名是内层那个。**
所以我们的生成器读到的**本来就是**规范签名 —— **忽略这层是默认行为,不是额外工作量**。

| 兼容糖 | 上游 | 我们写什么 |
|---|---|---|
| 参数别名 `concat(tensors=…, dim=…)` | 291 处 | `concat{x = …, axis = …}` |
| 变长 size `zeros(2, 3)` | 5 处 | `zeros{2, 3}` |
| 变长 tensor `block_diag(x, y, z)` | `variadic_tensor_decorator` | `block_diag{x, y, z}` |
| `reshape(2,3)` / `transpose(0,1)` / `expand(...)` | 各 1 处 | 传容器 |

**为什么这不是"少做了功能":**

1. **糖的收益在 Lua 侧本来就是 0** —— Python 要写 `zeros([2,3])`,又括号又方括号;
   Lua 的表调用本来就省掉括号,**`zeros{2,3}` 和 `zeros(2,3)` 一样长**
2. **我们没有存量代码要接** —— 兼容层服务的是"torch 用户的肌肉记忆 + 已有脚本",
   Lua 侧两者都是空的
3. **别名与具名表调用相冲**:`concat{x = a, tensors = b}` 怎么办?
   上游抛 `ValueError`(`:181-184`)—— 我们得把这个检查复制 291 遍
4. **`dim` / `axis` 两个名字 = index 语义标注表两条记录**,漏标一条就是静默 off-by-one(§2.2)

⚠️ **一条例外通道,判据写死:**
**「这个写法在 Lua 里比规范写法更短或更清楚吗?」不是就不做。理由不能是"上游有"。**

⚠️ **一条必须查的**:`decorator_utils.py` 里少数装饰器不只是改参数
(如 `legacy_reduction_decorator` / `view_decorator`)。跳过它们之前,
**逐个确认被装饰函数的语义不依赖那层壳** —— 判据是 §2.1.3 的证据第二行:
壳只动参数、内层签名不变,才可以跳过。发现有壳改了行为的,记 OPEN_QUESTION,不要默认放行。

**代替方案:一次性的教学式报错**,而不是一个永久的二义入口:

```
paddle.zeros(2, 3)
-> shape must be a container of integers, got number
     did you mean:  paddle.zeros{2, 3}
```

### 2.1.4 ★ 参数名一律用 **Paddle 自己的**,不用 PyTorch 那套

> 人的话:「**所以我们的 paddle-lua 也不要那些 PyTorch 的东西**。」

§2.1.3 挡的是 `decorator_utils.py` 那层**外壳**。但有一批 torch 风格的参数
**已经进了上游的规范签名本身**,壳挡不住,要在这里挡:

```python
# python/paddle/tensor/creation.py:1807  zeros 的真实签名
def zeros(shape, dtype=None, name=None, *,
          out=None, device=None, requires_grad=False, pin_memory=False)
```

而 Paddle 自己的名字在 `to_tensor` 里原样保留着(`creation.py:1124-1129`):
`place` / `stop_gradient`。**同一个概念,两套名字,我们只取 Paddle 那套。**

| ❌ 不用(PyTorch) | ✅ 用(Paddle 原生) | 备注 |
|---|---|---|
| `requires_grad` | **`stop_gradient`** | ⚠️ **两者取反**,见下 |
| `device` | **`place`** | `to_tensor` 用的就是 `place` |
| `dim` | **`axis`** | 少一个名字 = index 语义标注表少一条(§2.2) |
| `input` / `tensors` | **`x`** | |
| `size`(参数) | **`shape`** | ⚠️ `Tensor.size`(元素总数)是 Paddle 原生属性,**不在此列** |
| `pin_memory = true` | **`place = paddle.CUDAPinnedPlace()`** | 能力不丢,用原生方式表达 |

⚠️ **`requires_grad` 必须挡住的真正原因不是命名洁癖,是它和 `stop_gradient` 取反:**

| | 默认值 | 「要梯度」时写 |
|---|---|---|
| `requires_grad` | `False` | `true` |
| `stop_gradient` | `True` | `false` |

**默认值也是反的。** 两个名字同时存在,任何一次搞混都是**静默地训不动**
(梯度永远不流,loss 不降,而没有任何报错)。**一个概念只留一个名字。**

**这条规则约束的是「用户看得见的 API 表面」。实现层不受限** ——
GC 抄 Torch7 九层方案(D5)、规则表 schema 抄 argcheck(D30)都照做,
因为用户不会在自己的代码里写到它们。

⬜ **未决:`out = ` 参数。** 它不只是改名,是"写进预分配的显存"这个**能力**
(Paddle 原生的等价物是 `paddle.assign(x, output)`)。倾向 **v1 不做**,
有真实需求(显存复用)再单独论证 —— 记进 `process/open-questions.md`。

**CI 判据(机械可查):** 规则表里出现下列参数名即失败 ——

```
grep -rnE 'name *= *"(requires_grad|device|dim|input|tensors|pin_memory)"'  lua/   -> 失败
```

---

## 2.2 索引:全 1-based,且必须在文档里逐个标出来

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
