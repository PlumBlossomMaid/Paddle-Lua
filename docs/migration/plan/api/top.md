# paddle(顶层命名空间)

| | |
|---|---|
| 阶段 | P5(Tensor 构造与基础算子绑定)+ P3(生成器产出算子) |
| 依赖模块 | `paddle.Tensor`(类型)、`argrule`(签名层,D31)、Insight7(`from_insight` 入口,D26) |
| 可选依赖 | Insight7(仅 `from_insight` 入口;未装则入口报错,不静默降级) |
| 稳定性 | **稳定**(已迁移部分)/ 实验(新迁部分,逐批升) |
| 实现文档 | `plan/modules/03-codegen.md`(算子生成)、`plan/modules/05-tensor.md` |
| 迁移路书 | `inventory.md`(顶层 551 项机器分桶,DIRECT 196 / LEGACY 17 / PYTHON 328 / SPECIAL 10) |

> 📌 本文是 `paddle` 顶层命名空间的 api 契约。**迁移顺序与三态去向以
> `inventory.md` 为准**,本文只写已确认的导出契约,不重复分桶论证。
> 逐项抽查记录见 `inventory.md` §3.1(约 80 个核心成员,全部带上游 `file:line`)。

---

## 1. 导出清单

顶层命名空间共 551 项(`python/paddle/__init__.py` 的 `__all__`)。分桶:

| 桶 | 数量 | 处理 |
|---|---|---|
| **直接绑定** | 196 | P3 算子生成器(C8)产出,名字照搬 |
| **LEGACY** | 17 | 逐个评估(P3 时定,多数仍直接绑定) |
| **Lua 侧重设计 / 放放** | 328 | 按 §2 详细签名逐批迁移;依赖 Python 运行时的放放 |
| **跳过** | 10 | D10 / 无环境 / 兼容层(SPECIAL 桶) |

**本文第一批(已抽查,契约已定)**:创建 13 / 基础算子 22 / 归约统计 12 /
索引操作 14 / 逻辑比较 8 / 随机 5 / 激活 4 / 复数 4 / 杂项 6 —— 约 80 项,
见 §2。其余成员按 `inventory.md` 分桶在 P3 前后逐批补充进 §2。

---

## 2. 详细签名

> 调用约定:一张参数表(`paddle.f{x, opt=...}`,见 `api/README.md` §2.1.5)。
> 位置参数形式也支持;示例一律用混合表。**D35:PyTorch 兼容装饰器
> (别名、`size_args`、`variadic`)一律不移植** —— 下表只列 Lua 侧签名。

### 2.1 创建(Tensor 构造)

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.tensor{data, dtype, place, requires_grad}` | Lua 重写 | 数据→Tensor 构造(`creation.py:1009/1121`,inventory §3.1) |
| `paddle.to_tensor{data, dtype, place, stop_gradient}` | Lua 重写 | 独立入口,名字照搬(不合并);`requires_grad→stop_gradient` 取反(C12) |
| `paddle.from_insight{array}` | Lua 重写 | ⚠️ **唯一命名特例**:numpy→Insight(D26),底层走 DLPack 桥;未装 Insight7 报错 |
| `paddle.zeros{shape, dtype}` | 直接绑定 | `size_args` 不移植;`zeros{2,3}` 表调用(`creation.py:1807`) |
| `paddle.ones{shape, dtype}` | 直接绑定 | 同上 |
| `paddle.full{shape, fill_value, dtype}` | 直接绑定 | `full{{2,3},1.0}` 是合法表内位置形式(§2.1.5 判据) |
| `paddle.full_like{x, fill_value, dtype}` | 直接绑定 | |
| `paddle.empty{shape, dtype}` | 直接绑定 | ⚠️ `device/requires_grad/pin_memory` 是 PyTorch 参数,C12 不引入(`creation.py:3067-3071`) |
| `paddle.eye{n, m, dtype}` | 直接绑定 | `_C_ops.eye`(`creation.py:2038`) |
| `paddle.linspace{start, stop, num, dtype}` | 直接绑定 | dtype 默认推断 Lua 重写(`creation.py:428/486`) |
| `paddle.arange{start, end, step, dtype}` | 直接绑定 | `end=None` 折叠 + dtype 推断 Lua 重写(`creation.py:2265-2276`);LEGACY 桶 |
| `paddle.assign{x, output}` | 直接绑定 | 拷贝语义(非 view,`creation.py:3473`);LEGACY 桶 |

### 2.2 基础算子

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.add{x, y, alpha}` | 直接绑定 | `alpha` 缩放 y(`math.py:802`) |
| `paddle.subtract{x, y, alpha}` | 直接绑定 | 同上(`math.py:988`) |
| `paddle.multiply{x, y}` / `maximum{x, y}` / `minimum{x, y}` | 直接绑定 | 纯 re-export(`math.py:90-92`) |
| `paddle.divide{x, y}` / `floor_divide{x, y}` / `remainder{x, y}` | 直接绑定 | LEGACY 桶(`math.py:1093/1118/1305`) |
| `paddle.pow{x, y}` | 直接绑定 + Lua 适配 | 标量→`_C_ops.pow`、Tensor→`_C_ops.elementwise_pow`(`math.py:601-603`) |
| `paddle.exp` / `log` / `log2` / `log10` / `sqrt` / `square` / `abs` | 直接绑定 | 纯 re-export(`math.py:27`) |
| `paddle.sin/cos/tan/asin/acos/atan/sinh/cosh/tanh/asinh/acosh/atanh{x}` | 直接绑定 | 纯 re-export(`math.py:30-119`);`_` in-place 变体同列 |
| `paddle.ceil` / `floor` / `round` / `sign{x}` | 直接绑定 | 纯 re-export |
| `paddle.masked_fill{x, mask, value}` | 直接绑定 | `_C_ops.masked_fill`(`manipulation.py:7278`) |
| `paddle.masked_select{x, mask}` | 直接绑定 | `_C_ops.masked_select`(`search.py:1139`) |
| `paddle.lerp{x, y, weight}` | 直接绑定 | `_C_ops.lerp`(`math.py:4859`) |
| `paddle.einsum{equation, *operands}` | 直接绑定 | equation 字符串原样透传(`einsum.py:925`);LEGACY 桶 |
| `paddle.logaddexp{x, y}` | **Lua 重写** | ⚠️ **无 C++ 算子**:`log1p(exp(-abs(x-y)))+maximum`(`math.py:906-913`) |

### 2.3 归约与统计

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.sum{x, axis, dtype, keepdim}` | 直接绑定 | DIRECT 桶;`axis` 标 1-based |
| `paddle.max{x, axis, keepdim}` / `min{x, axis, keepdim}` | 直接绑定 | LEGACY 桶(min);`max` 传播梯度到所有相等极大值(与 amax 区分) |
| `paddle.mean{x, axis, keepdim, dtype}` | 直接绑定 + Lua 适配 | `dtype` 先 cast 再 mean(`stat.py:143-149`) |
| `paddle.prod{x, axis, keepdim, dtype}` | 直接绑定 + Lua 适配 | dtype cast + 0-size axis 特判 Lua 重写(`math.py:4127-4139`) |
| `paddle.var{x, axis, unbiased, keepdim, correction}` | 直接绑定 + **Lua 重写(CPU)** | ⚠️ 两路:CPU = 纯 Python 组合(mean+pow+sum+除,`stat.py:222-280`) |
| `paddle.std{x, axis, unbiased, keepdim, correction}` | 直接绑定 + **Lua 重写(CPU)** | 同上 + sqrt |
| `paddle.cumsum{x, axis, dtype}` / `cumprod{x, dim, dtype}` | 直接绑定 + Lua 适配 | 整数默认 cast int64(`math.py:3462-3480/3965`) |
| `paddle.sort{x, axis, descending, stable}` | 直接绑定 | `_C_ops.argsort` 取 values(`search.py:584`) |
| `paddle.argsort{x, axis, descending, stable}` | 直接绑定 | **返回双值**,indices 1-based(`search.py:162`) |
| `paddle.topk{x, k, axis, largest, sorted}` | 直接绑定 | **返回双值**,indices 1-based(`search.py:1236-1240`) |
| `paddle.median{x, axis, keepdim, mode}` | 直接绑定 + Lua 适配 | ⚠️ 别名时 mode 默认变 'min'(`stat.py:626`);Lua 无别名 → mode 固定默认 'avg' |
| `paddle.mode{x, axis, keepdim}` | 直接绑定 | **返回双值**,indices 1-based(`search.py:720-723`) |
| `paddle.nan_to_num{x, nan, posinf, neginf}` | **Lua 重写** | full_like + 3×where 组合 + finfo 边界表(`math.py:1587-1608`) |
| `paddle.norm{x, p, axis, keepdim}` | **Lua 重写(分发表)** | 分发给 vector_norm/matrix_norm(`linalg.py:1238-1271`) |

### 2.4 索引与操作

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.concat{x_list, axis}` | 直接绑定 | `axis` 1-based;别名 tensors/dim 不移植(`manipulation.py:1556`) |
| `paddle.stack{x_list, axis}` | 直接绑定 | `axis` 1-based |
| `paddle.transpose{x, perm}` | 直接绑定 | 参数名 `perm`(Paddle 原生) |
| `paddle.reshape{x, shape}` | 直接绑定 + Lua 适配 | 变长 `reshape(2,3)` 不移植;**共享内存** view |
| `paddle.view{x, shape_or_dtype}` | 直接绑定 + Lua 适配 | @overload/变长不移植;共享内存;`view_shape`/`view_dtype` 双路(`manipulation.py:8383-8391`) |
| `paddle.squeeze{x, axis}` / `unsqueeze{x, axis}` | 直接绑定 | `axis` 1-based;共享内存 view |
| `paddle.flatten{x, start_axis, stop_axis}` | 直接绑定 | 起止轴 1-based;共享内存 |
| `paddle.split{x, num_or_sections, axis}` | 直接绑定 | 返回 `pl.List`;`axis` 1-based |
| `paddle.expand{x, shape}` / `broadcast_to{x, shape}` | 直接绑定 | `broadcast_to` 是一行转发到 expand(`manipulation.py:5277`);共享内存 |
| `paddle.tile{x, repeat_times}` | 直接绑定 | 逐维复制,非广播 |
| `paddle.flip{x, axis}` / `roll{x, shifts, axis}` | 直接绑定 | `axis` 1-based |
| `paddle.diag{x}` / `tril{x, k}` / `triu{x, k}` | 直接绑定 | ⚠️ `k` 是**偏移量不是索引语义**,不标 1-based(§2.2 例外) |
| `paddle.rot90{x, k, axes}` | **Lua 重写** | ⚠️ 无 C++ 算子:`flip`+`transpose` 组合(`manipulation.py:1919-1932`);axes 1-based |
| `paddle.gather{x, index, axis}` / `gather_nd{x, index}` | 直接绑定 | **index 1-based**(C4,边界 -1) |
| `paddle.scatter{x, index, updates, overwrite}` | 直接绑定 | **index 1-based** |
| `paddle.index_select{x, index, axis}` | 直接绑定 + Lua 适配 | @index_select_decorator(PyTorch 参数序)不移植;index 1-based |
| `paddle.take_along_axis{x, indices, axis}` | 直接绑定 | indices 1-based;与 gather 同底层 |
| `paddle.take{x, index, mode}` | **Lua 重写** | mode 三态 + index_select 组合(`math.py:5855`);index 1-based |
| `paddle.nonzero{x, as_tuple}` | 直接绑定 + Lua 适配 | 返回坐标 tensor **1-based**;as_tuple 变体 Lua 重写 |
| `paddle.argwhere{x}` | 直接绑定 | 一行转发到 nonzero(`search.py:449-450`);1-based |
| `paddle.unique{x, return_index, return_inverse, return_counts, axis, dtype, sorted}` | 直接绑定 + Lua 适配 | @overload 族不移植;空张量特判照抄;返回 1–4 值按 flag 组合 |
| `paddle.isclose{x, y, rtol, atol, equal_nan}` / `allclose{...}` | 直接绑定 | 纯 re-export(`logic.py:22/32`) |

### 2.5 逻辑与比较

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.equal{x, y}` / `greater_equal` / `less_equal` / `less_than` / `not_equal{x, y}` | 直接绑定 + Lua 适配 | 标量升 0-d 逻辑 Lua 重写一次,五函数共用(`logic.py:221-227`);输出 bool 且 stop_gradient |
| `paddle.logical_and{x, y}` / `logical_or{x, y}` / `logical_not{x}` | 直接绑定 | 纯 re-export(`logic.py:33-35`);输入须 bool |
| `paddle.isfinite{x}` / `isinf{x}` / `isnan{x}` | 直接绑定 | 纯 re-export(`math.py:80-82`) |
| `paddle.where{condition, x, y}` | 直接绑定 + Lua 适配 | 三态广播显式照抄;`where{cond}` 单参 = nonzero;标量走 to_tensor(`search.py:797-847`) |
| `paddle.cast{x, dtype}` | 直接绑定 + Lua 适配 | dtype 字符串→枚举映射表 Lua 共用(`manipulation.py:251-259`) |
| `paddle.rank{x}` | **Lua 重写** | `#x.shape` + 0-d int32 打包(`attribute.py:69-70`);⚠️ 返回 0-d Tensor 非 number |
| `paddle.is_tensor{x}` | **Lua 重写** | 类型检查一行(`logic.py:756`) |

### 2.6 随机

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.rand{shape, dtype}` / `randn{shape, dtype}` | 直接绑定 + Lua 适配 | `size_args` 不移植;内核 `_C_ops.uniform/gaussian`(`random.py:1599/716`) |
| `paddle.randint{low, high, shape, dtype}` | 直接绑定 + Lua 适配 | `size_args` 不移植(`random.py:1891/1902`) |
| `paddle.randperm{n, dtype}` | 直接绑定 | **返回排列索引 tensor,1-based**(C4;上游是 0..n-1,Lua 侧 +1 或 doc 注明) |

### 2.7 激活(顶层入口;`nn.functional` 全量见 `nn-functional.md`)

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.relu{x, inplace}` | 直接绑定 + Lua 适配 | `inplace` 切 `_C_ops.relu_`/`_C_ops.relu`(`activation.py:836`);与 `nn.functional.relu` 同函数 |
| `paddle.sigmoid{x}` / `tanh{x}` | 直接绑定 | 纯 re-export(`math.py:104/119`) |
| `paddle.nn.functional.leaky_relu{x, negative_slope, inplace}` | 直接绑定 + Lua 适配 | inplace 双路(`activation.py:516-518`) |
| `paddle.nn.functional.silu{x, inplace}` | 直接绑定 | `_C_ops.silu`(`activation.py:1169`) |

### 2.8 复数

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.angle{x}` / `conj{x}` / `real{x}` / `imag{x}` | 直接绑定 | 纯 re-export(`math.py:39/58`、`attribute.py:23`) |
| `paddle.complex{real, imag}` | 直接绑定 | 与 as_complex 同族(`creation.py:3730`) |

---

## 3. index 语义参数

> `api/README.md` §2.2:每份文档必须列全,没有也要显式写。这是 CI 红线
> (`ci.md` §6 ④)的检查对象。**标注表人工维护**(`tools/gen/index_semantics.lua`),
> 生成器遇到未标注算子必须报错,不得默认放行(§8.2)。

| 参数 | 语义 | 基准 | 转换规则 |
|---|---|---|---|
| 所有 `axis` / `dim` / `axes` / `start_axis` / `stop_axis` / `perm` | 维度索引 | **1-based** | Lua `n` → C++ `n-1`;负数不变 |
| `concat.x_list` / `stack.x_list` | 张量列表 | —(非索引) | — |
| `gather.index` / `gather_nd.index` / `index_select.index` / `take.index` / `take_along_axis.indices` / `scatter.index` | **吃 index 的 tensor** | **1-based** | Lua 值 → C++ `-1` |
| `argsort`/`topk`/`mode`/`nonzero`/`argwhere`/`randperm` 返回的 indices | **吐 index 的 tensor** | **1-based** | C++ → Lua `+1` |
| `flatten.start_axis`/`stop_axis` | 闭区间轴 | 1-based | `{i,j}` → `[i-1, j)` |
| `median.axis` / `mode.axis` / `roll.shifts` 的下标 | 维度索引 | 1-based | 同上 |
| **`tril.k` / `triu.k`** | **偏移量,不是索引** | — | **不标 1-based**(§2.2 例外,实测确认 `creation.py:29` 直接透传) |
| `rot90.axes` | 维度索引(内部传给 flip/transpose) | 1-based | Lua → 0-based 再传底层 |
| `nn.functional.pool2d.padding` | **空间偏移,不是索引** | — | **不标 1-based**(§2.2 例外) |

**⚠️ 本模块没有 index 参数的成员**(显式列出,证明"看过"):`zeros/ones/full/
empty/eye/linspace/arange(轴无关)/add/subtract/multiply/divide/exp/log/sqrt/
square/abs/sin/cos/tan(一元逐元素)` 等 —— 均无维度或张量索引参数。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.zeros(2, 3)` | **`paddle.zeros{2, 3}`** | 表调用省括号(`api/README.md` §2.1.3);`size_args` 装饰器不移植(D35) |
| 2 | `paddle.to_tensor(x, device=...)` | **`paddle.to_tensor{data, place=...}`** | 参数名用 Paddle 原生(`place`/`stop_gradient`,C12) |
| 3 | `paddle.from_numpy(ndarray)` | **`paddle.from_insight{array}`** | numpy→Insight 特例(D26),唯一命名例外 |
| 4 | `paddle.concat(tensors=..., dim=...)` | **`paddle.concat{x_list, axis}`** | 别名 `tensors/dim/input` 不移植(D35) |
| 5 | 索引从 0 | **从 1**(axis / index 双轨) | D-R6;§3 表 |
| 6 | `paddle.sum(x, axis=1)` | **`paddle.sum{x, axis=1}`** | 一张参数表(`api/README.md` §2.1.5) |
| 7 | `paddle.split(...)` 返回 list | 返回 **`pl.List`** | D-R21(`api/README.md` §2.3) |
| 8 | `paddle.var/std` CPU 走 C++ 算子 | CPU 走 **Lua 组合**(mean+pow+sum+除) | 上游 CPU 就是纯 Python(`stat.py:222-280`);CUDA 走绑定 |
| 9 | `paddle.logaddexp` / `rot90` / `nan_to_num` / `rank` | **Lua 重写组合** | 上游无独立 C++ 算子,是纯 Python 组合 |
| 10 | `paddle.argsort` / `topk` / `mode` 返回 namedtuple | Lua 返回 **`values, indices` 两个值** | Lua 没有 namedtuple;indices 已 1-based |
| 11 | `paddle.relu(x, inplace=True)` | **`paddle.relu{x, inplace=true}`** | inplace 布尔切 `_ops.relu_` |
| 12 | `paddle.nn.functional.*` 与顶层同名 | 同一绑定,两个入口 | 命名铁律:名字照搬,不合并 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

-- 创建与基础算子(全部表调用)
local x = paddle.zeros{2, 3}                 -- 2x3 float32
local y = paddle.full{2, 3, 1.5}             -- full{{2,3},1.5} 等价
local z = paddle.add{x, y, alpha=2}

-- 归约(axis 1-based)
local s = paddle.sum{x, axis=1}              -- 按第 1 维求和
local m = paddle.mean{x, axis={1,2}, keepdim=true}

-- 索引(吃 index 的 tensor:1-based,边界 -1)
local idx = paddle.to_tensor{1, 3}           -- 取第 1、3 行
local rows = paddle.gather{x, idx, axis=1}

-- 双返回值(argsort/topk/mode)
local vals, idxs = paddle.topk{x, k=2, axis=-1}   -- idxs 是 1-based

-- view 语义(共享内存)
local v = paddle.reshape{x, {3, 2}}
v:fill_(0)                                   -- x 也被改(共享 buffer)
```

---

## 6. 本模块特有的坑

**① `axis` 与 index 的双轨 1-based 最容易漏。** `paddle.gather` 的 `index`
是**张量索引**(边界 -1),而 `axis` 是**维度索引**(边界 -1)。两个都 1-based,
但转换点不同:axis 在生成绑定的参数层转,index 在 C ABI 的 tensor 内容转
(吃 index 的 tensor 整体 -1、吐 index 的 tensor 整体 +1)。`index_semantics.lua`
标注表必须逐算子核对,生成器遇未标注即报错(`conventions.md` §8.2)。

**② `var/std` 的 CPU/CUDA 双路是语义分叉点。** 上游 CUDA 走 `_C_ops.var`,
CPU 走 `mean+pow+sum+除` 组合(`stat.py:222-280`)。Lua 侧若只绑定 C++ 路,
CPU 张量会拿到错误结果(没有那层 Python 组合兜底)。**CPU 组合必须 Lua 重写**,
且 `unbiased`/`correction` 互斥校验(`stat.py:198-201`)照抄。

**③ `logaddexp` / `rot90` 没有 C++ 算子。** 它们的"算子"是纯 Python 组合
(`math.py:906-913` / `manipulation.py:1919-1932`)。生成器遇到这俩名字要
**生成 Lua 组合而非绑定**——这就是 inventory 标注表存在的意义。

**④ `median.mode` 默认值是别名陷阱。** `@param_two_alias_one_default` 在
用户用 `input`/`dim` 别名时把 `mode` 默认从 'avg' 换成 'min'(`stat.py:626`)。
Lua 无别名 → **mode 固定默认 'avg'**,doc 必须写死,别学上游的"按写法变默认值"。

**⑤ `randperm` 返回 0..n-1。** 上游算子吐 0-based 排列(`random.py:2294`)。
Lua 侧若按 C4 直接 +1 会破坏与 Python 的互操作语义——**先与用户确认**:
是"忠实迁移 0-based"(破坏全局 1-based 一致性)还是"边界 +1"。当前建议:doc
注明并保持 1-based(D-R6 一致性优先),待 P5 实现时实测定。

**⑥ `view` 的 dtype 路是 reinterpret,不是 cast。** `_C_ops.view_dtype`
(`manipulation.py:8383-8391`)改 dtype 不重算元素(shape 变化)。用户拿
`view{x, "uint8"}` 当 cast 用会得到错误数值——doc 必须写 `paddle.cast`
才是逐元素转换。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| 未抽查成员(约 250 个 PYTHON 桶) | **逐批补进 §2**;顺序 = `inventory.md` 分桶 + M1 训练需求优先 |
| `paddle.svd` / `lstsq` / `slogdet` | **不在顶层命名空间**(实测 `tensor/__init__.py:91/107/109`,顶层 `__init__.py` 无)——归 `linalg.md` |
| `out` 参数(所有函数) | Q-19,v1 倾向不做,用 `assign` 表达 |
| `requires_grad` / `device` / `pin_memory` 参数 | C12 不引入(用 `stop_gradient` / `place` 表达);`empty` 等函数照此裁剪 |
| `paddle.clamp`(顶层别名) | `clamp = clip`(`__init__.py:1093`)——Paddle 自带别名,是否同步提供 P3 定(倾向:是,零成本) |
| `paddle.compat.*` | 兼容层,SPECIAL 桶跳过 |
| 依赖 Python 运行时(DLPack capsule / static 程序)的成员 | 放放,见 `inventory.md` §3 ③ |
