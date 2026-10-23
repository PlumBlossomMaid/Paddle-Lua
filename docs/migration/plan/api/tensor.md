# paddle.Tensor(类型)

| | |
|---|---|
| 阶段 | P5 |
| 依赖模块 | `paddle`(顶层,构造入口)、`argrule`(方法签名)、Insight7(`to_insight`/`from_insight` 互操作,D26) |
| 稳定性 | **稳定**(核心方法)/ 实验(算子方法随 P3 生成器逐步挂载) |
| 实现文档 | `plan/modules/05-tensor.md`、`plan/modules/02-binding.md`(sol2 usertype) |
| 上游方法来源 | `python/paddle/base/dygraph/tensor_patch_methods.py`(核心方法)、`.../generated_tensor_methods_patch.py`(算子方法挂载)、`.../math_op_patch.py`(转换/属性补丁)、`python/paddle/base/framework.py`(静态侧同名方法)、`python/paddle/base/variable_index.py`(索引解析,D14) |

> 📌 `paddle.Tensor` 在 Lua 侧是一个 **sol2 usertype**(D1)。方法分三层:
> **核心方法**(backward/item/numpy/clone 等,手写绑定)、**算子方法**
> (`x.abs()`/`x.sum()` 等,由 P3 生成器批量挂载,签名与顶层函数共用规则表)、
> **魔法方法**(`__getitem__`/`__setitem__`/`__len__`/`__tostring__` 等,元表)。

---

## 1. 导出清单

`paddle.Tensor` 本身是类型(不是函数),导出后:
`local Tensor = paddle.Tensor`(短名可 local 化,`api/README.md` §2.1)。

### 1.1 核心方法(手写绑定,P5)

| 方法 | 来源 | 说明 |
|---|---|---|
| `x:backward()` | `tensor_patch_methods.py:291` | 反向传播;与顶层 `paddle.autograd.backward` 同底层(`autograd.md`) |
| `x:item()` | `tensor_patch_methods.py:844` | 取元素为 Lua 标量(0-d 或带坐标) |
| `x:numpy()` → Lua `x:to_insight()` | `framework.py:1981` | **命名特例**:numpy→Insight(D26);返回 `insight.Array` |
| `x:clone()` | `framework.py:2645` / `tensor_patch_methods` | 深拷贝 |
| `x:detach()` | `framework.py:1930` | 断开 autograd(共享内存) |
| `x:clear_grad()` | `tensor_patch_methods.py:838` | 清梯度 |
| `x:to(place/dtype)` | `tensor_patch_methods.py:759` | 换 place/换 dtype |
| `x:cpu()` | `tensor_patch_methods.py:1111` | 拷贝到 CPU |
| `x:register_hook(fn)` | `framework.py:2148` | 反向钩子 |
| `x:element_size()` | `framework.py:2306` | 单元素字节数 |
| `x:tolist()` | 顶层同名 | Lua 返回嵌套表(D26 与 Insight 互转) |
| `x:fill_(v)` / `x:copy_(src)` | 算子 in-place 族 | 随生成器 |

### 1.2 属性(getter,绑定层直出)

| 属性 | 来源 | Lua 形式 |
|---|---|---|
| `x.shape` | `framework.py:2483` | `x:shape()` 方法(Paddle 顶层同名函数也是方法来源);返回 `pl.List`(1-based 尺寸,§2.2) |
| `x.ndim` | 属性 | `x:ndim()`;number |
| `x.dtype` | `framework.py:2507` | `x:dtype()`;`paddle.dtype` 值 |
| `x.place` | 属性 | `x:place()`;`paddle.Place` 值 |
| `x.stop_gradient` | `framework.py:2339/2372` | `x:stop_gradient()` getter + `x:set_stop_gradient(v)` setter(不是可写属性,见 §6) |
| `x.grad` | 属性 | `x:grad()`;无梯度时 nil |
| `x.name` | `framework.py:2435/2479` | `x:name()` / `x:set_name()`(调试用) |
| `x.T` | `framework.py:2589` | `x:T()`;转置视图 |
| `x.persistable` / `x.is_parameter` | `framework.py:2376/2408` | 静态侧属性,动态模式延后 |

### 1.3 魔法方法(元表)

| Lua 元方法 | 对应 Python | 来源 |
|---|---|---|
| `__index`(`x[i]`) | `__getitem__` | `tensor_patch_methods.py:1045` → `_getitem_dygraph` |
| `__newindex`(`x[i] = v`) | `__setitem__` | `:1052` → `_setitem_dygraph` |
| `__len`(5.2+)/ 显式 `x:len()` | `__len__` | 见 §6 ⚠️ 5.1 限制 |
| `__tostring` | `__str__` | `:927` → `tensor_to_string` |
| `__add`/`__sub`/`__mul`/`__div`/`__unm` 等 | `+ - * / -x` | 算术运算符 → 对应 `_C_ops`(sol2 元方法表,随生成器) |
| `__eq`(`x == y`) | `==`(逐元素) | ⚠️ Python `==` 是逐元素 Tensor,不是布尔(§6) |

### 1.4 算子方法(生成器批量挂载,P3)

上游 `generated_tensor_methods_patch.py` 把约 200 个 `_C_ops` 算子挂成方法
(`_abs` → `_C_ops.abs`、`_argmax` → `_C_ops.argmax`…`:1-80` 即全部形态)。
Lua 侧同一批算子**同时**出现在:
- 顶层函数 `paddle.abs{x}`(见 `top.md` §2)
- Tensor 方法 `x:abs()` / `x:argmax{axis}`

**方法签名与顶层函数共用同一张 `argrule` 规则表**(只差接收者 `self` 与去掉 `x` 参数)。
例如 `paddle.sum{x, axis=1}` ↔ `x:sum{axis=1}`。

---

## 2. 详细签名

### 2.1 索引(`__index` / `__newindex`)

```lua
x[i]              -- 标量索引:int(1-based)或 bool(掩码)
x[{i, j}]         -- 多维:闭区间?不——i,j 是各自维度的索引
x[{1, 3}]         -- 取第 1 维位置 1、第 2 维位置 3 的元素(1-based)
x[{{1,2}, {3}}]   -- 复合:{1,2} 是子区间(闭区间),{3} 是单点
x["1:3, :"]       -- ★ 字符串索引(D14,见 §2.2)
x[{1, 3}] = v     -- 赋值(__newindex)
```

**索引语义标注(本模块最关键,`conventions.md` §8.2)**:

| 索引形式 | 语义 | 基准 |
|---|---|---|
| 整数索引 `x[1]` | 位置索引 | **1-based** → C++ `-1` |
| 整数列表 `x[{1,2}]` | **闭区间**切片(`1..2`) | Lua `{i,j}` → C++ `[i-1, j)` |
| 布尔掩码 `x[{true,false,...}]` | 逐元素挑选 | 不转索引 |
| 省略号 `x[{"...", 2}]` | 扩展为完整切片(对应 Python `Ellipsis`) | — |
| 字符串 `x["1:3, :"]` | 见 §2.2 | — |

### 2.2 字符串索引(D14:移植 Insight7 的 `lua_spec_to_cpp`)

```lua
x["1:3, :"]        -- 第 1..3 行全列(闭区间,1-based)
x["::-1"]          -- 逆序(T-M0-14 已验证走 strided_slice,view 语义)
x[", 2"]           -- 所有行第 2 列
x["..., 2"]        -- 省略号展开
```

上游解析:`variable_index.py:26` `replace_ellipsis`、`:63` `replace_ndarray_and_range`、
`:75` `replace_none` —— 字符串规格先拆成切片对象,再交给 `_getitem_dygraph`。
Lua 侧:字符串解析在 Lua 层做(`lua_spec_to_cpp`,Insight7 已有实现),产出
C++ 可吃的索引元数据,再调同一个 `_getitem_dygraph` 绑定。

**这是 D14 决策的直接落点**:Insight7 的 `lua_spec_to_cpp` 是现成可移植实现,
本模块直接抄它的语义(1-based 闭区间),不用重写。

### 2.3 `item`

```lua
x:item()            -- 仅 0-d/单元素
x:item(2)           -- 拍平后第 2 个(1-based)
x:item(1, 3)        -- 多维坐标(1-based)
```

上游 `tensor_patch_methods.py:844-876`:坐标转 offset 后取标量。
Lua 侧 `x:item{...}` 坐标 1-based → C++ 0-based 转 offset。

### 2.4 `to_insight`(numpy 特例)

```lua
x:to_insight()      -- Tensor → insight.Array(DLPack 桥,Insight7 DLPack 已就绪)
paddle.from_insight{array}  -- insight.Array → Tensor
```

对应上游 `x.numpy()`(`framework.py:1981`,返回 numpy array)。
命名铁律特例(D26):numpy → Insight。

---

## 3. index 语义参数

> **本模块是 index 语义标注表的核心消费者。** 所有吃/吐 index 的方法与
> 索引形式都在这里标注,生成器读取 `tools/gen/index_semantics.lua` 时必须
> 命中这些条目,未命中即报错(§8.2)。

| 位置 | 语义 | 基准 | 转换 |
|---|---|---|---|
| `x[i]` 整数索引 | 位置 | **1-based** | Lua `n` → C++ `n-1` |
| `x[{i,j}]` 整数列表 | **闭区间** | 1-based | `{i,j}` → `[i-1, j)` |
| `x[{i}]` 单元素列表 | 点索引 | 1-based | `n` → `n-1` |
| `x["a:b"]` 字符串 | 闭区间 | 1-based | 同上 |
| `item(i)` / `item(i,j)` 坐标 | 位置 | 1-based | 逐维 `-1` |
| 算子方法的 `axis`/`dim` | 维度 | 1-based | 同 top.md §3 |
| 算子方法吃 index 的 tensor 参数 | 张量索引 | 1-based | 整体 `-1` |
| 算子方法吐 index 的 tensor 返回值 | 张量索引 | 1-based | 整体 `+1` |
| `x.T` / `transpose` 的 perm | 维度排列 | 1-based | 逐元素 `-1` |
| 布尔掩码 `x[{true,...}]` | 掩码 | —(非索引) | 不转换 |

**没有 index 参数的成员**(显式列出):`backward`/`detach`/`clone`/`clear_grad`/
`cpu`/`element_size`/`numpy`/`name` 等 —— 无维度或张量索引。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `x[1:3, :]` | **`x["1:3, :"]`** | Lua 没有切片语法;字符串索引是 D14 定案(`lua_spec_to_cpp`) |
| 2 | 索引从 0 | **从 1** | D-R6;§3 表 |
| 3 | `x.shape`(属性) | **`x:shape()`** | sol2 方法统一;⚠️ 返回 `pl.List`,不是裸表(§2.3 边界) |
| 4 | `x.stop_gradient = False`(可写属性) | **`x:set_stop_gradient(false)`** | sol2 usertype 没有可写属性;getter/setter 成对 |
| 5 | `x == y` 返回 Tensor | **`x:equal(y)` 或 `paddle.equal{x,y}`** | Lua `==` 是同一性比较(元表 `__eq` 若绑定逐元素会违反 Lua 语义)——**不绑 `__eq` 为逐元素**,用显式方法 |
| 6 | `len(x)` | **`x:len()`**(5.1 无 `#x`) | 5.1 的 `__len` 对表无效(C3);5.2+ 可补 `__len` 元方法 |
| 7 | `x.numpy()` | **`x:to_insight()`** | numpy→Insight 特例(D26) |
| 8 | 算子方法 `x.sum(axis=1)` | **`x:sum{axis=1}`** | 一张参数表;签名与顶层函数共用规则表 |
| 9 | `x.item(0, 2)` | **`x:item(1, 3)`** | 坐标 1-based |
| 10 | 布尔掩码 `x[[True, False]]` | `x[{true, false}]` | 掩码语义不转索引 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"
local x = paddle.to_tensor{{1, 2, 3}, {4, 5, 6}}   -- 2x3

-- 索引(全部 1-based)
print(x[1, 2])            -- 2(第 1 行第 2 列)
print(x["1:2, :"])        -- 前两行(闭区间)
local rev = x["::-1"]     -- 逆序 view(T-M0-14)

-- 属性与方法
print(x:shape())          -- pl.List{2, 3}
print(x:ndim())           -- 2
x:set_stop_gradient(false)

-- 算子方法(与顶层函数同签名)
local s = x:sum{axis=1}   -- 等价 paddle.sum{x, axis=1}
local a = x:argmax{axis=1}

-- 标量
local y = paddle.to_tensor(3.5)
print(y:item())           -- 3.5(Lua number)

-- 与 Insight 互操作
local arr = x:to_insight()       -- insight.Array
local back = paddle.from_insight(arr)
```

---

## 6. 本模块特有的坑

**① `==` 不绑逐元素。** Python 的 `x == y` 返回 Tensor,但 Lua 的 `==`
在元表层面语义是"同一性",绑逐元素会让 `if x == y then` 这类代码在 5.1 上
行为怪异(任何非零张量都"为真")。**`__eq` 不绑定**,用户显式用
`paddle.equal{x, y}`。

**② 5.1 的 `#x` 拿不到形状。** `__len` 元方法对 table 类型生效,对 usertype
在 5.1 上无效(C3)。`#x` 会得到 nil 或运行时错误 —— 文档和示例一律
`x:shape()` / `x:len()`。5.2+ 可补 `__len`(返回元素总数 `numel`,不是第一维)。

**③ `stop_gradient` 是 getter/setter 对。** sol2 没有可写属性;Lua 侧
`x:set_stop_gradient(v)`。⚠️ 与 `requires_grad`(PyTorch 名)是**取反且默认值也反**
的关系(C12),Lua 侧只有 Paddle 名。

**④ 字符串索引解析在 Lua 层,不在 C++。** `x["1:3, :"]` 的解析是
`lua_spec_to_cpp`(Insight7 实现),产出索引元数据再调 C++。这条边界写死:
**C ABI 层只收解析后的切片元数据**,不收字符串 —— 否则 P3 生成器要为一个
API 写两套解析。

**⑤ `item(i)` 是 offset 语义,但坐标是 1-based。** 上游 `item(*args)` 先把
坐标转 offset(`tensor_patch_methods.py:844`)。Lua 侧 `x:item{1, 3}` 的坐标
**逐个 -1** 转 C++ offset。⚠️ 拍平索引 `x:item(2)` 也是 1-based(第 2 个元素,
不是 Python 的第 2 个 = C++ 的 1 号)。

**⑥ 算子方法数量 = 顶层函数数量(约 200+)。** 上游 `generated_tensor_methods_patch.py`
每个算子一个 `_C_ops` 转发。Lua 侧不手写 —— P3 生成器把顶层函数的规则表
**镜像成方法**(`self` + 去掉 `x` 参数)。任何"手写一个算子方法"的冲动都违反 C8。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| 算子方法全量(约 200+) | **P3 生成器批量挂载**;P5 先挂训练必需(minimum: add/sub/mul/div/abs/exp/log/sum/mean/max/min/reshape/transpose/argmax/backward) |
| `__len` 元方法(5.2+) | P5 顺手做;5.1 用 `x:len()` |
| `register_hook` | P6 autograd 一起做(`autograd.md`) |
| `tolist` | P5,依赖 Insight7 互操作(D26) |
| `persistable` / `is_parameter` | 静态侧属性,动态模式延后 |
| `x.data`(原始 buffer 访问) | 延后,需 DLPack 协议细化后定 |
| `grad` 的自动释放策略 | P6 GC 机制(14-gc.md)一起定 |
