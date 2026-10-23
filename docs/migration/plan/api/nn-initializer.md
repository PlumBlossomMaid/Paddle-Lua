# paddle.nn.initializer

| | |
|---|---|
| 阶段 | **P9**(随 `paddle.nn` 同批;`Layer` 的 `weight_attr`/`bias_attr` 用它) |
| 依赖模块 | `paddle`、`paddle.Tensor`、随机数生成器(`paddle.seed`,P6) |
| 稳定性 | **稳定**(核心:Normal/Uniform/Xavier/Kaiming/Constant/Assign) |
| 实现文档 | `plan/modules/09-nn.md`、`plan/modules/03-codegen.md` |
| 上游来源 | `python/paddle/nn/initializer/__init__.py`(导出清单);`initializer.py`(基类 + calculate_gain);`normal.py`/`uniform.py`/`xavier.py`/`kaiming.py`/`constant.py`/`assign.py`/`orthogonal.py`/`dirac.py`/`bilinear.py` |

> 📌 初始化器是**可调用对象**(`layer.weight_attr = initializer`),在参数
> 创建时被调用。Lua 侧 = 带 `__call` 元方法的表/对象。**随机数必须走
> Paddle 的生成器**(api/README.md 模块清单)——不自己造 RNG。

---

## 1. 导出清单

`nn/initializer/__init__.py` 的真实导出:

| 导出 | 来源 | 说明 |
|---|---|---|
| `paddle.nn.initializer.Initializer` | `initializer.py`(基类) | 抽象基类 |
| `paddle.nn.initializer.calculate_gain` | `initializer.py:154` | 激活函数增益表 |
| `paddle.nn.initializer.Normal` | `normal.py:138` | 正态 `(mean, std)` |
| `paddle.nn.initializer.NormalInitializer` | `normal.py:33` | 旧名 |
| `paddle.nn.initializer.TruncatedNormal` | `normal.py:346` | 截断正态 |
| `paddle.nn.initializer.Uniform` | `uniform.py` | 均匀 `(low, high)` |
| `paddle.nn.initializer.XavierNormal` / `XavierUniform` | `xavier.py:49/50` | Glorot 初始化 |
| `paddle.nn.initializer.KaimingNormal` / `KaimingUniform` | `kaiming.py:288/343` | He 初始化 |
| `paddle.nn.initializer.MSRAInitializer` | `kaiming.py:38` | Kaiming 旧名 |
| `paddle.nn.initializer.Constant` | `constant.py:117` | 常值 `(value)` |
| `paddle.nn.initializer.ConstantInitializer` | `constant.py:32` | 旧名 |
| `paddle.nn.initializer.Assign` | `assign.py:193` | 用给定值/数组赋值 |
| `paddle.nn.initializer.NumpyArrayInitializer` | `assign.py:41` | 旧名 |
| `paddle.nn.initializer.Orthogonal` | `orthogonal.py:28` | 正交 |
| `paddle.nn.initializer.Dirac` | `dirac.py:31` | Dirac 分布 |
| `paddle.nn.initializer.Bilinear` | `bilinear.py:33` | 双线性(转置卷积用) |
| `paddle.nn.initializer.set_global_initializer` | `base/initializer.py` | 全局默认初始化器(旧机制) |

---

## 2. 详细签名

### 2.1 可调用约定

```lua
local init = paddle.nn.initializer
local layer = paddle.nn.Linear{
  in_features = 2, out_features = 4,
  weight_attr = init.KaimingUniform{ fan_in = 2, fan_out = 4 },  -- ⚠️ 参数名待核
  bias_attr   = init.Constant{ value = 0.0 },
}
```

**Lua 侧形态**:初始化器对象 = 带 `__call` 元方法的 usertype/表,
`weight_attr` 参数收它,参数创建时调用。

### 2.2 核心初始化器签名

| 初始化器 | Lua 签名 | 备注 |
|---|---|---|
| `Normal` | `Normal{ mean=0.0, std=1.0 }` | 正态采样 |
| `Uniform` | `Uniform{ low=-1.0, high=1.0 }` | 均匀采样 |
| `XavierNormal` / `XavierUniform` | `XavierNormal{ fan_in, fan_out, gain=1.0 }` | Glorot;**fan 计算随输入形状** |
| `KaimingNormal` / `KaimingUniform` | `KaimingUniform{ fan_in, fan_out, negative_slope=0.0 }` | He;**随机数必须走 Paddle 生成器** |
| `Constant` | `Constant{ value }` | 常值填充 |
| `Assign` | `Assign{ value }` | value 可为 number / insight.Array |
| `TruncatedNormal` | `TruncatedNormal{ mean=0.0, std=1.0, a=-2.0, b=2.0 }` | 截断正态 |
| `Orthogonal` | `Orthogonal{ gain=1.0 }` | 正交(QR 分解) |

⚠️ `fan_in`/`fan_out` 的参数名与是否显式传参:上游 Python 的 Xavier/Kaiming
是**传 fan 或按 shape 算**两类(旧 `*Initializer` 类显式传 fan,新 `Xavier*`
类按 shape 自动算)。Lua 侧 P9 时按新类实现核对签名(§6 ①)。

### 2.3 `calculate_gain`

```lua
paddle.nn.initializer.calculate_gain{ nonlinearity='relu', param=nil }
-- 'linear'=1, 'sigmoid'=1, 'tanh'=5/3, 'relu'=√2 ...
```

上游 `initializer.py:154`:激活名 → 增益值表。Lua 侧一张查表,Lua 重写。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** fan_in/fan_out 是**维度尺寸**不是索引;
value/mean/std 是数值。显式写"无"以证明看过(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `init.Normal(0.0, 1.0)`(位置) | **`init.Normal{ mean=0.0, std=1.0 }`** | 一张参数表 |
| 2 | `weight_attr=init.KaimingUniform(...)` | **`weight_attr=...`(初始化器对象)** | 可调用对象,形态一致 |
| 3 | 旧名 `NormalInitializer`/`XavierInitializer` | **只保留新名**(Normal/XavierNormal 等) | 旧名是兼容别名(# noqa 注释 `__init__.py:19/38/40`),不搬 |
| 4 | `set_global_initializer`(全局机制) | 延后 | 显式 `weight_attr` 优先;全局默认 P9 末再定 |
| 5 | `Assign` 收 numpy | **收 insight.Array / number** | numpy→Insight 特例(D26) |

---

## 5. 用法示例

```lua
local paddle = require "paddle"
local init = paddle.nn.initializer

local layer = paddle.nn.Linear{
  2, 4,
  weight_attr = init.XavierUniform{},
  bias_attr   = init.Constant{ value = 0.1 },
}

local conv = paddle.nn.Conv2D{
  1, 8, 3,
  weight_attr = init.KaimingUniform{ negative_slope = 0.0 },
}

-- calculate_gain 查表
local g = init.calculate_gain{ nonlinearity = 'relu' }   -- √2
```

---

## 6. 本模块特有的坑

**① 新类与旧类双轨的签名差异要 P9 核清楚。** 上游 `XavierInitializer`
(旧,显式传 `fan_in/fan_out`)与 `XavierNormal`(新,按 shape 自动算)并存
(`xavier.py`)。Python 侧 `nn.Linear` 默认用新类。Lua 侧**只移植新类**,
旧 `*Initializer` 别名不搬(§4 #3)——但签名到底是 `{fan_in, fan_out}` 还是
`{}`(自动算)必须在 P9 读 `xavier.py` 实体后定,本文留待决项。

**② 随机数必须走 Paddle 生成器。** `KaimingUniform` 的采样不能自己
`math.random` —— 与 Python 侧 `paddle.seed` 的可复现性契约挂钩(同 seed
同初始化)。Lua 侧初始化器内部调 `_ops` 的随机算子(如 `_C_ops.uniform`),
不引入第二套 RNG(api/README.md 模块清单红字)。

**③ `Assign` 收 Insight 数组是 D26 特例。** `Assign{value}` 的 value 是
`insight.Array` 或 number(对应 numpy)。内部走 DLPack 桥
(`from_insight`),与 `paddle.from_insight` 同链。

**④ 初始化器在参数创建时调用一次,不是惰性。** `weight_attr` 在
`Layer:_init` 时立即执行(创建 parameter tensor)。Lua 侧要保证**同一层
重复构建时初始化器对象可复用**(无内部状态残留)——初始化器对象应是
无状态的(参数全在构造时定死)。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `Orthogonal` / `Dirac` / `Bilinear` | P9 分批(非训练必需) |
| `set_global_initializer`(全局默认) | P9 末(显式 `weight_attr` 优先) |
| 新类签名的 fan 自动计算 | P9 读 `xavier.py` 实体后定 |
| 初始化器对象是否为 usertype vs 纯表 | P9 绑定层定(`01-c-abi.md` 约定) |