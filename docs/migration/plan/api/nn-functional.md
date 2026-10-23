# paddle.nn.functional

| | |
|---|---|
| 阶段 | **P9**(与 `paddle.nn` 同批;顶层同名函数如 `paddle.relu` 先于本模块 P5 绑定) |
| 依赖模块 | `paddle`(顶层)、`paddle.Tensor`、`_ops`(生成算子) |
| 稳定性 | **稳定**(激活/丢弃/池化核心)/ 实验(其余按需) |
| 实现文档 | `plan/modules/03-codegen.md`(算子生成)、`plan/modules/09-nn.md` |
| 上游来源 | `python/paddle/nn/functional/__init__.py`(导出清单);`activation.py`(激活);`common.py`(dropout/bilinear 等);`pooling.py`(池化);`norm.py`(归一化);`loss.py`(损失函数) |

> 📌 本模块**多数是对 `_ops` 的一行转发**(api/README.md 模块清单)。
> 与 `paddle.nn`(层类)区分:functional 是函数式调用,不持有参数。
> **顶层同名函数(如 `paddle.relu`/`paddle.sigmoid`)与 `nn.functional.*`
> 是同一绑定**(见 `top.md` §2.7)——Lua 侧名字照搬,不合并。

---

## 1. 导出清单

`nn/functional/__init__.py` 的真实导出:

### 1.1 激活(37 个,`:18-55`)

`celu` `elu` `elu_` `gelu` `glu` `gumbel_softmax` `hardshrink`
`hardsigmoid` `hardswish` `hardtanh` `hardtanh_` `leaky_relu` `leaky_relu_`
`log_sigmoid` `log_softmax` `maxout` `mish` `prelu` `relu` `relu6` `relu_`
`rrelu` `selu` `sigmoid` `silu` `softmax` `softmax_` `softplus`
`softshrink` `softsign` `swiglu` `swish` `tanh` `tanh_` `tanhshrink`
`thresholded_relu` `thresholded_relu_`

⚠️ `_` 后缀 = in-place 变体(同一算子,`_ops.x_`)。Lua 侧随生成器一起出,
**不手写**。

### 1.2 common(丢弃/其他,`:57-`)

`alpha_dropout` `bilinear` `class_center_sample` `cosine_similarity`
`dropout` `dropout1d` `dropout2d` `dropout3d` `embedding` `flatten`
`interpolate` `linear` `pad` `unfold` `fold` `upsample` 等

### 1.3 池化(18 个,`pooling.py`)

`avg_pool1d/2d/3d` `max_pool1d/2d/3d` + `adaptive_*` 6 个 +
`fractional_max_pool*` 2 个 + `lp_pool*` 2 个 + `max_unpool*` 3 个

### 1.4 归一化(norm.py)

`batch_norm` `instance_norm` `layer_norm` `group_norm` `normalize`
`spectral_norm` 等

### 1.5 损失函数(loss.py)

`binary_cross_entropy` `binary_cross_entropy_with_logits` `cross_entropy`
`nll_loss` `kl_div` `mse_loss` `l1_loss` `smooth_l1_loss` `margin_ranking_loss`
`triplet_margin_loss` `ctc_loss` 等(24 个,与 `nn.md` 损失类对应)

---

## 2. 详细签名(核心,已抽查的)

### 2.1 激活(复用 `top.md` §2.7 的标注)

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.nn.functional.relu{x, inplace}` | 直接绑定 + Lua 适配 | `_C_ops.relu_/relu` 双路(`activation.py:836`);顶层 `paddle.relu` 同函数 |
| `paddle.nn.functional.leaky_relu{x, negative_slope, inplace}` | 直接绑定 + Lua 适配 | `_C_ops.leaky_relu_/leaky_relu`(`activation.py:516-518`) |
| `paddle.nn.functional.silu{x, inplace}` | 直接绑定 | `_C_ops.silu`(`activation.py:1169`) |
| `paddle.nn.functional.sigmoid{x}` / `tanh{x}` | 直接绑定 | 纯 re-export(`math.py:104/119`) |
| `paddle.nn.functional.softmax{x, axis, dtype}` | 直接绑定 | ⚠️ 命名冲突见 `top.md` 软标注(P3 拍板) |
| `paddle.nn.functional.gelu{x, approximate}` | 直接绑定 | `_C_ops.gelu`(DIRECT 桶) |

### 2.2 丢弃

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.nn.functional.dropout{x, p, mode, training}` | 直接绑定 + Lua 适配 | `_C_ops.dropout(x, p, is_test, mode, seed, fix_seed, out=out)`(`common.py:1425`);⚠️ `p` 是**丢弃概率**;`training` 由 `Layer:train/eval` 状态驱动(Lua 侧 `paddle.is_grad_enabled` 不对,用层的 `_training` 字段) |

### 2.3 池化

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.nn.functional.avg_pool2d{x, kernel_size, stride, padding, ...}` | 直接绑定 + Lua 适配 | `_C_ops.pool2d(...)`(`pooling.py:398`);三态参数解析 Lua 重写 |
| `paddle.nn.functional.max_pool2d{...}` | 直接绑定 + Lua 适配 | 同上(`pooling.py:1294`) |
| `paddle.nn.functional.adaptive_avg_pool2d{x, output_size}` | 直接绑定 | `_C_ops.pool2d(adaptive=true)` 分支 |

### 2.4 归一化

| Lua 签名 | 去向 | 备注 |
|---|---|---|
| `paddle.nn.functional.layer_norm{x, normalized_shape, weight, bias, epsilon}` | 直接绑定 | `_C_ops.layer_norm`;⚠️ `normalized_shape` 是轴范围,**1-based**(§3) |
| `paddle.nn.functional.batch_norm{x, running_mean, running_var, weight, bias, training, momentum, epsilon}` | 直接绑定 | `_C_ops.batch_norm`;`training` 由层状态驱动 |

---

## 3. index 语义参数

| 参数 | 语义 | 基准 |
|---|---|---|
| `layer_norm.normalized_shape` | 归一化的**轴范围** | **1-based**(如 `{1, 2}` 归一化前两维) |
| `embedding` 的 input id tensor | 词表索引 | **1-based**(与 `nn.Embedding` 对齐,C4) |
| 全部 `axis` / `dim` 参数 | 维度索引 | **1-based** |
| `cross_entropy` 的 label | 类别索引 | **1-based**(C4,与 `nn.CrossEntropyLoss` 对齐) |
| `pool`/`pad` 的 `padding` | **空间偏移,不是索引** | **不标 1-based**(§2.2 例外) |
| `dropout.mode` | 枚举字符串 | 非索引 |

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `F.relu(x, inplace=False)` | **`F.relu{x, inplace=false}`** | 一张参数表 |
| 2 | `F.dropout(x, p=0.5)` | `F.dropout{x, p=0.5}` | `p` 语义照搬(丢弃概率) |
| 3 | 索引从 0 | **从 1**(label/axis/embedding id) | D-R6;C4 |
| 4 | `F.layer_norm(x, [2, 3])` | `F.layer_norm{x, normalized_shape={2, 3}}` | ⚠️ normalized_shape 是 1-based 轴范围 |
| 5 | 与顶层 `paddle.relu` 同名 | **同一绑定两入口** | 命名铁律:名字照搬,不合并 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"
local F = paddle.nn.functional

local x = paddle.randn{2, 4}

-- 激活(函数式)
local y = F.relu{x}
local z = F.gelu{x, approximate=true}

-- dropout(训练时)
local d = F.dropout{z, p=0.5, training=true}

-- 池化
local p = F.max_pool2d{ d, kernel_size=2, stride=2 }

-- 归一化(轴范围 1-based)
local n = F.layer_norm{p, normalized_shape={1, 2}}
```

---

## 6. 本模块特有的坑

**① `dropout.training` 不能简单用 `is_grad_enabled` 对位。** Python 侧
`training` 参数由 `Layer` 的 `train/eval` 状态 + `paddle.is_grad_enabled()`
共同决定(`common.py:1234` 附近)。Lua 侧功能式调用没有层上下文 ——
**调用方显式传 `training`**,doc 写明"与 `paddle.is_grad_enabled` 无关,
由你传"。

**② `normalized_shape` 是轴范围不是形状后缀。** 用户容易照 PyTorch 的
`normalized_shape=[4]` 理解(倒数维尺寸)。Paddle 的语义是轴闭区间
(`layer_norm.py` docstring)。Lua 侧 1-based 轴范围 + doc 对照示例,避免
与 `torch` 迁移用户混淆。

**③ `_` in-place 变体随生成器,不手写。** `relu_`/`tanh_`/`softmax_` 等
(C8:生成器范围)。任何"手写一个 in-place 变体"的冲动都违反 C8。

**④ 顶层同名函数先绑定(P5),本模块 P9 复用。** `paddle.relu` 在
`top.md` §2.7 已定契约;`nn.functional.relu` 是**同一绑定第二入口**。
实现上 `paddle.nn.functional` 表挂 `__index` 惰性装配(`api/README.md`
§2.5),不重复注册。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| 全部 37 个激活 + 24 个损失的详细签名 | P9 分批(导出清单已列全,签名随批补) |
| `interpolate` / `upsample` | P9(依赖 resize 算子族) |
| `spectral_norm` | P9(需要 hook 机制) |
| `gumbel_softmax` / `rrelu` / `maxout` | 实验,随 `nn.md` 批次 |
| `linear`(函数式) | P9 首批(与 `nn.Linear` 同底层) |