# paddle.nn

| | |
|---|---|
| 阶段 | **P9**(Layer 基类与核心层优先——M1 MNIST 需要 Linear/Conv2D/ReLU/Dropout/CrossEntropyLoss 等;其余层 P9 内分批) |
| 依赖模块 | `paddle`(Tensor)、`pl.class`(类系统,C11)、`paddle.nn.functional`(forward 内部调用) |
| 稳定性 | **稳定**(核心层:Linear/Embedding/Conv/Dropout/ReLU/BN/池化/损失)/ 实验(transformer/rnn 等) |
| 实现文档 | `plan/modules/09-nn.md`(自动注册、FIELDS、_create) |
| 上游来源 | `python/paddle/nn/__init__.py`(导出清单);`layer/layers.py`(Layer 基类);`layer/container.py`(容器);`layer/conv.py`;`layer/common.py`(Linear/Embedding);`layer/loss.py`(损失);`layer/norm.py`;`layer/pooling.py`;`layer/rnn.py`;`layer/transformer.py`;`layer/vision.py` |

> 📌 本文聚焦 **Layer 基类 + 容器 + 训练必需核心层** 的契约(P9 第一批)。
> 其余近 40 个层(池化/BN/RNN/transformer 等)导出清单全列,详细签名随
> P9 分批补充。**类的实现形态都是「pl.class 子类 + forward 方法」,
> 注册机制见 `09-nn.md` §3.2(D23:自动注册,实例 raw 表保持空 + 私有 FIELDS 键)**。

---

## 1. 导出清单

### 1.1 基类与容器

| 导出 | 来源 | 阶段 |
|---|---|---|
| `paddle.nn.Layer` | `layer/layers.py`(基类) | P9 首批 |
| `paddle.nn.Sequential` | `layer/container.py:750` | P9 首批 |
| `paddle.nn.LayerList` | `container.py:566`(⚠️ 继承 Layer,D27) | P9 首批 |
| `paddle.nn.LayerDict` / `ParameterList` / `ParameterDict` | `container.py` | P9 |
| `paddle.nn.Module` / `ModuleList` / `ModuleDict` | `modules/`(**新框架,⚠️ 见 §6 ⑤**) | 延后 |

### 1.2 核心层(P9 首批,训练必需)

| 导出 | 来源 | 说明 |
|---|---|---|
| `paddle.nn.Linear` | `layer/common.py:122` | 全连接 |
| `paddle.nn.Embedding` | `common.py:2291` | 词嵌入(索引语义!§3) |
| `paddle.nn.Conv1D` / `Conv2D` / `Conv3D` / Transpose 变体 | `layer/conv.py:253/712` | 卷积族 |
| `paddle.nn.ReLU` / `ReLU6` / `LeakyReLU` / `Sigmoid` / `Tanh` / `Softmax` 等 | `layer/activation.py` | 激活层(29 个全列于 `nn-functional.md`) |
| `paddle.nn.Dropout` / `Dropout2D` / `Dropout3D` | `layer/common.py` | 丢弃层 |
| `paddle.nn.Flatten` | `common.py` | 拍平 |
| `paddle.nn.BatchNorm1D`/`2D`/`3D`、`LayerNorm`、`GroupNorm` | `layer/norm.py` | 归一化族(13 个) |
| `paddle.nn.MaxPool2D` / `AvgPool2D` + 全部池化 | `layer/pooling.py` | 池化族(18 个) |

### 1.3 损失(24 个,`layer/loss.py`)

`CrossEntropyLoss`(`:208`)、`MSELoss`(`:684`)、`BCEWithLogitsLoss`(`:83`)、
`L1Loss`、`SmoothL1Loss`、`NLLLoss`、`KLDivLoss`、`BCELoss`、
`CosineEmbeddingLoss`、`CTCLoss`、`GaussianNLLLoss`、`HingeEmbeddingLoss`、
`HSigmoidLoss`、`MarginRankingLoss`、`MultiLabelMarginLoss`、
`MultiLabelSoftMarginLoss`、`MultiMarginLoss`、`PoissonNLLLoss`、`RNNTLoss`、
`SoftMarginLoss`、`TripletMarginLoss`、`TripletMarginWithDistanceLoss`、
`AdaptiveLogSoftmaxWithLoss`。

### 1.4 RNN / transformer / vision(P9 后期)

`GRU`/`LSTM`/`RNN`/`BiRNN`(+ Cell 族,`layer/rnn.py`);
`MultiHeadAttention`/`Transformer`(+ Encoder/Decoder 族,`layer/transformer.py`);
`ChannelShuffle`/`PixelShuffle`/`PixelUnshuffle`(`layer/vision.py`)。

---

## 2. 详细签名

### 2.1 `Layer` 基类(pl.class 子类)

```lua
local class = require "pl.class"
local MyLayer = class(paddle.nn.Layer)
MyLayer._name = "MyLayer"

function MyLayer:_init(in_features, out_features)
  self:super()                      -- pl.class 链 → Layer:_init(实例 raw 表保持空,D23)
  self.linear = paddle.nn.Linear(in_features, out_features)
end

function MyLayer:forward(x)
  return self.linear(x)             -- 子类必须实现
end
```

| 方法 | 说明 | 来源 |
|---|---|---|
| `forward(x)` | 前向,子类实现 | `layers.py`(抽象) |
| `train(mode)` / `eval()` | 切换训练/评估(影响 Dropout/BN) | `layers.py:756/819` |
| `apply(fn)` | 递归应用函数到所有子层 | `layers.py:872` |
| `full_name()` | 层名(调试) | `layers.py:929` |
| `register_forward_pre_hook(fn)` / `post_hook` | 前向钩子 | `layers.py:996/1083` |
| `state_dict()` / `set_state_dict(d)` | 参数序列化(P9,T-M0-19 格式) | `layers.py` |
| `parameters()` | 参数迭代器 | `layers.py` |
| `to(place)` / `cpu()` | 设备迁移 | `layers.py` |

⚠️ **`parameters()` / `state_dict()` 的返回走 `pl.List`**、`state_dict` 键为
`"layer_name.weight"` 点号路径(对应 Python 的 torch 风格命名,Paddle 用
`.` 还是 `/` 待 `09-nn.md` 核,见 §6 ④)。

### 2.2 `Sequential` / `LayerList`

```lua
-- Sequential:按序执行,自动 forward
local net = paddle.nn.Sequential{
  paddle.nn.Linear{2, 4},
  paddle.nn.ReLU{},
  paddle.nn.Linear{4, 2},
}
local out = net(x)                     -- forward 依序

-- LayerList:索引容器(⚠️ D27 继承 Layer)
local ml = paddle.nn.LayerList{ layer1, layer2 }
ml:len()                               -- ★ 不用 #ml(5.1 上 ipairs 跑空)
for i, l in ml:iter() do ... end
```

### 2.3 `Linear`

```lua
paddle.nn.Linear{ in_features, out_features, weight_attr=nil, bias_attr=nil }
```

`weight_attr`/`bias_attr` 是初始化器或 `False`(无偏置)。forward:
`y = x @ W^T + b`。

### 2.4 `Conv2D`

```lua
paddle.nn.Conv2D{ in_channels, out_channels, kernel_size,
                  stride=1, padding=0, dilation=1, groups=1,
                  padding_mode='zeros', weight_attr=nil, bias_attr=nil }
```

⚠️ `padding` 三态(int/tuple/'same'/'valid'),是**空间偏移不是索引**(§3);

### 2.5 `Dropout`

```lua
paddle.nn.Dropout{ p=0.5, mode='upscale_in_train' }
```

⚠️ `p` 是**丢弃概率**(与 keep_prob 相反);`train/eval` 切换由 `Layer:train()` 控制。

### 2.6 `CrossEntropyLoss`

```lua
paddle.nn.CrossEntropyLoss{ weight=nil, ignore_index=nil, reduction='mean' }
```

⚠️ **`ignore_index` 是索引语义的 label 值**(§3);reduction=`'mean'/'sum'/'none'`。

---

## 3. index 语义参数

| 参数 | 语义 | 基准 |
|---|---|---|
| `Embedding{num_embeddings, embedding_dim}` 的 **输入 id tensor** | 词表索引 | **1-based**(吃 index 的 tensor,C4) |
| `CrossEntropyLoss.ignore_index` | label 值(类别索引) | **1-based**(⚠️ 上游 0-based,对齐 `label` 约定 C4) |
| `LayerList` 的下标 `ml[i]` | 层序号 | **1-based** |
| 全部 `Conv`/`Pool` 的 `padding` / `dilation` / `stride` | **空间偏移,不是索引** | **不标 1-based**(§2.2 例外) |
| `transformer` 的 `num_heads` 等 | 超参,非索引 | 不标 |

⚠️ **Embedding/CrossEntropyLoss 的 index 张量是对齐 C4 的关键**:
`tools/gen/index_semantics.lua` 必须标注,生成器未标注即报错。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `class MyLayer(nn.Layer)` | **`class(paddle.nn.Layer)` + `:super()`** | pl.class(C11);`_create` 装配(D23) |
| 2 | `def forward(self, x)` | `function MyLayer:forward(x)` | pl.class 方法 |
| 3 | `nn.Sequential(l1, l2)` | `nn.Sequential{ l1, l2 }` | 一张参数表 |
| 4 | `len(layerlist)` | **`ml:len()`**(5.1 无 `#`) | D27;ipairs 在 5.1 跑空 |
| 5 | `nn.Linear(2, 4)`(位置) | `nn.Linear{2, 4}`(表内位置) | 表调用一致 |
| 6 | `weight_attr=xavier_uniform_()` | `weight_attr=...`(初始化器接口) | `nn-initializer.md` |
| 7 | `model.to(device)` | `model:to(place)` | place 值类型(C12) |

---

## 5. 用法示例

```lua
local paddle = require "paddle"
local class = require "pl.class"

local Net = class(paddle.nn.Layer)
Net._name = "Net"
function Net:_init()
  self:super()
  self.fc1 = paddle.nn.Linear{784, 128}
  self.relu = paddle.nn.ReLU{}
  self.fc2 = paddle.nn.Linear{128, 10}
end
function Net:forward(x)
  x = self.fc1(x)
  x = self.relu(x)
  return self.fc2(x)
end

local net = Net()
local critic = paddle.nn.CrossEntropyLoss{}

-- 训练循环(M1 验收形态)
for x, y in loader:iter() do
  local out = net(x)
  local loss = critic(out, y)
  loss:backward()
  opt:step(); opt:clear_grad()
end
```

---

## 6. 本模块特有的坑

**① `Layer` 自动注册依赖 `_create` 约定(D23)。** 实例 raw 表**保持空**、
子层注册在私有 `FIELDS` 键下。用 `class.cast` 造实例会绕过 `_create`,
产生没有 FIELDS 的破 Layer —— **`class.cast` 在 paddle 侧禁用**(§9)。

**② `LayerList` 继承 `Layer`(D27)。** `is_a(nn.Layer)` 必须为真;副作用是
5.1 上 `ipairs(ml)` 跑空 → **一律用 `ml:iter()` / `ml:len()`**。与 `io.md`
的 Dataset 同一个坑两次出现。

**③ `Embedding` / `CrossEntropyLoss` 的 label/index 1-based。** 上游
0-based(word2id、类别 id)。Lua 侧在 C ABI 边界对 index 张量整体 -1
(G4 吃 index 的 tensor)。⚠️ **训练脚本里用户写 1 表示"第一个词/第一类"**,
这是与 Python 代码迁移时最容易静默出错的地方——doc 必须写对照。

**④ `state_dict` 键名用点号还是斜杠待核。** Python 侧 `state_dict()` 的键
格式由 `full_name()` 决定(`layers.py:929`),是 `layer.weight` 还是
`layer/weight` 与 `paddle.save` 的变量命名约定相关。P9 核 `save_load` 链路
后定,本文只留待决项(T-M0-19 已验证 SaveTensor 格式,键格式待定)。

**⑤ `Module`/`ModuleList` 是新框架,延后。** `nn/__init__.py:107-110` 从
`modules/` 导入 `Module`/`ModuleList`/`ModuleDict`(与 `Layer` 并存的新类)。
⚠️ Paddle 在往 `Module` 迁移(Layer 是旧壳)。**P9 先用 Layer**(生态兼容),
Module 是否同步支持 P3 拍板——避免两套类系统并存(C11)。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| 全部 29 个激活层详细签名 | P9 首批做 ReLU 族;其余随 `nn-functional.md` |
| 池化/BN 全族详细签名 | P9 分批(导出清单已列) |
| RNN/transformer 族 | **P9 后期**(依赖容器与基础层稳定) |
| `Module`/`ModuleList`(新框架) | P3 拍板是否同步 |
| `state_dict` 键格式 | P9 核 save_load 链路后定 |
| `Parameter` 类 | P9 首批(参数绑定,依赖 `nn-initializer.md`) |