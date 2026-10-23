# paddle.optimizer

| | |
|---|---|
| 阶段 | **P10**(与 `paddle.nn` P9 衔接;`Layer` 参数 + `autograd.backward` 之后用) |
| 依赖模块 | `paddle`、`paddle.Tensor`、`paddle.autograd`、`paddle.nn.initializer`(部分) |
| 稳定性 | **稳定**(SGD/Momentum/Adam/AdamW——训练必需)/ 实验(其余) |
| 实现文档 | `plan/modules/10-optimizer.md`(待写) |
| 上游来源 | `python/paddle/optimizer/__init__.py`(导出);`optimizer.py`(基类 Optimizer,`state_dict:408`/`clear_grad:1911`/`step:2060`/`minimize:1964`);各优化器子模块(adam.py/adamw.py/sgd.py/momentum.py/rmsprop.py 等);`lr/`(调度器) |

> 📌 优化器是**带状态的对象**:持有参数引用 + 动量等状态。Lua 侧 =
> pl.class 实例(`Optimizer` 子类),**`step()` 在 `loss:backward()` 之后调用**,
> 然后 `clear_grad()`。这是 M1 MNIST 训练循环的固定四行。

---

## 1. 导出清单

`python/paddle/optimizer/__init__.py` 的真实导出:

| 导出 | 来源 | 阶段 |
|---|---|---|
| `paddle.optimizer.Optimizer` | `optimizer.py`(基类) | P10 首批 |
| `paddle.optimizer.SGD` | `sgd.py` | P10 首批(MNIST 用) |
| `paddle.optimizer.Momentum` | `momentum.py` | P10 首批 |
| `paddle.optimizer.Adam` | `adam.py` | P10 首批(MNIST 常用) |
| `paddle.optimizer.AdamW` | `adamw.py` | P10 首批 |
| `paddle.optimizer.Adadelta` / `Adagrad` / `Adamax` / `ASGD` / `Lamb` / `LBFGS` / `Muon` / `NAdam` / `RAdam` / `RMSProp` / `Rprop` | 各子模块 | P10 分批 |
| `paddle.optimizer.lr` | `lr/`(子模块) | P10 首批(LambdaDecay 等) |

---

## 2. 详细签名

### 2.1 基类 `Optimizer`

```lua
paddle.optimizer.Optimizer{
  parameters,           -- 参数列表(pl.List of Tensor)
  learning_rate = 0.1,  -- 或 lr 调度器对象
  weight_decay = nil,
  grad_clip = nil,
}
```

| 方法 | 说明 | 来源 |
|---|---|---|
| `opt:step()` | 一步更新参数(在 `loss:backward()` 后) | `optimizer.py:2060` |
| `opt:clear_grad()` | 清梯度(`set_to_zero=true`) | `optimizer.py:1911` |
| `opt:minimize(loss)` | ⚠️ 静态图遗留接口,动态模式不推荐 | `optimizer.py:1964` |
| `opt:state_dict()` / `opt:set_state_dict(d)` | 优化器状态序列化 | `optimizer.py:408/454` |
| `opt:get_parameters()` | 返回参数列表 | — |

⚠️ **`parameters` 是必填参数**:Python 侧 `SGD(parameters=net.parameters())`;
Lua 侧 `opt = paddle.optimizer.Adam{ parameters = net:parameters() }`。

### 2.2 核心优化器

| 优化器 | Lua 签名(要点) |
|---|---|
| `SGD` | `SGD{ parameters, learning_rate=0.001, weight_decay=nil, ... }` |
| `Momentum` | `Momentum{ parameters, learning_rate, momentum=0.9, use_nesterov=false }` |
| `Adam` | `Adam{ parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=nil }` |
| `AdamW` | `AdamW{ parameters, learning_rate, beta1, beta2, epsilon, weight_decay=0.01, ... }` |

⚠️ `learning_rate` 是**必填位置**(Python 侧也是 `lr` 首参)——Lua 侧
`SGD{ parameters, 0.01 }` 表内位置,或 `learning_rate=0.01` 具名。

### 2.3 lr 调度器(`paddle.optimizer.lr`)

```lua
local lr = paddle.optimizer.lr
local decay = lr.LambdaDecay{ learning_rate = 0.1, lr_lambda = function(epoch) return 0.1 ^ epoch end }
local sgd = paddle.optimizer.SGD{ parameters = net:parameters(), learning_rate = decay }
```

调度器是**可调用对象**(返回当前 lr),随 `step()` 推进 epoch 计数。
`lr/` 子模块导出: `LambdaDecay` / `StepDecay` / `MultiStepDecay` /
`ExponentialDecay` / `CosineAnnealingDecay` / `LinearWarmup` 等(随 P10 分批)。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** `learning_rate`/`weight_decay`/`beta1` 等是
数值;`parameters` 是张量列表(非索引)。显式写"无"以证明看过
(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.optimizer.SGD(parameters, lr=0.1)` | **`SGD{ parameters, learning_rate=0.1 }`** | 一张参数表;`lr` 缩写不搬(C12 用全名 `learning_rate`) |
| 2 | `optimizer.step()` 后 `optimizer.clear_grad()` | 同名为 `:step()` / `:clear_grad()` | 方法照搬 |
| 3 | `net.parameters()` 返回 list | **`net:parameters()` 返回 pl.List** | D-R21 |
| 4 | `learning_rate` 可传调度器对象 | 可传(可调用对象) | 形态一致 |
| 5 | `minimize` 静态图遗留 | 不移植(doc 注明) | 动态模式用 `step` |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

local net = ...                      -- paddle.nn.Layer(见 nn.md)
local opt = paddle.optimizer.Adam{
  parameters = net:parameters(),
  learning_rate = 0.001,
}

for x, y in loader:iter() do
  local loss = criterion(net(x), y)
  loss:backward()                    -- 先反向(autograd.md)
  opt:step()                         -- 再更新
  opt:clear_grad()                   -- 最后清梯度
end
```

带 lr 调度器:

```lua
local lr = paddle.optimizer.lr
local decay = lr.StepDecay{ learning_rate = 0.1, step_size = 10, decay = 0.5 }
local opt = paddle.optimizer.Momentum{ parameters = net:parameters(),
                                       learning_rate = decay, momentum = 0.9 }
```

---

## 6. 本模块特有的坑

**① 状态必须随参数在 `_init` 时建好,`step()` 无参。** Python 侧优化器
`_init` 里按参数建动量/二阶矩状态(`adam.py` 的 `_create_state`)。Lua 侧
同一形态:状态表在 `_init` 建,`step()` 不接收任何参数(除了可选
`set_gradients`)。**用户漏 `clear_grad()` 是最常见 bug** —— 梯度会跨 batch
累加,训练照常收敛但结果错。doc 必须把四行循环写死。

**② `learning_rate` 传调度器时,`step()` 要推进调度器计数。** 调度器
(如 `StepDecay`)的 epoch 计数随 `optimizer:step()` 自动推进(Python 侧
`optimizer.py:2060` 的 `step` 里 `self._global_lr` 更新)。Lua 侧若把
调度器实现成独立对象,必须**由 `step()` 驱动计数**而非用户手动调用——
否则用户忘推进,学习率恒为初始值。

**③ `weight_decay` 语义随优化器不同。** AdamW 的 weight_decay 是
**解耦的**(L2 直接加到参数更新),Adam/SGD 的 weight_decay 是
**耦合 L2 正则**(并入梯度)。两个名字同名不同义(`adamw.py` vs `adam.py`)
——doc 分开写,别混。

**④ `state_dict` / `set_state_dict` 是续训的关键。** T-M0-19 已验证
`SaveTensor/LoadTensor` 格式;优化器状态字典(动量/步数)用同一条链持久化
(`optimizer.py:408/454`)。续训语义:先 `set_state_dict` 再继续 `step()`,
否则学习率/动量状态错乱。

**⑤ `minimize` 是静态图遗留,不移植。** `optimizer.py:1964` 的 `minimize`
是 fluid 静态图接口。动态模式用户**必须** `loss:backward(); opt:step()`
两步。Lua 侧不提供 `minimize`,提供教学式报错
`did you mean: loss:backward(); opt:step()?`。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| 全部 16 个优化器 | P10 首批只做 SGD/Momentum/Adam/AdamW;其余分批 |
| `lr/` 全量调度器 | P10 首批 LambdaDecay/StepDecay;其余分批 |
| `grad_clip` | P10 首批(ClipGradByNorm 等,见 `nn/clip.py`) |
| `state_dict` 与 `paddle.save/load` 接线 | P10(链 T-M0-19 格式) |
| `LBFGS`(需闭包求值) | 延后(与函数式 API 冲突,先不做) |
| `Muon`(新优化器) | 延后(上游实验) |