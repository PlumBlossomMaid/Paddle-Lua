# P10 · paddle.optimizer

| | |
|---|---|
| 阶段 | P10 |
| 类别 | 手写 Lua(语义直抄) |
| 开工条件 | P9 完工 |
| 预估 | 2 周 |

---

## 1. 做什么 / 不做什么

**做:** SGD / Momentum / Adam / AdamW / RMSProp + 6 个学习率调度器 + 梯度裁剪。

**不做:** 不重写优化器的数学。**更新公式已经是算子** —— 见 §2。

---

## 2. 上游有什么可以用

**这一块的复用率被严重低估过:优化器的更新公式本身就是融合算子。**

| 出处 | 内容 |
|---|---|
| `.../forwards/dygraph_functions.h:30` | `adam__ad_func(param, grad, learning_rate, moment1, moment2, moment2_max, beta1_pow, beta2_pow, master_param, skip_update, beta1=0.9f, beta2=0.999f, epsilon=1.0e-8f, lazy_mode=false, min_row_size_to_use_multithread=1000, multi_precision=false, use_global_beta_pow=false, amsgrad=false)` |
| 同上 第 26 行 | `adagrad__ad_func(param, grad, moment, learning_rate, master_param, epsilon=1.0e-6f, multi_precision=false)` |
| 同上 第 24 行 | `adadelta__ad_func(param, grad, avg_squared_grad, avg_squared_update, learning_rate, master_param, rho=0.95f, epsilon=1.0e-6f, multi_precision=false)` |
| `python/paddle/optimizer/*.py` | 状态管理与调度逻辑的语义蓝本 |

**所以 Lua 侧的优化器不做数学,只做三件事:**
1. 管理状态张量(`moment1` / `moment2` / `beta_pow` …)
2. 按参数分组
3. 调一次融合算子

**这也解释了为什么只要 2 周** —— P3 已经把这些算子生成好了。

---

## 3. 设计

### 3.1 API

```lua
local opt = paddle.optimizer.Adam{
  learning_rate = 1e-3,
  parameters    = net:parameters(),
  weight_decay  = 0.01,
  beta1 = 0.9, beta2 = 0.999,
}

opt:step()
opt:clear_grad()
```

**用关键字表构造,不用位置参数。** 优化器参数多且顺序无自然含义,
`Adam(1e-3, params, 0.9, 0.999, 1e-8)` 是可读性灾难。
这与 `_wrap` 三模式一致(D-R8)。

### 3.2 参数组

```lua
paddle.optimizer.Adam{
  parameters = {
    { params = backbone:parameters(), learning_rate = 1e-4 },
    { params = head:parameters(),     learning_rate = 1e-3 },
  },
}
```

判定方式:`parameters` 的第一个元素是 Tensor 就是扁平列表,是 table 就是参数组。

### 3.3 状态张量的生命周期

Adam 的每个参数要配 2–3 个同尺寸状态张量。**这意味着优化器持有的显存
大约是模型参数的 2–3 倍**,而且从第一次 `step()` 起就一直活着。

这与 P6/P14 的 GC 讨论直接相关:**优化器状态是长生命周期的,不应该进 `paddle.scope`。**
文档里的标准训练循环(`06-autograd.md` §3.4)之所以把 `opt` 建在 scope 外,原因就在这里。

### 3.4 学习率调度器

| 调度器 | 说明 |
|---|---|
| `StepDecay` `MultiStepDecay` `ExponentialDecay` | 按 epoch |
| `CosineAnnealingDecay` `LinearWarmup` | 常用组合 |
| `ReduceOnPlateau` | 依赖指标,与 metrics-lua(M3)有接口关系 |

**调度器的 `step()` 与优化器的 `step()` 是两件事**,名字撞车是 Python 侧的历史包袱。
我们保持一致(互通性优先),但文档里要专门警示。

### 3.5 梯度裁剪

```lua
paddle.nn.utils.clip_grad_norm_(net:parameters(), 1.0)
```

**注意结尾的下划线** —— 表示 in-place。这是从 Python 抄来的约定,
在 Lua 里下划线结尾不是惯例,但**互通性和可迁移性优先于本地惯例**。

---

## 4. 已知的坑

**① `adam__ad_func` 名字里有两个下划线。** `adam_` 是 in-place 算子,
生成器加 `_ad_func` 后缀就成了 `adam__ad_func`。**不是 typo**,别"修"它。

**② in-place 算子的返回值是引用。** `std::tuple<paddle::Tensor&, ...>` ——
Lua 侧要小心,拿到的是同一块显存的别名而不是新张量。
C ABI 层(P1)必须正确处理:**不要为引用返回值 new 一个新句柄然后 free 两次**。

**③ `multi_precision` 与 `master_param` 是混合精度路径。** v1 传 `false` / `none`,
但接口要留着,不要在 Lua 层把这两个参数删掉 —— 删了以后加回来要改一圈。

**④ `clear_grad` 和 `zero_grad` 的语义差别。** Paddle 是 `clear_grad`(置 nil),
PyTorch 是 `zero_grad`(置 0)。**我们跟 Paddle。** 提供 `zero_grad` 作为别名会引起混淆,
不提供。

---

## 5. 验收

- [ ] 固定种子,Lua 与 Python 训练同一网络 100 步,loss 曲线一致到 1e-5
- [ ] 每个优化器单独对拍:同样的参数与梯度,一步更新后参数逐位一致
- [ ] 参数组:两组不同 lr 确实按各自 lr 更新
- [ ] 6 个调度器的 lr 序列与 Python 一致
- [ ] `clear_grad` 后 `param.grad` 为 nil
- [ ] 优化器状态能存能读(与 P7 联调),读回后继续训练曲线连续
- [ ] 1000 步训练,显存不增长(状态张量没有重复分配)

---

## 6. 未解问题

- 混合精度(AMP)是否在 v1 范围?**不在** —— 但接口参数保留,M2 再接
