# P6 · 自动微分与生命周期

| | |
|---|---|
| 阶段 | P6 |
| 类别 | 手写 C + Lua |
| 开工条件 | P5 完工 |
| 预估 | 2 周 |

---

## 1. 做什么 / 不做什么

**做:** `backward()` / `paddle.grad()` / `stop_gradient` / `no_grad()` /
梯度累积与清零 / 最小可用的显存生命周期控制。

**不做:** 不实现反向传播算法本身。**反向引擎是纯 C++ 的,已经在 `libpaddle` 里。**
我们只是调用它。这是整个项目"不重写框架"论断的最直接体现。

**不做:** 完整的 GC 九层机制(P14)。这里只做 `paddle.scope` 和一条 OOM 回调。

---

## 2. 上游有什么可以用

| 出处 | 内容 |
|---|---|
| `paddle/fluid/eager/backward.h:25-29` | `void Backward(const std::vector<Tensor>& tensors, const std::vector<Tensor>& grad_tensors, bool retain_graph = false, bool create_graph = false, std::string dump_backward_graph_path = "")` |
| `paddle/fluid/eager/backward.h:31-40` | `std::vector<Tensor> Grad(tensors, inputs, grad_tensors, retain_graph, create_graph, only_inputs, allow_unused, no_grad_vars, dump_path)` |
| `paddle/fluid/eager/utils.h:123` | `static AutogradMeta* EagerUtils::autograd_meta(paddle::Tensor* target)` |
| `paddle/fluid/eager/utils.h:149` | `static AutogradMeta* nullable_autograd_meta(const paddle::Tensor&)` |
| `paddle/fluid/eager/utils.h:68` | `element->StopGradient()` |
| `paddle/fluid/eager/utils.h:86` | `element->SetStopGradient(bool)` |

**注意 `Backward` 的第 5 个参数 `dump_backward_graph_path`** —— 上游自带反向图 dump,
这是免费的调试能力,应该在 Lua 侧暴露成 `paddle.set_backward_graph_dump(path)`。

---

## 3. 设计

### 3.1 Lua 侧 API

```lua
loss:backward()                       -- 最常见
loss:backward{ retain_graph = true }
paddle.grad({y}, {x})                 -- 返回梯度但不累积到 .grad

x.stop_gradient = false               -- 通过 __newindex
print(x.grad)                         -- 通过 __index,可能是 nil

paddle.no_grad(function()             -- 作用域式,不是 with 语句
  local y = net(x)
end)
```

**`no_grad` 用函数而不是语句块。** Lua 没有 `with`,
`with paddle.no_grad():` 的等价物只能是高阶函数。这比 Python 稍啰嗦,但完全可用,
而且**天然是异常安全的**(用 `pcall` 保证退出时恢复状态)。

### 3.2 `no_grad` 的实现必须异常安全

```lua
function M.no_grad(fn)
  local prev = core.get_grad_enabled()
  core.set_grad_enabled(false)
  local ok, err = pcall(fn)
  core.set_grad_enabled(prev)          -- 无论如何都恢复
  if not ok then error(err, 0) end
end
```

**`error(err, 0)` 的 `0` 很重要** —— 不加会把错误位置改写成 `no_grad` 内部这一行,
用户看到的堆栈就废了。这是 Lua 5.1 的常见坑,`process/conventions.md` 里有条目。

### 3.3 `paddle.scope` —— 显存生命周期的最小方案

Lua 的 mark-sweep GC 不知道一个 64 字节的 userdata 背后压着 300MB 显存。
**这是本项目最真实的语言级不匹配。** 完整方案在 P14,这里先做最有用的一件事:

```lua
paddle.scope(function()
  local h1 = big_layer(x)     -- 300MB
  local h2 = another(h1)      -- 300MB
  return h2:clone()           -- 只有返回值活下来
end)                          -- 退出时,作用域内的中间张量立即释放
```

实现:进入时记录一个 epoch,C 侧维护 epoch -> 句柄列表,退出时逐个 `pd_tensor_free`
(除了被 `return` 出去的)。

**文档里从第一天就把 `paddle.scope` 作为训练循环的默认写法**,而不是当成"高级技巧"。
养成习惯的成本远低于事后教育。

### 3.4 训练循环的标准形状

这是我们希望所有文档、示例都统一使用的形状:

```lua
for _, batch in loader do
  paddle.scope(function()
    local loss = F.cross_entropy(net(batch[1]), batch[2])
    loss:backward()
    opt:step()
    opt:clear_grad()
  end)
end
```

---

## 4. 已知的坑

**① Lua 的 GC 是不确定的,显存的释放时机也就是不确定的。**
不能指望 `h = nil` 之后显存立刻回来。这不是 bug,是语言语义。
唯一的解法是显式作用域(§3.3),或者显式 `h:free()`。

**② 循环引用会让张量永远活着。** `net.layers[1].parent = net` 这种写法在 Lua 里
会被 GC 正确处理(mark-sweep 能处理环),**但显存回收会推迟到下一次完整 GC**。
`nn.Layer` 的设计(P9)要避免不必要的反向引用。

**③ 别把 `retain_graph` 的默认值设成 `true`。** Paddle 默认 `false`,我们对齐。
设成 true 会让用户在不知情的情况下累积图,直到 OOM。

**④ `x.grad` 可能是 nil。** 未参与计算、或 `stop_gradient=true` 的张量没有梯度。
Lua 里 `nil` 参与算术会抛 "attempt to perform arithmetic on a nil value",
这个错误信息对用户毫无帮助。**要在 `__index` 里主动检测并给出更好的信息。**

---

## 5. 验收

- [ ] 一个 2 层 MLP,手算梯度与 `loss:backward()` 一致到 1e-6
- [ ] `paddle.grad` 与 `backward` 结果一致
- [ ] `no_grad` 内部 `error()` 后,梯度状态正确恢复(用 pcall 测)
- [ ] `no_grad` 内部创建的张量 `stop_gradient == true`
- [ ] `paddle.scope` 退出后显存下降,用 `core.memory_allocated()` 前后对比验证
- [ ] 不 `clear_grad` 连续 backward 两次,梯度确实累积(与 Python 行为一致)
- [ ] `x.grad` 为 nil 时的错误信息可读

---

## 6. 未解问题

- Lua 5.4 的 `__close` 能否让 `paddle.scope` 写成
  `local _ <close> = paddle.scope()` 的形式?若可以,5.4 上提供更优雅的语法糖,
  但**5.1 路径必须始终可用**(C3)。M0 第 9 项验证
