# paddle.autograd

| | |
|---|---|
| 阶段 | P6 |
| 依赖模块 | `paddle`(顶层)、`paddle.Tensor`(`:backward()` 共用底层) |
| 稳定性 | **稳定**(backward/grad/no_grad)/ 实验(jacobian/hessian/PyLayer) |
| 实现文档 | `plan/modules/06-autograd.md` |
| 上游来源 | `python/paddle/autograd/__init__.py`(装配);`backward_mode.py:34`(backward);`base/dygraph/base.py:706/328`(grad/no_grad);`autograd.py:461/587`(jacobian/hessian) |

> 📌 `paddle.autograd` 的导出主要 **re-export 自 `base/dygraph/base.py`**
> (`autograd/__init__.py:15-21`):`grad`/`no_grad`/`enable_grad`/
> `set_grad_enabled`/`is_grad_enabled`/`inference_mode` 全是那里的函数。
> `backward` 独立实现在 `backward_mode.py:34`。**Lua 侧 `Tensor:backward()`
> 与 `paddle.autograd.backward` 是同一个底层**(见 `tensor.md` §1.1)。

---

## 1. 导出清单

| 导出 | 来源 | 阶段 | 稳定性 |
|---|---|---|---|
| `paddle.autograd.backward` | `backward_mode.py:34` | P6 | 稳定 |
| `paddle.autograd.grad` | `base/dygraph/base.py:706` | P6 | 稳定 |
| `paddle.autograd.no_grad` | `base.py:328`(别名 `no_grad_`) | P6 | 稳定 |
| `paddle.autograd.enable_grad` | `base.py` | P6 | 稳定 |
| `paddle.autograd.set_grad_enabled` | `base.py` | P6 | 稳定 |
| `paddle.autograd.is_grad_enabled` | `base.py:442` | P6 | 稳定 |
| `paddle.autograd.inference_mode` | `base.py` | P6 | 稳定 |
| `paddle.autograd.jacobian` | `autograd.py:461` | P6 | 实验 |
| `paddle.autograd.hessian` | `autograd.py:587` | P6 | 实验 |
| `paddle.autograd.PyLayer` / `Function` | `py_layer.py`(`Function = PyLayer`,`__init__.py:35`) | 延后 | 实验 |
| `paddle.autograd.saved_tensors_hooks` | `saved_tensors_hooks.py` | 延后 | 实验 |

---

## 2. 详细签名

### 2.1 `backward`

```lua
paddle.autograd.backward{tensors, grad_tensors=nil, retain_graph=false}
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `tensors` | `TensorList` | 要反向的损失张量(可多个) |
| `grad_tensors` | `TensorList` 可选 | 初始梯度;nil = 全 1 |
| `retain_graph` | bool | 保留计算图(重复 backward 需要) |

⚠️ Python 的 `allow_unused`/`no_grad` 等参数见 `backward_mode.py:34-` 的签名;
Lua 侧第一批只做 `tensors/grad_tensors/retain_graph`(够 MNIST)。

### 2.2 `grad`(API 级梯度查询)

```lua
paddle.autograd.grad{outputs, inputs, grad_outputs=nil,
                     retain_graph=nil, create_graph=false,
                     only_inputs=true, allow_unused=false}
```

返回 `pl.List` of 梯度 Tensor(inputs 没有梯度的位置给 nil 或报错
取决于 `allow_unused`)。⚠️ 与 `Tensor:backward()` 的区别:grad 不反向传播
到非 inputs 的中间节点(只算 outputs→inputs 路径)。

### 2.3 `no_grad` / `enable_grad` / `set_grad_enabled` / `is_grad_enabled`

```lua
paddle.autograd.no_grad(function() ... end)   -- 函数式(上游 no_grad_ 装饰器)
paddle.autograd.no_grad{}                     -- ⚠️ 上下文管理器语义,见 §6

paddle.autograd.set_grad_enabled(false)
paddle.autograd.is_grad_enabled()             -- boolean
```

Lua 没有 with 语句。**no_grad 的上下文语义在 Lua 侧怎么表达**是
本模块第一个设计点(§6 ①)。

### 2.4 `jacobian` / `hessian`

```lua
paddle.autograd.jacobian{func, inputs, create_graph=false}
paddle.autograd.hessian{func, inputs, create_graph=false}
```

实验 API,基于 grad 的组合(`autograd.py:461/587`)。P6 延后或只做签名骨架。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** backward/grad/no_grad 都不涉及维度或张量索引。
`inputs`/`outputs`/`tensors` 是张量列表(非索引)。显式写"无"以证明看过
(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `loss.backward()` | **`loss:backward()`** | Tensor 方法,同底层(见 `tensor.md`) |
| 2 | `paddle.autograd.backward([loss], retain_graph=True)` | **`backward{tensors={loss}, retain_graph=true}`** | 一张参数表 |
| 3 | `with paddle.no_grad(): ...` | **待定(§6 ①)** | Lua 无 with;候选:函数式回调 / 显式 save/restore |
| 4 | `paddle.autograd.grad(...)` 返回 list | 返回 **`pl.List`** | D-R21 |
| 5 | `no_grad_`(别名) | `no_grad`(一个入口) | 命名铁律:主名照搬,下划线别名不搬 |
| 6 | `is_grad_enabled()` | 同名 | 全局状态,每 lane 一份(同 dtype-device.md §6 ④) |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

local x = paddle.to_tensor{1.0, 2.0}
x:set_stop_gradient(false)
local y = paddle.sum{x * 2}                 -- 4.0 + 6.0? 不,x 是两个元素

-- 方式一:Tensor:backward
y:backward()
print(x:grad())                             -- [2, 2]

-- 方式二:paddle.autograd.backward
local g = paddle.autograd.grad{outputs={y}, inputs={x}}
-- g[1] == [2, 2]

-- 方式三:no_grad
local z = paddle.autograd.no_grad(function()
  return paddle.sum{x * 2}
end)                                        -- 不记录梯度
```

---

## 6. 本模块特有的坑

**① `no_grad` 的上下文语义没有 Lua 原生对应。** Python 的
`with paddle.no_grad():` 是上下文管理器,Lua 只有函数调用。候选设计
(写进 `06-autograd.md`,P6 拍板):
1. **函数式**:`paddle.autograd.no_grad(fn)` —— 执行期间关闭梯度,结束恢复
2. **显式 save/restore**:`local prev = is_grad_enabled(); set_grad_enabled(false); ...; set_grad_enabled(prev)` —— 透传,零抽象
3. **装饰器**(函数签名)

**倾向 1(函数式)**:与 `pl.class` 的 `:super()` 调用风格一致,且不容易漏恢复。

**② `Tensor:backward()` 与 `paddle.autograd.backward` 必须同底层。**
Python 侧 `Tensor.backward`(`tensor_patch_methods.py:291`)最终也走
eager 的 `egr::Backward`(T-M0-02 已验证 `WITH_PYTHON=OFF` 可用)。Lua 侧
两个入口绑定同一个 C ABI 函数,不写两套 —— 否则保留计算图/梯度累加语义
会分叉。

**③ `grad` 的 `only_inputs`/`allow_unused` 默认值照抄。** Python 默认
`only_inputs=true`(只算到 inputs)、`allow_unused=false`(inputs 没参与
计算就报错)。Lua 侧照抄,别自作主张放宽 —— 静默返回 nil 会掩盖"图没连上"的 bug。

**④ `retain_graph` 默认 false 是常见的静默坑。** Python 的
`backward(retain_graph=False)` 一次反向后释放中间缓存,再次 `backward()`
报错"尝试对已释放的图反向"。Lua 侧报错信息要翻译成
`did you mean: backward{..., retain_graph=true}?`。

**⑤ `jacobian`/`hessian` 是组合不是算子。** 上游基于 `grad` 的组合
(`autograd.py:461/587`)。Lua 侧 P6 只做 `backward/grad/no_grad`,这两个
延后;但**接口名现在就写进导出清单**(避免 P6 后补时改名)。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `jacobian` / `hessian` | **延后**(组合,基于 grad;P6 末按需) |
| `PyLayer` / `Function` | **延后**(自定义反向,需要 C++ 扩展机制 P16 支持) |
| `saved_tensors_hooks` | **延后**(显存优化钩子,P14 GC 机制配合) |
| `inference_mode` | P6 只做 `no_grad`;inference_mode 差异(不建图)按需补 |
| `backward` 的 `allow_unused`/`no_grad` 等完整参数 | P6 末按需补全签名 |
