# P8 · jit::Layer 推理 ★

| | |
|---|---|
| 阶段 | P8 |
| 类别 | 薄绑定 |
| 开工条件 | P5 完工 |
| 预估 | 1 周 |

---

## 1. 做什么 / 不做什么

**做:** 绑定 `paddle::jit::Layer`,让 Lua 能加载 Python 端 `paddle.jit.save` 的产物并推理。

**不做:** 不做训练,不做动转静(那是 P15)。

**★ 为什么这一块要尽早做,尽管它编号靠后:**

它**不依赖 `paddle.nn` 的移植**。P9 那 6 周还没开始,这条路径就能给出一个完整的产品形态:

> **Python 训练 / Lua 推理。**

这个形态本身就有真实用户 —— 嵌入式、游戏引擎、OpenResty 里做推理的人,
他们本来就不想在生产环境装 Python。
**有真实用户比有完整 API 更能暴露设计错误**,这是 `plan/roadmap.md` §3 原则②。

投入 1 周,换一个可交付的产品。这是全项目 ROI 最高的一块。

---

## 2. 上游有什么可以用

| 出处 | 内容 |
|---|---|
| `paddle/fluid/jit/layer.h:43` | `std::vector<Tensor> forward(const std::vector<Tensor>& inputs)` |
| `paddle/fluid/jit/layer.h` | `void to(const phi::Place&)` / `FunctionNames()` / `Clone(void* stream)` |
| `paddle/fluid/jit/engine/base_engine.h:27-32` | `virtual std::vector<DenseTensor> operator()(const std::vector<DenseTensor>&)` / `virtual std::vector<Tensor> operator()(const std::vector<Tensor>&)` / `virtual std::unique_ptr<BaseEngine> Clone(void* stream = nullptr)` |
| `paddle/fluid/jit/engine/interpreter_engine.h:34-59` | `InterpreterEngine : public BaseEngine`,内部持 `framework::Scope scope_` 与 `std::shared_ptr<framework::InterpreterCore> inner_interpreter_` |

**三个关键事实:**

1. **接口货币就是 `paddle::Tensor`**(`base_engine.h:22` `using Tensor = paddle::Tensor;`)——
   我们的 `pd_tensor` 里装的正是它,**零转换**
2. **`Clone(void* stream)` 天然配合 Lanes**(P13):每个 worker 一份 Layer,共享权重
3. **中间变量活在 `framework::Scope` 里**(`interpreter_engine.h:56`),
   由 `InterpreterCoreGarbageCollector` 管,**永远不会变成 Lua 对象** ——
   这条路径天然免疫 P6/P14 的 GC 风险

---

## 3. 设计

### 3.1 API

```lua
local m = paddle.jit.load("resnet18")   -- 目录或前缀
m:to("gpu:0")

local y = m(x)                          -- __call → forward
local y = m:forward(x)                  -- 等价

print(m:function_names())               -- {"forward", ...}
```

### 3.2 绑定量

| 内容 | 行数(估) |
|---|---|
| `pd_jit_layer_load` / `_free` / `_forward` / `_to` / `_function_names` / `_clone` | 200(C ABI) |
| sol2 usertype | 100 |
| `lua/paddle/jit/init.lua` | 100 |
| **合计** | **~400** |

**四百行换一个产品形态。**

### 3.3 与 P13 的接口预留

`Clone(void* stream)` 在这一阶段就要绑出来,即使 P13 还没开始。
理由:接口留好了,P13 只是使用者;接口没留,P13 要回头改 P8 的 ABI。

---

## 4. 已知的坑

**① 这一切建立在 M0 第 16 项之上。** `WITH_PYTHON=OFF` 下 `jit` 目录是否编出、
是否可链接,尚未验证。**挂了则本阶段全部作废**,要在 P0 就一起验。

**② 输入输出是 `std::vector<Tensor>`,不是单个。** Lua 侧要处理变长返回:
一个输出时直接返回张量,多个输出时返回多个值(Lua 支持多返回值,比 Python 的元组更自然)。

**③ `Layer` 加载的是 Python 训练的模型,dtype/layout 可能与预期不同。**
加载后应该提供 `m:input_spec()` 让用户能查,而不是让他试到崩。
⚠️ 上游是否有对应接口未核实,若无则从 `FunctionNames` 与 program 里挖。

**④ 别试图让 `jit::Layer` 参与训练。** 它是推理引擎。训练路径是 P15 的
`run_program_ad_func`,两者不是一回事,不要混。

---

## 5. 验收

- [ ] Python 端 `paddle.jit.save` 一个 ResNet18,Lua 端加载并推理
- [ ] 输出与 Python 端**逐位一致**(不是 1e-6,是逐位)
- [ ] `m:to("gpu:0")` 后结果不变
- [ ] 加载 1000 次不泄漏
- [ ] `Clone()` 出的两个实例可以在两个线程同时推理(为 P13 铺路)
- [ ] 一个不含 Python 的容器里跑通全流程 —— **这是本阶段的宣传素材**

---

## 6. 未解问题

- Q-07 相关:`WITH_PYTHON=OFF` 下 `jit` 是否可用(M0 第 16、17 项)
- `jit::Layer` 能否读取不同 Paddle 版本存的模型?
