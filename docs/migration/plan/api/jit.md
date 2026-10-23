# paddle.jit

| | |
|---|---|
| 阶段 | **P8 `load`**(Python 训练 / Lua 推理互通)/ **P15 `trace`**(动转静 trace 模式,D20)/ **P18 `script`**(script 模式,延后) |
| 依赖模块 | `paddle`、`paddle.Tensor`、`paddle.static`(InterpreterCore,D16) |
| 稳定性 | **稳定**(`load`——T-M0-17 已验证 C++ 侧)/ 实验(trace)/ 延后(script) |
| 实现文档 | `plan/modules/08-jit.md`(待写)、`research/to-static.md`(动转静) |
| 上游来源 | `python/paddle/jit/__init__.py`(导出);`api.py:188-217`(to_static @overload)、`:889`(save)、`:1462`(load);`layer.py:20`(Layer);`translated_layer.py`(TranslatedLayer) |
| M0 实证 | **T-M0-17**(`tools/smoke/smoke_jit_load.cc`):`jit::Load` 在 `WITH_PYTHON=OFF` 下可链接可加载(`fluid/jit/serializer.h:78`),`.pdmodel/.pdiparams/.meta` 全读入 |

> 📌 本模块是 **P8「Python 训练 + Lua 推理」互通的桥头堡**(research/reuse.md §3,
> D18:`jit::Layer` 是 M1 可选项)。**`load` 的 C++ 侧已由 T-M0-17 验证**;
> `trace` 是 M3(M4 之前仅 trace,script 延后,D20)。

---

## 1. 导出清单

`python/paddle/jit/__init__.py:31` 的真实导出:

| 导出 | 阶段 | 说明 |
|---|---|---|
| `paddle.jit.save` | P8 | 动转静导出(`api.py:889`) |
| `paddle.jit.load` | P8 | 加载导出产物(`api.py:1462`)——**T-M0-17 已验证 C++ 链** |
| `paddle.jit.to_static` | P15 | 动转静装饰器(`api.py:188-217`,@overload 族) |
| `paddle.jit.TranslatedLayer` | P8 | load 的返回类型 |
| `paddle.jit.enable_to_static` | P15 | 全局开关(程序翻译器) |
| `paddle.jit.not_to_static` | P15 | 反标记(某函数不转) |
| `paddle.jit.set_code_level` / `set_verbosity` | P15 | 调试日志级别 |
| `paddle.jit.ignore_module` | P15 | 忽略模块 |

---

## 2. 详细签名

### 2.1 `paddle.jit.save`(P8,Python 侧导出)

```lua
-- ⚠️ save 是 Python 侧的导出工具。Lua 侧不实现 save——
--    它是"Python 训练 → 导出 → Lua 推理"链路里 Python 那一端。
paddle.jit.save{ layer_or_program, path, input_spec=nil }
```

Lua 用户**不调用 save**,只消费它的产物(见 §2.2)。若 Lua 侧需要导出,
走 T-M0-19 的 `SaveTensor`/`LoadTensor`(参数格式,`top.md` 已标)。

### 2.2 `paddle.jit.load`(P8,★ Lua 侧核心)

```lua
local layer = paddle.jit.load{ path, skip_load_weights=false }
-- layer:TranslatedLayer
local out = layer(x)
```

| 参数 | 说明 |
|---|---|
| `path` | 产物目录/文件名(不含扩展) |
| `skip_load_weights` | true = 只读结构不读权重 |

**T-M0-17 实证链**:
- `fluid/jit/serializer.h:78` `Load` 声明;`serializer.cc:176` 实现
- 产物 `.pdmodel/.pdiparams/.meta` 全部读入(上游 CI 产物实测)
- 内部走 `InterpreterCore`(`interpreter_engine.cc:72`,D16 复用)
- ⚠️ **已知链接缺口**:`PirFunctionInfo` 构造引用 `GetFeedTargetNames/GetFetchTargetNames`
  (定义在 pybind 目录,`WITH_PYTHON=OFF` 不产出)——绑定层需自带纯 PIR 等价 shim
  (T-M0-17 冒烟已实现,`smoke_jit_load.cc` 头注释)
- ⚠️ **Q-24**:forward 撞上游 fc+oneDNN bug(`onednn_context.cc:347`)——非无 Python 特有,
  P8 落地时用不含 fc 的新导出产物验证

### 2.3 `paddle.jit.to_static`(P15,trace 模式)

```lua
paddle.jit.to_static(fn)   -- 装饰器:fn 的调用被 trace 成静态程序
```

**Lua 侧形态待定**:trace 是「执行 Python/Lua 函数并记录算子序列」——
Lua 侧等价物是把 Lua 写的 forward 跑一遍、记录 `_ops` 调用序列生成程序。
⚠️ **D20:trace 先于 script**(零 AST);SOT/静态图编译/script 推迟到生态有需求。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** `path`/`skip_load_weights` 非索引;
load 返回的 `TranslatedLayer` 的输入输出按各自算子的语义(见 `top.md` §3)。
显式写"无"以证明看过(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.jit.load(path)` | **`paddle.jit.load{ path }`** | 一张参数表 |
| 2 | `save` 是 Python 端工具 | **Lua 不实现 save**(消费产物) | 「Python 训练 / Lua 推理」分工(D18) |
| 3 | `TranslatedLayer` 是类 | load 返回 TranslatedLayer 实例 | 同形态,方法调用 |
| 4 | `to_static` 装饰器语法 | **Lua 侧 trace 形态 P15 再定** | D20:trace 先于 script |
| 5 | `enable_to_static` 全局开关 | 延后 | 默认开/关 P15 定 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

-- Python 侧已:paddle.jit.save(model, "export/mnist", input_spec=...)
-- 产物:export/mnist.pdmodel + .pdiparams + .meta

local layer = paddle.jit.load{ path = "export/mnist" }  -- TranslatedLayer
local out = layer(input_tensor)                          -- 推理

-- ⚠️ T-M0-17:优先用不含 fc 的导出产物(避 Q-24)
```

---

## 6. 本模块特有的坑

**① 链接缺口必须自带 shim。** `GetFeedTargetNames/GetFetchTargetNames` 定义在
`fluid/pybind/pir_utils.cc:20/36`,`WITH_PYTHON=OFF` 构建不产出该库 →
链接 `libfluid_jit` 时报 LNK2019。绑定层需复制纯 PIR 等价实现(冒烟已验证,
`smoke_jit_load.cc` 注释说明)。P8 落地时要么修上游目录归属(把 pir_utils 移到
非 pybind 目录),要么绑定层常驻 shim —— **P8 定,写进 `08-jit.md`**。

**② fc+oneDNN forward bug(Q-24)影响 P8 验收。** forward 撞
`onednn_context.cc:347` 缺输入 W;New 引擎 0xC0000005。非无 Python 特有
(上游 bug)。**P8 用不含 fc 的新导出产物验证**,并在文档写明已知影响算子。

**③ `load` 返回的 layer 输出是 Tensor,不是 pl.List。** TranslatedLayer
的 forward 输出可能是多个张量(Python 侧 `layer(inputs)` 返回
`list[Tensor]` 或单 Tensor)。Lua 侧按上游 `__call__` 语义:单输出直接返回,
多输出返回 `pl.List`(T-M0-17 冒烟只验了单输出,多输出 P8 实测后定)。

**④ `save` 不在 Lua 侧 —— 别发明导出实现。** 用户需要导出时:
- Python 训练端:用 Python 的 `paddle.jit.save`(生态既有)
- 纯 Lua 端:参数/权重走 T-M0-19 的 `SaveTensor`/`LoadTensor`(见 `top.md` §2.1)
两套格式都是 Paddle 的,`jit.load` 只认 jit.save 产物。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `trace`(动转静) | **P15**(D20:trace 先于 script;零 AST,记录算子序列) |
| `script` 模式 | **P18 或延后**(D20:SOT/静态图编译/script 推迟到生态有需求) |
| `enable_to_static` / `not_to_static` / `set_code_level` | P15 |
| `save`(Lua 侧导出) | **不实现**(Python 端工具;Lua 导出走 SaveTensor) |
| 多输出 TranslatedLayer | P8 实测后定(§6 ③) |
| Q-24 修复路径 | P8(上游 patch 或新导出产物规避) |
