# CONVENTIONS.md · 代码约定

> 写代码前读。`CLAUDE.md` §8 是速查,本文件是完整规则。

---

## 1. Lua 5.1 语法子集(硬约束 C3)

### 1.1 禁止清单

| 语法 | 引入版本 | 替代 |
|---|---|---|
| `goto` / `::label::` | 5.2 | 重构控制流(通常是提取函数或用 flag) |
| `//` 整除 | 5.3 | `math.floor(a/b)` |
| `&` `\|` `~` `<<` `>>` | 5.3 | 自写函数;LuaJIT 用 `bit` 库需 `pcall` 探测 |
| `<close>` / `<const>` | 5.4 | 普通 `local` |
| `\z` `\xXX` `\u{XXXX}` | 5.2/5.3 | `string.char` / 显式拼接 |
| 整数子类型 / `math.type` | 5.3 | 只假设 double;整数用 `math.floor` 归一 |
| `table.unpack` | 5.2 | `local unpack = table.unpack or unpack` |
| `table.pack` | 5.2 | `{n = select("#", ...), ...}` |
| `os.exit(code, close)` 第二参 | 5.2 | 只传一个参数 |
| `load` 接受函数以外的 env 参 | 5.2 | `compat.load` 垫片 |
| `#` 作用于有洞的表 | 全版本 UB | 显式存 `n` 字段 |

### 1.2 需要垫片的 -> **用 `pl.compat`,不要自己写**

> ⚠️ **已变更(D25 / C11)。** 本节原先给了一份手写的 `lua/paddle/compat.lua`。
> 现在兼容层统一走 vendored 的 `pl.compat`,**不再自写**。

```lua
local compat = require "pl.compat"

compat.load(s, name, "t", env)   -- 5.1 走 setfenv,5.2+ 走 load 的 env 参数
compat.lua51                     -- boolean
compat.jit                       -- 是否 LuaJIT
compat.is_windows
compat.dir_separator
```

`unpack` 这种一行的仍然可以就地写 `local unpack = table.unpack or unpack`
(`pl.utils` 也导出了一份),但**凡是涉及 `load` / 环境 / 沙箱的,一律走 `pl.compat`** ——
这是最容易在某个 Lua 版本上静默出错的地方,不该有第二份实现。

### 1.3 但要区分「语法」与「运行时特性」

`research/gc.md` §3.3 已论证。下列**不受 C3 约束**:

| | 为什么 |
|---|---|
| `collectgarbage("generational")` | 运行时字符串参数,5.1 上 `pcall` 保护即可安全降级 |
| C 侧给 userdata 挂 `__close` 元方法 | C API 操作,不是 Lua 语法。5.1/5.2/5.3/LuaJIT 忽略,5.4 生效 |
| 用户**自己**写 `local t <close> = ...` | 用户代码不受我们的约束 |

**结论:我们的库代码严守 5.1,但不妨碍给 5.4 用户提供 5.4 的能力。**

### 1.4 检查

`tools/lint_51.lua`(T-M1-01)。原理:用 Lua 5.1 自身的 `loadstring` 去
load 每个源文件 —— **5.1 编译器本身就是最权威的 5.1 语法检查器。**
CI 里对 `lua/**/*.lua` 全量跑,**包括 `lua/paddle/_vendor/pl/`**
(Penlight 声称支持 5.1,但我们要自己验证,不能只信 rockspec)。

---

## 2. 生态基座(硬约束 C11)

详细论证见 `plan/foundations.md`。这里只放规则。

| 需求 | 用这个 | **禁止** |
|---|---|---|
| 类 / 继承 | `pl.class` | 自写 class 系统、`middleclass`、`30log` |
| 不定长集合 | `pl.List` | 自写 list、第二套集合类型 |
| 5.1–5.4 兼容 | `pl.compat` | 自写 `compat.lua` |
| 读 Lua table 字面量 | `pl.pretty.read` / `pretty.load(...,paranoid)` | 自写 `sandbox_load` |
| 深拷贝 / 深比较 | `pl.tablex` | 自写 |
| 数组 / numpy 的位置 | Insight7 | 自写数组类型 |
| **文件系统** | **`paddle.utils.fs`**(我们的 C++17 层) | ~~❌ `pl.path` / `pl.dir` / `pl.file`~~ 见下 |

关于文件系统那一行:`pl.path` 一族依赖 C 库 `lfs`。
**「不引入新的强制 C 依赖」已于 2026-08-03 取消(R23),所以它们不再是禁止项。**
但**结论不变,理由换了**:我们的绑定 `.so` 已经链着 libpaddle,
加一组 C++17 `<filesystem>` 转发函数的边际成本 ≈ 0,
而多一个 rock = 5 Lua × 3 OS = 15 个构建组合。**是划算,不是被禁。**
`pl.path` 的纯字符串函数(`splitext` / `normcase`)不碰 lfs,需要时可以用。

三条使用红线:

1. **`class.cast` 禁用。** 它绕过 `_create`,造出没有 `FIELDS` 的破 Layer 实例
2. **`super` 是保留的实例字段名。** Penlight 的 `call_ctor` 会 rawset 它
3. **`List` 不进热路径。** `x:shape()`、`_wrap` 的参数解析、生成代码 —— 一律裸表

CI 必须有的 grep 规则:

```
class%.cast                             -> 失败
rawset%(self                            -> 失败(除 FIELDS 那一处;排除 _vendor/)
ipairs%(                                -> 在 lua/paddle/nn/ 下警告(见 §3 第 3 行)
```

~~`require ["']pl%.(path|dir|file|app)` -> 失败~~ —— 随 R23 取消。
真正要守的是 `rockspec` 的依赖清单,不是某个 `require` 语句。

---

## 3. 与 Python 的已知差异(用户可见,必须写进用户文档)

| # | Python | Paddle-Lua | 原因 |
|---|---|---|---|
| 1 | 索引 0-based,`axis` 0-based | **全 1-based** | D13 / R6 |
| 2 | `state_dict` 键名容器下标 0-based | **也是 0-based**(唯一破例) | 权重互通,R15 |
| 3 | `for m in module_list:` 直接迭代 | **`ipairs(ml)` 在 5.1/LuaJIT 上一次都不迭代,必须用 `ml:iter()` / `ml:len()`** | 实例 raw 表为空 + 5.1 的 `ipairs` 用 `lua_rawgeti`;`plan/foundations.md` §2.4。**`LayerList` 本身是 `Layer` 的子类,`is_a(nn.Layer)` 为真**(R22) |
| 4 | `for k, v in model.__dict__.items()` | `pairs(layer)` **看不到任何字段** | 自动注册要求 raw 表为空,用 `named_*` |
| 5 | `parameters()` 返回 generator | 返回 `pl.List` | R21 |
| 6 | 层的构造函数是 `__init__` | **`_init`**(Penlight 约定) | D25 |
| 7 | `super().__init__()` | **`self:super()`** | Penlight 约定 |
| 8 | 实例字段可叫任何名字 | **不能叫 `super`** | `pl.class` 的 `call_ctor` 占用 |
| 9 | `paddle.sum(x, axis=1)` | **`paddle.sum{x, axis=1}`** —— 只把 `(` 换成 `{`,**别的一个字不动** | Lua 没有 kwargs,一张混合表就是 kwargs(§5.4)。⛔ **不是** `paddle.sum(x, {axis=1})`,那是错的(R36) |
| 10 | `f(x, axis=None)` 与不传可区分 | **不可区分,`nil` = 没给** | 区分三态 = 3^N 枚举,D30 |

---

## 4. 模块与命名

```lua
-- 文件:lua/paddle/nn/linear.lua
local compat = require "paddle.compat"
local M = {}
function M.foo() end
return M
```

| 对象 | 风格 | 例 |
|---|---|---|
| 模块 | 全小写,下划线 | `paddle.nn.functional` |
| 类 | 大驼峰 | `Layer` `Linear` `DataLoader` |
| 方法/函数 | 小写下划线,**与 Paddle Python 同名** | `add_sublayer` `clear_grad` |
| 私有 | 前缀 `_` | `_build_once` |
| 常量 | 全大写 | `MAX_NDIM` |

**方法名与 Paddle Python 保持一致是硬要求** —— API 对齐是项目卖点。
Lua 惯例(如 `tostring`)只在语言层面(元方法)使用。

---

## 5. 索引与参数(硬约束 C4)

### 5.1 转换表

| 类别 | Lua 侧 | → C++ |
|---|---|---|
| 正 `axis`/`dim` | `n` | `n-1` |
| 负 `axis`/`dim` | `-1` | **`-1` 不变** |
| 切片区间 | `{i, j}` 闭 | `[i-1, j)` |
| 字符串切片 | `"1:3"` | `"0:2"`(移植 `lua_spec_to_cpp`) |
| 吃 index 的 tensor | — | 整体 `-1` |
| 吐 index 的 tensor | — | 整体 `+1` |
| label / token id | 1..C | 0..C-1 |

### 5.2 index 语义标注表

`tools/gen/index_semantics.lua`:

```lua
return {
  ["argmax"]       = { returns_index = {1} },
  ["topk"]         = { returns_index = {2} },
  ["index_select"] = { takes_index = {"index"} },
  ["gather"]       = { takes_index = {"index"} },
  -- ...
}
```

⚠️ **`ops.yaml` 没有这个信息,只能人工维护。**
**生成器遇到未在此表出现的新算子必须报错并中止,不得默认放行。**
默认放行 = 静默 off-by-one = 本项目最难查的 bug 类别。

### 5.3 1-based 护栏(M1 必做)

```
cross_entropy / embedding / one_hot:
  label 出现 0     → 报错 "looks like 0-based data, use paddle.index_from_zero()"
  label 出现 C+1   → 报错 "out of range"
```

1-based 语义下 `0` 本身非法,**两个方向都有天然护栏**。
这是 1-based 方案能安全落地的关键,不是"有空再加"。

### 5.4 调用约定

签名层是 `argrule`(R27/D31),**四种写法**,解析顺序定死(`plan/argrule.md` §2.5⑧):

```lua
paddle.sum(x, 1, true)                              -- ① 位置
paddle.sum{ x = x, axis = 1, keepdim = true }       -- ② 全具名
paddle.sum{ x, 1, true }                            -- ③ 表内位置
paddle.sum{ x, axis = 1, keepdim = true }           -- ④ 混合表  ← 默认写法
```

**★ 文档、示例、教程一律用 ④,除非调用只有位置参数(那就用 ①)。**
理由不是好看,是**迁移成本**:玩深度学习的人绝大多数是从 Python 过来的,
而 ④ 与 Python 的 kwargs 写法**只差一个字符**:

```python
paddle.sum(x, axis=1, keepdim=True)     # Python
```
```lua
paddle.sum{x, axis=1, keepdim=true}     -- Lua:( -> {,别的一个字没动
```

⛔ **没有第五种。** `paddle.sum(x, {axis = 1})`(位置参数 + 尾随选项表)是**错的** ——
那是两个实参的位置调用,规则 #2 `axis:number` 会收到一张 table(R36)。
它是大多数 Lua 库的习惯写法,也因此是最容易写错的一种,由 CI 挡(`plan/ci.md` §6①b)。

⚠️ **④ 与 Python kwargs 的三处不等价,必须写进用户文档:**

| | Python | Paddle-Lua |
|---|---|---|
| 显式传 `None` | `f(x, axis=None)` 与不传**可区分** | **不可区分**,`axis = nil` 就是「没给」 |
| 跳过中间的位置参数 | `f(x, keepdim=True)` 照写 | **表内位置写法 `{x, nil, true}` 不行**(数组有洞,`#t` 未定义)—— 只能用 ② 或 ④ |
| 动态拼参数 | 要 `f(x, **kwargs)` | **表本身就是 kwargs**,`f(t)` 直接就行,反而更简单 |

第一条是 D30 的直接后果:要区分「给了 / 没给 / 显式 nil」就是 argcheck 的 3^N 枚举,
`DataLoader` 的 16 个可选参数会直接挂死(`plan/foundations.md` §4.5)。
所以 **`nil` 一律等于「没给」**。⬜ 上游若有「`None` 与缺省含义不同」的参数,
必须在 api 文档里逐个点名并给一个显式写法 —— **尚未普查,记 Q-21,P3 前查完**。

第二条正好是**推荐 ④ 而不是 ③ 的第二个理由**:Lua 的数组不能有洞,
③ 一旦想跳过中间参数就没法写,而 ④ 天然没有这个问题。

---

## 6. 错误处理

### 6.1 C++ 异常绝对不能穿过 Lua(C7)

```
C++ 层     throw paddle::Exception
             ↓  中间层 try/catch,全部转 status code
C ABI 层   paddle_status_t + 错误消息缓冲区   ← extern "C",noexcept
             ↓  sol2 层检查 status
Lua 层     error(msg)  →  用户 pcall 捕获
```

**中间层每个导出函数都必须是 `noexcept` 且包 `try/catch(...)`。**
5.1/LuaJIT 用 `longjmp` 实现 error,C++ 异常穿过它是 UB。

### 6.2 错误消息

```
paddle: <模块>.<函数>: <发生了什么>. <怎么修>
```

例:`paddle: nn.Linear: in_features must be > 0, got 0`

---

## 7. 生成代码

```
csrc/capi_gen/**      生成
lua/paddle/_ops.lua   生成
csrc/sol/gen_*.cpp    生成
```

文件头必须有:

```
-- GENERATED BY tools/gen/gen_ops.py
-- FROM  <$PADDLE_ROOT>/paddle/phi/ops/yaml/ops.yaml @ <paddle git sha>
-- DO NOT EDIT
```

**改生成结果 = 改生成器,不是改产物。** CI 检查生成产物与重新生成的一致性。

---

## 8. Paddle API 引用(反幻觉,`CLAUDE.md` §4)

每处调用必须带出处:

```cpp
// paddle/phi/api/include/tensor.h:388
const void* raw = t.data();
```

```lua
-- 经 C ABI: paddle_tensor_data() -> paddle/phi/api/include/tensor.h:388
```

---

## 9. 测试

- 框架:**busted**
- 位置:`tests/` 镜像 `lua/paddle/` 结构
- **每个测试必须在目标 Lua 版本上跑过**,不能只跑一个版本
- 数值测试给出容差,不用 `==` 比浮点
- 测试名描述行为:`it("returns 1-based indices for argmax")`

---

## 10. 提交

```
<scope>: <祈使句摘要>

<为什么这么改;如果推翻了已有结论,写明推翻了哪条>

Refs: <文档章节 / 任务 ID>
```

scope ∈ `capi` `sol` `lua` `gen` `docs` `ci` `test`

**一次提交只做一件事。** 生成产物与生成器改动放同一提交。
