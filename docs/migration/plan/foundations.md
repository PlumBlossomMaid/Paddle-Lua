# 生态基座:Penlight 与 Insight7

| | |
|---|---|
| 类别 | **跨阶段的地基决策**,不属于任何单一阶段 |
| 影响 | P5 / P7 / P9 / P10 / P11 / P12 / P13 |
| 开工条件 | 无(P0 之前就应读完) |
| 状态 | 论证完成,已核实 Penlight 源码;Insight7 侧有 3 个未解问题 |

> **本文件回答一个问题:Lua 侧的"标准库"从哪来。**
> 结论:**类系统、容器、跨版本兼容层用 Penlight;数值数组(numpy 的位置)用 Insight7。**
> 两者都是**一等公民**,不是可选加分项。

---

## 0. 一句话结论

```
Python 侧             Lua 侧(本项目)
--------------------------------------------------
class / dict / list   Penlight (pl.class / pl.List / pl.Map)
numpy                 Insight7
paddle.Tensor         paddle.Tensor(本项目自己实现)
```

**为什么要定这件事:** Lua 的标准库极小,没有类、没有 list、没有跨版本兼容层。
如果不指定,每个模块会各自造轮子,最后出现三套 class、两种 list、
以及散落各处的 `local unpack = table.unpack or unpack`。
**指定一个基座的价值不在省代码,在于让所有模块的对象长得一样。**

---

## 1. Penlight

### 1.1 为什么是 Penlight

| 候选 | 判决 |
|---|---|
| 自写 ~80 行最小类系统 | ❌ **已推翻(R18)**,理由见 §1.6 |
| `middleclass` | 只有 class,没有容器/兼容层,还要再选两个库 |
| `30log` | 同上,更小但更少 |
| **Penlight** | ✅ **选这个** |

Penlight 的四条硬指标(均已核实):

| 指标 | 事实 | 出处 |
|---|---|---|
| 许可证 | MIT/X11 | `penlight-dev-1.rockspec` 的 `license` 字段 |
| Lua 版本 | `lua >= 5.1`,官方支持 5.1-5.4 + LuaJIT | 同上 `dependencies` |
| 纯 Lua | 我们要用的子集**没有一行 C** | §1.2 已实测 |
| 维护 | `lunarmodules/Penlight`,活跃 | GitHub |

### 1.2 依赖闭包:核心 11 个文件,`pl.path` 一族按需再加

> 🔄 **2026-08-03 变更(人的决定):`CLAUDE.md` §9 的「不引入新的强制 C 依赖」禁令已取消。**
> 本节原先花了很大篇幅论证怎么绕开 `luafilesystem`,**那部分论证作废**。
> 需要 C 库就引入,判据改为「边际成本」,见 `CLAUDE.md` §9.1。

Penlight 的 rockspec 声明 `dependencies = {"lua >= 5.1", "luafilesystem"}`。
`lfs` 是 C 库,但**现在这不再是障碍**。

不过闭包的事实仍然有用 —— 实测我们**核心**要用的模块的传递闭包
(2026-08-03 拉取 master 逐个 grep require):

```
pl.class  pl.List  pl.compat  pl.pretty  pl.tablex
pl.utils  pl.stringx  pl.lexer  pl.operator  pl.text  pl.types
```

11 个文件,5374 行,157 KB,**grep 计数 lfs 全部为 0**,闭包已封闭。

**结论(修订后):**

| 模块 | 处置 |
|---|---|
| 上面 11 个 | **vendored,核心强制**。零 C 依赖,`require "paddle"` 就能用 |
| `pl.path` / `pl.dir` / `pl.file` | **允许使用**,按需引入 `lfs` 依赖。P12 的数据集下载/解压会用到 |
| `pl.app` / `pl.lapp` 等 CLI 模块 | 不用 —— 与框架无关,不是依赖问题 |

**但文件系统的默认实现仍然是 `paddle.utils.fs`(C++17 `std::filesystem`),不是 lfs。**
理由**变了**:不再是「禁止 C 依赖」,而是 `CLAUDE.md` §9.1 第 1 步 ——
我们已经有一个链着 libpaddle 的 `.so`,往里加 `mkdir`/`exists`/`listdir`
的边际成本约等于零,而 `lfs` 是 15 个构建组合。
**同样的结论,但现在是因为划算,不是因为被禁。**
碰到 `pl.path` 的路径处理确实更顺手的地方,直接用,不必绕。

### 1.3 分发方式:vendor,不走 luarocks 依赖

| 方案 | 判决 |
|---|---|
| rockspec 里写 `dependencies = {"penlight"}` | ❌ 会把 lfs 拖进来(见 §1.2) |
| **vendor 那 11 个文件到 `lua/paddle/_vendor/pl/`** | ✅ **选这个** |

理由:
1. ~~避开 lfs~~ —— **此理由已作废**(禁令取消)
2. **版本锁定** —— Penlight 的 `pl.class` 若某天改了 `_create` / `_class_init` 语义,
   我们的 `nn.Layer` 会**静默**坏掉(不是编译错误,是参数注册悄悄失效)。
   **这条现在是 vendor 的主要理由,而且它本来就比第 1 条强。**
3. 开箱即用 —— `require "paddle"` 不应该因为用户没装 penlight 而失败
4. 与 `research/to-static.md` §5 vendor luacheck parser 的做法一致,不新增机制

**但要暴露出去:** `paddle.pl` 指向 vendored 副本,用户可以直接
`local List = require("paddle").pl.List`。同时若用户环境里已有系统 Penlight,
`require "pl.List"` 拿到的是系统的那份 —— 两份 `List` 的实例互不 `is_a`。
⚠️ 这是**已知代价**,必须写进用户文档:
**跨库传 `List` 时传裸表,不传 `List` 实例。**

### 1.4 `pl.class` 与 `nn.Layer` 自动注册的兼容性(核心问题)

`plan/modules/09-nn.md` §3.2 的自动注册依赖一条不变量:

> **实例的 raw 表里永远只有一个私有键 `FIELDS`,用户可见字段一个都不进 raw 表** ——
> 这样 `__newindex` 才会在**每一次**赋值时触发。

而 Penlight 的 `class` 默认把实例字段直接 rawset 在实例上。**看起来是正面冲突。**

**实测结论:不冲突。Penlight 提供了两个官方钩子,刚好够用。**

已核实的 `pl/class.lua` 关键行为(master,265 行,2026-08-03 拉取):

| 行为 | 代码 | 对我们的意义 |
|---|---|---|
| `_create` 钩子 | `if rawget(c,'_create') then obj = c._create(...) end` | **我们可以自己造实例表**,在 `_init` 之前就把 `FIELDS` 装好 |
| `_class_init` 钩子 | `if base and rawget(base,'_class_init') then base._class_init(c,c_arg) end`,**在 `c.__index = c` 之后执行** | **我们可以在每个子类创建后覆盖 `__index`/`__newindex`** |
| `c.__index = c` 是硬编码的 | `_class` 内无条件执行 | ⚠️ 若只在基类设 `__index`,`tupdate` 复制过去后会被这一行**覆盖掉** —— 所以必须走 `_class_init` |
| `_create` / `_class_init` 会被继承 | `tupdate(c,base,plain)` 浅拷贝基类全部字段 | 孙类也自动获得,不需要用户做任何事 |
| Penlight 自己就用 `_create` | `pl/List.lua:53` 的 `function List._create (src)` | 这不是我们发现的野路子,是官方用法 |

于是 `nn.Layer` 的定义大致是:

```lua
local class = require "pl.class"
local FIELDS = {}                   -- 私有键,用户拼不出来

local Layer = class()

Layer._create = function()          -- 每个实例创建时先跑这个
  local o = {}
  rawset(o, FIELDS, {
    params = {}, sublayers = {}, buffers = {}, plain = {},
    order = {}, training = true,
  })
  return o
end

Layer._class_init = function(c)     -- 每个子类创建后跑这个
  c.__index    = layer_index        -- 覆盖 Penlight 硬编码的 c.__index = c
  c.__newindex = layer_newindex
end

Layer.__index    = layer_index      -- 基类自己也要设一次
Layer.__newindex = layer_newindex
```

`layer_index` 的类查找回退写成 `cls[k]` 即可 ——
`cls.__index` 是给**实例**用的,而 `cls[k]` 走的是**类自己的元表** `{__index = base}`,
两者互不干扰,继承链能正确走通。

### 1.5 `pl.class` 的四个坑(已核实,必须写进 09-nn)

**坑① `super` 是保留的实例字段名。**
`call_ctor` 里有 `rawset(obj,'super',function(obj,...) ... end)`,
构造期间实例上真的存在 `super` 这个 raw 键(构造完会 `rawset(obj,'super',nil)` 清掉)。
后果:用户不能给自己的层起一个叫 `super` 的参数;
我们的 `__newindex` 也不会看到这次 rawset(rawset 绕过元方法)——
**清理是 Penlight 做的,不会污染 `FIELDS`,但这条要在文档里写明。**

**坑② `class.cast` 不能用。**
`cast(klass,obj)` 就是 `setmetatable(obj,klass)`,它换元表但不动 raw 表 ——
把一个普通表 cast 成 `Layer` 会得到一个**没有 `FIELDS` 的破实例**,
之后第一次赋值就崩在 `rawget(self,FIELDS)` 上。禁用,CI grep。

**坑③ `require("pl.class").class` 会往 stderr 写弃用警告。**
只用 `local class = require "pl.class"`,不要再取 `.class`。

**坑④ 默认 `__tostring`。**
`_class` 末尾有 `if not rawget(c,'__tostring') then c.__tostring = _class_tostring end`,
而 `_class_tostring` 会读 `obj._class`。我们的 `layer_index` 必须能把 `_class` 回退到类表上,
否则 `print(layer)` 会崩。(按 §1.4 的写法能走通,但必须有测试。)

### 1.6 翻案:为什么放弃「自写 80 行类系统」

原理由是「没有依赖,行为完全可控」。**这条不成立**,因为:

1. **依赖不是新增的,是本来就要有的。** 我们还需要 list、需要 `load`/`setfenv` 的
   5.1-5.4 兼容层、需要安全读 Lua table 字面量(见 §1.7)。
   自写类系统只是把「一个依赖」换成「一个依赖 + 三个自己造的轮子」。
2. **「行为完全可控」是负资产。** 我们自己写的类系统只有我们自己会调试;
   Penlight 的 `class` 被大量项目用了十几年,`_create` / `_class_init` 有文档、有测试。
3. **用户侧的收益被低估了。** 用户要继承 `nn.Layer` 写自己的网络。
   如果用的是 Penlight 的 class,用户已有的 Penlight 知识**直接可用**,
   而不是要学一个只在本项目里存在的 `:extend()`。

原来担心的「用户可能已有自己的 OOP 库,冲突」 —— **恰恰反了**:
Penlight 是 Lua 生态事实上的通用库,选它才是冲突最小的选择。

### 1.7 Penlight 顺带解决的四个已有问题

| 已有设计 | 被 Penlight 替代 | 收益 |
|---|---|---|
| `plan/modules/07-serialization.md` 手写的 manifest 沙箱读取 | `pl.pretty.read` | 已核实其实现:`utils.load(s,'tbl','t',{})` 空环境 + 拒绝 `function` 关键字 + 用 `pl.lexer` 二次确认 |
| 各处散落的 `local unpack = table.unpack or unpack` | `pl.compat` | 还顺带给了 `compat.load`(5.1 走 `setfenv`)、`compat.lua51`、`compat.jit`、`compat.is_windows` |
| `state_dict` 的深比较 / 深拷贝 | `pl.tablex.deepcopy` / `deepcompare` | 测试里用得最多 |
| 调试打印 tensor 元信息 / 配置 | `pl.pretty.write` | 免写 table 序列化 |

⚠️ 注意 `pl.pretty.read` 不防死循环(`while true do end` 能过)。
若 manifest 来自不可信来源,要用带 `paranoid` 的 `pretty.load`。
**结论:自己存的 manifest 用 `read`,用户给的路径下的用 `load(..., paranoid=true)`。**

---

## 2. `pl.List` 作为一等公民

### 2.1 哪些 API 返回 `List`

原则:**返回「个数不定的同类东西」就返回 `List`;返回定长元组就返回多值。**

| API | 返回 |
|---|---|
| `layer:parameters()` | `List` |
| `layer:sublayers()` | `List` |
| `layer:buffers()` | `List` |
| `nn.LayerList` | **是** `List` 的子类,见 §2.4 |
| `optimizer:param_groups()` | `List` |
| dataset 的 batch collate 结果 | `List` |
| `paddle.split(x, n)` | `List` |
| `x:shape()` | ⚠️ **裸表**,见 §2.3 |
| `named_parameters()` | **迭代器**,不是 `List`(与 Python 一致) |
| `x:size()` / `x:dim()` | 数字,多值返回 |

### 2.2 为什么值得

`pl.List` 是 1-based 的,与 D-R6 天然一致。它免费给出:

```lua
local n = net:parameters()
  :filter(function(p) return not p.stop_gradient end)
  :map(function(p) return p:numel() end)
  :reduce("+")                       -- 可训练参数总量,一行
```

对比自己造:`filter` / `map` / `reduce` / `slice` / `index` / `sort` / `join` /
`__concat` / `__eq` 要写 200 行并逐个测。而且 `List` 的 `__tostring` 让
`print(net:parameters())` 直接可读,对交互式调试是实打实的体验差别。

### 2.3 边界:`List` 不进热路径,不进 Tensor 形状

**`x:shape()` 必须返回裸表,不返回 `List`。**
理由:形状会被喂回 C ABI 层(`reshape` 等),
`List` 实例带元表,过 sol2 时多一次元表查找;而且形状表在训练循环里每步都造。

**一等公民不等于无处不在** —— `List` 出现在**用户手里的集合**上,
不出现在**框架内部的高频小数据**上。
同理:`_wrap` 三模式调用的参数解析、算子生成代码,**一律裸表**。

### 2.4 与 `LayerList` 的关系:**继承 `Layer`,Layer 优先**

`nn.LayerList` 需要同时是「`List`」和「`Layer`」。Lua 单继承 -> 只能选一个基类。

> 🔄 **2026-08-03 翻案(R22,人的决定)。** 本节原选 `class(List)`,
> 理由是「主要用法是当序列用」。**这个理由站不住,已推翻。选 `class(Layer)`。**

**为什么 `Layer` 优先:**

| | |
|---|---|
| ① `is_a(nn.Layer)` 必须为真 | 这是**语义正确性**,不是便利性。`LayerList` 装的是层、参与前向、有参数、要 train/eval —— 它**就是**一个 Layer。让它不是,是为了实现方便去撒谎 |
| ② 递归遍历是 `Layer` 的核心 | `parameters()` / `state_dict()` / `to()` / `train()` / `eval()` / 钩子 —— 每一个都要递归子层。继承 `List` 意味着这**六处**全部要加 `LayerList` 特例,而继承 `Layer` 只需要实现一次 `List` 的接口 |
| ③ 与 Python 一致 | Python 里 `ModuleList` 就是 `Module` 的子类。从 Python 迁移的代码里有 `isinstance(m, nn.Module)`,直译过来是 `m:is_a(nn.Layer)` —— 原来的方案会让这行**静默返回 false** |
| ④ 特例会繁殖 | 「父层的 `layer_newindex` 识别出 `LayerList` 后展开」是一个特例。有了它,`LayerDict` 要一个,`Sequential` 要一个,用户自定义容器还要一个 |

**代价:失去 `pl.List` 的方法(`map` / `filter` / `reduce` / `__concat`)。**
这个代价小,而且可补 —— `LayerList` 内部持一个 `pl.List`,
类表上挂 `map` / `filter` 等转发方法即可。
`layer_index` 的最后一步是 `getmetatable(self)[k]`(§1.4),会正常查到这些方法。

#### ⚠️ 但因此暴露一个新问题:`#ml` 和 `ipairs(ml)` 在 5.1 上是静默失效的

这是我在改这一节时才想清楚的,**它对两个方案都成立**,不是选 `Layer` 带来的:

自动注册要求实例的 raw 表保持空(§1.4 / `09-nn.md` §3.2),于是:

| 写法 | 5.1 / LuaJIT | 5.3 / 5.4 |
|---|---|---|
| `#ml` | **0**(`#` 直接看 raw 表的数组部分) | 0(`__len` 对表是 5.2+,但我们的 raw 表仍是空的) |
| `ipairs(ml)` | **一次都不迭代** —— Lua 5.1 的 `ipairsaux` 用的是 `lua_rawgeti`(`lbaselib.c`) | **能正常迭代** —— 5.3 的 `ipairs` 走 `lua_geti`,尊重 `__index` |

**同一段代码在 5.1 上循环体一次都不进,在 5.4 上正常跑。**
这是跨版本行为不一致,比单纯不支持更糟。
(5.2 的确切行为待核实,见 Q-16。)

**处置:**
- `LayerList` 必须提供 `ml:len()` 与 `ml:iter()`,**文档与示例一律用这两个**
- **不去救 `ipairs`** —— 5.1 上无法救(没有 `__ipairs`,`ipairs` 是 rawgeti)。
  能在 5.4 上工作而在 5.1 上不能的写法,我们不写进任何示例
- 加一条测试:`ipairs(ml)` 在五个版本上的行为被显式断言(不管结果是什么),
  防止将来有人「顺手修好了 5.4」而不知道 5.1 还是坏的

**考虑过但否决的方案:让 `LayerList` 把整数键 rawset 进 raw 表。**
这样 `#` 和 `ipairs` 在所有版本上都能用。**否决理由:**
那样 `ml[2] = otherLayer` 就**不会触发 `__newindex`**(键已存在),
FIELDS 里还挂着旧层 -> `parameters()` 返回旧层的参数 -> **优化器优化了一个不在网络里的层**。
这正是 R16 当年担心的那个 bug,在一个具体角落里复活。

**两个失败模式对比,选了较响的那个:**
- 空 raw 表 -> 循环体不执行 -> **前向立刻出错,一眼看见**
- 整数键进 raw 表 -> 训练照常跑,**收敛到错的地方** ← 不能接受

---

## 3. Insight7 顶替 numpy 的位置

### 3.1 定位

```
insight.Array   ~  numpy.ndarray   无梯度、CPU/GPU 数值容器、数据预处理
paddle.Tensor   ~  torch.Tensor    有梯度、参与自动微分
```

**Insight7 不参与自动微分,也不打算参与。**
它在本项目里的职责就是 numpy 在 PyTorch 生态里的职责:
数据加载与预处理、指标的中间计算、可视化前的整形、单元测试里造期望值。

### 3.2 为什么是 Insight7 而不是别的

| 事实 | 出处 |
|---|---|
| Lua 绑定的 dtype 常量**已经是 Paddle 命名** `ins.float32` / `ins.int64` | `bindings/lua/insight/init.lua` 头部注释写着 "DType Shortcuts (Paddle-style)" |
| Place 构造器**已经是 Paddle 命名** `ins.CPUPlace()` / `ins.GPUPlace(0)` | 同上 |
| 字符串索引 `x["1:3, :"]` 本来就是从它移植的(D14 / R7) | `plan/overview.md` §6.1.2 |
| `_wrap` 三模式调用本来就是从它移植的(D15 / R8) | `bindings/lua/insight/_wrap.lua` |
| dtype 覆盖 Paddle 常用全集,另有 `F8_E4M3` / `F8_E5M2` | `include/insight/core/dtype.h:16-36` |
| 绑定方式同为 sol2 | `bindings/lua/insight_lua.cpp` |
| 许可证 MIT | `bindings/lua/insight/init.lua` 头部 |

**关键论点:Insight7 的 Lua API 本来就是照着 Paddle 设计的。**
选它不是「引入一个数组库」,是「把已经对齐的那半边接上」。
换成任何别的库,用户都要同时记两套 dtype 名、两套 place 写法、两套切片语法。

Lua 生态里的其它候选(torch7 的 `torch.Tensor`、各种纯 Lua 矩阵库)
要么已停止维护,要么没有 dtype / GPU / 广播,要么与我们的索引约定冲突。

### 3.3 互操作:零拷贝的条件

两侧都是 strided buffer,理论上可零拷贝共享:

| Insight7 | Paddle |
|---|---|
| `Array::data()` -> `void*`,`include/insight/core/array.h:134-135` | `Tensor::data()`,`paddle/phi/api/include/tensor.h:388,397` |
| `Array::shape()`,`array.h:109` | `Tensor::shape()`,`tensor.h:199` |
| `Array::dtype()`,`array.h:112` | `Tensor::dtype()`,`tensor.h:224` |
| `Array::place()`,`array.h:115` | `Tensor::place()` |
| 视图构造 `Array(const Array&, const Shape&, ...)`,`array.h:84` | `Tensor(std::shared_ptr<phi::TensorBase>, const std::string&)`,`tensor.h:142` |

对外接口:

```lua
paddle.from_insight(arr)   -- 尽量零拷贝;不满足条件时拷贝并 warn
paddle.to_insight(t)       -- 同上
```

**零拷贝的四个前提**(任一不满足即退化为拷贝):
1. 同 place(CPU-CPU,或同一 GPU 设备号)
2. dtype 双向可映射
3. 内存连续(或 strides 双方都能表达)
4. **生命周期能挂钩** —— 最难的一条,见 Q-13

⚠️ **Q-13(未解):** 让 `paddle::Tensor` 接管外部指针,需要一个 `phi::Allocation`,
且它必须持有对 Insight7 `Array` 的引用,否则 Array 先析构会造成悬垂。
`tensor.h:142` 那个构造函数收的是 `shared_ptr<phi::TensorBase>` ——
**我们能否自己实现一个持有 `insight::Array` 的 `TensorBase` 子类,未核实。**
若不行,第一版只做拷贝互转,零拷贝降级为 P12 的加分项。
**注意这不影响可用性,只影响大数组预处理的吞吐。**

### 3.4 ✅ 已决:Insight7 的 `axis` 改 1-based(成员索引与维度索引**都要** 1-based)

> **2026-08-03 人的决定:「Insight7 的 bug 以后是要修的,以后会顺手修复的,
> 成员索引和维度索引都要 1-base。」**
> 本节从「⚠️ 阻塞项,待拍板」改为「已定方向,待实施」。待拍板 P1 关闭,Q-12 转为已决。

#### 现状:一半做了,一半漏了

| | 现在 | 目标 |
|---|---|---|
| **成员索引 / 切片**(`a[1]`、`a["1:3, :"]`) | ✅ **已经是 1-based** —— `bindings/lua/insight_lua.cpp:212-266` 的 `lua_spec_to_cpp` 在解析切片串时做了转换 | 保持 |
| **维度索引 / `axis`**(`ins.sum(a, 1)`) | ❌ **0-based** —— `bindings/lua/insight/reduction.lua:18` `native.sum(x, axis, keepdims or false)`、`indexing.lua:40` `native.argsort(x, axis or -1)`,**直接透传** | 改成 1-based |

**这是「漏做」,不是「设计如此」。** 两条佐证:
1. 同一个绑定里,切片已经转了(`lua_spec_to_cpp`),说明作者的意图就是 Lua 侧 1-based
2. `bindings/julia/Insight.jl:726-729` **做了 axis 转换** —— 「绑定层负责转换」是 Insight7 既定设计,Lua 侧属于遗漏

所以这是**按 bug 修**,不是 breaking change。

#### 为什么必须修

```lua
local a = ins.sum(arr, 1)     -- 现在:0-based 的第 1 轴,即第 2 维
local b = paddle.sum(t, 1)    -- Paddle-Lua:第 1 维
```

同一个脚本、同一个字面量 `1`,两个意思 —— 正是 D13(统统 1-based)要消灭的静默 off-by-one。
Insight7 从「参考实现」升为一等公民(R20)之后,这两行会真的出现在同一个用户脚本里。

**而且更糟的是它在同一个库内部就不自洽:**
`a["1:3"]` 是 1-based,`ins.sum(a, 1)` 是 0-based。
用户学会前者之后,后者的错误是**静默的**(维度存在,只是错了一维,结果 shape 合法)。

#### 落地方式

| | |
|---|---|
| 在哪修 | **Insight7 仓库的 Lua 绑定**(不是在 paddle-lua 里包一层) |
| 改什么 | `bindings/lua/insight/*.lua` 里所有透传 `axis` 的位置,统一走一个 `to_c_axis(axis, ndim)` |
| 负数轴 | `-1` 表示最后一维,**Lua 侧保持 `-1` 不变**(负数轴在两种基准下都是从末尾数,无歧义)。转换只作用于**正数** |
| `axis = 0` | 转换后是非法值 -> **必须报错**,不能静默当成 `-1` 或第一维。这是 D13 「1-based 让 off-by-one 从静默变可检测」的兑现点 |
| 多轴 | `axis` 收 table 时逐元素转换 |
| 时机 | **顺手修,不专门排期。** 但必须在 P12(vision transforms 大量用 axis)之前完成 |
| 验收 | Insight7 侧加一条测试:对 3 维数组,`ins.sum(a, 1)` 与 Julia 的 `sum(a, dims=1)` 结果一致 |

#### 修好之前 paddle-lua 怎么办

不做包装层(包装层挡不住用户直接 `require "insight"`)。
在 `12-dataset-vision.md` 的 transforms 里**避免依赖 axis 语义**,
或在调用处显式写 `--[[ Insight7 axis 尚未 1-based,见 foundations §3.4 ]]` 注释,
修好后统一清理。

### 3.5 Insight7 与 Lanes(P13)

DataLoader 的 worker 线程里会用 Insight7 做预处理 -> `insight::Array` 要能跨 lane。

我们对 `paddle::Tensor` 的解法是 `__lanesclone` + `shared_ptr` 原子计数(D-R17)。
**Insight7 是否有等价的原子引用计数,未核实(Q-14)。**
`array.h:72` 的 `Array(const Array &other)` 是拷贝构造,但**语义未知**(深拷贝?共享缓冲?)。

保底方案与 Tensor 一致:整数句柄表过 Linda。
不阻塞 P13,但要在 P12 之前查清楚。

### 3.6 Insight7 与显存

⚠️ Insight7 有自己的 GPU 后端和分配器。两个框架各自持有显存池,**会互相饿死**。

第一版的约定:
- **文档建议:Insight7 用于 CPU 侧数据准备,GPU 计算交给 Paddle**
- 不做统一分配器(那是跨仓库的大改造)
- `paddle.to_insight` 遇到 GPU tensor 时 warn 一次

这条要写进 `research/gc.md` 的风险登记。

### 3.7 依赖形态

| | |
|---|---|
| Penlight | **vendored 强制依赖**,11 个纯 Lua 文件进发布包 |
| Insight7 | **rockspec 声明依赖**,不 vendor(它是 C++ 库,有自己的构建) |

Insight7 是**软强制**:`paddle` 核心不 `require "insight"`;
只有 `paddle.np` / `paddle.from_insight` / `paddle.vision.transforms` 里
惰性 `require`,缺失时给出清晰错误。

理由:核心训练路径不该被一个数组库阻塞掉安装。
但**文档、教程、示例一律按「装了 Insight7」来写** —— 这才是「一等公民」的含义。

---

## 4. 参数检查:argcheck(vendored + 去 Torch 化,但**只用在冷路径**)

> 基座的第四块。前三块是「类/集合」「numpy」,这一块是「函数签名」。
> 结论先说:**采纳,但分层用 —— 冷路径用 argcheck,热路径继续用 `_wrap`,生成代码两者都不用。**

### 4.1 结论表

| 场景 | 调用频次 | 方案 |
|---|---|---|
| **构造期 API**:`nn.Linear{...}`、`DataLoader{...}`、optimizer、`transforms.*` | 一个模型几十~几百次 | **argcheck** |
| **热路径**:`paddle.add`、`t:reshape`、`Dataset:get(i)` | 每 step 上千次 | **`_wrap` 三模式**(D-R8 保留) |
| **P3 生成的 2000+ 算子** | 每 step 上万次 | **生成期静态展开**,见 §4.5 |

### 4.2 「硬编码了 Torch7」这个前提不成立

实测(commit `b3b32c0`,2016-06-29,842 行 / 7 文件),Torch 耦合只有**两处、共 32 行**:

| 位置 | 内容 | 性质 |
|---|---|---|
| `env.lua:26-52` | `if pcall(require, 'torch') then` 覆写 `env.istype` / `env.type` | **它覆盖的那个默认实现本来就是通用的** —— 读 `getmetatable(o).__typename`,回退 `type(o)`。两个函数上方原注释写着 `-- user configurable function` |
| `usage.lua:83` | 有 `sundown` + torch + tty 时把 markdown 渲染成 ANSI 颜色 | 纯装饰。没有 Torch 时错误信息里会**留下字面的 `**number**` 标记**,要替换掉 |

也就是说:**扩展点是作者留好的,不是我们要凿开的。**
我们只需要把 `env.istype` 指向「认识 `paddle.Tensor` / `Parameter` / `nn.Layer` / `insight.Array`」的版本。

### 4.3 真正让它「不好用」的是一个 2 行的 Windows bug

`graph.lua:13-21`:

```lua
local function table2id(tbl)
   -- DEBUG: gros hack de misere        ← 作者自己写的:「悲惨的脏 hack」
   return tostring(tbl):match('0x([^%s]+)')
end
```

它拿对象地址当生成代码里的唯一标识。而 `tostring(t)` 的格式来自 C 的 `%p`:

| 平台 | `tostring({})` | `match('0x…')` |
|---|---|---|
| glibc / macOS | `table: 0x7f9c...` | 命中 |
| **MSVC(Windows 上的 Lua/LuaJIT 官方构建)** | `table: 00B62A68` | **nil** |

后果:**任何一条规则带 `default` / `defaulta` / `defaultf` / `opt` 的 argcheck,在 Windows 上直接抛
`graph.lua:281: bad argument #4 to 'format' (string expected, got nil)`。**
不带默认值的最简用法反而正常 —— 所以它表现为「时好时坏」,而不是「装不上」。

**这不是「过时」,是「Torch7 只跑 Linux/macOS,没人在 Windows 上用过」。**
十年没人修,是因为十年没人在 Windows 上用它。

实测(本机 `lua5.1`,MSVC 构建,**未安装 Torch**):

```
未打补丁    argcheck{{name="x",type="number"}}                  -> OK
未打补丁    argcheck{{name="x",type="number",default=0}}        -> 崩
改掉这 2 行 上游全套 test/test.lua                              -> 全部通过
```

> test/test.lua 里另有 1 处**测试自身**的 5.1 不兼容:`string.format('%s', nil)`
> 在 5.3+ 合法(走 `luaL_tolstring`),在 5.1 是 `bad argument`。
> 这是测试的问题,不是 argcheck 的问题。我们的 vendored 副本要连测试一起修。

我们的替换:用**弱键计数器**发号,不碰地址。
副作用是生成代码里的标识符变成确定性的(`arg1_1d` 而不是 `arg7f9c1a_1d`),
**对 CI 的 regen-diff(`ci.md` §6)反而是好事** —— 地址每次运行都不同,没法比对。

### 4.4 但性能是真问题,而且比想象中大一个量级

「生成特化代码 = 零开销」这个说法**实测不成立**。本机 Lua 5.1,3 参数
(`number` + `number` 带默认 + `string` 可选),30 万次:

| | ns/call | 相对裸调用 |
|---|---|---|
| 裸 Lua 函数调用 | **43** | 1× |
| Insight7 `_wrap` 那种定参包装 | **420** | 10× |
| **argcheck(全位置参数)** | **2597** | **60×** |
| **argcheck(具名表 `f{...}`)** | **2733** | 63× |
| argcheck(只传 1 个,走默认值) | 660 | 15× |

拆解(实测归因):

| 来源 | 代价 |
|---|---|
| `env.istype` 被调 **6 次**(决策树要逐分支试) | 6 × 300 ns = **1800 ns** |
| 生成的函数体里有 `local usage = require "argcheck.usage"` —— **每次调用都 `require` 一遍** | 193 ns |
| `select(n, ...)` 链 | 107 ns × n |

即使把这两条都修掉(`require` 提到 upvalue、基础类型直接内联成 `type(x)=="number"`
而不走 `istype`),手写等价决策树实测仍要 **897 ns** ——
因为 `...` + `select` 的架构本身就比定参函数贵。**这是结构性的,不是调优能拿掉的。**

**为什么这条足以否掉「全面采用」:**
选 sol2 的理由(D1/R1)是「绑定开销 50-200 ns vs kernel 1-10 µs,差 1-2 个数量级」。
**2.6 µs 不满足这个论证** —— 它和一个小 kernel 同量级,在 elementwise 这种算子上是 **25%-50% 的净损耗**。
**同一把尺子必须两头都用。**

反过来,构造期 API 用它是**完全免费**的:一个 200 层的网络构造一次 = 200 × 2.6 µs ≈ **0.5 ms**,
一次性,发生在训练开始之前。

### 4.5 生成代码不用 argcheck —— 它和我们的 codegen 是重复建设

argcheck 的核心机制是**运行时**生成 Lua 源码 + `loadstring` + `debug.setupvalue` 注入 upvalue。
而 P3(`tools/gen/`)已经是一条**构建期**代码生成流水线,要为 2000+ 算子产出 `lua/paddle/_ops/`。

对生成的算子:
- 签名从 yaml 来,**构建期就完全已知** —— 没有理由推迟到运行时再生成检查
- 构建期展开能做 argcheck 做不到的事:直接生成定参函数(不用 `...`/`select`)、
  把类型检查内联、把 **1-based -> 0-based 的 axis 转换**和检查合并成一次
- 运行时 `loadstring` 还有附带成本:2000 个算子 = 2000 次 `loadstring` + 编译,拖慢 `require`,
  与 `plan/api/README.md` §2.5 的惰性加载目标冲突

**但 argcheck 的 usage/help 渲染(`usage.lua`)要复用** —— 让生成的算子和手写 API
报同一种形状的错误信息,比省那 151 行更有价值。

### 4.6 落地形态

```
lua/paddle/_vendor/argcheck/    【P4】vendored,842 行纯 Lua,BSD-3
  ├── init.lua  graph.lua  usage.lua  utils.lua  doc.lua  dump.lua
  └── env.lua                 ← 删掉 26-52 行的 torch 分支
lua/paddle/_argcheck_env.lua   【P5】我们的 istype:认 Tensor/Parameter/Layer/insight.Array
```

改动清单(**每一条都要在 vendored 目录里留 `-- PADDLE-LUA PATCH:` 注释**,
否则下次同步上游时会被无声覆盖):

| # | 文件 | 改什么 | 理由 |
|---|---|---|---|
| ① | `graph.lua:13-21` | `table2id`/`func2id` 改弱键计数器 | §4.3,**不改在 Windows 上不能用** |
| ② | `env.lua:26-52` | 删掉 torch 分支 | §4.2 |
| ③ | `usage.lua:83` | 去掉 sundown/torch,自己剥 markdown 标记 | 否则错误信息里有字面 `**` |
| ④ | `graph.lua` 生成模板 | `require "argcheck.usage"` 提到 upvalue | §4.4,白捡 193 ns |
| ⑤ | `test/test.lua` | `string.format('%s', nil)` -> `tostring(nil)` | 测试自身的 5.1 不兼容 |

### 4.7 三个必须验的点

**① `debug` 库是硬依赖。** `utils.lua` 全靠 `debug.getupvalue` / `debug.setupvalue` 注入 upvalue。
沙箱化的宿主(游戏引擎、把 `debug` 摘掉的嵌入环境)里 argcheck **直接不可用**。
我们的目标人群包含嵌入式宿主 -> **`_wrap` 那条路必须一直保留可用,不能全仓库改成 argcheck。**
这条和 §4.1 的分层是同一个结论的两个理由。

**② `loadstring` vs `load`。** `init.lua:9` 是 `local loadstring = loadstring or load`,5.1/5.2+ 都覆盖。
但 **LuaJIT + `debug.setupvalue` 的组合要单独验**(M0 新增一项)。

**③ 与 `pl.class` 的交互。** `nn.Layer` 的实例 raw 表是空的(§2 / 09-nn §3.2),
`env.istype` 若用 `rawget(getmetatable(o), '__typename')`,拿到的是**类表**而不是实例表 —— 这恰好是对的,
但**必须写测试钉死**,因为它看起来像是碰巧成立的。

---

## 5. 对硬约束的影响

~~`CLAUDE.md` §9 原文:「引入新的强制 C 依赖(除 sol2 / Lanes)」-> 禁止。~~
🔄 **该禁令已于 2026-08-03 由人取消(R23)。判据改为「边际成本」,见 `CLAUDE.md` §9.1。**

本决策在**新旧两套判据下都成立**:
- Penlight 的受限子集是**纯 Lua**,vendored,零 C 依赖 —— 旧判据下合规
- Insight7 是**软强制**,且不进核心路径 —— 它带来的价值(API 本就是 Paddle 命名)
  远大于一个 rock 的构建成本,新判据下划算

`CLAUDE.md` 新增了一条约束 **C11**,把「基座只有这一套」写死,
防止后来者再引入第二套 class 或第二个 list 实现。
**注意 C11 与被取消的 §9 是两回事:C11 管的是「不要有两套做同一件事的东西」,
不是「不要有依赖」。取消 §9 不动摇 C11。**

---

## 6. 落到别处的改动清单

| 文档 | 改什么 |
|---|---|
| `CLAUDE.md` §1 | 新增 C11:类系统用 `pl.class`,集合用 `pl.List`,兼容层用 `pl.compat` |
| `CLAUDE.md` §2 | 新增 D23-D26 |
| `CLAUDE.md` §9 | ~~补注:Penlight 受限子集是纯 Lua 例外~~ -> **整条禁令删除(R23)**,改为 §9.1 的边际成本判据 |
| `plan/modules/09-nn.md` §3.1 | 翻案为 `pl.class` + `_create` / `_class_init` |
| `plan/modules/09-nn.md` §3.2 | `FIELDS` 的装配点改到 `_create` |
| `plan/modules/07-serialization.md` | manifest 读取改用 `pl.pretty.read` |
| `plan/modules/11-io.md` / `12-dataset-vision.md` | 返回 `List`;Insight7 预处理路径 |
| `plan/modules/04-packaging.md` | vendor Penlight;rockspec 声明 Insight7 |
| `process/conventions.md` | 基座使用规则 + 「与 Python 的已知差异」表 |
| `process/open-questions.md` | Q-12 / Q-13 / Q-14 / Q-15 |
| `process/decisions.md` | R18-R21 翻案与决策;~~待拍板 P1 升级为阻塞~~ -> **P1 已拍板:改 Insight7(R24)** |
| `process/decisions.md` | R22(`LayerList` 继承 `Layer`)、R23(C 依赖禁令取消)、R24(Insight7 axis 按 bug 修)|
| `process/open-questions.md` | Q-12 转「已决待实施」;Q-08 风险下降;新增 Q-16(Lua 5.2 的 `ipairs`)|
| **`plan/api/README.md` §2.1** | 关键字参数那一行改成**分层**:冷路径 argcheck / 热路径 `_wrap`(R25) |
| **`plan/layout.md`** | `_vendor/argcheck/` 进目录树与模块清单;`_argcheck_env.lua` 归 P5 |
| **`plan/ci.md` §6** | 新增红线:vendored argcheck 的 5 处改动必须带 `-- PADDLE-LUA PATCH:`;`lua/paddle/_ops/` 里出现 `argcheck` 即失败 |
| **`process/status.md` §4** | M0 新增一项:LuaJIT 上 `debug.setupvalue` 注入 upvalue 是否可用 |
| **`process/decisions.md`** | R25(argcheck 分层采纳)+ §2.11 |

---

## 7. 未解问题

| # | 问题 | 阻塞谁 |
|---|---|---|
| ~~Q-12~~ | ✅ **已决(2026-08-03):改 Insight7,axis 也 1-based。** 顺手修,P12 之前完成 | 不再阻塞,转为待实施 |
| Q-16 | Lua 5.2 的 `ipairs` 是否尊重 `__index` | 不阻塞,只影响测试断言 |
| Q-13 | 能否实现持有 `insight::Array` 的 `phi::TensorBase` 子类以做零拷贝 | 只影响吞吐,不阻塞拷贝版 |
| Q-14 | `insight::Array` 拷贝构造是否共享缓冲 + 原子计数(能否 `__lanesclone`) | P13 的 Insight7 预处理路径 |
| Q-15 | vendored Penlight 与用户系统 Penlight 并存时两份 `List` 不互认,是否需要提供桥 | 用户体验,不阻塞 |
| **Q-17** | **LuaJIT 上 `debug.setupvalue` 能否注入 upvalue**(argcheck 的命脉,§4.7②) | **P5 起的所有冷路径 API;挂了就全线退回 `_wrap`** |
| **Q-18** | 「冷/热路径」的边界要不要机器判定,还是靠人在 code review 里认 | 不阻塞,但不定会慢慢烂掉 |

---

## 8. 验收

- [ ] 11 个 Penlight 文件 vendored 进 `lua/paddle/_vendor/pl/`,CI 校验其 sha256 与上游一致
- [ ] CI 有 grep 规则:命中 `pl.path` / `pl.dir` / `pl.file` / `pl.app` / `class.cast` 即失败
- [ ] `nn.Layer` 的自动注册在 `pl.class` 的**三层继承**(Layer -> MyBlock -> MyNet)上验证通过
- [ ] `print(layer)` 在 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT 上都不崩(坑④)
- [ ] `paddle.from_insight` / `to_insight` 往返 dtype 与数值一致(先拷贝版即可)
- [ ] Q-12 有明确结论后才允许在教程里出现 `ins.sum(x, axis)` 一类写法
- [ ] **vendored argcheck 在 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT × Linux / macOS / Windows 上跑通上游全套 `test/test.lua`**
      (Windows 那一格是 §4.3 的回归测试,**不能只测 Linux**)
- [ ] **`argcheck{{name="x",type="number",default=0}}` 在 Windows 上不崩** —— 单独立一条,因为这就是上游坏掉的那个点
- [ ] **`grep -rn "argcheck" lua/paddle/_ops/` 无输出**(生成代码不许用,§4.5)
- [ ] **vendored argcheck 与上游的 diff 逐行可解释**,每处改动带 `-- PADDLE-LUA PATCH:` 注释
- [ ] **`env.istype` 能正确识别 `nn.Layer` 实例**(raw 表为空,§4.7③)
