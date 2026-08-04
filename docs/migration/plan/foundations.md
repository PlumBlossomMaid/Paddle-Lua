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

### 1.3 ✅ 已决:声明 rock 依赖,**不 vendor**(R30,推翻本节原方案)

> **2026-08-03 人拍板(原待拍板 P9):**
> 「**pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西。**」

~~原方案:vendor 那 11 个文件到 `lua/paddle/_vendor/pl/`。~~ **已推翻。**

| 方案 | 判决 |
|---|---|
| **rockspec 里写 `dependencies = {"penlight"}`** | ✅ **选这个**,生态内所有仓库统一 |
| ~~vendor 11 个文件~~ | ❌ 见下 |

**为什么改:** vendor 的四条理由里,**第 1 条本来就作废了**(R23 取消禁 C 依赖),
**第 3、4 条是便利性**,而**第 2 条(版本锁定)不需要靠 vendor 达成** ——
rockspec 锁 minor + CI 跑一组 `pl.class` 语义测试,同样挡得住 `_create` 语义漂移,
而且升级路径是显式的(改一行版本号),不是"改 5374 行 vendored 代码"。

**真正决定性的是原方案自己写在下面的那条代价:**
vendor 之后用户环境里若已有系统 Penlight,`paddle.pl.List` 与 `require "pl.List"`
**是两个类,实例互不 `is_a`**。原文的处理是"写进用户文档:跨库传裸表"。
但一旦 Penlight 成为**整个生态的地基**(paddle-lua / Insight7 / `argsig` / metrics / ocean 都用),
这个坑就从"一条文档注意事项"变成"每两个库之间都有一次"。
**声明依赖让它彻底消失** —— 全生态只有一份 `pl.class`,`is_a` 处处成立。

**连带:**

| | |
|---|---|
| `lua/paddle/_vendor/pl/` | **删除**(`layout.md` / `04-packaging.md` / `overview.md` 同步) |
| `luafilesystem` | 进入依赖闭包(Penlight 的 rockspec 声明了它)。可接受:R23/D28 已取消禁令,且 `lfs` 是 luarocks 上可移植性最好的 rock 之一 |
| **我们自己不 `require "lfs"`** | 文件系统仍走 `paddle.utils.fs`(C++17 `std::filesystem`)。`lfs` 只是被 Penlight 拖进来的传递依赖,**不是我们的 API 面** |
| 版本锁定 | 从"锁 commit + 校验 sha256"改为**rockspec 锁 minor + CI 语义测试**(待拍板 P7 相应改写) |
| `paddle.pl` | 仍然暴露,但它现在就是 `require "pl"` 的那一份,**不再是副本** |
| CI | 新增红线:`grep -rn "_vendor/pl" -> 失败`(全生态只许有一份 Penlight) |

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

### 3.6.1 `DType` / `Place` 是两边共有的类型

Insight7 的 API 本来就是照着 Paddle 设计的(§3.2:dtype 常量与 Place 构造器已是 Paddle 命名),
所以 `DType` / `Place` **不是撞名,是同一个概念**。参数签名层里用 `extend` 把两边的谓词取或,
不逼着共有概念起两个名字(`plan/argsig.md` §2.2)。

⬜ 更远的方向:它们最好在 C 层就是**同一个值类型** —— 零拷贝互操作(§3.3)本来就要求
内存布局对齐,dtype 却还要在边界上转换一次,是不必要的。属于 Insight7 侧的长期演进。

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

## 4. 参数检查:**取 argcheck 的形,不取它的实**

> 基座的第四块。前三块是「类/集合」「numpy」,这一块是「函数签名」。
>
> ⚠️ **本节于 2026-08-03 内部翻案一次(R25 -> R26)。**
> 初版结论是「vendored argcheck,冷路径用」。**该结论错了** ——
> 补测发现 argcheck **编不出 `Conv2D` / `Adam` / `DataLoader` 的签名**,
> 而这三个恰恰就是「冷路径」的全部内容。翻案过程保留在 §4.5,因为
> **这个错误的形状值得记住:性能数据看着难看但可接受,把人的注意力吸引走了,
> 而真正的否决理由是「能不能表达」,一个我当时根本没去测的维度。**

### 4.1 结论

| | 结论 |
|---|---|
| **vendored argcheck 本体** | ❌ **不用**(理由 §4.5,是硬性的「编不出来」,不是性能取舍) |
| **argcheck 的规则表 schema** | ✅ **照抄**:`{name=, type=, default=, defaulta=, defaultf=, opt=, check=, help=}` |
| **argcheck 的 `usage.lua`(151 行)** | ✅ **移植**:错误信息与 `help` 的渲染,这部分设计是好的 |
| **代码生成这个机制本身** | ✅ **保留** —— 它不是问题所在(§4.4) |
| **3^N 的组合枚举** | ❌ **丢掉** —— 它才是问题所在(§4.5) |
| 落地 | `lua/paddle/_args.lua`,约 150 行,**我们自己写**(§4.7) |

**一句话:argcheck 的声明式接口是对的,它的求解器是错的。**

### 4.2 先澄清:「硬编码了 Torch7」这个前提不成立

实测(commit `b3b32c0`,2016-06-29,842 行 / 7 文件),Torch 耦合只有**两处、共 32 行**:

| 位置 | 内容 | 性质 |
|---|---|---|
| `env.lua:26-52` | `if pcall(require, 'torch') then` 覆写 `env.istype` / `env.type` | **它覆盖的那个默认实现本来就是通用的** —— 读 `getmetatable(o).__typename`,回退 `type(o)`。两个函数上方原注释写着 `-- user configurable function` |
| `usage.lua:83` | 有 `sundown` + torch + tty 时把 markdown 渲染成 ANSI 颜色 | 纯装饰 |

**扩展点是作者留好的,不是要凿开的。** 这条记下来是因为:
它说明我们移植 `usage.lua` 时**不会被 Torch 拖住**。

### 4.3 也澄清:它不是「过时」,是「在 Windows 上从来没能用」

`graph.lua:13-21`:

```lua
local function table2id(tbl)
   -- DEBUG: gros hack de misere        ← 作者自己写的:「悲惨的脏 hack」
   return tostring(tbl):match('0x([^%s]+)')
end
```

拿对象地址当生成代码里的唯一标识。而 `tostring(t)` 的格式来自 C 的 `%p`:

| 平台 | `tostring({})` | `match('0x…')` |
|---|---|---|
| glibc / macOS | `table: 0x7f9c...` | 命中 |
| **MSVC(Windows 上的 Lua/LuaJIT 官方构建)** | `table: 00B62A68` | **nil** |

后果:**任何一条规则带 `default` / `opt` 的 argcheck,在 Windows 上直接抛
`graph.lua:281: bad argument #4 to 'format' (string expected, got nil)`。**
不带默认值的最简用法反而正常 —— 所以它表现为「时好时坏」。

实测(本机 `lua5.1`,MSVC 构建,**未安装 Torch**):改掉这 2 行之后,
上游全套 `test/test.lua` **全部通过**。

> 另有 1 处是**测试自身**的 5.1 不兼容:`string.format('%s', nil)` 在 5.3+ 合法、5.1 报错。

**十年没人修,是因为 Torch7 只跑 Linux/macOS。** 这条对我们的意义不是「能不能修」
(能,2 行),而是**它是一个维护状态的读数**:最后一次提交 2016 年,
一个让整个 Windows 平台不可用的 bug 十年无人发现。**这个库没有在被使用。**

### 4.4 代码生成机制本身没问题 —— 但它只特化了一半

**它是干嘛用的:** 朴素做法是**解释式**的 —— 把规则表留到运行时,每次调用
循环遍历规则、查 `rule.type`、比对 `type(arg)`、套默认值。这是「每参数每调用
若干次表查找 + 一个循环」。

argcheck 把这件事**在定义期算完**,产出一段直线 Lua 源码,`loadstring` 编译,
再用 `debug.setupvalue` 把默认值和 `istype` 作为 upvalue 注入进去
(所以生成的 chunk 里不做任何全局查找)。运行时规则表已经不存在了。

**这个技术是对的,我们要留下。** 问题在于它**特化了控制流,却没特化类型判断**:

实测 3 参数、30 万次:

| | ns/call |
|---|---|
| 裸 Lua 函数调用 | 43 |
| **argcheck** | **2597** |

归因:

| 来源 | 代价 | 为什么 |
|---|---|---|
| `env.istype` 被调 **6 次** | 6 × 300 ns = **1800 ns** | 生成的代码里写的是 `istype(x, "number")` —— 一次经 upvalue 的函数调用 + `getmetatable` + `rawget` + `type`。**如果都要生成代码了,最该内联的就是这一句**:`type(x) == "number"` |
| 函数体里有 `local usage = require "argcheck.usage"` | 193 ns | **每次调用都 `require` 一遍**,应该提到 upvalue |
| `select(n, ...)` 链 | 107 ns × n | 架构决定,见下 |

**70% 的开销来自「生成了代码却还在调解释期的函数」。** 这不是代码生成太重,
是代码生成做得不够彻底。

### 4.5 ★ 致命伤:3^N 的组合枚举 —— 它编不出 `Conv2D`

这才是否决理由。

`init.lua:24-51`:每条规则有三种状态(**给了 / 没给 / 显式给 nil**),
于是枚举 **3^N** 种组合,每种作为一条路径插进决策树,再把整棵树摊成嵌套 if。

实测(1 个必填 + 其余带默认值):

| 参数个数 | 组合数 | 生成代码 | 构建耗时 |
|---|---|---|---|
| 6 | 243 | 50 KB | 9 ms |
| 8 | 2,187 | 267 KB | 89 ms |
| 10 | 19,683 | 1.37 MB | 840 ms |
| **11** | **59,049** | 3.1 MB | ❌ **`control structure too long`(Lua 编译器跳转偏移上限)** |

**上限是 9 个可选参数。** 而 Paddle 的签名是这样的:

| 类 | 参数 | 可选 | 3^可选 | argcheck |
|---|---|---|---|---|
| `Linear` | 5 | 3 | 27 | ✅ |
| `Dropout` | 5 | 5 | 243 | ✅ |
| `MultiHeadAttention` | 8 | 6 | 729 | ✅ |
| `LayerNorm` | 9 | 8 | 6,561 | ⚠️ 0.3 s / 0.6 MB |
| `Momentum` | 10 | 10 | 59,049 | ❌ **编不出** |
| **`Conv2D` / `Conv3D`** | 14 | 11 | 177,147 | ❌ **编不出** |
| `Embedding` | 13 | 11 | 177,147 | ❌ **编不出** |
| `Adam` | 12 | 12 | 531,441 | ❌ **编不出** |
| `LSTM` | 15 | 13 | 1,594,323 | ❌ **编不出** |
| `AdamW` | 15 | 15 | 14,348,907 | ❌ **编不出** |
| **`DataLoader`** | 17 | 16 | **43,046,721** | ❌ **直接挂死,不是报错** |

**初版结论说「冷路径(构造期 API)用 argcheck」。冷路径就是这张表。
也就是说那条建议在它自己举的每一个例子上都不成立。**

逃生舱也没有:`noordered=true`(只允许具名调用)只是把常数压小,
指数没变 —— N=12 时仍要 1.6 MB / 5.1 s,N=17 一样挂死。

**而最讽刺的是:这个指数完全是为一个我们不想要的功能付的。**

3^N 里那个「没给」的状态,是为了支持**位置参数的中间省略**:
`f(1, "x")` 里的 `"x"` 到底是第 2 个参数还是第 3 个?argcheck 靠类型去猜,
所以要把所有可能的「哪些位置被跳过」组合都枚举出来。

- **尾部省略**(`f(1,2)` 后面全默认)只需要 **N+1** 种情况,不是 3^N
- **中间省略**才是 3^N 的来源
- **而 Python 根本不允许位置参数中间省略** —— 想跳过就必须用关键字

我们的用户是从 Paddle Python 过来的。**我们在为一个 Python 里非法、
用户不会写、写了也该报错的调用形式,付指数级的代价。**

### 4.6 去掉枚举之后:同样的机制,好一个数量级

保留「定义期生成源码 + loadstring + upvalue 注入」,只把 3^N 枚举换成
「尾部省略 + 具名表」两条 O(N) 的直线代码。原型实测:

| | argcheck | 原型 |
|---|---|---|
| 3 参数,位置调用 | 2597 ns | **403 ns** |
| 3 参数,具名表 | 2733 ns | **367 ns** |
| 11 参数(`Conv2D`) | ❌ 编不出 | 2.3 KB,**< 1 ms** |
| 17 参数(`DataLoader`) | ❌ 挂死 | 3.6 KB,**< 1 ms** |
| 17 参数的具名表调用 | — | 803 ns |
| 30 参数 | ❌ | 6.4 KB,2 ms |

**代码量从指数变成线性,速度快 6-7 倍,而且能表达真实签名。**
这两件事是同一个原因:不枚举了。

原型约 40 行生成器。加上 `defaulta` / `defaultf` / `check` / 自定义 istype /
usage 渲染,预计 **~150 行**。

### 4.7 落地形态

```
lua/paddle/_args.lua        【P5】~150 行:规则表 -> 生成源码 -> loadstring
                                   schema 与 argcheck 兼容;usage 渲染移植自
                                   argcheck/usage.lua(BSD-3,保留版权头)
lua/paddle/_wrap.lua        【P5】保留。移植自 Insight7,不依赖 debug 库,
                                   作为 `_args` 的兜底(见下)
```

**为什么 `_wrap` 必须保留:** `_args` 和 argcheck 一样依赖
`debug.getupvalue` / `debug.setupvalue` 与 `loadstring`。
把 `debug` 摘掉或禁掉 `load` 的嵌入宿主(游戏引擎、沙箱)里这条路直接不通。
`_args` 在检测不到 `debug.setupvalue` 时**降级为解释式实现**(慢,但能跑),
这是它比 argcheck 多出来的一件事。

**与 P3 的关系:** 生成的 2000+ 算子**不在运行时调 `_args`** ——
它们的签名构建期就完全已知,由 `tools/gen/` 直接展开成定参函数,
顺带把 1-based -> 0-based 的 axis 转换合并进去(`§4.5` 的教训:能在构建期
算完的事不要推到运行时)。**但两边共用同一份规则表 schema 和同一套错误信息格式** ——
这是「基座只有一套」(C11)在参数检查上的落点。

### 4.8 重载:**用 `if`,和 Python 一样**

> ⚠️ 本小节两次修正。§4.5 说「我们不需要类型重载」——过头了;
> 随后改成「显式列举变体 `args.overload{}`」——**也错了,而且错得更微妙。**

**先看事实:Paddle 确实做类型分派,但它写在函数体里:**

| API | 分派 | 上游 |
|---|---|---|
| `paddle.to_tensor(data)` | scalar / ndarray / Tensor / list | `creation.py:794-798` 一串 `isinstance` |
| `paddle.reshape(x, shape)` | list / `Variable` / `pir.Value` | `manipulation.py:1056,1105` |
| `optimizer(parameters=)` | 参数列表 / 参数组 dict 列表 | 函数体内判断 |

**再看「显式列举变体」错在哪:** 上面这些函数的候选变体,**参数列表是一样的**。
`to_tensor` 无论 `data` 是 number 还是 table,签名都是 `(data, dtype?)`。
把它写成三个变体,是在用重载机制表达一件不是重载的事 ——
**真实情况是「一个参数的类型是联合类型」+「函数体内行为不同」。**

#### 所以分两层,各归各

| 层 | 谁负责 | 形式 |
|---|---|---|
| **参数是否合法**(类型、默认值、缺失) | `_args` 的规则表 | `type = {"number", "table", "paddle.Tensor"}` —— **联合类型,一条规则** |
| **拿到之后怎么做**(行为分支) | **函数体里的 `if`** | 直接照抄上游的 `isinstance` 链 |

```lua
paddle.to_tensor = args.check{
  {name="data",  type={"number","table","paddle.Tensor","insight.Array"}},
  {name="dtype", type="string", opt=true},
}(function(data, dtype)
  if is_tensor(data)  then return data:astype(dtype)     -- 对应上游 isinstance(data, Tensor)
  elseif is_array(data) then return from_insight(data, dtype)
  elseif type(data) == "number" then return scalar_tensor(data, dtype)
  else return from_table(data, dtype) end
end)
```

#### 为什么这个比重载引擎好

1. **它是直译。** 上游是 `if isinstance(...)`,我们就是 `if is_...(...)`。
   这属于 A1 类「语义可以直抄」—— 项目的核心方法就是逐行翻译,
   **在这里发明一套声明式重载,等于给一段本来能直抄的代码加了一层要自己维护的抽象**
2. **`_args` 少一个机制。** 不用写变体匹配、不用设计「所有变体都不匹配」时的错误信息 ——
   而那个错误信息恰恰很难写好(argcheck 那个「列出所有 3^N 分支」的 usage 就没人看得懂)
3. **错误信息更好。** 手写 `if` 的 `else` 分支可以说
   「`to_tensor` 的 data 必须是 number/table/Tensor,你给的是 function」,
   比通用的「没有匹配的签名」有用
4. **Python 本来就没有真重载**,所以 Paddle 里**不存在**参数列表不同的情况。
   Torch7 那种 `torch.rand([gen,] sizes)`(前置可选参数)才需要引擎,
   **而 Python 的语法根本写不出那种签名** —— 这也正是 3^N 在今天没有客户的根本原因

#### 一句话

**argcheck 那个时代:函数小,库替你穷举组合。**
**关键字参数时代:参数多,组合穷举不起 —— 而且也不需要,因为分派本来就该是作者写的 `if`。**

**要换掉的只是求解器,声明式规则表这个接口形式仍然成立。**
`_args` 只做一件事:**验证并规格化一组参数**。分派是控制流,不是签名的事。

---

### 4.9 必须验的三点

**① `debug.setupvalue` 在 LuaJIT 上给 `load` 出来的函数注入 upvalue 是否可用**(Q-17,M0 #22)。
挂了就全线走解释式降级路径。**纯 Lua,不需要 libpaddle,必须在 P5 之前有答案。**

**② 与 `pl.class` 的交互。** `nn.Layer` 的实例 raw 表是空的(§2 / 09-nn §3.2),
自定义 `istype` 若用 `rawget(getmetatable(o), '__typename')` 拿到的是**类表**而非实例表 ——
这恰好是对的,但**必须写测试钉死**,因为它看起来像是碰巧成立的。

**③ 生成的 chunk 名与错误信息。** argcheck 的报错是 `[string "argcheck"]:75:` ——
用户看到一个不存在的文件和一个对不上的行号。
我们生成时要把 chunkname 设成 `@paddle.nn.Conv2D` 一类**能指回真实 API 的名字**。

---

## 5. 基座 vs 外围:哪些必须我们自己拥有

> 到这里基座已经有三块了:`pl.class`(类)、Insight7(numpy)、`_args`(argcheck)。
> 值得停下来问一句:**还要几块?边界在哪?**

### 5.1 先量一下:框架本身其实只依赖两个外部东西

扫 `python/paddle/vision/**` + `python/paddle/io/**` 的全部 import,
去掉标准库和 paddle 自身:

| 第三方 | 出现次数 |
|---|---|
| `numpy` | 30 |
| `PIL` | 11 |
| (没有别的) | — |

**框架本体的外部生态面就这么大。** 其余全部是标准库 ——
而标准库的缺口正是 Penlight 在补的那一块。

### 5.2 关键区分:基座是「形状适配器」,外围是「功能」

三块基座有一个共同点,而且这个共同点才是它们成为基座的原因:

**它们都不计算任何东西。**

| | 它决定的是 | 换掉它会怎样 |
|---|---|---|
| `pl.class` | 每个类**怎么写** | 全仓库的 `_init` / `super` / 继承写法要改 |
| Insight7 | 每个数组**怎么写** | 每个 `ins.*` 调用点、每个 dtype 常量要改 |
| `_args` | 每个函数**怎么声明** | 每个公开 API 的签名要重写 |

**它们决定代码长什么样。** 所以必须**在 P5 之前定死** —— 之后再换就是全仓库重写。

对照,外围:

| | 它决定的是 | 换掉它会怎样 |
|---|---|---|
| 图像解码(PIL 的位置) | 你**能不能**读 JPEG | 换一个解码库,改 `_decode` 一个文件 |
| HTTP 下载 | 你能不能自动下数据集 | 同上 |
| 进度条 / 日志 | 好不好看 | 改一处 |

**外围决定你能做什么,随时可加,不影响任何已写的代码。**

> **判据:换掉它要改几处?一处 = 外围。全仓库 = 基座。**
> 基座在 P5 前定死,外围什么时候补都行。

### 5.3 纪律:基座要尽可能少,而且要尽可能不是我们的

「设计一套生态」是个**没有边界**的目标,而这个项目会死在别的地方 ——
`WITH_PYTHON=OFF` 编不出来(G0)就全盘作废,和生态多完整无关。

而且每多拥有一块,就多一份**永久**的维护面:**5 Lua 版本 × 3 OS,长期**。
所以顺序是:

1. **能用现成 rock 就用**(Penlight 就是这么来的 —— 它是 Lua 生态的通用库,不是我们的)
2. **能塞进已有的 `.so` 就塞**(边际成本 ≈ 0,`CLAUDE.md` §9.1)
3. **实在没有才自己写,而且写完要能独立于 paddle-lua 存在**(见 §5.4)

**三块基座已经到顶了。** 再想加第四块,先回答「换掉它要改几处」——
答案不是「全仓库」就不是基座,不进这一层。

### 5.4 ✅ 已决:参数解析器独立成项目,paddle-lua 与 Insight7 共用

> **2026-08-03 人拍板(原待拍板 P8):**
> 「Insight 的那个解析器还是太简陋了,我打算弄一个单独的解析器项目,然后给 paddle 和 Insight 用。」
> 记为 **R27 / D31**。`plan/layout.md` 里的 `lua/paddle/_args.lua` 相应改为**外部依赖**。

理由(§5.2 的判据直接给出答案):参数解析器属于**基座** ——
换掉它要改全仓库。而基座的作用域**本来就该是跨仓库的**:
Insight7 的 dtype 常量和 Place 构造器已经是 Paddle 命名(§3.2),
接口层对齐在这个生态里是既定做法。否则用户会在同一个脚本里
面对两套规则表方言和两种报错格式。

#### 5.4.1 范围必须现在钉死 —— 两个消费者的库最容易长歪

**它做三件事,不多不少:**

| ✅ 在范围内 | ❌ 明确不做 |
|---|---|
| 规则表 -> 校验并规格化一组实参 | 类系统(那是 `pl.class`) |
| `usage` / `help` 渲染(移植自 argcheck `usage.lua`) | 集合类型(那是 `pl.List`) |
| **自定义类型的注册接口**(宿主注入 `istype`) | 类型分派 / 重载(那是函数体里的 `if`,§4.8) |
| 无 `debug` 库时降级为解释式实现 | 序列化、日志、任何与参数无关的东西 |

**「解析器」这个名字有歧义,要防住:** 它解析的是**实参列表**,
不是字符串、不是配置文件、不是 Lua 源码。
`x["1:3, :"]` 那个字符串索引的解析**不在这里**(它属于 paddle-lua 的 `slice.lua`,
因为语义是张量的,不是通用的)。

#### 5.4.2 依赖策略(2026-08-03 人的两条追加约束)

> 「新时代 argcheck **不能有任何的别的框架哪怕是 insight 和 paddle 的硬编码**,
> 最好是**纯 lua**,**支持扩展类型**什么的,最好是**基于 penlight** 的。
> 然后也**不能叫 argcheck 了**,名字不能在 luarocks 上面装了(= 名字不能已被占用)。
> **一定要考虑大量参数如 50 个函数参数、类构造函数参数的情况。**」
>
> 「其实**最好别轻易造轮子**,如果有优秀的类似于 argcheck 的现代库也是可以接受的,
> 而且**别太难用**,比如那种**要颠覆我们的语法**的,我们肯定也不会用。」

| 约束 | 落地 | 展开在 |
|---|---|---|
| 零框架硬编码 | 自定义类型**全部走注册接口**,库里搜不到 `paddle` / `insight` 字样 | 下面 |
| 纯 Lua | 5.1 语法子集,5.1/5.2/5.3/5.4/LuaJIT 全覆盖;**不引入 C 模块** | 下面 |
| 基于 Penlight | ⚠️ 与 §1.3「vendor 不走 rock 依赖」冲突,**新增待拍板 P9** | 下面 |
| 别轻易造轮子 | 先做普查,再决定写多少 | **§5.4.5** |
| 50 个参数 | Lua 自身有三道墙,其中一道 argcheck 也撞 | **§5.4.6** |
| 换个名字 | luarocks 可用性实测 | **§5.4.7** |

**零框架硬编码怎么做到:**

```lua
args.register_type("paddle.Tensor",  function(o) return ... end)
args.register_type("insight.Array",  function(o) return ... end)
```

这正是 argcheck `env.istype` 那个形状(§4.2),它的设计对,我们照抄。
库自己只认:Lua 基本类型、`pl.class` 的 `is_a`、以及注册进来的东西。

**「基于 Penlight」值得做,因为它正好覆盖这个库最脆的地方:**

| 用到 | 干什么 | 省掉的坑 |
|---|---|---|
| `pl.compat` | `load` / `loadstring` / `setfenv` / `unpack` 的跨版本差异 | 这库的命脉就是 `loadstring`(M0 #23) |
| `pl.pretty.write` | usage 文本里渲染默认值(表、函数) | argcheck `usage.lua` 里那段自写序列化 |
| `pl.utils.pack/unpack` | 带 `nil` 的变长实参 | `select('#', ...)` 的一堆边界 |
| `pl.class` 的 `is_a` | 识别 Penlight 类实例 | 生态里的类**本来就是** `pl.class`(D25) |

⚠️ **但它带来一个必须先解决的冲突,记为待拍板 P9:**
Penlight 的 rockspec 声明 `dependencies = {"lua >= 5.1", "luafilesystem"}`
(已核实,`penlight-dev-1.rockspec:31-34`)。而 §1.3 决定 paddle-lua **vendor**
Penlight 正是为了避开 `lfs`。若解析器声明 `penlight` 依赖而 paddle-lua 继续 vendor,
**生态里就有两份 Penlight —— 正好违反 C11。** 三个出路:

| | 方案 | 代价 |
|---|---|---|
| **A(建议)** | 解析器声明 `penlight` 依赖;**paddle-lua 也改成声明依赖,不再 vendor** | 多一个 `lfs`(C 模块)。但 R23/D28 已取消「不引入 C 依赖」禁令,且 `lfs` 是 luarocks 上可移植性最好的 rock 之一 |
| B | 解析器只**兼容** Penlight:有就用,没有就走自带的 20 行兜底 | 运行时分支 = 两条路径都要测,且违反「基座只有一套」的精神 |
| C | 解析器依赖 rock,paddle-lua 继续 vendor | ❌ 两份 Penlight,违反 C11 |

**在 P9 拍板前,按 A 写。** 理由:vendor 的三条理由(§1.3)里有两条是冲着 `lfs` 去的,
而那两条成立的前提(禁 C 依赖)已经在 R23 被人自己取消了。

#### 5.4.3 两个消费者带来的三个新问题

**① 版本兼容一破就是两个库一起破。**
规则表 schema **就是**它的 API 契约。
-> 锁版本到 minor(`>= 0.2, < 0.3`),schema 变更走废弃流程。
-> **schema 与 argcheck 保持兼容**(`name/type/default/defaulta/defaultf/opt/check/help`),
   这样既有知识可迁移,也给了一个天然的稳定锚点。

**② 排期变成外部依赖。** paddle-lua 的 **P5 需要它** ——
它必须在 P5 之前可用。好消息是它**不依赖 G0**(纯 Lua),
可以和 M0 完全并行,甚至现在就能写。
-> 工程树里挂成 **1.7**,排在 M0 内部但标注「与 1.1-1.6 无依赖关系」。

**③ 谁先用它,谁就替另一个把坑踩了。**
建议 **Insight7 先接** —— 它体量小、`_wrap` 已经在那里,替换是可度量的;
而 paddle-lua 还没有一行代码,拿它当第一个白鼠等于同时调试两个未知数。

#### 5.4.4 对本仓库的影响

| 之前 | 现在 |
|---|---|
| `lua/paddle/_args.lua`(~150 行,我们的文件) | **外部 rock**,rockspec 声明依赖 |
| `_wrap.lua` 作为无 `debug` 库的兜底 | **降级路径归解析器自己管**,paddle-lua 不再需要 `_wrap` |
| C11 的作用域 = 本仓库 | **C11 的作用域 = 整个生态** |

⚠️ **`_wrap.lua` 从 `layout.md` 的 P5 清单里删掉** ——
留着它就是第二套参数处理,正好违反 C11。

#### 5.4.5 ★ 先普查再动手:现成的轮子有,但**不是这个形状的轮子**

「别轻易造轮子」是对的,所以先把 luarocks 上的候选全过了一遍
(2026-08-03 实测,`luarocks.org/search`):

| 库 | 许可 / 依赖 | 位置调用 | **具名表调用** | **默认值** | **usage 渲染** | 自定义类型 | 50 参数 | 语法侵入 |
|---|---|---|---|---|---|---|---|---|
| **Torch `argcheck`** `b3b32c0` (2016) | BSD-3 / 无 | ✅ | ✅ | ✅ | ✅ | ✅ `env.istype` | ❌ **10 个就编不出**(3^N,§4.5) | 低 |
| **`tableshape`** (leafo) | MIT / **仅 `lua>=5.1`** | ❌ | ✅ | ⚠️ 靠 transform | ❌ | ✅ 类型对象**本身可调用** | ✅(自带 `tableshape.codegen`) | 中(`+`/`*` 操作符 DSL,可不用) |
| **`typecheck`** (gvvaughan) | MIT / `std._debug` | ✅ | ❌ | ❌ | ❌ 只有错误消息 | ⚠️ | ✅ | 中(`argscheck "f (int, ?table) => int" .. function ...`) |
| **`checks`** (fab13n) | MIT / **C 模块** | ✅ | ❌ | ❌ | ❌ | ✅ 全局 `checkers` 表 | ✅ | 低 |
| **`ltypekit`** (daelvn) | / Moonscript 味 | ✅ | ❌ | ❌ | ❌ | ✅ | — | ❌ **`number -> number` 柯里化签名 = 颠覆语法** |
| `geoffleyland/argcheck` | MIT | ✅ | ❌ | ❌ | ❌ | ❌ | — | ❌ 规格写在**注释**里,且要 `lua -largcheck` 预加载;作者自称 proof of concept |
| `lua-livr` / `valua` / `lua-schema-validation` | — | — | — | — | — | — | — | 面向 **web 表单/数据输入**,不是函数签名,不在同一问题域 |

**结论:没有任何一个提供「同一份签名同时吃位置调用和具名表调用 + 默认值 + usage」。**
这不是巧合 —— 那四件事凑在一起是 **Python kwargs 的形状**,
而这些库是给 Lua 自己的 API 写的,Lua 自己没有 kwargs。
我们要移植的恰恰是一套 kwargs API(`Conv2D(in_channels=3, ...)`),
所以缺的那块必须自己写。**它大约 200 行,而且没人写过。**

**但「类型判定」这块有轮子,不要自己写。** 做法是**把 `type` 槽定义成可调用契约**:

```lua
{name="x", type="paddle.Tensor"}                    -- 字符串 -> 查注册表(同 checks 的 checkers)
{name="x", type={"number","table"}}                 -- 数组 -> 联合类型
{name="x", type=function(o) return ... end}         -- 任意谓词
{name="x", type=ts.array_of(ts.number)}             -- ★ tableshape 的类型对象**本身就是 callable**,直接塞进来就能用
```

于是 `tableshape` 是**可选增强**而不是依赖:要复杂结构校验的人自己
`luarocks install tableshape`,我们一行代码都不用为它写。
`checks` / `typecheck` 的谓词同理(包一层 `function(o) ... end`)。

> **判据(§5.2 的同一把尺子):** 借来的是**功能**(类型判定),自己拥有的是**形状适配器**(调用约定)。
> 这与「基座要尽可能不是我们的」(§5.3)一致 —— 我们只拥有那块必须由我们拥有的。

#### 5.4.6 ★ 50 个参数:Lua 自身有三道墙,argcheck 撞了两道

先确认这不是假想需求。Paddle Python API 的实测分布(`ast` 扫 `python/paddle/**`):

| | 函数 | 参数总数 / 其中可选 |
|---|---|---|
| 最大 | `ResNetBasicBlock.__init__` `python/paddle/incubate/xpu/resnet_block.py:434` | **43 / 33** |
| 次大 | `block_multihead_attention_xpu` `.../block_multihead_attention.py:430` | 38 / 25 |
| | `fused_multi_transformer` `.../fused_transformer.py:1057` | 34 / 21 |
| 分布 | ≥30 参数 **9 个**;≥20 参数 **18 个**;≥12 参数 **156 个** | |

**所以「50 个参数」是对的量级,而 argcheck 在 10 个可选参数就编不出来了(§4.5)。**
3^33 这个数字不需要评论。

**Lua 5.1 自身的三道墙(本机实测,`lua5.1.exe`):**

| 墙 | 上限 | 撞上时的报错 |
|---|---|---|
| 一个函数的 upvalue 数 | **60** | `function at line N has more than 60 upvalues` |
| 一个函数的局部变量数 | **200** | `has more than 200 local variables` |
| 寄存器 / 表达式复杂度 | **N=122**(见下表) | `function or expression too complex` |

**第一道墙 argcheck 也撞:** 它是**每条规则若干个 upvalue**
(`init.lua:22` 收集 `upvalues` 表,`init.lua:115-116` 逐个 `setupvalue`)。
43 个参数 ×(类型谓词 + 默认值)≈ 86 个 upvalue > 60,**即使没有 3^N 也编不出来**。

**由此定死三条生成规则:**

1. **所有 per-rule 数据放进一个表 upvalue** —— 生成函数的 upvalue 数是**常数 2**(`R`, `err`),与 N 无关。代价是每次检查多一次 `R[i]` 表索引
2. **每条规则最多一个局部变量**(`a1..aN`),不生成临时局部
3. **N > 100 切换到「表形态」**:实参存进一个表 `a[i]`,不再一规则一局部 —— 实测 **N=1000 仍能编译**(139 KB)

**原型实测(Lua 5.1,上述规则 1+2 的形态):**

| N | 生成字节数 | 编译耗时 | 每次调用 |
|---|---|---|---|
| 3 | 609 B | <1 ms | **350 ns** |
| 17 | 2 587 B | <1 ms | 1.5 µs |
| **50** | **7 405 B** | **~1 ms** | **3.75 µs** |
| 80 | 11 785 B | <1 ms | 5.85 µs |
| 120 | 17 856 B | ~1 ms | 8.75 µs |
| **123** | — | ❌ `function or expression too complex` | — |
| 表形态 50 / 150 / 300 / 1000 | 6.7 KB / 20 KB / 41 KB / 139 KB | 全部通过 | (未测) |

调用耗时含测试循环里 `unpack(args)` 的开销,增量约 **73 ns/参数,线性**。
对照:argcheck 3 个参数 2597 ns,9 个可选参数生成 1.37 MB / 耗时 840 ms(§4.5-4.6)。

> ⚠️ **这三道墙要写进 CI**(`plan/ci.md` 红线①b 已有生成体积回归,再加一条):
> 用 43 参数(Paddle 的真实最大签名)做基准,**编不出来即失败**。

#### 5.4.7 名字:不能叫 argcheck,且 luarocks 上不能已被占用

`luarocks.org/search` 实测(2026-08-03,子串匹配):

| 候选 | 结果 |
|---|---|
| `argcheck` | ❌ **已被占用**(`geoffleyland/argcheck`;Torch 的那个另算) |
| `signature` / `atrium` / `portico` / `usher` | ❌ 有同名或近名 rock |
| **`argsig`** | ✅ **空的** |
| `callsig` / `sigcheck` / `arglet` / `argrule` | ✅ 空的 |
| `signet` / `concierge` / `warden` / `clerk` / `lintel` / `sieve` | ✅ 空的 |

**建议 `argsig`**(argument signature):
- 说清了它是什么 —— **签名**层,不是"校验器"也不是"解析器"(§5.4.1 提过后者有歧义)
- 与 argcheck 的知识可迁移关系一眼可见,又不冒名
- 短,`local args = require "argsig"` 读起来正常
- 不含任何框架名(C11 / 零硬编码)

⬜ **最终定名由人拍板;定名当天在 luarocks 上占位,避免被抢。**

---

## 6. 对硬约束的影响

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

## 7. 落到别处的改动清单

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
| **`plan/layout.md`** | `_args.lua` 归 P5(**不再是 `_vendor/argcheck/`**,R26)|
| **`plan/ci.md` §6** | 新增红线:`lua/paddle/_ops/` 里出现 `_args` 即失败(生成算子构建期展开);`_args.lua` 的生成代码量必须 O(N) —— 加一条 30 参数的回归 |
| **`process/status.md` §4** | M0 新增一项:LuaJIT 上 `debug.setupvalue` 注入 upvalue 是否可用 |
| **`process/decisions.md`** | ~~R25(vendored argcheck,冷路径)~~ -> **R26(不 vendor,取 schema + usage,自写 O(N) 生成器)** + §2.11 |

---

## 8. 未解问题

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

## 9. 验收

- [ ] 11 个 Penlight 文件 vendored 进 `lua/paddle/_vendor/pl/`,CI 校验其 sha256 与上游一致
- [ ] CI 有 grep 规则:命中 `pl.path` / `pl.dir` / `pl.file` / `pl.app` / `class.cast` 即失败
- [ ] `nn.Layer` 的自动注册在 `pl.class` 的**三层继承**(Layer -> MyBlock -> MyNet)上验证通过
- [ ] `print(layer)` 在 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT 上都不崩(坑④)
- [ ] `paddle.from_insight` / `to_insight` 往返 dtype 与数值一致(先拷贝版即可)
- [ ] Q-12 有明确结论后才允许在教程里出现 `ins.sum(x, axis)` 一类写法
- [ ] **`_args.lua` 能表达 `DataLoader`(17 参 / 16 可选)与 `Conv2D`(14 参 / 11 可选)**
      —— 这是 §4.5 的回归测试,**argcheck 就是死在这里的**
- [ ] **生成代码量对参数个数是线性的**:30 参数 < 10 KB、构建 < 10 ms(指数回归会立刻在这条上暴露)
- [ ] `_args` 在 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT × Linux / macOS / Windows 上行为一致
      (**Windows 那一格不能省** —— argcheck 正是死在只测 Linux 上,§4.3)
- [ ] **摘掉 `debug` 库后 `_args` 降级为解释式实现且仍然正确**(§4.7)
- [ ] **`grep -rn "_args" lua/paddle/_ops/` 无输出**(生成算子构建期展开,§4.7)
- [ ] **`istype` 能正确识别 `nn.Layer` 实例**(raw 表为空,§4.8②)
- [ ] **报错信息里的 chunkname 指回真实 API 名**,不是 `[string "argcheck"]`(§4.8③)
