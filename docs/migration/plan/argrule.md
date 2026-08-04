# `argrule` —— 新时代的 argcheck(孵化说明书)

> **一句话:对着 argcheck 重写一个,继承它的**规则表**与**usage**,
> 丢掉它的**组合枚举**与**重载引擎**,把**类型判定**开放成可插拔契约。**
>
> ✅ **名字已拍板(2026-08-03):`argrule`**(P10 关闭)。luarocks 上实测未被占用,**定名当天占位**。
> **本文件是孵化说明书** —— 独立仓库建起来之后整体迁走,这里只留一行指针。
> 选型过程与实测数据不在这里,在 `plan/foundations.md` §4(为什么不 vendor argcheck)
> 与 §5.4(为什么独立成项目、轮子普查、三道墙、命名)。

---

## 0. 定位

| | |
|---|---|
| 它是什么 | 一个**函数签名层**:一份规则表,同时支持位置调用与具名表调用,负责校验、填默认值、渲染 usage |
| 它不是什么 | 不是类型系统、不是结构校验器(那是 `tableshape`)、不是 CLI 参数解析(那是 `argparse`)、不是字符串索引解析 |
| 消费者 | paddle-lua、Insight7,以及任何想要 kwargs 风格 API 的 Lua 库 |
| 许可 | 新代码 MIT;`usage` 渲染部分移植自 Torch argcheck,保留其 **BSD-3** 头 |
| 目标版本 | Lua 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT,纯 Lua |
| 依赖 | **Penlight**(`pl.compat` / `pl.pretty` / `pl.utils` / `pl.class`)—— 全生态地基级默认依赖,**不 vendor**(R30/D34)。除此之外零依赖 |
| 体量估计 | 生成器 ~200 行 + usage ~120 行 + 降级路径 ~80 行 ≈ **400 行** |

**为什么必须自己写**:普查了 luarocks 上全部候选,没有一个提供
「同一份签名同时吃位置调用与具名表调用 + 默认值 + usage」——
那是 **Python kwargs 的形状**,而 Lua 自己没有 kwargs
(`plan/foundations.md` §5.4.5 有完整对比表)。

---

## 1. 对着 argcheck 的三列账

### 1.1 继承(照抄,不重新设计)

| 继承什么 | 出处 | 为什么 |
|---|---|---|
| **规则表 schema** `{name, type, doc, default, defaulta, defaultf, opt, check}` | `init.lua:82-96` | 十年验证过的字段划分,既有知识可迁移;这也是我们与 argcheck 的兼容锚点。**只保留 `doc`,不要 `help`** —— argcheck 两个都有还要 assert「二选一」(`init.lua:80`),那是历史包袱;`doc` 更短,且写 `doc` 的规则表在 argcheck 里也能跑 |
| **表级选项** `doc` / `noordered` / `nonamed` / `quiet` | `init.lua:74-80` | 同上。`noordered` / `nonamed` 是**解歧义的逃生舱**,见 §2.6 |
| **`usage` 两段式渲染**:一行签名 `generateargp` + 对齐的参数表 `generateargt` | `usage.lua:9-60` | 这是 argcheck 最好的部分,报错时直接把整份 usage 打出来 |
| **默认值三态**:`default`(常量)/ `defaulta`(取另一个参数的值)/ `defaultf`(函数求值) | `init.lua:87-88` | `defaulta` 正好对上 Paddle 里大量的 `if b is None: b = a` |
| **生成源码 -> `loadstring` -> 注入 upvalue** 的机制 | `init.lua:105-116` | 机制本身没问题,问题在它特化了什么(`foundations.md` §4.4) |
| **类型判定是注册点,不是硬编码** | `env.lua:4` 原注释 `-- user configurable function` | 它的设计对 |

### 1.2 丢弃(明确不要)

| 丢弃什么 | 出处 | 为什么 |
|---|---|---|
| **3^N 组合枚举** | `init.lua:24-51`、`graph.lua` 全部 | 9 个可选参数 = 1.37 MB / 840 ms,10 个编不出。Paddle 最大签名 43 个参数 |
| **`chain` / `overload` 重载引擎** | `init.lua:13-21, 78-79` | 类型分派用函数体里的 `if`,直译上游的 `isinstance` 链(`foundations.md` §4.8) |
| **位置参数中间省略、靠类型猜** | 上面那个指数就是为它付的 | Python 根本不允许这么写。要跳过参数就用具名表 |
| **`tostring(t):match('0x…')` 当标识符** | `graph.lua:13-21` | MSVC 的 `%p` 不带 `0x` -> Windows 上必崩(`foundations.md` §4.3)。用显式分配的自增 id |
| **每条规则若干个 upvalue** | `init.lua:22, 115-116` | Lua 5.1 只有 60 个 upvalue,43 参数 ≈ 86 个,**即使没有 3^N 也编不出来** |
| **`sundown.ascii` 的 markdown 渲染** | `usage.lua:2-5` | 一个可选的外部渲染器,不属于签名层 |
| **`rule.type` 只能是字符串** | `init.lua:83` | 见下 |

### 1.3 新增

| 新增什么 | 为什么 |
|---|---|
| **`type` 是可调用契约**(字符串 / 数组=联合 / 任意谓词 / 任何 callable 类型对象) | 让 `tableshape` 的类型对象**直接可用**,类型判定层就不必自己写(D32) |
| **单表 upvalue**,生成函数的 upvalue 数恒为 2 | 绕开 60 upvalue 墙(D33) |
| **N > 100 切「表形态」** | 绕开 122 的寄存器墙;实测 1000 参数仍可编译 |
| **无 `debug` / 无 `loadstring` 时的解释式降级** | LuaJIT 沙箱、`-DLUA_NODEBUG` 之类的环境 |
| **报错指向调用点**,不是库内部 | `checks` 做得对的地方:`debug.getinfo(2)` 拿调用者的 `file:line` |
| **容器协议分 plain / opaque** | 逐元素访问贵不贵是**语义相关**的:对 GPU 上的 Tensor 逐元素查类型 = n 次设备同步。选路机械,且算出来的行为与上游一字不差(§2.3) |
| **类型层只查「消歧需要的」** | 其余检查下放到转换层(反正要遍历一次)。判据机械:去掉它之后调用仍唯一 -> 可下放(§2.3) |
| **`pl.compat` / `pl.pretty` 复用** | 跨版本 `load`/`unpack`/`setfenv` 与默认值渲染,不自己写(R30:Penlight 是地基级依赖) |

---

## 2. API 面(最小,先冻这个)

```lua
local rule   = require "argrule.rule"        -- 主角:定义签名。一个词,天天用
local method = require "argrule.method"      -- 同上,自动吃掉 self
local argrule = require "argrule"            -- 管理面:注册类型、内省。少用
```

**模块分工**:热路径的 API 是两个装饰器,`require` 出来正好叫 `rule` / `method`,
读起来就是自然语言。根模块 `argrule` 再导出一遍,外加
`register` / `alias` / `usage` / `source` / `strict`。

**规则表自己也遵守它自己的调用约定** —— `name` / `type` 可以位置写:

```lua
{"x", "Tensor"}                    -- 等价于 {name = "x", type = "Tensor"}
{"axis", "number", default = -1}   -- 混写:前两个位置,其余具名
{name = "axis", type = "number"}   -- 全具名。**推荐在公开 API 里这么写**
```

⚠️ **位置槽只到 2(`name`、`type`),再往后必须具名。**
`{"x", "number", 1}` 里的 `1` 是 `default` 还是 `doc`?—— 不留这个问题。

**推荐 vs 允许**:内部代码、原型、一次性脚本用短形式;
**paddle-lua / Insight7 的公开 API 一律全具名** —— 那些规则表要拿来生成文档(Q-18),
而且会被人当范例抄。这条写进 `process/conventions.md`,**不进 CI**(短形式本身没错)。

### 2.1 完整签名

```lua
rule{
  -- 规则,按位置顺序
  {"x",    "paddle.Tensor", doc = "input"},
  {"axis", "number", default = -1, doc = "1-based; -1 = last dim"},
  -- 表级选项
  doc       = "Softmax along an axis.",
  nonamed   = false,   -- 禁用 f{...} 的两种表形式(见 §2.6)
  noordered = false,   -- 禁用 f(a, b) 位置形式
  quiet     = false,   -- 不抛错,返回 nil, err
}(fn)                  -- fn 可以是任何函数,包括直接来自 C 侧的
```

| 规则字段 | 含义 |
|---|---|
| `name`(位置 1) | 参数名。具名调用的键,usage 里的显示名 |
| `type`(位置 2) | **可调用契约**:字符串(查注册表)/ 数组(联合)/ 谓词函数 / 任何 callable 类型对象 |
| `doc` | 一行说明。进 usage,也进将来生成的 API 文档 |
| `default` | 常量默认值 |
| `defaulta` | 取另一个参数的值(直译 `if b is None: b = a`) |
| `defaultf` | 函数,每次调用求值 |
| `opt` | 可省且**没有**默认值(省了就是 `nil`) |
| `check` | 额外的取值校验(范围、枚举),类型之外的那一层 |

**`name` 和 `type` 都可以省**:`{"x"}` = 只要有这个位置,不管类型;
`{name = "opts", opt = true}` = 什么都收。但**公开 API 不要这么写**。

### 2.2 类型短名与注册表冲突

`{"x", "Tensor"}` 里的 `"Tensor"` 是**宿主注册进来的**,库自己不认识任何框架类型:

```lua
argrule.register("paddle.Tensor", function(o) return paddle.is_tensor(o) end)  -- 全名,规范
argrule.alias("Tensor", "paddle.Tensor")                                       -- 短名,给人用
```

⚠️ **短名会撞** —— 一个进程里 paddle 和 Insight7 同时在。规则:
**全名是规范,短名是 alias,重复注册同一个短名直接报错**(不是后者覆盖前者)。

**但有些类型是两边共有的,不是撞名,是同一个概念:**

| 短名 | 谁提供 | 怎么办 |
|---|---|---|
| `Tensor` | 只有 paddle | `register` |
| `Array` | 只有 Insight7 | `register` |
| **`DType` / `Place`** | **两边都有** —— Insight7 的 API 本来就是照着 Paddle 设计的(§3.2:dtype 常量与 Place 构造器已是 Paddle 命名) | **`extend`,谓词取或** |

```lua
-- 谁先加载谁 register,后来者 extend,不报错
argrule.register("DType", paddle.is_dtype)
argrule.extend  ("DType", insight.is_dtype)      -- 现在两边的 dtype 都过
```

`register` 撞名报错,`extend` 是显式的"我知道这个类型已存在,我要往里加一支"。
两者区分开,既挡住了意外覆盖,又不逼着共有概念去起两个名字。

> ⬜ **更远的方向(记一笔,不在本项目范围内):**
> `DType` / `Place` 最好在 C 层就是**同一个值类型**,而不是两个 `.so` 各造一份 ——
> paddle-lua 与 Insight7 的零拷贝互操作(§3.3)本来就要求内存布局对齐,
> dtype 却还要在边界上做一次转换,是不必要的。这属于 Insight7 侧的长期演进。

**短名不是随便起的,它必须等于用户能直接 local 化的那个导出名:**

```lua
local Tensor = paddle.Tensor          -- 对应 Python 的 from paddle import Tensor
local Array  = insight.Array
```

Python 侧 `from paddle import Tensor` 之后代码里直接写 `Tensor`,很顺手;
Lua 的等价写法就是上面这一行。既然用户会这么写,
规则表里的 `type = "Tensor"` 就该是**同一个标识符** ——
否则会出现"代码里叫 `Tensor`,规则表里叫别的"这种两套词汇。
这条同时写进 `api/README.md` §2.1 的命名映射表。

---

### 2.3 容器类型:判据是「是个容器 + 装的是整数」

⛔ **不要把它写成三个类的联合。**

```lua
-- ❌ 错:把判据写成「是这三个之一」
type = {"table", "pl.List", "insight.Array"}

-- ✅ 对:判据是结构性的
type = "IntList"     -- 是个容器,且装的是整数
```

> 人的原话:「**反正需要是个容器**」「**而且装的是整数**」「**Tensor 也应该满足容器协议,
> 毕竟只要是个 int 容器按说都应该行**」。这三句就是完整定义 ——
> 那几个类只是它今天已知的实例,不是它的定义。
> 写成联合类型,等于把「容器」硬编码成一张框架名单:
> 用户自己的 `Vector` 类、别人的 `array` rock、明天多出来的第四种容器,全被挡在外面,
> 而它们本来完全合法。这也正是 §4「零框架硬编码」要挡的东西。

**✅ 上游核实过,`Tensor` 确实算**(`CLAUDE.md` §4 的出处要求):

| 事实 | 出处 |
|---|---|
| `ShapeLike = Sequence[int \| Tensor \| None] \| Tensor` —— **里面没有 `int`** | `python/paddle/_typing/shape.py:22-33` |
| "If `shape` is a Tensor, it should be a **1-D Tensor** which represents a list." | `python/paddle/tensor/creation.py:1832` |
| "If `shape` is a list or tuple, each element should be **integer or 0-D Tensor**" | 同上 `:1831` |
| `concat` 的 `x` 是 `Sequence[Tensor]`,文档写 "``x`` is a Tensor **list or tuple**" —— **不收单个 Tensor** | `python/paddle/tensor/manipulation.py:1482, 1507` |

#### 容器协议(库只认协议,不认类名)

容器分两级,**区别不是"谁写的",是"逐元素访问贵不贵"**:

| 级别 | 谁 | 长度 | 取元素 | 自报 dtype |
|---|---|---|---|---|
| **plain** | 裸 table、`pl.List`、任何纯 Lua 序列 | `#o` | `o[i]` 便宜 | 没有 |
| **opaque** | `insight.Array`、**`paddle.Tensor`** | `#o` / `:len()` / `:size()` | `o[i]` **贵**(可能触发设备同步) | `o:dtype()` + `o:ndim()` |

长度探测顺序固定:`#o` 可用 -> 用它;否则 `o:len()`;否则 `o:size()`。取元素一律 `o[i]`,**1-based**。

> ⚠️ `#o` 在 Lua 5.1 上对 **table** 的 `__len` 无效(5.1 只对 userdata 认 `__len`)。
> 而 `Tensor` / `Array` 都是 userdata,**它们反而没这个问题** —— 有问题的是 `pl.List`
> 这类带元表的 table(靠数组部分,`#` 照样对)。探测顺序必须有 fallback,进 M0 #24 实测。

#### 元素怎么查:**类型层只查消歧需要的,剩下的下放**

> 人的提议:「**我想的是在函数里面有个 if 判断,如果有小数直接 error 抛出异常**」。
> **对,而且比放进类型层更好** —— 但要分两种情况,分界线是机械的:

```
某个元素检查,去掉它之后调用仍然只有唯一解释?
    是  -> 下放到转换层(反正要遍历,顺手查,零额外成本)
    否  -> 必须留在类型层(解析器要靠它选解释)
```

| 类型 | 类型层查什么 | 元素检查在哪 | 为什么 |
|---|---|---|---|
| **`IntList`** | **只查「是不是容器」**(opaque 容器另看 `ndim==1` + dtype,O(1)) | **转换层的 `if`,`error` 抛出** | 去掉元素检查,`zeros{2,3}` 仍然唯一 —— ② 里 `2` 本来就不是容器 |
| **`TensorList`** | 容器 **+ 逐元素是不是 Tensor** | 类型层 | 去掉就有歧义:`concat{{a,b},2}` 的 ③ 正是靠「元素 `2` 不是 Tensor」出局(⑧) |

**为什么 `IntList` 的元素检查放转换层更好,而不只是"也行":**

1. **零额外成本。** table -> `std::vector<int64_t>` 那一步**本来就要遍历 n 次**,
   顺手判一下是不是整数是免费的;放类型层就是**同一个表扫两遍**
2. **0-D Tensor 元素只有那一层处理得了。** 上游允许元素是 0-D Tensor(`creation.py:1831`),
   取值要 device->host;类型层要么重复这套逻辑,要么假装看不见
3. **报错反而更准** —— 转换层知道下标:

```
shape[3] must be an integer, got 2.5
  in paddle.zeros, called from train.lua:42
```

⚠️ **两条硬要求,否则这个下放就变成放水:**
- **必须 `error`,不许静默取整。** `math.floor(2.5)` 式的宽容会造出静默错误的 shape
- **必须指向调用点**(`debug.getinfo` 拿调用者的 `file:line`),不是报在绑定层内部
- C++ 侧的检查不得让异常穿过 Lua(C7),走 C ABI 的 status 码转 `lua_error`

#### opaque 容器:只有 O(1) 一条路

```
list_of(E) 判一个 opaque 容器 o(Tensor / Array):
  E 是标量类型(Int / number / …)?
      是 -> 用 o 自报的 dtype + ndim == 1 证明,**不碰元素**      O(1)
      否 -> 没有 O(1) 证明,且**不许逐元素** -> 判否
```

⛔ **绝不对 opaque 容器逐元素访问。** 这不是性能优化,是红线:
`t[i]` 对一个 GPU 上的 Tensor 是一次 device->host 同步,n 个元素就是 n 次 ——
**类型检查触发设备同步**,用户永远查不出自己的训练为什么慢。

#### 一般形式:`list_of(elem)`

```lua
-- 库提供的组合子(库自己不认识 int,也不认识 Tensor)
argrule.list_of(elem)          -- 容器 + 每个元素满足 elem

-- 宿主注册别名
argrule.register("Int",        -- ⚠️ 不是 Lua 的 integer:上游允许元素是 0-D 整数 Tensor
                 function(o) return is_lua_int(o) or is_0d_int_tensor(o) end)
argrule.register("IntList",    argrule.list_of("Int"))
argrule.register("TensorList", argrule.list_of("Tensor"))   -- concat / stack 的第一个参数
```

`list_of` 里没有一个字是框架相关的:它只做「探容器协议 + 逐元素套 `elem`(或走 O(1) 证明)」,
`elem` 本身又是一个可调用契约(§2.1),所以 `list_of(list_of("Int"))` 这种嵌套是免费的。

**为什么值得单起一个名字,而不是每处都展开写判据:**

1. 2000+ 个生成算子里 `shape` / `axes` / `perm` / `strides` 反复出现,
   一份定义改一处 vs 改两千处
2. **报错信息**:`IntList expected, got number` 比一串联合类型可读
3. 容器协议的探测顺序、O(1)/O(n) 的选路只写一遍,不会有的地方查有的地方不查

| ✅ 接受 | ❌ 不接受 |
|---|---|
| 整数的裸 table:`{2, 3}` | 有小数:`{2.5}` —— **在转换层报错,不在类型层** |
| 整数的 `pl.List`:`List{2, 3}` | 二维 / 非整数 dtype 的 Array 或 Tensor |
| 整数 dtype 的**一维** `insight.Array` / **`paddle.Tensor`** | 裸数字 —— **没有出路,报错并建议 `{n}` 写法**,§2.4 |
| 元素是 0-D 整数 Tensor 的 table(上游明说) | |
| **任何满足容器协议且装整数的第三方类型** | 空容器?—— 见 §6 未决 |

> **顺带一个好性质:`pl.List` / `insight.Array` / `Tensor` 都带元表**,
> 而 §2.6 的表形式判定要求 `getmetatable(t) == nil`。
> 所以 `paddle.zeros(List{2,3})` 永远不会被误认成"具名表调用" —— **带元表的实参天然免疫歧义**。

---

### 2.4 `zeros(2, 3)`:❌ **不支持,连装饰器也不做**

> 这一节记录一个提出又砍掉的东西,留着是为了不被第二次提出来(§7.1 保留翻案痕迹)。
>
> - 我上一轮临时提了个表级选项 `sizeargs = "shape"`(**之前从没讨论过**)
> - 人问「之前有讨论过吗」-> 我降级成宿主侧 ~10 行装饰器
> - 人再问「**用得少的东西,为了 API 简洁是不是就该去掉**」-> ✅ **对,整个去掉**

⚠️ 先纠正我更早的一个错误断言:「`paddle.zeros(5)` 本来就是错的」在当前 develop 上**不成立**,
上游确实支持 `ones(1,2,3)` / `ones(5)` / `ones(size=[1,2,3])`
(`python/paddle/utils/decorator_utils.py:406-437`,用在 5 个函数上)。
**它合法。我们仍然不做。**

**决定性的理由:上游加这个糖的原因,在 Lua 侧不存在。**

| | Python | Lua |
|---|---|---|
| 规范写法 | `paddle.zeros([2, 3])` | `paddle.zeros{2, 3}` |
| 糖 | `paddle.zeros(2, 3)` | —— **不需要,上面那行已经一样短** |

Python 需要这个糖,是因为 `([...])` 又是括号又是方括号,而且 torch 用户手上有存量代码。
Lua 的表调用**本来就省掉了括号**,`zeros{2,3}` 与 `zeros(2,3)` 一样长 ——
**糖的全部收益是 0,成本却照付**:

1. 同一件事**两种写法**,文档要写两遍,示例要选一种(C11 的精神:基座只有一套)
2. 制造一个反直觉的坑:`zeros(2, "int64")` **不是**「shape=2, dtype=int64」而是错的
   (上游行为:第一个位置实参是 int 就把**全部**位置实参当 shape)
3. 多一条解析分支要测,而它只服务 5 个函数

**代替方案:把它变成一次性的教学,而不是一个入口。**

`IntList` 参数收到裸数字时,报错里带上写法建议(只在错误路径上,零运行时成本):

```
paddle.zeros(2, 3)
-> shape must be a container of integers, got number
     did you mean:  paddle.zeros{2, 3}
     at train.lua:42
```

**教一次,好过永久养一个二义入口。**

## 2.5 用户端长什么样(按候选名 `argrule` 写,定名后全局替换)

> `rule{...}` 吃一张规则表、返回一个装饰器,装饰器吃**任何**函数 ——
> 包括直接来自 C 侧的 `_C_ops.softmax`,不需要包一层 lambda。
> 这个形状是从 argcheck 继承的(它那张表在源码里就叫 `rules`),名字正好把 schema 写在了外面。

### ① 最简:一份规则表,三种调用方式

```lua
local rule = require "argrule.rule"

-- 短形式:原型和内部代码这么写
softmax = rule{ {"x", "Tensor"}, {"axis", "number"}, doc = "blah blah" }(_C_ops.softmax)

-- 全具名:公开 API 这么写
paddle.nn.functional.softmax = rule{
  {name = "x",    type = "paddle.Tensor", doc = "input"},
  {name = "axis", type = "number", default = -1, doc = "1-based; -1 = last dim"},
  doc = "Softmax along an axis.",
}(_C_ops.softmax)

softmax(t)                    -- 位置
softmax(t, 2)
softmax{x = t, axis = 2}      -- 具名表
softmax{t, 2}                 -- 表内位置
```

传错时报错指向**调用点**,并把整份 usage 打出来(argcheck 的做法):

```
train.lua:42: softmax: bad argument #2 'axis' (number expected, got string)

usage: softmax(
   x    = paddle.Tensor   -- input
  [axis = number]         -- 1-based; -1 = last dim  [default=-1]
)
  Softmax along an axis.
```

### ② 默认值三态:常量 / 取另一个参数 / 惰性求值

```lua
paddle.nn.functional.conv2d = rule{
  {name = "x",        type = "paddle.Tensor"},
  {name = "weight",   type = "paddle.Tensor"},
  {name = "bias",     type = "paddle.Tensor", opt = true},        -- 可空,且没有默认值
  {name = "stride",   type = {"number", "IntList"}, default = 1},  -- 常量
  {name = "padding",  type = {"number", "IntList", "string"}, default = 0},
  {name = "dilation", type = {"number", "IntList"}, defaulta = "stride"}, -- 直译 `if d is None: d = stride`
  {name = "groups",   type = "number", default = 1},
  {name = "place",    type = "paddle.Place",
                      defaultf = function() return paddle.get_device() end},  -- 每次调用才求值
}(function(x, weight, bias, stride, padding, dilation, groups, place)
  ...
end)
```

`opt` 与 `default` 的区别是**有没有值**,不是"能不能省" —— 两者都能省。

### ③ 类构造函数:`method` 自动吃掉 `self`

```lua
local Conv2D = paddle.nn.Layer:subclass "Conv2D"

Conv2D.__init = method{
  {name = "in_channels",  type = "number"},
  {name = "out_channels", type = "number"},
  {name = "kernel_size",  type = {"number", "IntList"}},
  {name = "stride",       type = {"number", "IntList"}, default = 1},
  {name = "padding",      type = {"number", "IntList", "string"}, default = 0},
  {name = "dilation",     type = {"number", "IntList"}, default = 1},
  {name = "groups",       type = "number", default = 1},
  {name = "padding_mode", type = "string", default = "zeros"},
  {name = "weight_attr",  type = "table", opt = true},
  {name = "bias_attr",    type = {"table", "boolean"}, opt = true},
  {name = "data_format",  type = "string", default = "NCHW"},
  doc = "2D convolution layer.",
}(function(self, in_channels, out_channels, kernel_size, stride, padding, ...)
  self.weight = self:create_parameter{...}
end)

local m = Conv2D{ in_channels = 3, out_channels = 64, kernel_size = 3, padding = "same" }
```

**11 个参数、8 个可选 —— 这正是 argcheck 编不出来的那张表**(3^11 = 177147 条路径,
`foundations.md` §4.5)。这里是 11 行 `if`,生成的源码约 1.6 KB。

### ④ 类型分派在函数体里,用 `if`,和 Python 一样

```lua
paddle.to_tensor = rule{
  {name = "data", type = {"number", "boolean", "table",
                          "paddle.Tensor", "insight.Array"}},   -- 联合类型
  {name = "dtype", type = "string", opt = true},
  {name = "stop_gradient", type = "boolean", default = true},
}(function(data, dtype, stop_gradient)
  if paddle.is_tensor(data)       then return data:astype(dtype)
  elseif insight.is_array(data)   then return paddle.from_insight(data, dtype)
  elseif type(data) == "number"   then return paddle._scalar(data, dtype)
  else                                 return paddle._from_table(data, dtype) end
end)
```

签名层只回答"**允许什么进来**",不回答"**进来之后走哪条路**"。
后者是函数体的事 —— 直译上游的 `isinstance` 链(D30 / `foundations.md` §4.8)。

### ⑤ 宿主注册自定义类型(库里零框架硬编码)

```lua
-- lua/paddle/init.lua,全进程一次
argrule.register("paddle.Tensor", function(o) return paddle.is_tensor(o) end)
argrule.register("paddle.Place",  function(o) return paddle.is_place(o)  end)

-- Insight7 那边同理
argrule.register("insight.Array", function(o) return insight.is_array(o) end)
```

`pl.class` 的类**不用注册** —— 库直接认 `is_a`(生态里的类本来就是 `pl.class`,D25)。

### ⑥ 想要结构校验?把 `tableshape` 的类型对象塞进 `type`

```lua
local ts = require("tableshape").types

paddle.optimizer.Adam = method{
  {name = "learning_rate", type = {"number", "paddle.lr.LRScheduler"}, default = 0.001},
  {name = "betas", type = ts.array_of(ts.number):length(2), default = {0.9, 0.999}},
}(function(self, lr, betas) ... end)
```

`tableshape` 的类型对象本身就是 callable,**我们一行代码都不为它写**(D32)。
它是可选增强,不是依赖。

### ⑦ 内省

```lua
print(argrule.usage(softmax))    -- 那份 usage 文本
print(argrule.source(softmax))   -- 生成的 Lua 源码,排错用(对应 argcheck 的 rules.debug)
argrule.strict(false)            -- 全局关校验(只对此后定义的函数生效)
```

### ⑧ ⚠️ 第一个参数就是 table:解析顺序是**定死的**,不是猜的

这是这套调用约定唯一的真歧义,踩中的正好是最常用的那批函数:

```lua
paddle.zeros{2, 3}      -- shape = {2,3}?还是「表内位置」调用 shape=2, dtype=3?
```

**解析顺序(生成的代码里就是这个顺序,四步,确定性):**

```
n == 1 且实参是无元表的 table 时:
  ① 表里有任何一个键等于某条规则的 name    -> 具名模式
  ② 否则,试「表内位置」:t[1], t[2], …
       全部类型通过 **且必填参数都有值**    -> 表内位置模式
  ③ 否则,且规则 #1 的 type 接受 table
       **且其余必填参数都有默认值**         -> 整个表就是第一个实参
  ④ 否则                                    -> 按表内位置报错(错得最像用户的本意)
```

⚠️ **一个解释「成立」= 类型全过 + 必填参数全有值。** 只查类型是不够的 ——
少了后半句会凭空造出一堆假歧义。

**`paddle.zeros{2, 3}` 因此是确定的:** ② 里 `shape = 2` 过不了 `IntList`(数字不是容器,
§2.3),`dtype = 3` 也过不了枚举检查 -> 落到 ③ -> `shape = {2, 3}`。**不需要写任何标注。**
(而 `zeros(2, 3)` 这种写法我们**根本不支持**,§2.4 —— 少一条分支就少一处歧义。)

> 这一步能成立,靠的是 `api/README.md` §2.1.1 那条:
> **枚举参数不接受裸数字**。如果 `dtype` 能收 `3`,②就通过了,歧义就真的存在。
> **一条 API 约定消掉了一整类调用歧义** —— 这两条要一起改,不能只动一边。

**先看一个「看着像歧义、其实不是」的例子** —— `paddle.full(shape, fill_value)`:

```lua
paddle.full{ {2, 3}, 1.0 }
-- ② shape={2,3} ✓  fill_value=1.0 ✓                     -> 成立
-- ③ shape={{2,3},1.0} ✓ 是 table,但 **fill_value 没人给了**
--    它是必填、无默认                                     -> 不成立
-- 结论:②,确定的,不用标注
```

**必填参数是天然的消歧器。** `full` 的第二个参数没有默认值,
所以「整个表当 shape」这条路本身就走不通。

**②③ 真的同时成立,才是真歧义** —— 加上 §2.3 的**元素逐个查**之后,
这个条件苛刻到近乎不存在了。先看两个「以为是歧义、其实不是」的:

```lua
paddle.concat = rule{
  {name = "x",    type = "TensorList"},        -- = list_of("Tensor"),§2.3
  {name = "axis", type = "number", default = 1},
}(_C_ops.concat)

paddle.concat{a, b}
-- ② x = a          -> a 是 Tensor,不是 TensorList         ✗
-- ③ x = {a, b}     -> 每个元素都是 Tensor                  ✓   -> 整表当 x

paddle.concat{ {a, b}, 2 }
-- ② x = {a,b} ✓, axis = 2 ✓                                ✓   -> 表内位置
-- ③ x = {{a,b}, 2} -> 元素 2 是数字,不是 Tensor           ✗
```

**两种写法都是确定的,`concat` 不需要 `nonamed`。**
`stack` / `meshgrid` / `broadcast_tensors` 同理。

> ⚠️ **我上一版在这里写错了**,说这批「第一个参数是列表」的函数**必须**写 `nonamed = true`。
> 那是把 `type = "table"` 当判据算出来的 —— 只查容器不查元素,②③ 才会同时成立。
> 换成 `list_of(elem)` 之后,**元素类型自己就是消歧器**。
> 教训与「必填参数是天然消歧器」是同一条:**判据越弱,假歧义越多。**

**那什么时候才是真歧义?** 要让 ②③ 同时成立,需要
`t[1]` 既满足规则 #1(是个 `list_of(E)`),又满足 `E`(因为整表也要过)——
也就是 **`E` 自己得接受容器**,例如 `list_of("table")`。
再叠上「其余参数全可选」。Paddle 与 Insight7 的现有 API 里**一个都没有**;
留着 `nonamed` 是给用户和将来用的逃生舱。

**CI 判据(静态可查,在生成期就知道):**

```
规则 #1 的 type 是 list_of(E)
  且 E 接受容器(list_of / "table" / 未约束)
  且其余参数全部可选
    -> 必须显式写 nonamed 或 noordered,否则失败
```

不留启发式的理由:猜错的那次是**静默的错误结果**,不是报错 ——
`zeros{2,3}` 会安安静静给你一个 shape 为 2 的东西。

---

## 3. 生成策略(三条硬规则)

来自 Lua 自身的三道墙(5.1 实测:**60 upvalue / 200 局部变量 / N=122 寄存器**,
`foundations.md` §5.4.6):

1. **所有 per-rule 数据放进一个表 upvalue** -> 生成函数的 upvalue 数恒为 2(`R`, `err`),与 N 无关
2. **每条规则最多一个局部变量**,不生成临时局部
3. **N > 100 切换到表形态**(实参存进 `a[i]`),实测 1000 参数 / 139 KB 仍可编译

生成出来的骨架(N ≤ 100 的快形态):

```lua
local R, err = ...
return function(...)
  local n = select("#", ...)
  local a1, a2, ..., aN
  local t = ...
  if n == 1 and type(t) == "table" and getmetatable(t) == nil then
    a1 = t.name1  a2 = t.name2  ...          -- 具名表;表内位置在此后补 a_i = a_i or t[i]
  else
    a1, a2, ..., aN = ...                     -- 位置
  end
  if a1 == nil then err(R, 1, "missing") end  -- 必填
  if a3 == nil then a3 = R[3].default end     -- 默认值
  if a1 ~= nil and not R[1].is(a1) then err(R, 1, "type") end
  return R.fn(a1, a2, ..., aN)
end
```

**实测(Lua 5.1)**:3 参 609 B / 350 ns;**50 参 7.4 KB / ~1 ms / 3.75 µs**;
120 参 17.8 KB;123 参撞寄存器墙。增量约 **73 ns/参数,线性**。
对照 argcheck:3 参 2597 ns,9 个可选参数 1.37 MB / 840 ms,10 个编不出。

---

## 4. 零框架硬编码(CI 挡)

库里搜不到 `paddle` / `insight` 字样。它自己只认三样东西:

1. Lua 基本类型(`type()`)
2. `pl.class` 的 `is_a`(生态里的类**本来就是** `pl.class`)
3. `register_type` 注册进来的
4. **容器协议**(`#` / `:len()` / `:size()` + `o[i]`)—— 结构,不是类名(§2.3)

```bash
grep -rniE "paddle|insight|torch"  lua/   -> 非空即失败
```

---

## 5. 验收

| # | 项 | 判据 |
|---|---|---|
| 1 | 5 个 Lua 实现 × 3 OS 全绿 | 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT × Win / Linux / macOS |
| 2 | **43 参数基准**(Paddle 真实最大签名 `ResNetBasicBlock.__init__`) | 生成成功,且在**每个** Lua 上成功 —— 5.1 与 5.2+ 的 upvalue 上限不同(60 vs 255) |
| 3 | 三种调用方式等价 | `f(1,2)` == `f{a=1,b=2}` == `f{1,2}` |
| 4 | 零框架硬编码 | 上面那条 grep |
| 5 | 无 `debug` 库时降级可用 | 同一套测试在 `debug=nil` 下重跑 |
| 6 | 生成体积回归 | 30 个可选参数 > 10 KB 即失败(挡指数回归) |
| 7 | `usage` 与 argcheck 的输出形状一致 | 拿 argcheck 的测试用例对拍 |
| 8 | Insight7 先接入并通过它自己的测试 | 它体量小,替 paddle-lua 踩坑 |

---

## 6. 未决

| # | 问题 | 状态 |
|---|---|---|
| ~~P10~~ | 叫什么名字 | ✅ **已定:`argrule`**。剩下的动作是**去 luarocks 占位**(`plan/foundations.md` §5.4.7)|
| — | `defaulta` 的求值顺序 | argcheck 是声明顺序;若 A 的默认值取自 B 而 B 在 A 之后,要报错还是拓扑排序?**倾向报错**,简单可预测 |
| — | 返回值要不要也校验 | `typecheck` 有(`=> int`)。**倾向不做** —— 我们的返回值几乎都是 Tensor,校验收益低于噪声 |
| — | 空容器算不算合法 `IntList` | `paddle.zeros({})` = 0-D?**倾向放行,由 C 层报错** —— 「非空」是语义约束不是类型约束 |
| ~~一维 int Tensor 能不能当 `shape`~~ | ✅ **已定:收。** 人:「Tensor 也应该满足容器协议,只要是个 int 容器按说都应该行」。上游核实通过(`creation.py:1832`),且 2-D Tensor 不算 `TensorList` 是同一条判据算出来的(§2.3),没写特例 |
