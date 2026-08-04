# `argsig` —— 新时代的 argcheck(孵化说明书)

> **一句话:对着 argcheck 重写一个,继承它的**规则表**与**usage**,
> 丢掉它的**组合枚举**与**重载引擎**,把**类型判定**开放成可插拔契约。**
>
> **文件名 `argsig` 是暂定名**(待拍板 P10)。
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
| **规则表 schema** `{name, type, help/doc, default, defaulta, defaultf, opt, check}` | `init.lua:82-96` | 十年验证过的字段划分,既有知识可迁移;这也是我们与 argcheck 的兼容锚点 |
| **表级选项** `help` / `doc` / `quiet` / `debug` | `init.lua:74-80` | 同上 |
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
| **`pl.compat` / `pl.pretty` 复用** | 跨版本 `load`/`unpack`/`setfenv` 与默认值渲染,不自己写(待拍板 P9) |

---

## 2. API 面(最小,先冻这个)

```lua
local sig = require "argsig"

-- 主形态:装饰器
paddle.to_tensor = sig.check{
  {name = "data",  type = {"number", "table", "paddle.Tensor", "insight.Array"},
                   help = "input data"},
  {name = "dtype", type = "string", opt = true},
  {name = "place", type = "paddle.Place", defaultf = function() return paddle.get_device() end},
  help = "Create a paddle.Tensor from data.",
}(function(data, dtype, place)
  -- 类型分派在这里,用 if,和 Python 一样
  if is_tensor(data) then return data:astype(dtype) end
  ...
end)

-- argcheck 兼容形态(同一张表里带 call)
local f = sig.check{ {name="x", type="number"}, call = function(x) ... end }

sig.register_type("paddle.Tensor", function(o) return ... end)  -- 宿主注入
sig.usage(f)   --> string,报错时自动打印的那份
sig.source(f)  --> string,生成的 Lua 源码(排错用,对应 argcheck 的 rules.debug)
```

**三种调用方式落到同一份规则表:**

```lua
f(1, "float32")                    -- 位置
f{data = 1, dtype = "float32"}     -- 具名表
f{1, "float32"}                    -- 表内位置(Insight7 `_wrap` 的第三种)
```

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
| **P10** | 叫什么名字 | 建议 `argsig`;备选 `callsig` / `sigcheck` / `arglet` / `argrule` / `signet` / `clerk`(均实测未被占用)。**定名当天在 luarocks 占位** |
| **P9** | 依赖 Penlight 还是自带兜底 | 建议两边都改成声明 rock 依赖,不再 vendor(`foundations.md` §5.4.2) |
| — | `defaulta` 的求值顺序 | argcheck 是声明顺序;若 A 的默认值取自 B 而 B 在 A 之后,要报错还是拓扑排序?**倾向报错**,简单可预测 |
| — | 返回值要不要也校验 | `typecheck` 有(`=> int`)。**倾向不做** —— 我们的返回值几乎都是 Tensor,校验收益低于噪声 |
