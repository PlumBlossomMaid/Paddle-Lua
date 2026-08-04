# 动转静 · Lua 源码获取的"君子之约"与 AST 库选型

> 起因:「AST 的话我也想用第三方 Lua 库,但是和用户之间要有个君子之约:
> 必须让我们能够找到 Lua 源码,而且不能加密。」
>
> **结论:这个约定是对的,而且比"君子之约"更强 —— 它可以做到"可验证"。**
> **更重要的是:它只约束 script 模式,而我们规划的第一版是 trace 模式,根本不需要 AST。**

---

## 0. 先把范围缩小:AST 是可选的,不是必需的

PyTorch 有两条动转静路径,Paddle 只有后者:

| 模式 | 怎么工作 | 控制流 | 需要源码/AST? |
|---|---|---|---|
| **trace**(`torch.jit.trace`) | **跑一遍**,记录实际发生的算子调用 | **按当次执行"烤死"** | ❌ **完全不需要** |
| **script**(`torch.jit.script` / Paddle `to_static`) | **读源码 → 改 AST → 重新执行** | 真正捕获 `if`/`while` | ✅ 必需 |

`plan/overview.md` §2.1.3 已定:**第一版只做直线图 tracing。**

**所以整个"君子之约 + AST 库"是 M4 的话题,不是 M3 的。**
这一点很重要 —— 它意味着:

- **M3 的自研静态图零 AST 依赖**,也就零源码可得性约束
- 用户拿到的第一版静态图,对加密/字节码分发的项目**照样能用**
- AST 只在用户想要"带控制流的图"时才进场,那是明确的**增值功能**,不是基线功能

**这个分层本身就是对君子之约最好的处理:把约束限制在自愿使用的高级功能里,
而不是让它污染基线能力。**

---

## 1. 这个约定不是 Lua 的弱点 —— Python 也一样

必须先说清楚,免得把它当成"Lua 不如 Python"的证据。

`paddle.jit.to_static` 底层是 `inspect.getsource(func)`。它在这些情况下同样失败:

| Python 失败场景 | 对应的 Lua 场景 |
|---|---|
| REPL / Jupyter 里定义的函数 | `load(string)` 定义的函数 |
| `exec()` 出来的代码 | 同上 |
| 只分发 `.pyc`,删掉 `.py` | 只分发 `luac -s` 字节码 |
| 加密/打包工具(PyArmor 等) | Lua 加密 loader |
| C 扩展里的函数 | `what == "C"` |

**PyTorch/Paddle 已经和用户签了完全一样的君子之约,只是没人这么叫它。**
所以我们不是在提出一个额外要求,而是在**把一个业界既有的隐含契约写到明面上** ——
这反而比 Python 生态做得更诚实。

---

## 2. Lua 侧怎么拿到源码(机制)

```lua
local info = debug.getinfo(fn, "S")
-- info.source        "@path/to/file.lua"  或  "=stdin"  或  源码字符串本身
-- info.short_src     人类可读的短名
-- info.linedefined   函数首行
-- info.lastlinedefined 函数末行
-- info.what          "Lua" / "C" / "main"
```

规则(5.1/5.2/5.3/5.4/LuaJIT 一致):

| `source` 首字符 | 含义 | 我们能做什么 |
|---|---|---|
| `@` | 后面是**文件路径** | ✅ 读文件,切 `linedefined..lastlinedefined` |
| `=` | 宿主自定义的 chunk 名,**无源码** | ❌ 放弃 |
| 其他 | `source` **本身就是源码** | ✅ 直接用,最理想 |

---

## 3. 五种失败模式,以及能否检测

君子之约只有在**违约可检测**时才有意义。逐条核对:

| # | 情况 | 检测方式 | 可检测? |
|---|---|---|:-:|
| 1 | C 函数 | `info.what == "C"` | ✅ |
| 2 | 剥离调试信息的字节码(`luac -s` / `luajit -b -s`) | `info.linedefined <= 0` 或 `source == "=?"` | ✅ |
| 3 | 未剥离字节码(有行号,无文件) | `source` 是 `@path` 但 `io.open` 失败 | ✅ |
| 4 | 加密 / 自定义 loader | 同上,`io.open` 失败 | ✅ |
| 5 | `load(string)` 且宿主传了 `=name` | `source:sub(1,1) == "="` | ✅ |
| **6** | **文件在加载后被改过** —— 读到的源码和实际运行的函数不一致 | 见 §4 | **⚠️ 朴素方法检测不了** |

**前五种都能给出清晰的报错。第 6 种是真正危险的那个:**
它不报错,而是**默默地按错误的源码建图**,产出一个语义不对的模型。

---

## 4. 把"君子之约"升级成"可验证" —— `string.dump` 比对

第 6 种其实是可以证伪的,方法是往返验证:

```lua
-- 1. 从文件抽出 linedefined..lastlinedefined 的源码
-- 2. 前面补 (linedefined-1) 个换行,让行号对齐
-- 3. 用相同的 chunkname 重新 load
local padded = string.rep("\n", info.linedefined - 1) .. extracted_src
local rebuilt = assert(load(padded, info.source))   -- 5.1 用 loadstring/setfenv 垫片

-- 4. 比对字节码
if string.dump(rebuilt) == string.dump(fn) then
  -- ✅ 已证明:抽出来的源码就是这个函数的源码,逐字节一致
end
```

**这把契约从"我相信你没骗我"变成"我验过了"。**

诚实的三条限制:

| 限制 | 说明 | 影响 |
|---|---|---|
| `string.dump` 对带 upvalue 的闭包 | 只 dump 函数原型,upvalue **不进字节码** | 不影响比对结论(我们比的是代码,不是绑定) |
| 提取的是 `function ... end` 整段 | `rebuilt` 是外层 chunk,`fn` 是内层函数 | 需要 `load` 后再取出内层函数再 dump,不是直接比 |
| 5.4 的 `string.dump(f, strip)` | 默认不 strip,含调试信息 | 一致即可,但要确保两边参数相同 |

⚠️ **列为 M0 验证项:这个往返比对在 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT 上是否都成立。**
LuaJIT 的字节码格式与标准 Lua 完全不同,但只要**自己和自己比**就行,不需要跨实现一致。

**如果这条验证通过,建议的策略是:**

```
默认:  尝试往返验证 → 通过则静默继续,失败则 warning(源码可能已变更)
严格:  paddle.jit.strict_source = true → 验证失败直接报错
宽松:  用户显式传 source 覆盖
```

---

## 5. AST 库选型

### 5.1 候选对比(已联网核对)

| 库 | 语法覆盖 | 自身可跑版本 | 许可 | 最近更新 | 纯 Lua | 有代码生成 |
|---|---|---|---|---|:-:|:-:|
| **luacheck 的 `parser.lua`** | **5.1-5.4 + LuaJIT** | **5.1-5.4 + LuaJIT** | MIT | **2026-07-31** | ✅ | ❌ |
| LuaMinify `ParseLua.lua` | 5.1 为主 | 5.1 | MIT | 2022-11 | ✅ | ✅ |
| andremm/lua-parser | 5.3 | — | MIT | 2026-01 | ❌ 需 LPegLabel | ❌ |
| lua-language-server | 5.1-5.4+LJ | 5.4 | MIT | 2026-06 | ✅ | ❌ |
| Metalua | 5.1 | 5.1 | MIT | 停更 | ✅ | ✅ |

### 5.2 推荐:luacheck 的 parser,vendored

**三条硬理由:**

1. **它的语法覆盖正好是我们的目标集。** README 原文:
   > "Luacheck supports checking Lua files using syntax of **Lua 5.1 - 5.4, and LuaJIT**."

2. **它自己就跑在 5.1 上。** README 原文:
   > "Luacheck itself is written in Lua and **runs on all of mentioned Lua versions**."
   >
   > 这直接满足我们"库代码必须是 Lua 5.1 语法子集"的约束 ——
   > **不用改一行就能用,这是所有候选里唯一做到的。**

3. **依赖极小,可直接 vendor:**

   ```
   src/luacheck/parser.lua    31 KB   → parser.lua 只 require lexer + utils
   src/luacheck/lexer.lua     20 KB
   src/luacheck/utils.lua      8 KB
   ────────────────────────────────
   合计 ~59 KB,3 个文件,MIT
   ```

   AST 节点自带 `line` / `offset` / `end_offset`(`parser.lua:6-9`),
   **源码级变换和错误回溯都需要这个,不是所有解析器都有。**

**一个有意思的旁证:** luacheck 的 Windows 单文件发行版里打包了
`Lua 5.4.4 + LuaFileSystem + **LuaLanes**`。
**和我们选的强制依赖是同一个** —— 说明这套组合在 Windows 上是走得通的、有人趟过。

### 5.3 缺的那块:代码生成器要自己写

**luacheck 是 linter,只有 parser 没有 printer。** 这是它唯一的短板。

动转静的标准做法(Paddle `dygraph_to_static` 就是这么干的):

```
源码 → AST → 变换(if/while → 图节点构造调用) → 生成新源码 → load() 执行
                                                    ↑
                                              需要 AST→源码 打印器
```

选项:

| 方案 | 成本 | 评价 |
|---|---|---|
| 自己写 printer 配 luacheck AST | **~400-600 行** | ✅ **推荐**。printer 是纯机械代码,好写好测 |
| 改用 LuaMinify(自带 printer) | 0 | ❌ 5.1-only、2022 停更,语法覆盖不够 |
| 跳过源码生成,AST **直接**建 `pir::Program` | 高 | ❌ 等于自己实现一个 Lua 解释器子集 |

**推荐第一条。** 理由:parser 是难写的部分(要处理运算符优先级、各版本语法差异),
printer 是简单的部分(遍历 + 拼字符串)。**用现成的难的,自己写简单的**,分工正确。

---

## 6. 建议写进文档的用户契约原文

```
paddle.jit.script(fn) 需要读取 fn 的 Lua 源码。使用它意味着你同意:

  1. fn 所在的 .lua 文件在调用时可被 io.open 读取
  2. 该文件未被加密、未被剥离调试信息(luac -s / luajit -b -s)
  3. 该文件自加载后未被修改

不满足时,paddle.jit.script 会报错并指明原因,不会静默产出错误的图。

如果你的项目必须加密分发,请使用 paddle.jit.trace(fn, example_inputs)。
它不读取源码,代价是控制流按追踪时的实际路径固定。
```

**关键是最后一段:给违约者一条明确的退路。**
一个只有惩罚没有出路的约定不会被遵守;
而 trace 模式本来就存在,把它写成"加密分发的官方方案"零成本。

---

## 7. 净变化与新增 M0 项

| 项 | 结论 |
|---|---|
| 君子之约是否成立 | ✅ 成立,且与 Python 生态既有契约等价 |
| 违约能否检测 | ✅ 6 种情况中 5 种可直接检测 |
| 第 6 种(文件被改) | ✅ **可用 `string.dump` 往返比对证伪** —— 从"约定"升级为"验证" |
| AST 库 | **luacheck 的 parser + lexer + utils,3 文件 ~59 KB,MIT,vendored** |
| 代码生成器 | **自己写,~400-600 行** |
| 这些何时需要 | **M4**。M3 的 trace 模式零 AST 依赖 |
| 对加密分发用户的影响 | **基线功能不受影响**,只有 script 模式不可用 |

| # | 新增 M0 验证项 | 影响 | 预估 |
|---|---|---|---|
| 20 | `string.dump` 往返比对在 5.1/5.2/5.3/5.4/LuaJIT 上是否都成立 | 🟢 退回纯君子之约(仍可用) | 0.5 天 |
| 21 | luacheck `parser.lua` 抽出 3 个文件后能否独立运行于 5.1 | 🟢 换库或多带几个文件 | 0.5 天 |

> 两项都不紧急 —— 它们服务的是 M4。**M0 阶段可以跳过,不影响任何生死判定。**
> 记在这里只是为了不遗忘。
