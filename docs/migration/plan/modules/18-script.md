# P18 · script 模式(AST)

| | |
|---|---|
| 阶段 | P18 —— **最后一块,最难的一块** |
| 类别 | 手写 Lua |
| 开工条件 | P15 完工 |
| 预估 | 6 周 |

---

## 1. 做什么 / 不做什么

**做:** 解析用户的 Lua 前向函数源码,把 `if` / `while` / `for` 转换成
Paddle 的控制流算子,生成不依赖运行时分支的静态图。

**不做:** 不支持任意 Lua。**支持一个明确的子集**,子集外的写法**报错而不是猜**。

**这是 trace 模式(P15)的补集。** trace 把控制流固化成本次走过的分支;
script 把控制流真正编进图里。PyTorch 的 `torch.jit.trace` 与 `torch.jit.script`
是同一组关系。

---

## 2. 有什么可以用

### 2.1 AST 库:一次自我纠错

> ⚠️ **翻案记录:** 曾断言"Lua 没有官方 AST 库,要自己写 parser"。**这是错的。**

调研后的结论:

| 候选 | 星数 | 许可 | 能否用 |
|---|---|---|---|
| **luacheck**(lunarmodules) | 454 | MIT | ✅ **选它** |
| LuaMinify | 273 | MIT | ⚠️ 2022 年后无更新 |
| andremm/lua-parser | 209 | MIT | ❌ 需要 LPegLabel(C 依赖) |
| LuaLS/lua-language-server | 4330 | MIT | ❌ 体量过大,为 LSP 设计 |

**luacheck 唯一地同时满足两个条件:**

1. README 明载"supports checking Lua files using syntax of Lua 5.1 – 5.4, and LuaJIT"
   —— **能解析我们要支持的全部方言**
2. README 明载"Luacheck itself is written in Lua and runs on all of mentioned Lua versions"
   —— **它自己能在 5.1 上跑**,满足 C3

第 2 条是决定性的。一个只能在 5.4 上跑的 parser 对我们毫无用处。

**vendor 哪些文件:**

| 文件 | 大小 |
|---|---|
| `parser.lua` | 31136 B |
| `lexer.lua` | 19818 B |
| `utils.lua` | 8318 B |
| **合计** | **~59 KB** |

`parser.lua:6-9` 显示 AST 节点带 `line` / `offset` / `end_offset` ——
**有位置信息,错误消息可以精确到列。**

**要自己写的:printer(AST -> Lua 源码),约 400–600 行。**
luacheck 是个 linter,只需要读不需要写,所以没有 printer。

### 2.2 与用户的君子之约

script 模式需要拿到用户函数的**源码**。方式是 `debug.getinfo(f, "S")`,
返回 `source` / `linedefined` / `lastlinedefined`。

**这要求:**

1. 用户不能把库预编译成字节码后丢掉源码
2. 用户不能加密源码

**这不是我们发明的额外约束。** Python 的 `paddle.jit.to_static` 走
`inspect.getsource`,面对的是完全相同的隐含契约 ——
只是 Python 用户很少遇到,因为 `.py` 源码通常都在。

**约定要写在文档最显眼处,而不是埋在 FAQ 里。**

### 2.3 契约从"信任"升级为"验证"

光靠约定不够,要能**检测**用户是否违约:

```
1. debug.getinfo(f, "S") 拿到 source 与行号
2. 读出那段源码,用同样的 chunkname 重新 load
3. string.dump 两者,比对
4. 不一致 → 报错:"函数源码与实际字节码不符,script 模式无法使用"
```

**坑:** `load` 时的行号必须对齐,否则 dump 出来的调试信息不同。
方法是在源码前补 `linedefined - 1` 个换行,并使用与原函数相同的 chunkname。

M0 第 20 项验证这条路可行(可跳过项,不阻塞 M0)。

---

## 3. 设计

### 3.1 支持的 Lua 子集

| 语法 | 支持 | 说明 |
|---|---|---|
| 算术、比较、逻辑运算 | ✅ | |
| `local` 声明与赋值 | ✅ | |
| `if / elseif / else` | ✅ | -> `cond` 算子 |
| `while` | ✅ | -> `while_loop` 算子 |
| 数值 `for` | ✅ | 展开或转 `while` |
| 函数调用(算子、其它 scripted 函数) | ✅ | |
| 泛型 `for`(`pairs` / `ipairs`) | ❌ | 迭代次数不确定 |
| `goto`(5.2+) | ❌ | 且违反 C3 |
| 闭包、upvalue 写入 | ❌ | |
| `table` 动态构造 | ⚠️ | 只支持常量表 |
| 协程 | ❌ | |
| 元表操作 | ❌ | |

**子集外的写法必须报错并给出行号与建议**,例如:

```
paddle.jit.script: my_forward.lua:23: 不支持泛型 for
  提示:改用数值 for,或把这段逻辑移到 script 函数外
```

### 3.2 流水线

```
用户函数 f
  ├─ debug.getinfo(f, "S")           取源码位置
  ├─ 读源码 + string.dump 验证        契约检查
  ├─ luacheck parser → AST
  ├─ 子集检查(遍历 AST,遇到不支持的节点立即报错带行号)
  ├─ 变换:控制流 → Paddle 控制流算子调用
  ├─ printer:AST → Lua 源码
  ├─ load() 新函数
  └─ 交给 P15 的 trace 录制
```

**注意最后一步:script 不绕开 trace,而是"改写成 trace 能处理的形式"再交给 trace。**
这让 P18 完全建立在 P15 之上,不需要第二套图组装逻辑 ——
这也是为什么 P18 必须排在 P15 之后。

---

## 4. 已知的坑

**① 这是全项目最容易做砸的一块。** 6 周是乐观估计。
**如果时间紧张,砍掉它比做半个更好** —— trace 模式已经覆盖绝大多数场景。

**② 报错质量决定这个功能的成败。** 用户写的 Lua 有 95% 的概率触及子集边界。
如果报错是 "unexpected node type 17",这个功能就没人用。
**报错信息的工作量应该按功能本身的 30% 来预留。**

**③ vendored 代码要标清来源。** `lua/paddle/_vendor/luacheck/` 下必须有
`ORIGIN.md` 写明:仓库 URL、commit、许可证、我们做过的修改(理想情况是零修改)。

**④ printer 的正确性验证。** 写完 printer 后的第一个测试应该是:
**AST -> 源码 -> AST,两次 AST 相同**(round-trip)。
这比逐个特性测试更能暴露问题。

---

## 5. 验收

- [ ] 带 `if` 的前向,script 化后数值与动态图一致
- [ ] 带 `while` 的前向同上
- [ ] printer 的 round-trip 测试:100 个 Lua 文件,AST 往返不变
- [ ] 子集外的 10 种写法,逐个验证报错含行号与建议
- [ ] 源码契约:故意用 `string.dump` 剥离源码,报出可读错误
- [ ] script 产物能被 Python `jit.load` 读回
- [ ] 五个 Lua 版本上 parser 都能跑

---

## 6. 未解问题

- P-04(待人拍板):源码契约走"纯君子之约"还是"约定 + `string.dump` 验证"?
  倾向后者,但会增加约 200 行与一些边界情况
- 数值 `for` 是展开还是转 `while`?循环次数已知时展开更快,但会让图变大
