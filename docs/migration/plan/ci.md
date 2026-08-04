# CI · 持续集成计划

> 每个阶段解锁哪些 job、每个 job 挡住什么。目录与阶段对照见 `plan/layout.md`。

---

## 0. 中心约束:**没有现成的 libpaddle 可用**

先把这条摆在最前面,因为它决定了 CI 的全部形状:

- 我们要的是 `WITH_PYTHON=OFF` + `ON_INFER=OFF` 的构建 —— **上游不发布这种产物**
- 从源码编 Paddle 是**小时级**,不是分钟级
- GitHub Actions 的免费 runner **没有 GPU**,磁盘和时长也扛不住编 Paddle

**推论:CI 必须分层,而且大部分 job 不能依赖 libpaddle。**

天真的做法(每次 push 编一遍 Paddle 再跑测试)会让 CI 从第一天就是废的 ——
跑一次两小时,没人会等它,然后所有人开始 `--no-verify`。

---

## 1. 四层 CI

| 层 | 需要 libpaddle? | 时长 | 触发 | 作用 |
|---|:-:|---|---|---|
| **L0 静态层** | ❌ | **< 30s** | 每次 push | 语法、红线、文档树完整性。**现在就能建** |
| **L1 主构建** | ✅(缓存) | ~5 min | 每个 PR | 单一配置编译 + 单元测试。**PR 的必过门** |
| **L2 兼容矩阵** | ✅(缓存) | ~40 min | nightly + release | 5 Lua × 3 OS。**跨版本差异只能靠它抓** |
| **L3 数值/GPU** | ✅ + GPU | 小时级 | nightly + release | 与 Python 对拍、GPU 路径、多 worker 吞吐 |

**libpaddle 怎么来:** 由一个独立的、手动/定期触发的 workflow 编译一次,
产物作为 **artifact + cache key = (paddle_commit, os, build_type)** 存起来,
L1/L2/L3 直接拉缓存。**Paddle 不变就永远不重编。**
这个 workflow 的脚本就是 P0 阶段的产物 `ci/build-libpaddle.sh`(`layout.md` §3)。

---

## 2. L0 · 静态层(不需要任何构建)

这一层**今天就可以建**,不必等 P0。而且它挡的恰恰是最容易在 review 里漏掉的东西。

| job | 内容 | 挡住什么 |
|---|---|---|
| `lua-syntax` | 用 5.1 的 `luac -p` 过一遍所有 `.lua` | **5.1 语法子集**(C3)。在 5.4 上开发很容易写出 `goto`、整数除法 `//`、`<close>` |
| `luacheck` | 全局变量泄漏、未使用变量 | Lua 没有 `local` 声明检查,一个拼错的变量名会静默变成全局 nil |
| `redlines` | grep 红线,见 §6 | 基座使用规则(`conventions.md` §2) |
| `doctree` | **校验工程树的链接完整性**,见 §5 | 文档树是执行计划,断链 = 有任务永远不会被执行 |
| `rockspec` | `luarocks lint` | 发布时才发现 rockspec 写错,太晚 |
| `no-crlf` | 行尾一致性 | Windows 开发 + Linux CI 的经典噪音源 |

---

## 3. L1 · 主构建(PR 必过)

固定一个配置:**Linux + Lua 5.4 + CPU + Release**。

```
1. 拉 libpaddle 缓存(miss 则整个 job 直接失败,不现场编)
2. cmake 配置 + 编译 csrc/
3. busted tests/unit/
4. regen-diff(见 §6)
```

**为什么 PR 门只用一个配置:** 5 个 Lua × 3 OS 的完整矩阵放进 PR 门,
等待时间会长到没人愿意提小 PR。矩阵放 nightly,PR 门只保证"没编坏、没测挂"。

**为什么缓存 miss 直接失败而不是现场编:** 现场编会让某个倒霉的 PR 等两小时。
缓存 miss 是**基础设施的问题**,应该由维护者手动触发重建,不该由随机的 PR 承担。

---

## 4. L2 · 兼容矩阵(nightly)

```
Lua:  5.1  5.2  5.3  5.4  LuaJIT-2.1(GC64)
OS:   Linux  macOS  Windows
```

**这一层不是奢侈品,它是唯一能抓到跨版本静默差异的地方。** 已经踩到的实例:

| 差异 | 后果 | 只有矩阵能发现 |
|---|---|---|
| 5.1/LuaJIT 的 `ipairs` 用 `lua_rawgeti` | `ipairs(layer_list)` 在 5.1 上**一次都不迭代**,5.4 正常(D27) | ✅ 单版本 CI 100% 漏掉 |
| `__len` / `__pairs` 是 5.2+ | 5.1 上元方法静默失效 | ✅ |
| 整数除法、`goto`、位运算 | 5.1 上语法错误 | L0 能挡一部分 |

`tests/compat/` 存在的意义就是把这些差异**显式断言下来** ——
断言的不是"行为一致",而是"我们知道它不一致,且知道不一致在哪"。

---

## 5. `doctree` job:把文档树当代码检查

工程树(`WORKPLAN.md`)是**执行计划**,不是说明文字。断链的后果是有任务永远不被执行。

| 检查 | 失败条件 |
|---|---|
| **✅/🔵 节点**的指针目标存在 | 文件不存在 |
| **⬜ 节点**的指针目标 | **不检查存在性** —— 那是"待创建的产出",见下 |
| 每个非叶节点至少有一个子节点指针 | 空的中间节点 = 计划漏了一块 |
| 每个叶节点有「完工判据」小节 | 没有判据的任务无法判断做完没有 |
| 每个叶节点有「前置」字段(可以是"无") | 缺前置 = DFS 会在错误的时机执行它(见 `WORKPLAN.md` §2) |
| 树里的阶段编号与 `roadmap.md` 一致 | 两处不同步 |

**这是个纯文本 job,几十行脚本,但它保证的是"计划本身是完整的"。**

两条实现细节(第一次跑校验就撞上了,记下来免得重踩):

1. **⬜ 节点的指针是"产出",不是"引用"。**
   `2.10.0 API 设计 -> plan/api/optimizer.md` 里那个文件**现在就不该存在** ——
   它是那个节点做完之后才有的东西。
   一刀切检查存在性会让整棵树从第一天就是红的,然后没人再看这个 job。
   **规则:节点状态决定校验强度。** ✅/🔵 必须存在;⬜ 只校验路径**格式**合法。
2. **指针里的 `{a,b}.md` 花括号要展开**再逐个校验,
   否则会去找一个名叫 `{top,tensor}.md` 的文件。

---

## 6. 必须由 CI 挡住的红线

人会忘,review 会漏。这些全部机器检查:

```bash
# ① 基座红线(conventions.md §2)
grep -rn "class%.cast"                lua/                      -> 失败
grep -rn "rawset(self"                lua/paddle/nn/            -> 失败(仅 FIELDS 一处豁免)
grep -rn "require *['\"]middleclass"  lua/                      -> 失败(第二套 class)
grep -rn "_vendor/pl"                 .                         -> 失败(D34,全生态只许一份 Penlight)

# ①b 参数检查的四条(foundations.md §4 §5.4.6)
grep -rn "argrule"                    lua/paddle/_ops/          -> 失败(生成算子构建期展开,§4.7)
grep -rniE "paddle|insight"           <解析器仓库>/lua/          -> 失败(零框架硬编码,§5.4.2)
lua tools/ci/args_codegen_size.lua                              -> 30 个可选参数的规则表
                                                                   生成 >10 KB 即失败(指数回归)
lua tools/ci/args_codegen_43.lua                                -> 用 Paddle 真实最大签名
                                                                   (43 参数 / 33 可选,resnet_block.py:434)
                                                                   编不出来即失败(upvalue/寄存器墙)
lua tools/ci/args_nonamed.lua                                   -> 规则 #1 是 list_of(E)、E 自己也接受容器、
                                                                   且其余参数全可选,却没声明
                                                                   nonamed/noordered 即失败(argrule ⑧)
                                                                   —— 必填参数与元素类型本身就消歧,
                                                                   所以这条现在几乎永远不触发,留着挡将来
lua tools/ci/args_no_enum_number.lua                            -> dtype/device/layout/reduction 这类
                                                                   枚举参数的 type 里出现 "number"
                                                                   即失败(api/README §2.1.1)
lua tools/ci/args_intlist.lua                                   -> ① shape/axes/perm/strides 这类参数的
                                                                   type 不是 "IntList" 即失败;
                                                                   ② 任何 type 里出现 "pl.List" /
                                                                   "insight.Array" 即失败 —— 容器是
                                                                   **协议**不是类名单(argrule §2.3)
lua tools/ci/args_no_opts_table.lua                             -> 文档与代码里出现
                                                                   `paddle.f(…, {ident = …})` 这一形状
                                                                   即失败 —— 最后一个实参是「键全是
                                                                   标识符」的 table,几乎必然是写成了
                                                                   「选项表」。正确形状是 `f{…, ident = …}`
                                                                   (api/README §2.1.5)。**扫 docs/ 与 lua/**

# ② 5.1 语法子集(C3)
grep -rnE "goto |::[a-z]+::|[^/]//[^/]|<close>|<const>"  lua/   -> 失败

# ③ 生成代码不许手改(layout.md §5)
python tools/gen/emit_capi.py && python tools/gen/emit_lua.py
git diff --exit-code csrc/capi_gen/ lua/paddle/_ops/            -> 非零即失败

# ④ 1-based 标注表必须覆盖全部算子(overview §6.1.1)
python tools/gen/check_index_semantics.py                       -> 未标注即失败

# ⑤ 上游零改动(项目级硬约束)
git -C $PADDLE_ROOT diff --exit-code                            -> 非零即失败
```

`args_codegen_43.lua` 挡的是**第二类指数**:不是生成代码的长度,而是 **Lua 自身的三道墙**
(5.1 实测:60 个 upvalue / 200 个局部变量 / N=122 的寄存器上限,`foundations.md` §5.4.6)。
它必须在**全部 5 个 Lua 实现**上跑 —— 5.1 与 5.2+ 的 upvalue 上限不同(60 vs 255),
只测一个版本会漏。基准签名用 Paddle 真实存在的最大那个,不用假想值。

第 ③ 条是 `layout.md` §5 的安全绳:60k 行生成代码里手改一处,
当时能跑,下次重新生成被**静默覆盖**,改动消失且无任何报错。

第 ④ 条来自风险登记表里那条"index 语义标注表的长期维护" ——
Paddle 每加一个算子都可能要补标注。**默认放行会让 1-based 承诺慢慢烂掉**,
所以策略必须是**未标注即报错**,而不是未标注则按 0-based 透传。

第 ⑤ 条挡的是"为了让某个测试过,顺手改了一行 Paddle" ——
这会让整个项目的"零上游改动"承诺在无人察觉时失效。

①b 的第二条是在防一类**具体的**退化。argcheck 之所以被否,
就是因为它对参数「给了/没给/显式 nil」做 3^N 枚举,9 个可选参数就 1.37 MB、
10 个直接编不出来,而 `Conv2D` 有 11 个、`DataLoader` 有 16 个(`foundations.md` §4.5)。

`args_no_opts_table.lua` 挡的是**一个字符的差**:`f(x, {opt=1})` 与 `f{x, opt=1}`。
前者是绝大多数 Lua 库的习惯写法,后者才是我们的约定,而**人眼几乎分辨不出来** ——
证据是项目自己三份 README 的首屏示例就写错了,评审多次都没看出来(R36)。
这类「读起来完全正常、但按自家规则跑不通」的错误只能靠机器挡,
而且**必须扫文档** —— 文档是用户抄走的第一份代码。

**「为了支持某种更灵活的调用形式,顺手枚举一下组合」是一个非常自然的改动** ——
它在 3 个参数的测试上完全正常,要到某个真实的层才炸。
所以判据必须是**生成代码的体量**,而不是「有没有人想着别这么写」。

**L0 就能跑这两条 —— 它们不需要 libpaddle。**

---

## 7. 数值对拍与 golden 数据(L3)

**CI 端有 Python,用户端没有。** 这个区分要在文档里反复强调 ——
`tools/gen/` 和 `tests/parity/` 的 Python 是**开发期依赖**,
不出现在 rockspec 的 `dependencies` 里,也不出现在用户的安装步骤里。

| 项 | 做法 |
|---|---|
| golden 数据存哪 | **不进 git。** `tests/fixtures/` 只放生成脚本,CI 用固定种子现场生成并缓存 |
| 对拍什么 | 40 个层的前向输出(1e-6)、`state_dict` 键名逐字符、固定种子的参数初始化逐位 |
| 谁产生基准 | CI 里装一份**官方 Paddle Python 包**(不是我们编的那个),现跑现比 |

用官方 pip 包做基准而不是我们自己编的 libpaddle:
否则我们编错了,两边一起错,对拍**永远是绿的**。

---

## 8. 阶段解锁表

每个阶段完工时,必须同时交付它的 CI job。**没有 CI 的阶段不算完工。**

| 阶段 | 新增 job |
|---|---|
| **现在** | L0 全部(`lua-syntax` 除外,还没有 .lua) |
| P0 | `build-libpaddle`(缓存生产者)+ `probe`(C++ 反向验证) |
| P1 | L1 编译 + `tests/unit/capi`(纯 C 测试) |
| P2 | `require "paddle_core"` 冒烟 + L2 矩阵**首次启用** |
| P3 | `regen-diff` + `check_index_semantics` |
| P4 | rockspec 安装冒烟(从 tarball 装一遍) |
| P5 | 索引/切片的 1-based 断言;`tests/compat/` 建立 |
| P6 | 梯度数值检查(有限差分对拍) |
| P7 | 跨语言往返:Python 存 -> Lua 读 -> Lua 存 -> Python 读 |
| P8 | `jit.save` 产物加载一致性 |
| P9 | **40 层 parity + state_dict 键名 + `LayerList` 五版本行为断言** |
| P10 | 优化器与 Python 的训练曲线对拍(固定种子) |
| P11 | DataLoader 顺序确定性 |
| **M1** | `examples/mnist.lua` 端到端,准确率 > 95% 作为**门** |
| P12 | 数据集下载走 mock server(CI 不打真实网络) |
| P13 | `num_workers=4` 吞吐 >= 3× 单 worker;**ThreadSanitizer** |
| P14 | OOM 注入后能被救回 |
| P15+ | trace 产物与动态图数值一致 |

P13 的 ThreadSanitizer 单列出来,是因为多线程的 bug **不会稳定复现** ——
靠跑几次测试碰运气是抓不到的,必须上工具。

---

## 9. 未解问题

| # | 问题 |
|---|---|
| CI-1 | GPU runner 从哪来?没有的话 L3 的 GPU 部分只能本地手动跑,**GPU 路径就没有回归保护** |
| CI-2 | Windows 上编 libpaddle 的可行性未验证(P0 只验证了当前机器) |
| CI-3 | libpaddle 缓存的体积是否超出 GitHub Actions 的 cache 限额(10 GB/repo) |
