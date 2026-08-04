# DECISIONS.md · 决策记录与翻案记录

> `CLAUDE.md` §2 是决策**索引**;本文件是决策的**理由与翻案痕迹**。
> **翻案不是失败,不留翻案痕迹才是。**

---

## 1. 翻案记录(重要:先看这个)

已经被推翻的结论列在这里。**看到旧文档里的旧结论时,先查本节。**

| # | 原结论 | 现结论 | 推翻理由 | 出处 |
|---|---|---|---|---|
| R1 | LuaJIT 走 FFI,普通 Lua 走 sol2 | **全部走 sol2** | 绑定开销 50-200ns,比 kernel 1-10µs 低 1-2 个数量级,不值得维护两套 | `research/architecture.md` §B |
| R2 | 建议向 Paddle 上游 PR 一个 C++ DLPack 导出 | **撤回**,零上游改动可行 | `tensor.h` 的公开 API 足够在我们仓库里实现 | `research/architecture.md` §A |
| R3 | Torch7 堆追踪是可选、默认关 | **默认开** | `torch7/init.lua:189` `torch.setheaptracking(true)` | `research/gc.md` §6.3 |
| R4 | 不要用 lua_lanes/effil,自建线程池 | **用 Lanes** | Lanes 的 deep userdata 正是为我们这种场景设计的 | `research/dataloader.md` §9 |
| R5 | Lanes 做成可选依赖 | **强制依赖** | 单一 Tensor 表示、单一代码路径、少一个验证项 | `research/dataloader.md` §9.5(a) |
| R6 | 索引 1-based + `axis` 0-based | **统统 1-based** | 混合方案的认知负担是持续的;Torch7 先例;边界在库里不在用户代码里 | `plan/overview.md` §6.1.1 |
| R7 | 切片写法 `x:slice{{1,3},{}}` | **字符串索引 `x["1:3, :"]`** | Insight7 已有跨语言实现;语法零学习成本;绕开 Lua 的 `__index` 单参数限制 | `plan/overview.md` §6.1.2 |
| R8 | 关键字参数用"位置 + 末位 table" | **`_wrap` 三模式** | Insight7 `_wrap.lua` 更接近 kwargs 体验,且是纯 5.1 | `plan/overview.md` §6.1.2 |
| R9 | `dataset`/`hub`/`cpp_extension`/`jit`/`distributed` **排除** | **改判"延后"** | 把"难"和"该不该做"混为一谈了。难度决定排期,不决定范围 | `plan/overview.md` §2.2 |
| R10 | `cpp_extension` 可以直接抄 | **只有 1/3 能抄** | `cpp_extension.py` 那 1400 行本身就是一个 setuptools 插件,整块作废 | `plan/overview.md` §2.1.2 |
| R11 | 自研静态图 = 独立项目体量 | **只需 tracer + 组装 `pir::Program`** | `InterpreterCore` 提供 IR/调度/GC/pass/序列化 | `research/reuse.md` §2.1 |
| R12 | 静态图要自己建反向图 | **`run_program_ad_func` 免费提供** | 它是 `_ad_func`,自动挂 GradNode | `research/reuse.md` §4 |
| R13 | Lua 没有 AST 库,要自己写 parser | **luacheck 的 parser 覆盖 5.1-5.4+LJ 且自身跑在 5.1** | 已联网核对 README 与文件结构 | `research/to-static.md` §5 |
| R14 | `paddle.save/load` 阻塞于 pickle 解析器 | **原生 `SaveTensor`/`LoadTensor` 先行** | pickle 解决的是「读 Python 存的」;「Lua 存 Lua 读」不需要它 | `research/reuse.md` §5.1 |
| R15 | 统统 1-based,无例外 | **`state_dict` 键名里的容器下标破例为 0-based** | 键名必须与 Python 逐字符一致,否则权重互不可读,摧毁"Python 训练 / Lua 推理"的全部价值。破例范围严格限于**序列化键名字符串**,`seq[1]` 访问仍是 1-based | `plan/modules/09-nn.md` §3.5 |
| R16 | `nn.Layer` 照抄 Python 的 `__setattr__` 自动注册 | ~~改为显式 `add_sublayer` / `add_parameter`~~ **本条已被 R17 再次推翻** | (当时的理由)Lua 的 `__newindex` 只在键**不存在**时触发,重复赋值会静默漏注册 | `plan/modules/09-nn.md` §3.2 |
| R17 | (R16 的结论)显式 `add_sublayer` / `add_parameter` | **恢复自动注册** | R16 的推理**只在朴素实现下成立**。把实例的 raw 表保持为空(只放一个用户拼不出来的私有键 `FIELDS`),`__newindex` 就会在**每一次**赋值时触发 —— Python 的 `nn.Module.__setattr__` 做的是同一件事(把 param 从 `__dict__` 移进 `_parameters`)。放弃自动注册会实打实抬高用户门槛,而代价只是每次字段访问多 ~60ns(kernel 是 µs 级) | `plan/modules/09-nn.md` §3.2 |
| R18 | Tensor 必须做成 Lanes deep userdata(`DeepPrelude` / `DeepFactory`),绑定层要为此重新设计 | **sol2 usertype + `__lanesclone`** | deep userdata 的存在意义是给共享 C 对象一个原子引用计数,而 `paddle::Tensor` **本来就有** —— `tensor.h:142` 收的是 `std::shared_ptr<phi::TensorBase>`,`shared_ptr` 的计数是原子的。每个 lane 各拿一个 sol2 外壳,底下缓冲共享,零拷贝。绑定层不必为 Lanes 改任何设计 | `plan/modules/13-lanes.md` §4.1 |
| R19 | `nn.Layer` 自写 ~80 行最小类系统,明确否决第三方 OOP 库 | **用 `pl.class`(Penlight)** | 「引入依赖」的顾虑不成立:我们本来就需要 list、5.1–5.4 兼容层、安全 table 读取,自写只是把「一个依赖」换成「一个依赖 + 三个自造轮子」。「用户可能已有自己的 OOP 库」**恰恰是选 Penlight 的理由** —— 它是 Lua 生态的通用库。已核实 `pl/class.lua` 的 `_create` / `_class_init` 两个官方钩子刚好能承载自动注册,与 R17 不冲突 | `plan/foundations.md` §1 |
| R20 | Insight7 只是「字符串索引 / `_wrap` 三模式的参考实现」 | **Insight7 顶替 numpy 的位置,一等公民** | 它的 Lua 绑定的 dtype 常量与 Place 构造器**本来就是 Paddle 命名**(`bindings/lua/insight/init.lua` 头部注释写着 "Paddle-style"),切片语法与 `_wrap` 我们已经在抄它。选它不是「引入一个数组库」,是「把已经对齐的那半边接上」 | `plan/foundations.md` §3 |
| R22 | `nn.LayerList` 继承 `pl.List`,`Layer` 能力靠父层特例展开 | **继承 `Layer`,Layer 优先** | (人的决定)`is_a(nn.Layer)` 为假是**语义错误**,不是可接受的差异:`LayerList` 装的是层、参与前向、有参数、要 train/eval,它就是一个 Layer。而且遍历特例会繁殖(`LayerDict`/`Sequential`/用户自定义容器各要一份),Python 侧 `ModuleList` 本就是 `Module` 子类,直译过来的 `isinstance` 检查会静默返回 false。丢掉的 `pl.List` 方法用转发补 | `plan/foundations.md` §2.4 |
| R23 | 「不引入新的强制 C 依赖」是硬约束 | **取消。需要就引入,判据改为「边际成本」** | (人的决定)该规则把 §3.1 的下载能力和 §3.3 的图像解码逼进死胡同,是用一条数量规则替代逐案判断。新判据:先问我们自己那个已链着 libpaddle 的 `.so` 能不能顺手做(边际成本 ≈ 0),做不了就直接引入,但必须给出 5 Lua × 3 OS 的覆盖证据与降级路径 | `CLAUDE.md` §9.1 |
| R24 | Insight7 `axis` 0-based 是「已发布库的既定行为」,只能绕 | **按 bug 修,改成 1-based** | (人的决定,「以后会顺手修复的」)同一个 Lua 绑定里切片已经是 1-based(`insight_lua.cpp:212-266` 的 `lua_spec_to_cpp`),Julia 绑定也做了 axis 转换(`Insight.jl:726-729`)—— 说明「绑定层负责转换」是它既定设计,Lua 侧 axis 属于**漏做**。库内部本就不自洽 | `plan/foundations.md` §3.4 |
| R21 | 集合类返回裸 Lua 表 | **`parameters()` 一类返回 `pl.List`** | `pl.List` 是 1-based,与 R6 天然一致;`filter`/`map`/`reduce`/`__tostring` 免费。**但边界明确**:热路径与框架内部小数据(`x:shape()`、`_wrap` 参数解析、生成代码)一律裸表 | `plan/foundations.md` §2 |
| ~~R25~~ | ~~分层:冷路径 vendored `argcheck`,热路径 `_wrap`~~ | **本条已被 R26 推翻,当天** | 当时只测了性能(2597 ns/call),**没测「能不能表达」**。见 R26 | — |
| R26 | (R25 的结论)冷路径 vendored `argcheck` | **不 vendor argcheck。取它的规则表 schema + `usage.lua`,自写 ~150 行 O(N) 生成器 `_args.lua`** | argcheck 对每条规则枚举「给了/没给/显式 nil」三态,共 **3^N** 条路径。实测 9 个可选参数即 1.37 MB 生成代码 / 840 ms,**10 个就 `control structure too long` 编不出来**。而 `Conv2D` 11 个可选、`Adam` 12 个、`DataLoader` 16 个(3^16 = 4300 万,**直接挂死**)—— **R25 说的「冷路径」就是这张表,那条建议在它自己举的每个例子上都不成立**。更讽刺的是这个指数是为「位置参数中间省略、靠类型猜」付的,**而 Python 根本不允许这么写**。去掉枚举后同样的机制:代码量 O(N),3 参数 403 ns(快 6 倍),17 参数 3.6 KB / < 1 ms | `plan/foundations.md` §4.5 §4.6 |
| R27 | 参数解析器是 paddle-lua 的内部文件 `lua/paddle/_args.lua` | **独立项目 / 独立 rock,paddle-lua 与 Insight7 共用** | (人的决定)基座的作用域**本来就该是跨仓库的** —— 否则用户在同一脚本里面对两套规则表方言和两种报错格式,而 Insight7 的 dtype 常量与 Place 构造器已经是 Paddle 命名,接口层对齐是这个生态的既定做法。连带:**`_wrap.lua` 从 P5 清单删除**(留着就是第二套参数处理,违反 C11);降级路径归解析器自己管 | `plan/foundations.md` §5.4 |
| R28 | (R26 的落地形态)自写 `_args.lua`,零依赖 | **先普查再写:类型判定层借现成的,只自写「调用约定层」(~200 行);库基于 Penlight** | (人的追加约束)「最好别轻易造轮子,如果有优秀的类似于 argcheck 的现代库也是可以接受的」。普查了 luarocks 上全部候选:`tableshape`(MIT,零依赖,类型对象**本身可调用**)、`typecheck`(MIT,argcheck 式,但只有位置参数)、`checks`(**C 模块**)、`ltypekit`(`number -> number` 柯里化签名,颠覆语法)。**没有一个提供「同一份签名同时吃位置调用与具名表调用 + 默认值 + usage」** —— 因为那是 **Python kwargs 的形状**,而 Lua 自己没有 kwargs。所以:缺的那 ~200 行自己写;**类型判定不自己写** —— `type` 槽定义成可调用契约,`tableshape` 的类型对象直接塞进去就能用,它是可选增强不是依赖 | `plan/foundations.md` §5.4.5 |
| R29 | 生成器「一条规则一个 upvalue」(argcheck 的做法) | **所有 per-rule 数据放进一个表 upvalue,生成函数的 upvalue 数恒为 2** | Lua 5.1 实测三道墙:**60 个 upvalue** / 200 个局部变量 / **N=122 时 `function or expression too complex`**。Paddle 的真实最大签名是 `ResNetBasicBlock.__init__` **43 参数(33 可选)**(`python/paddle/incubate/xpu/resnet_block.py:434`),≥30 参数的有 9 个、≥12 参数的有 156 个。43 ×(谓词 + 默认值)≈ 86 个 upvalue **> 60** —— **argcheck 即使没有 3^N 也编不出 Paddle 最大的那个签名。** 改成单表 upvalue 后实测:50 参数 7.4 KB / ~1 ms / 3.75 µs;N > 100 切「表形态」,实测 1000 参数仍能编译 | `plan/foundations.md` §5.4.6 |
| R30 | **vendor Penlight 受限子集**到 `lua/paddle/_vendor/pl/`(D25 / `foundations.md` §1.3) | **声明为 rock 依赖,全生态统一,不 vendor** | (人的决定)「pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西」。原方案四条理由:第 1 条(避开 `lfs`)在 R23 已作废,第 3、4 条是便利性,**只剩版本锁定** —— 而版本锁定不需要靠 vendor(rockspec 锁 minor + CI 跑一组 `pl.class` 语义测试同样挡得住,且升级路径是改一行而不是改 5374 行)。决定性的是原文自己写下的代价:vendor 后**系统 Penlight 与我们那份互不 `is_a`**。当 Penlight 成为**整个生态的地基**(paddle-lua / Insight7 / `argrule` / metrics / ocean 五个消费者),这个坑就从一条文档注意事项变成**每两个库之间各出现一次**。声明依赖让它彻底消失。代价:`luafilesystem` 进入传递依赖 —— 但**我们自己不 `require "lfs"`**,文件系统仍走 `paddle.utils.fs` | `plan/foundations.md` §1.3 |
| R31 | `shape` 一类参数写成三个类的联合 `{"table", "pl.List", "insight.Array"}`(本条是我几小时前刚写下的) | **判据是结构:`IntList` = `list_of("integer")` = 「是个容器 + 装的是整数」** | (人的原话)「反正需要是个容器」「而且装的是整数」。联合类型把「容器」硬编码成一张框架名单 —— 用户自己的 `Vector`、别人的 `array` rock、明天的第四种容器全被无理由挡住,而这正是 §4「零框架硬编码」要挡的。改成 `argrule.list_of(elem)` 组合子(探容器协议 + 逐元素套 `elem`)后,库里一个框架名都没有,`IntList` / `TensorList` 只是宿主注册的别名。**连带一个意外收获:元素逐个查把 `concat` 的调用歧义也消掉了** —— `concat{a,b}` 的 ② 解释里 `x = a` 是 Tensor 不是 TensorList,直接出局;`concat{{a,b},2}` 的 ③ 解释里元素 `2` 不是 Tensor,也出局。**所以「第一个参数是列表的函数必须写 `nonamed`」这条(同样是我上一版写的)作废** | `plan/argrule.md` §2.3 ⑧ |

---

## 2. 决策理由(按主题)

### 2.1 为什么全部用 sol2 而不是裸 C API / FFI(D1)

绑定层开销 50-200ns,kernel 启动 1-10µs。**差 1-2 个数量级。**
为了省这个而维护 FFI + sol2 两套绑定,是典型的过早优化。
sol2 的类型安全和元表管理能力值得这点开销。

### 2.2 为什么还要 C ABI 中间层(D2)

三条,任一条都够:
1. sol2 是重模板库,绑 2000+ 算子的 TU **×5 个 Lua 版本**重编 = 编译时间灾难。中间层是纯 C,只编一次
2. C++ 异常不能穿过 Lua 的 `longjmp`(5.1/LuaJIT 尤甚)→ 必须有一层强制转 status
3. 将来某个 Lua 版本 sol2 不支持时(如 5.5),可绕过 sol2 直接接中间层

### 2.3 为什么砍 5.5(D4)

sol2 无 `compat-5.5.h`;上游 3 个 issue 全 open(#1721/#1723/#1747,最新 2026-02)。
且失败形态很坑:`lua_version.hpp:105` 会让 `SOL_LUA_VERSION=505` 一路放行,
**在真正用到 5.5 变更过的 C API 时才炸,表现为一堆莫名编译错误而非清晰报错。**

代价近乎为零且可逆:中间层是纯 C ABI,将来加 5.5 有两条路(等 sol2,或 5.5 单独走裸 C API),
**架构上不需要为 5.5 预留任何东西。**

### 2.4 为什么 Lanes 强制依赖(D8)

可选依赖 → Tensor 双表示 → 两条代码路径 → 各自测试 → 用户 bug 环境发散。
强制依赖 → 单一表示,少一个 M0 验证项。
代价只有发行包体积,Lanes 是纯 C++ 无外部依赖小库。

### 2.5 为什么优先级里 B 优先于 A(D12)

**测不了的代码即使"直接能抄"也是负债。**
它以"已完成"姿态进入代码库,在第一个真实用户手里炸。
典型例:单卡 `all_reduce` 等于恒等映射,**永远看起来是对的**。

A2(要重写)的成本是**看得见的时间**;B2(测不了)的成本是**看不见的债**。
**宁可先付看得见的。**

### 2.6 为什么统统 1-based(D13)

1. 混合方案的认知负担**是持续的** —— 纯 1-based 只学一次,混合方案要在每个调用点判断
2. Torch7 先例:切片、`dim`、返回的 index tensor、`ClassNLLCriterion` target 全 1-based,活了很多年
3. 边界在库里(生成器一处转换),不在用户代码里

**代价(已知且已接受)**:
- (a) 需人工维护 index 语义标注表,**长期成本**,必须配 CI 强制
- (b) 跨语言数据互操作会碰壁(Python 存的 label 是 0-based)
- (c) **但 1-based 下 `label==0` 本身非法 → off-by-one 从静默变成可检测**

### 2.7 为什么先主框架、Ocean/Metrics 排 M3(D11)

除了依赖方向,还有一条:
**Ocean 和 Metrics 是核心层 API 质量的验收器。**
先写它们,只会用一个还没定型的底座试错;
核心稳定后再写,它们能诚实暴露 API 缺口,而不是让核心去迁就它们。

### 2.8 为什么 trace 先于 script(D20)

script 需要读源码 → 引入"用户必须提供未加密源码"的契约。
trace 零源码依赖。

**把约束限制在自愿使用的高级功能里,而不是让它污染基线能力。**
加密分发的项目照样能用基线静态图。

### 2.9 为什么要指定一个"生态基座"(D23–D26)

Lua 的标准库极小:没有类、没有 list、没有跨版本兼容层、没有安全的 table 字面量读取。
**不指定基座的后果不是"多写点代码",是"最后有三套 class、两种 list"** ——
`nn.Layer` 一套、`optimizer` 一套、`dataset` 又一套,互相不认。

选 Penlight 的价值**不在省代码,在于让所有模块的对象长得一样**,
并且让用户已有的 Penlight 知识直接可用。

选 Insight7 的价值在于它的 Lua API **本来就是照着 Paddle 设计的**
(dtype 常量、Place 构造器、字符串切片、`_wrap` 三模式),
换任何别的数组库,用户都要同时记两套写法。

~~⚠️ 代价是一个真实的阻塞项:Insight7 的 Lua 绑定 `axis` 是 0-based。~~
✅ **已解决方向(2026-08-03,人的决定):按 bug 修 Insight7,改成 1-based,
「以后会顺手修复的」。成员索引与维度索引都要 1-based。**

它成为一等公民之后,用户会在同一个脚本里写 `ins.sum(a, 1)` 和 `paddle.sum(t, 1)` ——
这两个 `1` 含义不同,正是 D13 要消灭的静默 off-by-one。
但这不是「Insight7 的设计」而是「Lua 绑定的遗漏」:
同一个绑定里切片已经是 1-based,Julia 绑定的 axis 也已转换。
详见 `plan/foundations.md` §3.4。**待拍板 P1 关闭,Insight7 的一等公民定位(D26)不再是有条件的。**

### 2.10 为什么取消「不引入新的强制 C 依赖」(R23)

这条规则的问题不在严格,在于它是**数量规则**:它把所有 C 依赖当成同一件事。
实际上代价差得很远 ——

- 往我们**已经存在**的那个 `.so` 里加一个函数:边际成本 ≈ 0
- 新增一个 rock:边际成本 = **5 Lua 版本 × 3 OS = 15 个构建组合**,永久

按数量禁,结果是为了守住"0"而把下载(P12 §3.1)和 JPEG 解码(P12 §3.3)
硬拗成"调外部 curl"、"v1 只支持二进制原始格式" —— **代价转嫁给了用户,而不是消失了。**

新判据见 `CLAUDE.md` §9.1。**取消的是禁令,不是成本核算。**

被这条解禁的两件事:
| 能力 | 之前的形态 | 现在 |
|---|---|---|
| HTTP 下载 | 调外部 `curl`,或让用户手工下载 | `luasocket` + `luasec`,**Q-08 的风险大幅下降** |
| JPEG/PNG 解码 | v1 不支持,延到 M2 | 可在 v1 做,**ImageNet 级数据集不必等 M2** |

### 2.11 argcheck:为什么最后是「取形不取实」(R25 -> R26)

人的判断是:「argcheck 更全更好用,就是过时而且硬编码了 Torch7」。
实测下来**三个词全不准**,但**不准的方向和我第一次给的结论也不一样**:

| 人的印象 | 实测 |
|---|---|
| 硬编码 Torch7 | **不成立。** 耦合共 32 行、2 处,且 `env.istype` 上方原注释就是 `-- user configurable function` |
| 过时 | **不是过时,是在 Windows 上从来没能用。** `graph.lua:13` 用 `tostring(t):match('0x…')` 取地址,MSVC 的 `%p` 不带 `0x` -> 带默认值的规则集必崩。改 2 行,上游全套测试通过 |
| 更全更好用 | **声明式接口确实好,求解器不行。** 见下 |

#### 翻案的形状值得记住

第一次的结论是「vendored,冷路径用」,依据是性能:2597 ns/call vs `_wrap` 420 ns，
和小 kernel 同量级,所以「热路径不能用,冷路径随便用」。

**这个推理本身没错,错在它把注意力锁死在性能这一个维度上。**
性能数字难看但可接受,于是我停在了「划一条线」上,
没有再问一句**「冷路径的那些函数,它到底编不编得出来」**。

答案是编不出来。argcheck 对每条规则枚举三态、共 3^N 条路径:

| 可选参数 | 生成代码 | 结果 |
|---|---|---|
| 8 | 267 KB | 能编 |
| 9 | 1.37 MB / 840 ms | 能编,已经离谱 |
| **10** | 3.1 MB | ❌ `control structure too long` |
| **16**(`DataLoader`) | — | ❌ **挂死** |

而 `Conv2D` 11 个可选、`Embedding` 11 个、`Adam` 12 个、`LSTM` 13 个、`AdamW` 15 个。
**「冷路径」这个词指的就是这些类。**

**教训:一个方案被否,可能有多个独立的理由,而找到第一个「可接受的缺点」
会让人停止寻找第二个。「多慢」是程度问题,「能不能表达」是有无问题 ——
应该先问有无。**

#### 但代码生成这个机制是对的,要留下

argcheck 的思路是:把「遍历规则表、逐个查类型」这件解释式的工作
在**定义期**算完,产出直线源码,`loadstring` 编译,`debug.setupvalue` 注入默认值。
运行时规则表已经不存在。**这是好技术,我们保留。**

它真正的两个错误是:
1. **只特化了控制流,没特化类型判断** —— 生成的代码里还在调 `istype(x,"number")`
   这个解释期函数,70% 的运行时开销在这。既然都生成代码了,最该内联的就是它
2. **3^N 枚举** —— 为「位置参数中间省略靠类型猜」付指数代价,
   而这个功能 Python 根本不提供,我们的用户既不会写也不该写

去掉②、修掉①,同一个机制:3 参数 403 ns(快 6.4 倍),17 参数 3.6 KB / < 1 ms。
**代码量从指数变线性和速度快 6 倍是同一个原因。**

#### 所以最终形态

- **schema 照抄** argcheck(`name/type/default/defaulta/defaultf/opt/check/help`)—— 设计是好的,而且知识可迁移
- **`usage.lua` 移植**(151 行,BSD-3,保留版权头)—— 错误信息和 help 渲染重写没有价值
- **求解器自写**(~150 行)—— 只做「尾部省略 + 具名表」两条 O(N) 路径
- **`_wrap` 保留** —— `_args` 依赖 `debug` + `loadstring`,沙箱宿主里不通,
  这时降级为解释式实现。**这是 `_args` 比 argcheck 多出来的一件事**
- **不做重载引擎。** 类型分派用**函数体里的 `if`**,直译上游的 `isinstance` 链;
  参数校验层只需要 `type` 支持**联合类型**。理由:`to_tensor` / `reshape` 这些函数的
  候选「变体」**参数列表是一样的** —— 那不是重载,是联合类型 + 行为分支。
  且 Python 语法写不出 Torch7 那种前置可选参数,**Paddle 里不存在参数列表不同的重载**
  (`foundations.md` §4.8)
- **生成的 2000+ 算子不调 `_args`** —— 构建期展开,但**共用同一份 schema 与错误信息格式**(C11)

---

## 3. 待拍板(需要人决定,智能体不要自行决定)

| # | 问题 | 选项 | 倾向 |
|---|---|---|---|
| ~~P1~~ | ~~Insight7 的 Lua 绑定 `axis` 是否改成 1-based~~ | — | ✅ **已拍板(2026-08-03):改。** 人的原话:"Insight7 的 bug 以后是要修的,以后会顺手修复的,**成员索引和维度索引都要 1-base**"。按 bug 修(不是 breaking change),在 Insight7 仓库的 Lua 绑定里改,**顺手做、不专门排期,但要在 P12 之前完成**。落地细节见 `plan/foundations.md` §3.4,记为 R24 |
| P2 | 多卡测试环境何时具备 | — | 无环境则 `distributed` 一行不写 |
| P3 | 自研静态图第一版是否碰控制流 | 碰 / 不碰 | 建议不碰(trace 先行) |
| P4 | 源码契约走纯君子之约还是 + `string.dump` 验证 | — | 建议后者,待 M0 #20 |

---

## 4. 补充待拍板事项

| # | 事项 | 现状 |
|---|---|---|
| ~~P5~~ | ~~`state_dict` 容器下标破例为 0-based(R15)是否认可?~~ | ✅ **已认可(2026-08-03)。** 认可理由(人):"确实是被迫的 0-based,不过一般人也不会去看权重里面的东西"。已写进 `plan/modules/09-nn.md` §3.5 |
| ~~P8~~ | ~~`_args` 是否做成独立 rock~~ | ✅ **已拍板(2026-08-03):做。** 人的原话:「Insight 的那个解析器还是太简陋了,我打算弄一个单独的解析器项目,然后给 paddle 和 Insight 用」。记为 R27,落地细节见 `plan/foundations.md` §5.4 |
| ~~P10~~ | ~~解析器叫什么名字~~ | ✅ **已拍板(2026-08-03):`argrule`**(我建议的是 `argsig`,未采纳)。`local rule = require "argrule.rule"` 之后 `rule{...}(f)` 读起来是完整一句话;luarocks 实测未被占用;不含任何框架名。连带:`plan/argsig.md` -> `plan/argrule.md`,rockspec 依赖项改名。**剩下的动作:去 luarocks 占位** |
| P6 | P18 script 模式若工期紧张,是否同意直接砍掉? | 建议同意 —— trace 模式已覆盖绝大多数场景,做半个比不做更糟 |
| ~~P9~~ | ~~解析器「基于 Penlight」,那 Penlight 还 vendor 吗?~~ ✅ **已拍板(2026-08-03):不 vendor,全生态声明 rock 依赖。** 人的原话:「pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西」。记为 R30。~~原文:** Penlight 的 rockspec 声明 `luafilesystem`(已核实 `penlight-dev-1.rockspec:31-34`),而 §1.3 决定 vendor 正是为了避开 `lfs`。解析器声明 rock 依赖 + paddle-lua 继续 vendor = **生态里两份 Penlight,违反 C11** | **建议 A:两边都改成声明 rock 依赖,不再 vendor。** 代价是多一个 `lfs`,但 R23/D28 已取消「不引入 C 依赖」禁令,而 vendor 的三条理由里有两条是冲着 `lfs` 去的 —— 前提没了,结论也该重估。**拍板前按 A 写**~~(结果就是 A)(`plan/foundations.md` §1.3 §5.4.2) |
| P7 | **(R30 后改写)** Penlight 既然是 rock 依赖,版本区间怎么定? | 建议 `>= 1.13, < 2.0` + **CI 跑一组 `pl.class` 语义测试**(`_create` / `_class_init` / `call_ctor` 的 `super` 注入)。`pl.class` 若改了 `_create` 语义,我们的 `nn.Layer` 会**静默**坏掉,不是编译错误 —— 靠版本号挡不住,只能靠测试挡 |
