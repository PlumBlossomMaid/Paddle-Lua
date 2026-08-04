# OPEN_QUESTIONS.md · 未解问题

> 卡住时写这里,开工时扫一眼。
> 格式:**现象 / 已排查 / 需要什么才能继续 / 影响哪个任务**
> 解决后移到"已关闭",**不要删除** —— 后来者需要知道这条为什么不再是问题。

---

## 1. 开放中

### Q-01 · `WITH_PYTHON=OFF` + `ON_INFER=OFF` 是否被上游支持过

| | |
|---|---|
| 现象 | `cmake/configure.cmake:15` 有 `-DPADDLE_NO_PYTHON`,但守卫是 `(NOT WITH_PYTHON) AND ON_INFER` |
| 推断 | 上游预期的"无 Python"场景**是推理**;训练态无 Python 可能从未被编译过 |
| 需要 | 实际跑一次 T-M0-01 |
| 影响 | **全部**。项目级生死判定 |

### Q-02 · Lanes 的 `__lanesclone` 可用性与签名

| | |
|---|---|
| 现象 | 已决定用 sol2 usertype + `__lanesclone` 传 Tensor(D-R17),但该元方法在哪个 Lanes 版本引入、签名如何,未核实 |
| 已排查 | deep userdata 路线已调研并**主动放弃** —— `paddle::Tensor` 自带 `shared_ptr` 原子计数,不需要第二套 |
| 需要 | 拉目标版本 tag,确认 `__lanesclone` 存在与签名;确认 Lanes 不会对 userdata 做按位 memcpy |
| 影响 | T-M0-04 / T-M0-05 / P13。**不通过则走句柄表退路,多线程 DataLoader 依然成立** |

### Q-03 · Lanes 是否有 `on_state_create` 类钩子

| | |
|---|---|
| 现象 | 每个 lane 需初始化 Paddle tracer(`g_current_tracer` 初值 `nullptr`),但 Lanes 不知道 Paddle 存在 |
| 需要 | 查 Lanes 文档/源码确认钩子存在;不存在则需另找注入点 |
| 影响 | T-M0-06 / DataLoader 多 worker |

### Q-04 · `AttributeMap` 在 C++ 侧手工构造的难度

| | |
|---|---|
| 现象 | Python 侧靠 pybind 的 dict→AttributeMap 转换喂 `run_program_ad_func`;我们要在 C ABI 里手工构造 |
| 需要 | 读 `framework::AttributeMap` 定义 + 看 `eager_custom_python_api.h:136` 附近怎么组装的 |
| 影响 | T-M0-18 / 静态图嵌动态图路径(M3) |

### Q-05 · `x["::-1"]` 是否零拷贝

| | |
|---|---|
| 现象 | Insight7 的 `Array` 直接改 strides;Paddle 走 `strided_slice` 算子,是否 view 取决于 `FLAGS_use_stride_kernel` |
| 需要 | T-M0-14 |
| 影响 | 性能承诺;不通不致命(退回 `flip` + 拷贝语义) |

### Q-06 · `SaveTensor`/`LoadTensor` 的文件格式是否跨版本稳定

| | |
|---|---|
| 现象 | 它是 `framework/io/save_load_tensor.h` 的内部工具,没有公开的格式规范 |
| 风险 | 若格式随 Paddle 版本变,不适合做默认存档格式 |
| 需要 | T-M0-19;必要时改用 safetensors |
| 影响 | D19 / T-M1-15 |

### Q-07 · `OpProtoHolder` 链在 `WITH_PYTHON=OFF` 下是否还活着

| | |
|---|---|
| 现象 | `cpp_extension` 的 `parse_op_info` 走旧 fluid 静态图算子表 |
| 影响 | `cpp_extension` 可行性(M3);不通则整块延到 M3+ |

---

### Q-08 · HTTP 下载 / 解压 / 图像解码的来源

| | |
|---|---|
| 现象 | `paddle.dataset` / `paddle.vision` 需要下载、解压、解码 JPEG;Lua 无标准库支持 |
| ~~原退路~~ | ~~调外部 `curl`;v1 只支持二进制原始格式数据集~~ —— **这两条退路是"不引入新的强制 C 依赖"逼出来的,该规则已于 2026-08-03 取消(R23)** |
| 现在的选项 | ① 我们自己那个已链着 libpaddle 的 `.so` 顺手导出(边际成本 ≈ 0,优先);② 引入 `luasocket` + `luasec`(HTTP+TLS)、图像解码库 —— 现在**允许**,但要按 `CLAUDE.md` §9.1 给出 5 Lua × 3 OS 的覆盖证据与降级路径 |
| 需要 | 先查 Paddle C++ 侧有没有现成的(方案 ①);没有就按 ② 选库并核实 rock 覆盖 |
| 影响 | P12 的工期。**风险已大幅下降** —— 原本的"做不了"变成了"选哪条路" |

### Q-09 · `pir::Program` 的构造 API

| | |
|---|---|
| 现象 | P15 要在 C++ 侧手工组装图,但具体构造接口尚未调研 |
| 需要 | P15 开工前做一次专门调研,结论回填 `plan/modules/15-to-static.md` §2 |
| 影响 | P15。**在调研完成前,`15-to-static.md` §3.2 是纸面推测** |

### Q-10 · 可绑定的集合通信 C++ 接口清单

| | |
|---|---|
| 现象 | `plan/modules/17-distributed.md` §4 的设计是纸面推测,未核实上游可绑定接口 |
| 需要 | 调研 `paddle/fluid/distributed/collective/` |
| 影响 | P17。但因该阶段本就无法验收,优先级低 |

### Q-13 · 能否为 `insight::Array` 实现 `phi::TensorBase` 子类以做零拷贝

| | |
|---|---|
| 现象 | `paddle::Tensor` 的可用构造是 `Tensor(std::shared_ptr<phi::TensorBase>, const std::string&)`(`paddle/phi/api/include/tensor.h:142`)。要零拷贝接管 `insight::Array` 的缓冲,需要一个持有该 `Array` 引用的 `TensorBase` 子类,否则 Array 先析构会悬垂 |
| 已知 | Insight7 侧接口齐全:`data()` `array.h:134-135`、`shape()` `:109`、`dtype()` `:112`、`place()` `:115`、视图构造 `:84` |
| 需要 | 核实 `phi::TensorBase` 是否是可在下游继承的公开抽象类,以及 `DenseTensor` 对 `Allocation` 的要求 |
| 影响 | 只影响大数组预处理的吞吐。**不阻塞** —— 第一版走拷贝互转即可 |

### Q-14 · `insight::Array` 的拷贝构造是共享还是深拷贝

| | |
|---|---|
| 现象 | `include/insight/core/array.h:72` `Array(const Array &other)` 存在,但语义未核实。若是共享缓冲 + 原子计数,则可照搬 Tensor 的 `__lanesclone` 方案(D24) |
| 需要 | 读 `src/core/array.cpp` 的拷贝构造实现 |
| 影响 | P13 的 Insight7 预处理路径。保底方案与 Tensor 一致(整数句柄表过 Linda),**不阻塞** |

### Q-15 · vendored Penlight 与用户系统 Penlight 并存

| | |
|---|---|
| 现象 | 我们 vendor `pl.*` 到 `lua/paddle/_vendor/pl/`;用户环境里可能另有一份系统 Penlight。两份 `List` 类的实例互不 `is_a` |
| 倾向 | 文档说明"跨库传集合时传裸表,不传 `List` 实例",并暴露 `paddle.pl` 让用户能拿到我们这一份 |
| 影响 | 用户体验,**不阻塞** |

### Q-16 · Lua 5.2 的 `ipairs` 是否尊重 `__index`

| | |
|---|---|
| 现象 | 实例 raw 表保持空(D-R17)导致 `ipairs(layer)` 的行为**跨版本不一致**:5.1/LuaJIT 的 `ipairsaux` 用 `lua_rawgeti`(`lbaselib.c`),一次都不迭代;5.3/5.4 用 `lua_geti`,能正常走 `__index`。**5.2 属于哪一边未核实** |
| 为什么要问 | `nn.LayerList` 改为继承 `Layer`(R22)后,`ipairs(ml)` 是最自然的写法。我们已决定**不救它**、改用 `ml:iter()`,但测试要断言五个版本的实际行为,得先知道 5.2 的答案 |
| 需要 | 读 Lua 5.2 的 `lbaselib.c`,或直接在 5.2 上跑一次(M0 可顺手做,不需要 libpaddle) |
| 影响 | 只影响测试断言与文档措辞。**不阻塞** —— 结论无论哪一边,处置都是"文档只用 `ml:iter()`" |

### Q-17 · LuaJIT 上 `debug.setupvalue` 能否给 `load` 出来的函数注入 upvalue

| | |
|---|---|
| 现象 | argcheck 的整个机制是「生成源码 -> `loadstring` -> 逐个 `debug.setupvalue` 注入 upvalue」(`utils.lua`)。**这条路在 LuaJIT 上没验过。** |
| 为什么要问 | 它是 D-R25 的**命脉**。挂了不是"慢一点",是**冷路径 API 全线退回 `_wrap`** |
| 需要 | M0 #22。**纯 Lua,不需要 libpaddle,现在就能做** |
| 影响 | **必须在 P5 之前有答案。** P5 之后再翻案,所有已写的构造期签名要重写一遍 |

### Q-18 · 「冷路径 / 热路径」的边界靠人认还是机器判

| | |
|---|---|
| 现象 | R25 把参数检查分成两层,判据是「调用频次」。这是个**语义**判据,grep 判不了 |
| 倾向 | 先靠 api 文档里逐项标注(`plan/api/README.md` §2.1)+ code review;**如果开始出现误用,再上基准测试门禁**(某函数 >1 µs/call 即失败) |
| 为什么先不上 | 过早的机器门禁会卡住合理的例外,而现在还没有足够的样本知道例外长什么样 |
| 影响 | 不阻塞。**但不定会慢慢烂掉** —— 和 index 语义标注表是同一类长期维护债 |

## 2. 已关闭

| # | 问题 | 结论 | 关闭时间 |
|---|---|---|---|
| C-01 | sol2 能否支持 Lua 5.5 | ❌ 不能。无 `compat-5.5.h`,3 个 issue 全 open。**5.5 出局** | 2026-08 |
| C-02 | Lanes 能否做成可选依赖 | 不需要回答 —— **已决策强制依赖** | 2026-08 |
| C-03 | Torch7 堆追踪是否默认开启 | ✅ 默认开(`init.lua:189`)。我们也应默认开 | 2026-08 |
| C-04 | `.pdparams` 是不是 pickle | ✅ 是,protocol 4,内容只有 `dict[str, np.ndarray]`,无 Paddle 类 | 2026-08 |
| C-05 | 是否需要改 Paddle 上游 | ❌ 不需要。`tensor.h` 公开 API 足够 | 2026-08 |
| C-06 | Lua 有没有可用的 AST 库 | ✅ 有。luacheck `parser.lua`,5.1-5.4+LJ,MIT,自身跑在 5.1 | 2026-08 |

### ✅ Q-11 · `LuaTensor` 用 sol2 usertype 还是 Lanes deep userdata(2026-08-03 关闭)

| | |
|---|---|
| 原冲突 | `02-binding.md` 写普通 sol2 usertype,`13-lanes.md` 要求 deep userdata,两者不兼容 |
| 结论 | **与 sol2 对齐,不用 deep userdata。** 跨 lane 靠 `__lanesclone` + `paddle::Tensor` 自带的 `shared_ptr` 原子计数 |
| 为什么 | deep userdata 的价值是"给共享 C 对象做原子引用计数",而这件事 `shared_ptr` 已经做了。套两层是重复劳动,还把 P2 绑死在 Lanes 上 |
| 遗留 | `__lanesclone` 的签名待核实(转为 Q-02);不可用时走句柄表退路 |
| 决策记录 | D-R17 |

### Q-12 · Insight7 Lua 绑定的 `axis` 是 0-based —— ✅ 2026-08-03 已决(待实施)

| | |
|---|---|
| 现象 | `bindings/lua/insight/reduction.lua:18` `native.sum(x, axis, keepdims or false)`、`indexing.lua:40` `native.argsort(x, axis or -1)` —— **直接透传,无 1-based 转换**。而 `bindings/julia/Insight.jl:726-729` **做了转换** |
| 关键发现 | 同一个 Lua 绑定里,**成员索引/切片已经是 1-based** —— `bindings/lua/insight_lua.cpp:212-266` 的 `lua_spec_to_cpp` 做了转换。所以 Insight7 的 Lua 绑定**内部就不自洽**:`a["1:3"]` 1-based,`ins.sum(a,1)` 0-based |
| 结论 | **人已拍板(P1):改。** 原话"Insight7 的 bug 以后是要修的,以后会顺手修复的,成员索引和维度索引都要 1-base"。按 **bug** 修(不是 breaking change),依据是 Julia 绑定已转换 + Lua 侧切片已转换,axis 属于漏做 |
| 待办 | 在 Insight7 仓库改 `bindings/lua/insight/*.lua`,统一走 `to_c_axis(axis, ndim)`;负数轴保持原样;`axis = 0` 报错。**顺手做,但要在 P12 之前完成。** 落地细节见 `plan/foundations.md` §3.4 |
| 为什么留在这里 | 它从"未解问题"变成"已决待实施",**不是已完成**。修完之前 P12 的 transforms 不能依赖 axis 语义 |
