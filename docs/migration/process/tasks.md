# TASKS.md · 任务板

> 状态:`TODO` / `DOING` / `DONE` / `BLOCKED` / `BLOCKED-ENV`
> **只做前置条件已满足、且在当前闸门允许范围内的任务(见 `process/status.md` §1)。**
> 完成后更新状态与实际耗时。

---

## 闸门 G0 前 · 只允许这些

### T-M0-01 · WITH_PYTHON=OFF + ON_INFER=OFF 编译 🔴 生死判定

| | |
|---|---|
| 状态 | ✅ **DONE(2026-10-22,实际 2 天)** |
| 前置 | 无 |
| 预估 | 3-5 天 |
| 依据 | `plan/overview.md` §11.3 |

**做什么**
```bash
cd $PADDLE_ROOT && mkdir build-nopy && cd build-nopy
cmake .. -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=OFF \
         -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

**同时收集(= M0 #16,不额外花时间)**
- `new_executor` / `jit` 两个 target 是否产出
- 最终产物里 `egr::Backward`、`paddle::Tensor`、`jit::Layer::Load`、
  `framework::LoadOpMetaInfoAndRegisterOp`、`RegisterOOMCallback` 是否导出
  (`nm -D` / `dumpbin /EXPORTS`)

**验收**
- [ ] 编译成功,或失败原因被完整记录到 `process/m0-report.md`
- [ ] 上述 5 个符号的导出情况已确认
- [ ] `$PADDLE_ROOT` 的 `git status` 仍然干净(C2)

**失败时**:不要试图 patch Paddle。记录全部编译错误,分类(缺 pybind 头 /
条件编译遗漏 / 链接缺失),报告给人。

---

### T-M0-02 · 反向传播冒烟

| | |
|---|---|
| 状态 | ✅ **DONE(2026-10-22,实际 2 天,与 T-M0-01 并行)** |
| 前置 | T-M0-01 |
| 预估 | 1 天 |

写一个纯 C++ 程序,链接 T-M0-01 的产物:
```cpp
// 出处:paddle/fluid/eager/backward.h
paddle::Tensor x = ...;  x.set_stop_gradient(false);
auto y = /* 某个 *_ad_func */;
egr::Backward({y}, {}, false);
// 检查 x.grad() 有值且数值正确
```

**验收**:`x.grad()` 数值与手算一致。

**结果**:✅ 通过。
- 冒烟源:`tools/smoke/smoke.cc`(留在仓库,含全部 `文件:行号` 出处注释)
- 计算图:`a=full({2,2},3)`,`b=full({2,2},4)`,`c=multiply_ad_func(a,b)`,`d=sum_ad_func(c)`,`egr::Backward({d},{})`
- 手算:`d/da == b == 4.0`,`d/db == a == 3.0`
- 实测输出:`SMOKE PASS: a.grad == 4.0, b.grad == 3.0, max_err=0`
- 前置陷阱(已写进 smoke.cc 注释):必须显式 `paddle::framework::InitMemoryMethod();`
  (出处 `paddle_gtest_main.cc:73`,`init.cc:443`),否则 fail-fast 0xC0000409;
  kernel 注册需写 `PD_DECLARE_KERNEL(full, ...)` 等(照 `backward_test.cc` 范例),
  实测无 `PD_DECLARE_KERNEL` 时 `full` 报 `operator is not registered`
- 链接:EXE 链接需 `/FORCE:MULTIPLE`(见 `process/m0-report.md` §4 的怪癖记录)

---

### T-DOC-01 · 文档一致性巡检

| | |
|---|---|
| 状态 | TODO |
| 前置 | 无(闸门 G0 前允许) |
| 预估 | 0.5 天 |

- [ ] 所有文档里的 `文件:行号` 出处在当前 Paddle 版本上仍然成立
- [ ] `AGENTS.md` §2 决策台账与各专题文档结论一致
- [ ] `process/status.md` §4 与 `plan/overview.md` §8 的 M0 清单条目一一对应

---

## 闸门 G0 后 · M0 其余验证

> 每项产出写入 `process/m0-report.md`,并更新 `process/status.md` §4。

| ID | 任务 | 前置 | 预估 | 状态 |
|---|---|---|---|---|
| T-M0-03 | phi kernel / DeviceContext 多线程共享安全性 | T-M0-01 | 2d | ✅ **DONE(2026-10-22)**:`tools/smoke/smoke_mt.cc` 多线程冒烟 —— 4 线程×50 轮 + 8 线程×200 轮(共 1800 次 full→multiply→sum→Backward 完整图),共享框架 DeviceContextPool,全部 `a.grad==4.0 / b.grad==3.0 / max_err=0`。结论:CPU 上多线程共享 DeviceContext 安全,无需 per-thread context |
| T-M0-04 | Lanes v3.17.x deep API 与 4.x 差异核对 | — | 1d | ✅ **DONE(2026-10-22)**:已核对(v3.17.2 tag + master@4aa54c9 双 clone)。**`__lanesclone` 两版都有且调用约定一致**(`clone:__lanesclone(dest, source, size)`,`lua_call(L,3,0)`;`tools.c:1616-1618` vs `intercopycontext.cpp:869-871`)——D24 依赖的接口面不变。**deep userdata API 两版结构不同**:v3.17 是 C API(`deep.h`: `luaG_newdeepuserdata`/`luaG_todeep`/`DeepPrelude` struct),4.x 是 C++ class(`deep.hpp`: `DeepFactory`/`DeepPrelude`,NVI 虚函数)——但 D24 已改走 `__lanesclone`,不依赖 deep userdata,故差异无影响。**`on_state_create` 两版都存在**(v3.17 `state.h:19-20`/`universe.h:60`;4.x `state.cpp:212`/`keeper.cpp:1126`),T-M0-06 顺带确认。详见 `m0-report.md` §9 |
| T-M0-05 | Lanes 与绑定的 C++ ABI 一致性(MSVC) | T-M0-04 | 1d | ✅ **DONE(2026-10-22)**:`tools/smoke/lanes_abi_check.cc` 实测 —— Lanes v3.17.2 的 10 个 C 源文件用 MSVC 编译为 obj,从 **C++ TU** 链接 Lua 5.1(LuaJIT ABI,`lua51.lib`)+ 调用 `luaopen_lanes_core` 创建模块表,运行输出 `LANES ABI OK`(exit 0)。**v3.17.2 是纯 C 实现**(10 .c / 0 .cpp,`PROJECT(lanes C)`)—— §9.5(c) 的"虚函数 ABI"担忧只适用于 4.x,不适用于 v3.17。**实测陷阱**:Lanes 与调用方必须用**同一份 Lua 头**(先前 PUC 5.1 头编 Lanes + LuaJIT 头编调用方 → 运行即崩;统一 LuaJIT 头后通过)。详见 `m0-report.md` §10 |
| T-M0-06 | Lanes `on_state_create` 钩子是否存在 | T-M0-04 | 0.5d | ✅ **DONE(2026-10-22)**:存在(v3.17.2 `src/state.h:19-20` `initialize_on_state_create`/`call_on_state_create` + `universe.h:60` `on_state_create_func` 字段;4.x 同有 `state.cpp:212`/`keeper.cpp:1126`)。语义:每个新 lane 创建时由 keeper 调用一次,`lua_CFunction` 签名;正好承载"每个 worker lane 重新 `require "paddle"`"的初始化(13-lanes.md §4.4,Q-03 关闭)。详见 `m0-report.md` §9/§10 |
| T-M0-07 | `RegisterOOMCallback` + `lua_gc` 救回 OOM | T-M0-01 | 1d | ✅ **DONE(2026-10-22)**:`tools/smoke/oom_smoke.cc` 实测 PASS。用 MockAllocator(前 N 次抛 `BadAlloc`)+ `RetryAllocator` 验证语义:① 回调返回释放字节数 >0 -> 重试直至成功(2 次回调);② 回调返回 0 -> 只调一次、`BadAlloc` 逃逸;③ 无回调 -> 不触发回调重试。输出 `OOM SMOKE PASS`。**两个实测要点**:① `offload_retry_times` 默认 **-1**(`common/flags.cc:2363`),不开重试路径,必须设为 >0;② `RetryAllocator` 外层还有 waiter 重试(`retry_allocator.cc:88-115`),即使无回调也会等 `retry_time` 后重试最后一次。**绑定层落地**:启动时设 flag + 注册回调(内做 `lua_gc`),CPU 分配链需手动包 `RetryAllocator` 或依赖 GPU 路径自动包。详见 `m0-report.md` §11 |
| T-M0-08 | LuaJIT GC64 确认 | — | 0.5d | ✅ **DONE(2026-10-22)**:`tools/ci/lua_gc64_check.lua` 实测 PASS。vcpkg LuaJIT 2.1.1774896198(x64/Windows)输出 `lua-gc64 OK: 16-byte GC64 layout confirmed`。判定依据:① 源码 `lj_arch.h:168/181/216-217/303/459` x64 默认 `LJ_TARGET_GC64=1`(除非 `LUAJIT_DISABLE_GC64`);② 运行时 userdata 地址间隙为 16 的倍数(GC64 布局);③ `ffi.sizeof("void*")==8`。**结论**:本机/CI 的 LuaJIT 是 GC64,userdata 头 16 字节 —— 绑定层按此布局,无需兼容非 GC64 |
| T-M0-09 | `__close` 对 userdata 在 5.4 上的行为 | — | 0.5d | ✅ **DONE(2026-10-22,实际 ~1d)**:`tools/ci/lua_close_check.lua` 在 Lua 5.4.7(源码 MSVC 编译)与 5.1.5(choco)上实测 PASS,已登记 l0.yml lua-matrix 五版本。**Case A**:`<close>` 变量作用域退出触发 `__close`(表也触发 —— to-be-closed 机制对任何有 `__close` 的对象生效)。**Case B 关键发现**:GC 只走 `__gc`,**不调 `__close`** —— 用 `io.tmpfile()` 真实 userdata + `debug.setmetatable` 同挂 `__gc`/`__close`,实测 GC 后 `closed_by=="gc"`。**绑定层结论:RAII usertype 必须同时挂 `__close`(5.4 `<close>` 确定性释放)与 `__gc`(兜底释放),两者需幂等**。详见 `m0-report.md` §13 |
| T-M0-10 | 堆追踪 vs Lanes 自定义分配器 | T-M0-04 | 1d | ✅ **DONE(2026-10-22,实际 ~2d)**:`tools/smoke/lanes_alloc_check.cc` 实测 PASS(`LANES ALLOC PASS`)。用计数 `lua_Alloc` 装在 master state 上(模拟堆追踪),加载真实 `lanes.lua`,`on_state_create` 钩子里 `lua_getallocf` 实测 3 个 lane state 的分配器**全部 != counting_alloc**(inherited=0)。**两个关键结论**:① **LuaJIT 下 master 的自定义 allocator 完全到不了 lane**(`state.c:253-255` `LUAJIT_FLAVOR()==64` 分支 lane state 一律 `luaL_newstate()`);PUC 才会复用(`state.c:257-271`)。② **实测陷阱:`lua_setallocf` 不能拿裸 `realloc/free` 替换**(vcpkg LuaJIT 是 `lua51.dll`,自己的 CRT 堆,free 跨堆 0xC0000374 堆损坏),**必须链式转发原分配器**(先 `lua_getallocf` 再换)—— 这正是 Lanes `protected` 模式的做法(`tools.c:173-190`)。**对绑定层的落地:堆追踪必须按 14-gc.md §3.1 做张量字节计数器(与 Lua allocator 无关),LuaJIT 上靠 allocator 钩子根本看不到 lane 分配。详见 `m0-report.md` §14 |
| T-M0-11 | 纯 Lua pickle 读真实 `.pdparams` | — | 1d | ✅ **DONE(2026-10-23,实际 ~1d)**:实现在独立仓库 `E:\code\lua-pickle`(rock/模块名 `pickle`,双向可逆设计,见 `lua-pickle/docs/design.md`)。unpickler 核心(reader.lua 栈机 + registry + ndarray 还原 + types 值模型)读通真实样本 `m0-11-sample.pdparams`(326B, protocol 4):`linear.weight`(2x3 f4)、`linear.bias`(2 f4)、`StructuredToParameterName@@` 全对,`tools/check.lua` 在**本机 Lua 5.1.5** 上与 Python `pickle.load` 生成的权威基准(m0-11-expected.lua)逐字节对拍 PASS(keys/shape/dtype/raw bytes)。实现细节:① opcode 分派对照 CPython `Lib/pickle.py`(68 个)、numpy 还原对照 numpy 2.4.6 实测(`dtype` REDUCE+BUILD 拼 `<f4`;ndarray `_reconstruct`+BUILD 5 元组 `(1,shape,dtype,fortran,raw)`);② 踩坑:Lua 5.1 `#` 对洞表未定义,`load_tuple2/3` 初版用 `stack[#stack]=nil` 收缩栈导致后续 REDUCE 索引错位,改 `table.remove` 修复;③ CI 框架定案(busted + pytest,见 design §8.1)。详见 `m0-report.md` §15 |
| T-M0-14 | `x["::-1"]` 走 `strided_slice` 是否 view + autograd | T-M0-01 | 0.5d | ✅ **DONE(2026-10-23)**:`tools/smoke/smoke_strided_slice.cc` 实测 PASS —— **eager 层 strided_slice 默认是 view,不是 copy**(旧记录"拷贝语义"是错的,那只是 fallback)。证据:`FLAGS_use_stride_kernel` 默认 **true**(`phi/core/kernel_factory.cc:31`),STRIDED kernel 做 `out->ResetHolder(input.Holder())` + 重写 meta.offset/strides(`phi/kernels/stride/strided_slice_kernel.cc:95-99`);实测 `py == px + 12B`(offset=3 元素)、写穿透 `py[0]=99` 后 `px[3]==99`、逻辑值 `y[i]==x[3-i]`。autograd 正常:seed=[1,2,3,4] → `x.grad=[4,3,2,1]`(`egr::Backward`,`strided_slice_grad_ad_func` 存在于 `dygraph_grad_functions.h:656`)。详见 `m0-report.md` §16 |
| T-M0-15 | `WITH_PYTHON=OFF` 下 `OpProtoHolder` 链是否可用 | T-M0-01 | 0.5d | ✅ **DONE(2026-10-23)**:**可用,且不依赖 pybind**。证据链:Python 侧 `OpProtoHolder`(`framework.py:3202`)只是 pybind 薄壳,`__init__` → `get_all_op_protos()`(`:3187`)→ `core.get_all_op_protos()`(pybind 绑定 `pybind.cc:2591`)= 遍历 `OpInfoMap::Instance().map()` + `HasOpProtoAndChecker()` + `Proto().SerializeToString()`(`pybind.cc:2591-2603`);C++ 侧 `OpInfo`/`OpInfoMap` 纯 C++(`op_info.h:42/139`,`op_info.cc:22`),且**上游无 Python 环境路径自己就在用**(`custom_operator.cc:1270` `OpInfoMap::Instance().Insert`)。build-nopy dumpbin 实测:`libop_info.lib` 里 `OpInfoMap::Instance` 与 `libop_registry.lib` 里 `OpInfoMap::Get` 均为 External 定义。**意义**:Lua 侧写一个等价 C ABI 函数即可替代 Python `OpProtoHolder`(运行期查 op proto),无需补丁;生成器构建期从 yaml 拿 schema(不依赖此链)。详见 `m0-report.md` §17 |
| T-M0-17 | `jit::Layer Load()` 加载 `jit.save` 产物并 forward | T-M0-01 | 1d | ✅ **DONE(2026-10-23,部分验证)**:`tools/smoke/smoke_jit_load.cc` 实测(下载上游 CI 产物 `multi_program_load_with_property.tar.gz`,4 文件 .pdmodel/.infer.pdmodel/.pdiparams/.meta)。**结论:`jit::Load` 在 `WITH_PYTHON=OFF` 下可链接、可加载**(`libfluid_jit.lib` 导出 `?Load@jit@paddle`;ProgramDesc/params/attrs 全读入;`FLAGS_jit_engine_type="New"` 时 InterpreterEngine 启动 `New Executor is Running` `program_interpreter.cc:255`)。**两个发现**:① 链接缺口 —— `PirFunctionInfo` 构造(`function_schema.cc:112/115`)无条件引用 `GetFeedTargetNames/GetFetchTargetNames`(`pybind/pir_utils.cc:20/36`),pybind 目录在 WITH_PYTHON=OFF 不产出 → 冒烟自带等价 shim 验证(纯 PIR);② forward 撞上游 fc 算子 + oneDNN bug(Predictor 抛 `OneDnnContext does not have the input W` `onednn_context.cc:347`;New 引擎 0xC0000005),**非无 Python 特有,记 Q-24**。不阻塞 G1;P8 jit 落地时用新导出产物。详见 `m0-report.md` §18 |
| T-M0-18 | 手工构造 `AttributeMap` 喂 `run_program_ad_func` | T-M0-01 | 1d | ✅ **DONE(2026-10-23)**:`tools/smoke/smoke_run_program.cc` 实测 PASS(exit 0,输出 `SMOKE PASS: hand-built AttributeMap reached C++ logic`)。签名纯 C++(`run_program_func.h:24-28`),`AttributeMap`/`Attribute` 可手工构造(`type_defs.h:51/73`,variant 成员含 `pir::Value` 向量/`shared_ptr<pir::Program>`),冒烟按 `RunProgramImpl` 消费的键全量填充(`run_program_impl.cc:449-490` + 461-466 scope 要求),空 program 触发预期 C++ Traceback、无 pybind 泄漏。**意义**:P16 静态图运行时参数可纯 C++ 构造,绑定层只需薄 C ABI 透传字符串键 + 原子值。详见 `m0-report.md` §19 |
| T-M0-19 | `SaveTensor`/`LoadTensor` 格式稳定性 | T-M0-01 | 0.5d | ✅ **DONE(2026-10-23)**:`tools/smoke/smoke_save_load.cc` 实测 PASS(exit 0,输出 `SMOKE PASS: round-trip ok (dims/dtype/data), version=0, file_size=50, re-save byte-identical`)。链 = `framework::SaveTensor/LoadTensor`(`save_load_tensor.h:20-24`)→ `phi::SerializeToStream/DeserializeFromStream`(`save_load_tensor.cc:24-45`);磁盘格式 = uint32 版本 0 + LoD + TensorDesc proto(dtype+dims)+ 原始数据(`dense_tensor_serialize.cc:69-81`/`dense_tensor_tostream.cc:97-228`),反序列化强校验版本与 LoD(`:133-140`/`:153-157`)。**Python 互通确认**:`pybind/io.cc:140` `load_dense_tensor` 就是 `framework::LoadTensor`(D19 成立)。重存字节级幂等(无时间戳/随机性)。**意义**:Lua 侧 `paddle.save/load`(单张量)直接以它为线格式,不用 pickle;view 保存时自动实体化(`dense_tensor_tostream.cc:99-107`)。详见 `m0-report.md` §20 |
| T-M0-20 | `string.dump` 往返比对(M4 用,可跳过) | — | 0.5d | TODO |
| T-M0-21 | luacheck parser 独立跑 5.1(M4 用,可跳过) | — | 0.5d | TODO |

---

## 闸门 G1 后 · M1

> ⛔ **G1 未开不要动。** 此处仅列骨架,细化推迟到 G1 通过后。

| ID | 任务 | 依据 | 预估 |
|---|---|---|---|
| T-M1-01 | `tools/lint_51.lua`(Lua 5.1 子集检查器) | C3 | 2d |
| T-M1-02 | vendor Penlight 受限子集(11 文件)+ 校验其在 5.1–5.4/LJ 上可加载 | `plan/foundations.md` §1.2 | 1d |
| T-M1-03 | `class.lua` extend/super 类系统 | `plan/overview.md` §6.1 | 3d |
| T-M1-04 | `_wrap.lua` 三模式调用(移植 Insight7) | D15 | 1d |
| T-M1-05 | C ABI 中间层骨架(生命周期/错误/dtype/place) | D2 | 5d | ✅ **DONE(2026-10-24)**:paddle_c.h(2.1.1)+ status.cc(2.1.2)+ tensor.cc(2.1.3)+ autograd.cc(2.1.4)+ 纯 C 迷你训练循环(2.1.5)+ **device.cc / jit.cc(2.1.6)**全部实现并实测 **MINI_TRAIN PASS + DEVICE_SMOKE PASS**(e069c9c→18ce534)。三处链接坑已解:① @rsp 必须 link.exe 直调 + 当前目录裸 `@capi_libs.rsp`(cl 转交 LNK1181);② 上游 **/MT + 30 宏**(build.ninja:2028/120),缺 PHI_INNER 致 LNK2019 dllimport;③ `Tensor(const Place&)`/`(Place,shape)` 是幽灵构造(tensor.h:121/135 只声明不实现)→ 换 `paddle::experimental::empty`(api.h:384)。2.1.6 新增坑:DeviceManager 不管理 CPU(count 硬返 1);jit::Load 对坏路径不抛异常返回空 Layer;jit 链接缺口 shim(GetFeedTarget/GetFetchTarget)必要。验收:梯度==手算、坏参数→PD_INVALID_ARG、双线程错误槽隔离,32 项全过。详见 status.md §2 |
| T-M1-06 | sol2 `usertype<Tensor>` + `__lanesclone` 槽位(D24) | D1/D8 | 4d | ✅ **DONE(2026-10-24)**:`csrc/sol/tensor_binding.cc` LuaTensor usertype(ndim/numel/dtype/shape/is_contiguous)+ `__lanesclone` 槽位已预留(aab6526),`paddle_core.tensor{dims}` 工厂 + `paddle_core.init()`(幂等,收进 C ABI `pd_init_once`)。SOL_SMOKE PASS。坑:① PD_GUARD 入口清错误槽——`pd::check()` 必须紧跟失败调用,中间不能插 `pd_place_free`(会清槽);② 嵌入 EXE 静态链 Paddle 又 require 同样静态链的 dll → gflags 双注册(`dygraph_debug`,flags_native.cc:309),嵌入者永不静态链 Paddle;③ sol2 编译 include 顺序:Lua 5.1 头在前、vcpkg include(仅取 sol)在后,否则按 Lua 5.5 头生成 5.1 没有的 API 致 LNK2019 |
| T-M1-07 | `slice.lua` + `lua_spec_to_cpp`(移植 Insight7) | D14 | 2d | ✅ **DONE(2026-10-25)**,实现形态演进过(layout.md P5 注记):`lua_spec_to_cpp` **在 C++ `csrc/sol/slice.cc`(2.2.4)以 `paddle_core.slice_spec` 落地**(纯字符串 1-based→0-based 重写,验证 `"1:3"→"0:2"` `"::-1"` 透传),不再需要独立 Lua 文件;Lua 侧消费在 `lua/paddle/tensor.lua`(2.5):`spec_to_strided` 把 0-based 字符串解析成 axes/starts/ends/strides 喂 `pd_op_strided_slice`。**关键语义(实测)**:字符串切片是**半开区间**(`"1:3"` = 第 1、2 行,1-based [1,3) → 0-based [0,2));负步长 stop 保持 `-dim-1` sentinel 交给 kernel `normalize_interval`(slice_utils.h:128)权威处理——Lua 侧只填 Python slice.indices 默认值,不许提前 clamp(否则 `"::-1"` 压成空)。TENSOR_SMOKE PASS |
| T-M1-08 | 算子生成器原型(yaml → C ABI + sol2 + `_ops.lua`) | C8 | 10d | 🟡 **2.3.2+2.3.3 已完(2026-10-24)**:`yaml_subset.lua` 4 份 yaml 对拍全绿 + `emit_capi.lua`/`emit_lua.lua` 端到端跑通,add/matmul/relu 产物落位(`csrc/capi_gen/` + `lua/paddle/_ops/`,lint 全绿 + 可重入)。**关键发现:add/matmul 不在 ops.yaml(529 块),在 `inconsistent/dygraph_ops.yaml`(38 块)**——生成器输入必须含它。**2.3.4 墙 #2 已完(2026-10-24,c7768da)**:capi_type 覆盖全部 attr 类型 + **4 种输入形态**(`Tensor?`→NULL 句柄 + `paddle::make_optional`、`std::vector<Tensor>`/`?`→`(const pd_tensor*, int64_t)` + prelude 构造),白名单 9 算子 MSVC 真实编译通过;emit_lua `lua_default_literal` 扩到 130 种组合(IntArray/vector/Scalar/DataType/Place)。**2.3.4 墙 #3 已完(2026-10-24,16a2917)**:多输出 C ABI = `int64_t pd_op_x(pd_tensor* outs, int64_t nout, ...)`(与 pd_jit_layer_forward 先例一致),emit_op_multi 逐元素 `std::get<N>` 解 tuple;**发现 `intermediate:` 输出不进 ad_func tuple**(lstm yaml 4 输出实际 2 个,yaml_subset 补 `ir.intermediate` + emit_capi `effective_returns` 过滤);4 个 runtime-length 输出算子(coalesce_tensor/distribute_fpn_proposals/einsum/rnn,tuple 含 `std::vector<Tensor>`)skip(需两阶段 ABI,另行设计)→ 全量 **529 ops** 生成全绿 + 可重入一致 + 白名单 8 算子(含 accuracy 3 出/argsort/topk/eigh/lstm)MSVC 编译通过;emit_lua 多输出 wrapper 放行(sol2 2.4 包多值)。**2.3.4 收尾①全量 1-based 转换已完(2026-10-25)**:转换分工 = **Tensor/`vector<Tensor>` index 整体 ±1 在 C ABI 层**(emit_capi `pd_idx_minus1/plus1`,调 `subtract/add(full_like(t,1,dtype,place))`)**、标量 axis/dim + 标量 takes_index attr 在 Lua wrapper**(emit_lua,含 yaml 默认值 `shift_axis_default` 平移为 1-based 字面量 + wrapper 内 `if v > 0 then v = v-1` 还原);intermediate 折叠分发修复(.cc 用 `#effective_returns > 1`,16 算子);**单 vector 输出 emit_op_vec**(9 算子:ap_facade/split/unstack/unbind/broadcast_tensors/meshgrid/split_with_num/sync_comm_stream/ap_variadic,yaml `Tensor[]{...}` 形态,非 skip)。**2.3.4 收尾② regen-diff CI 已完(2026-10-25)**:产物全量落库(emit_capi 529 ops → 24 .cc + ops_gen.h;emit_lua 529 wrapper)+ `tools/ci/regen_diff.lua`(原地重生成 → `git status --porcelain` 两目录必须为空);**顺带修掉 3 个生成器缺陷**(lint 529 全量抓出):① yaml 参数名撞 Lua 关键字(`arange`/`range_v2` 的 `end` → wrapper 局部名 `end_` + 具名表 `t["end"]`);② 浮点默认值中缀 `f`(`rrelu` 的 `1.0f/8` → `1.0/8`,`f$` 只剥结尾);③ emit_lua 头部 sha 作用域(emit_wrapper 拿不到 CLI 段 local)→ 改传参,头部从写死 `@ sha` 改为真实 sha。验证:lint 全绿、可重入 0 diff、MSVC 6 代表 .cc EXIT=0。**coverage ④ 已核实(2026-10-25)**:ops.yaml 529 forward/35 skip/494 生成,v1 口径(剔 25 个 in-place,M2 再加)= 494/504 = **98.0% ≥ 95% ✅**(raw 93.4%,缺口全在 skip_rules 注明理由)。**2.3.4 收尾③ sol2 算子绑定层已完(2026-10-25)**:`emit_sol.lua`(纯 Lua 5.1)生成 `csrc/sol/gen_ops_*.cc`(25 字母分片 + gen_ops_all.cc registry),529 个 `pd_op_*` 注册成 sol2 lambda(单出→LuaTensor / 多出→`sol::variadic_results` / 单 vector 出→query-then-allocate + `sol::as_table` 1-based);`enums_place.cc`(PD_DTYPE_*/PD_PLACE_* 常量 + LuaPlace + `pd_place_new`);LuaTensor 移入 `sol_common.hpp`。**三个运行期根因修复(全实测)**:① `const std::vector<T>&` 参数本环境必崩 0xC0000005 → 全改按值 `std::vector<T>`;② `Scalar(int)` 语义(yaml_subset 保留 `sub` + emit_capi 调用点 `Scalar(static_cast<int64_t>(...))`,ABI 仍 double)修 split "must be int, but got float";③ **`SOL_SAFE_FUNCTION_CALLS=1` 在 LuaJIT 下无效**——真正门控是 `stack.hpp:198` 的 `#if SOL_IS_OFF(SOL_PROPAGATE_EXCEPTIONS)`,LuaJIT 2.1+ 强制该宏 ON(`compatibility/lua_version.hpp:142-150`)→ 须加 `SOL_EXCEPTIONS_SAFE_PROPAGATION=0`,错误路径(add(1,2))从崩溃变 Lua error。accuracy 报错核实为**输入形状契约非绑定层**(AccuracyInferMeta 要 label rank2+第二维=1 `ternary.cc:55-58`,kernel 要 int64 `accuracy_kernel.cc:22`)。验证:20-op 白名单 `OPS_SMOKE PASS`(单出/多出/单 vector 出/容器/错误路径全覆盖)+ emit_sol 全量 529 生成 + paddle_core 重编 `[26/26]` 后 OPS_SMOKE 仍 PASS;regen_diff.lua 纳入 emit_sol(products: csrc/capi_gen + lua/paddle/_ops + csrc/sol/gen_ops_*) |
| T-M1-09 | `index_semantics` 标注表 + 未标注即报错的 CI | §8.2 | 3d | ✅ **DONE(2026-10-24,068b2ff)** 首批 12 算子标注 + 后续 2.3.4 墙 #1 扩到 **533 个算子全覆盖(0 未标注)**:takes_index 55 / returns_index 54 / no_index 424;新增 `tools/gen/skip_rules.lua`(skip 单一事实来源,emit_capi/emit_lua/统计共用;graph 家族 4 + dgc + tdm_sampler 因 CSC 结构索引语义走 P17 skip)。lint 全绿、全量 emit 门控 533 ops 全通。**教训:启发式「签名无 index 关键词」分箱不可靠**(median/nanmedian/multinomial 被误分进无索引组,实际是 returns_index)——判定必须逐个读 yaml/源码 |
| T-M1-10 | autograd `backward`/`grad` 绑定 | — | 2d | ✅ **DONE(2026-10-25,cd61d8e)**:`csrc/capi/autograd.cc` 补 `pd_grad`/`pd_has_grad`/`pd_set_has_grad` + `pd_backward` 签名扩成 (tensors, grad_tensors, retain_graph);`csrc/sol/tensor_binding.cc` 注册 `core.backward/core.grad/core.has_grad/core.set_has_grad` + usertype `backward/grad/stop_gradient/set_stop_gradient`;`lua/paddle/autograd.lua`(backward/grad/no_grad/enable_grad/set_grad_enabled/is_grad_enabled,函数式 no_grad 用 pcall 恢复 + `error(err,0)` 保留调用方帧)+ `tensor.lua` `METHODS.backward`(与 `paddle.autograd.backward` 共用 `core.backward` 单一底层,api/autograd.md §6②)。**收尾 2.6.2(ee03f65)**:`paddle.scope` 纯 Lua(api/scope.md:进入水位 + 退出 `collectgarbage("collect")` 施压)+ `core.memory_allocated` 绑定(module.cc);scope_smoke 显存下降 PASS(probe 实证 memory_allocated 真实跟踪 +4000004B→0)。AUTOGRAD_SMOKE PASS(2 层 MLP 手算梯度 1e-6 / grad==backward / no_grad 内部停用+错误恢复 / 两次 backward 累积到 8 / stop_gradient 表面 + nil-grad 断言)+ 全部旧 smoke 无回归 + lint 全绿。**核心踩坑(实测,已修)**:sol2 把 **nil 中间实参当参数链终止符**——`core.backward({loss}, nil, true)` 的第三实参 true 被静默吞掉(retain 丢,`_probe_optional` 探针矩阵实证:`({1},nil,true)→c_has=0`、`({1},{},true)→c_has=1`;类型不匹配实参则跳过留给下一 optional)。修法:`tensor.lua`/`autograd.lua` 调 core 时中间参数**永不传 nil**,缺省用 `{}` 空表(语义同 nil:无 seeds)/显式默认值填满 |
| T-M1-11 | `nn.Layer` 类系统 | — | 5d | ✅ **DONE(2026-10-25)**:`lua/paddle/nn/layer.lua`(Layer 基类:pl.class `_create`/`_class_init` 钩子 + FIELDS 私有键自动注册 + order 书 + pre-order DFS 迭代器 + state_dict 键名与 Python 逐字符一致 + set_state_dict/to/cpu/apply/add_sublayer/add_parameter/create_parameter)+ `lua/paddle/nn/init.lua` + 顶层 lazy 挂 `nn`。**NN_LAYER_SMOKE PASS(15 节)** + 8 个旧 smoke 回归全绿 + lint 全绿。**三个实测 bug(全修)**:① `named_*` 迭代器 stack 每次调用重建 → 死循环无限产出第一项(stack/seen 提进闭包);② pre-order DFS 缺 `cur` 跨调用状态 → 本层剩余参数被跳过(b/fc.b 丢失);③ `training` 若落 FIELDS 顶层 → `layer_index` 读不到 → 改走 plain 槽(`self.training = m` 经 `__newindex`,`Layer:_init` 里 `self.training = true` 播种,不能放 `_create`——裸表期 rawset 会锁死 `__newindex`)。**运行环境坑**:build-nopy 运行时 DLL 在 `paddle\common\phi\pir\fluid\inference` + `third_party\install\{onednn,bin,mklml\lib,lapack,libuv,cryptopp,warpctc,warprnnt,zlib}` + MSVC `VC145.OpenMP\vcomp140.dll`(mkldnn 传递依赖);cmd `set "P=..." && set "PATH=%P%;%PATH%"` 整行展开陷阱(P 在赋值前已展开为空)→ 必须单条 set 写全 |
| T-M1-12 | Linear/Conv2D/BN/ReLU/Dropout/Sequential | — | 8d | 🟡 **2.9.2 Linear 完成(2026-10-25)**:`lua/paddle/nn/layers/linear.lua`(Paddle Linear:y=xW+b,weight=[in,out] bias=[out],forward 走 `pd_op_linear_v2`(ops.yaml:3246)/无 bias 走 `pd_op_matmul`(common.py:2596))+ `nn/init.lua` 挂载。**按用户指示构造函数与正式 API 一律 argrule 包装(D31)**:Linear._init、Layer.create_parameter/add_parameter/add_sublayer 全部规则表化(self 规则 + 表内位置/具名/位置三形态,`LayerContract`/`TensorContract` 用 argrule.alias{谓词};argrule 独立仓库 `E:\code\argrule` 经 luarocks make 装到本机,rockspec 已声明 `argrule >= 0.1`)。**NN_LINEAR_SMOKE PASS**(注册/state_dict 键名/零初始化 y=0/no-bias matmul/数值 13/backward 梯度)+ 10 smoke 全绿 + lint 全绿。**踩坑**:argrule `default=nil` 不算 optional(用 `opt=true`);表内位置洞(nil 截断 `table_array_len`)→ 混合表 `{2,3,bias_attr=false}`。剩余 Conv2D/BN/ReLU/Dropout/Sequential 随 2.9.3-2.9.5 |
| T-M1-13 | `optimizer.SGD` / `Adam` | — | 5d |
| T-M1-14 | `io.Dataset` / `DataLoader`(单 worker) | — | 5d |
| T-M1-15 | `paddle.save/load` 走 `SaveTensor`/`LoadTensor` | D19 | 2d | ✅ **DONE(2026-10-25,ea94bc5)**:`csrc/capi/serialize.cc` 新增 `pd_tensor_save`/`pd_tensor_load`(save_load_tensor.h:20/24,Group 7);`csrc/sol/tensor_binding.cc` `register_serialize_fns`(`core.save_tensor/load_tensor`);`lua/paddle/serialization.lua`(顶层 `paddle.save{obj,path}`/`paddle.load{path}`,inventory.md:439-440)+ init.lua 挂载。**SERIALIZATION_SMOKE PASS(两段式 --save/--load 独立进程,覆盖 07-serialization.md §5 重启进程读回)** + 六个旧 smoke 无回归 + lint 全绿。**踩坑**:CMake `file(GLOB)` 不自动拾取新 .cc → 重跑 `cmake -S . -B build-test` 后 LNK2019 消失。**剩余**:state_dict/manifest 目录格式归 P9/M2(lua-pickle 接线);pickle 只读归 M2 |
| T-M1-16 | `paddle.dataset` MNIST | `plan/overview.md` §2.1.1 | 3d |
| T-M1-17 | `jit::Layer` 绑定(可选,高性价比) | D18 | 2d | ✅ **DONE(2026-10-25)**:`csrc/capi/jit.cc` 新增 Group 6 `pd_jit_layer_load/free/forward/to/function_names/clone`(serializer.h:78 / layer.h:57-78)+ 链接缺口 shim(`GetFeedTargetNames/GetFetchTargetNames`,pybind 目录 WITH_PYTHON=OFF 不产出,纯 PIR 等价实现)+ `JitFlagsOnce`(`enable_pir_api=false` / `jit_engine_type="New"` / `use_mkldnn=false`);`csrc/sol/jit_binding.cc` LuaLayer usertype(`forward`/`__call` 共享 lambda,`to`/`function_names`/`clone`/`__lanesclone`);`lua/paddle/jit/init.lua`(`paddle.jit.load{path}`)。**JIT_SMOKE PASS**:加载 Python 导出的非 PIR tiny 网络(add+add+relu)→ function_names={forward} → forward(2,4)=2.5 → to/clone 稳定。**关键翻案(Q-24)**:New 引擎 0xC0000005 真根因是上游 `interpreter_engine.cc` 的 `@fetch` 后缀断裂(FunctionSchema::OutputArgNames 加后缀,New executor 建裸名变量 → FetchOuts 空指针 + skip_gc_vars 不匹配),非 fc+oneDNN —— 上游 Python 默认 Predictor 引擎从不经过该路径。已修:patch **0002**(`StripFetchSuffix` 两处),`git apply -R` 可逆验证通过。**已知残留(Q-25)**:宿主 Lua 进程 forward 后 lua_close 阶段确定性 0xC0000005(New executor 异步线程池与退出时序竞态,C++ 直连干净),smoke 以 os.exit(0) 收尾,记 open-questions。详见 status.md |
| T-M1-18 | GC 九层机制 | D5/D6 | 3d |
| T-M1-19 | 1-based 护栏(label 范围检查) | C4 | 1d |
| T-M1-20 | **M1 验收:MNIST 训练收敛** | G2 | 3d |

---

## 完成记录

| ID | 完成日期 | 实际耗时 | 备注 |
|---|---|---|---|
| T-M0-01 | 2026-10-22 | 2 天 | 生死判定编译通过(5195 files / 9.90 GB,`build-nopy`),5 验收符号全导出;1 个补丁 0001(C2 满足);详见 `process/m0-report.md` |
| T-M0-02 | 2026-10-22 | 2 天(与 T-M0-01 并行) | `tools/smoke/smoke.cc` 冒烟 PASS(`a.grad==4.0, b.grad==3.0, max_err=0`),G0 闸门打开 |
| T-M0-14 | 2026-10-23 | 0.5d | `tools/smoke/smoke_strided_slice.cc` PASS:**eager strided_slice 是 view**(写穿透 + autograd);详见 `m0-report.md` §16 |
| T-M0-15 | 2026-10-23 | 0.5d | `OpProtoHolder` 链无 pybind 依赖(C++ `OpInfoMap` 纯 C++);详见 `m0-report.md` §17 |
| T-M0-17 | 2026-10-23 | 1d | `jit::Load` 可链接可加载;发现链接缺口 + fc/oneDNN bug(Q-24);详见 `m0-report.md` §18 |
| T-M0-18 | 2026-10-23 | 1d | 手工 `AttributeMap` 喂 `run_program_ad_func` PASS;详见 `m0-report.md` §19 |
| T-M0-19 | 2026-10-23 | 0.5d | `SaveTensor`/`LoadTensor` 往返无损、重存字节一致;详见 `m0-report.md` §20 |
