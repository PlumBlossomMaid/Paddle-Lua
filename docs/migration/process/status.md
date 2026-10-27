# STATUS.md · 当前状态

> **每次开工第一个读的文件。每次收工最后一个更新的文件。**
> 本文件记录**易变**信息;稳定的约束和决策在 `AGENTS.md`。

---

## 1. 当前闸门

```
G0  ✅ 已通过(2026-10-22)  WITH_PYTHON=OFF + ON_INFER=OFF 能编出可用 libpaddle
G1  ✅ 已通过(2026-10-23)  M0 必做 17 项全部完成,产出 M0_REPORT(见 m0-report.md §0)
G2  ✅ 已通过(2026-10-27)  M1 验收:MNIST 训练收敛(lua examples/mnist.lua,2 epoch,test accuracy 0.9869 > 0.95)
```

**当前允许做的事:M2 全部工作(3.1 dataset+vision / 3.2 Lanes / 3.3 GC 九层 / 3.4 CIFAR 验收)。**
**⛔ 尚未开 G3:M2 验收(CIFAR + 吞吐 3×)先做。**

---

## 2. 当前任务

> **位置以 `docs/migration/WORKPLAN.md` 里唯一的 🔵 为准。** 本节只记现场笔记。

| | |
|---|---|
| 阶段 | **M2(已开闸,2026-10-27)** —— 3.1 dataset+vision / 3.2 Lanes 多 worker / 3.3 GC 九层 / 3.4 CIFAR 验收 |
| 工程树位置 | `WORKPLAN.md` 节点 **3. M2 可用**(🔵;2.12 M1 验收 ✅ G2 已开) |
| 下一个动作 | **3.1.1 下载底座完成(2026-10-27,WORKPLAN 3.1.1 ✅)**:x64 版 luasocket 3.1.0 + luasec 1.3.2 编译部署(openssl 3.6.2 经 vcpkg,手动 curl 走代理入缓存突破 error56)+ `paddle.utils.download{url,path,md5}` + 纯 Lua MD5,**UTILS_DOWNLOAD_SMOKE PASS(8 断言)**。**下一步 = 3.1.2 MNIST dataset 适配**(URL/MD5 已核 mnist.py:33-37;从 io.Dataset 形态走,reader 生成器不搬,见 status.md 3.0 前置段的 4 个收获) |
| 人点名的三件调研 | **① Q-22** phi 层有没有不依赖 pybind 的 DLPack 通道(决定 Insight7 内存协议怎么定);**② 数出「跑通 MNIST 训练,Insight7 至少要哪些函数」**(从上游代码里数,不凭感觉 —— 已知 `python/paddle` 全层只用 132 个 `np.*` 符号、真函数约 15 个,见 `research/ecosystem.md` §5.1);**③ Q-23** Array API Standard 是否如记忆所述(要走代理核实)。三件都**不阻塞** `WORKPLAN` 主线 |
| 全部 ⛔ 阻塞节点 | `WORKPLAN.md` 4.3 distributed —— 无多卡环境(待拍板 P2) |
| 待人拍板 | 剩余待拍板:P3 / P4 / P6 / P7(改写)+ 1.7 的 P9(Penlight 分发)与 luarocks 占位 —— 见 `process/decisions.md` §3 §4 |
| 上次卡在哪 | **2.3.4 收尾①全量 emit_lua 1-based 转换完成(2026-10-25)**,前置三堵墙已全过:**墙 #1 index 标注**(068b2ff → 4d4dc64,533 算子 0 未标注);**墙 #2 capi type gaps**(c7768da,4 种输入形态 + 全部 attr 类型,白名单 9 算子 MSVC 编译通过);**墙 #3 multi-output**(16a2917,out-array C ABI + intermediate 过滤)。本次收尾①核心:**转换分工定案**(Tensor/vector\<Tensor\> 整体 ±1 在 C ABI 层 `pd_idx_minus1/plus1`;标量 axis/dim 与标量 takes_index attr 在 Lua wrapper)、**axis 默认值平移**(0-based yaml 默认 → 1-based 字面量,wrapper 统一 `if v > 0 then v = v - 1` 还原)、**intermediate 折叠分发修复**(.cc 改 `#effective_returns > 1`)、**单 vector 输出 emit_op_vec**(9 算子)。验证:lint 4 文件过、529+529 ops 全量生成、可重入 0 diff、MSVC 6 代表 .cc 编译 EXIT=0、抽查 6 算子转换正确。**下一步 = 收尾② regen-diff CI / ③ 20-op parity / ④ 95% coverage** |

---

## 3. 环境记录

> 智能体开工时若发现与当前机器不符,**重新探测并覆盖本节**,附机器标识。

| 项 | 值 | 探测时间 |
|---|---|---|
| 机器标识 | `E:\code\paddle-lua` (Windows, 2026-10-21 复测) | 2026-10-21 |
| `$PADDLE_ROOT` | `E:\code\paddle` | 2026-08-03 |
| Paddle 分支 | `windows_dataloader_multiprocess` | 2026-08-03 |
| Paddle 版本 | 3.5.0 | 2026-08-03 |
| 本仓库 | `E:\code\paddle-lua` | 2026-10-21 |
| Insight7(参考实现) | `E:\code\Insight7` | 2026-08-03 |
| vcpkg | `E:\code\vcpkg`(有 lua 5.5.0 / sol2 3.5.0 / luajit) | 2026-08-03 |
| Lua 版本 | **5.1.5** `C:\Program Files (x86)\Lua\5.1\`(choco 也有 lua5.1);luajit 不在 PATH | 2026-10-21 |
| Python | 3.11.4 | 2026-10-21 |
| 编译器 | **MSVC 14.50.35730(cl.exe,VS 18.x)** —— 需 vcvars64 环境;冒烟编译脚本里已固化 `call "C:\Program Files (x86)\Microsoft Visual Studio\...\VC\Auxiliary\Build\vcvars64.bat"` | 2026-10-22 |
| 构建工具链 | **CMake + Ninja**(`build-nopy` 用 Ninja 生成器);dumpbin 可用(/symbols /exports) | 2026-10-22 |
| GPU | Quadro P620 4GB | 2026-10-21 |
| git remote | `github.com/PlumBlossomMaid/Paddle-Lua` | 2026-10-21 |
| 代理 | `http://127.0.0.1:7890`(http/https 均可) | 2026-08-03 |

---

## 3.1 构建记录(build-nopy,2026-10-22)

| 项 | 值 |
|---|---|
| 构建目录 | `E:\code\Paddle\build-nopy` |
| cmake 配置 | `-DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_SHARED_IR=OFF -GNinja` |
| 编译并行度 | `-j6`(12 核吃一半,用户 2026-10-21 指令) |
| 产物规模 | **5195 files / 9.90 GB**(含测试二进制) |
| 冒烟产物 | `tools/smoke/smoke.exe`(链接 483 库,`/FORCE:MULTIPLE`) |
| 结果 | ✅ 编译成功;`smoke.exe` 输出 `SMOKE PASS: a.grad == 4.0, b.grad == 3.0, max_err=0` |

---

## 4. M0 验证进度

| # | 项 | 状态 |
|---|---|---|
| 1 | `WITH_PYTHON=OFF`+`ON_INFER=OFF` 编译 | ✅ **DONE(2026-10-22)**:编译成功(5195 files / 9.90 GB),5 个验收符号全部确认导出。详见 `process/m0-report.md` §3。构建目录 `$PADDLE_ROOT\build-nopy`,CPU-only(`-DWITH_GPU=OFF`) |
| 2 | 该配置下 `egr::Backward` 跑通反向 | ✅ **DONE(2026-10-22)**:`tools/smoke/smoke.exe` 输出 `SMOKE PASS: a.grad == 4.0, b.grad == 3.0, max_err=0`,数值与手算一致。详见 `process/m0-report.md` §4 |
| 3 | phi kernel / DeviceContext 多线程安全 | ✅ **DONE(2026-10-22)**:`tools/smoke/smoke_mt.cc` 多线程冒烟 PASS —— 4 线程×50 轮 + 8 线程×200 轮(1800 次完整图)共享框架 DeviceContextPool,`a.grad==4.0 / b.grad==3.0 / max_err=0`。CPU 多线程共享上下文安全,无需 per-thread context |
| 4 | Lanes v3.17.x deep API 与 4.x 差异 | ✅ **DONE(2026-10-22)**:`__lanesclone` 两版调用约定一致(3 参 `lua_call(L,3,0)`);deep userdata API 两版结构不同(C vs C++ class),但 D24 走 `__lanesclone` 不受影响;`on_state_create` 两版都有(T-M0-06 顺带确认)。详见 `m0-report.md` §9 |
| 5 | Lanes 与绑定的 C++ ABI 一致性 | ✅ **DONE(2026-10-22)**:`lanes_abi_check.cc` 实测 PASS(`LANES ABI OK`)。v3.17.2 纯 C(10 .c / 0 .cpp),无虚函数 ABI 问题(那是 4.x);MSVC 编译 Lanes obj 从 C++ TU 链接 Lua 5.1 成功。**陷阱:Lanes 与调用方必须用同一份 Lua 头,混用 PUC/LuaJIT 头运行即崩**。详见 `m0-report.md` §10 |
| 6 | Lanes `on_state_create` 钩子 | ✅ **DONE(2026-10-22)**:存在。v3.17.2 `state.h:19-20`(`initialize_on_state_create`/`call_on_state_create`)+ `universe.h:60`(`on_state_create_func` 字段);4.x 同有(`state.cpp:212`/`keeper.cpp:1126`)。承载"每个 worker lane 重新 `require "paddle"`"(13-lanes.md §4.4,Q-03 关闭) |
| 7 | `RegisterOOMCallback`+`lua_gc` 救回 OOM | ✅ **DONE(2026-10-22)**:`tools/smoke/oom_smoke.cc` 实测 PASS。MockAllocator 抛 `BadAlloc` + `RetryAllocator` 验证:回调返回释放字节数 >0 -> 重试至成功;返回 0 -> 只调一次、异常逃逸;无回调 -> 不重试。**要点**① `offload_retry_times` 默认 **-1**(`common/flags.cc:2363`),必须设 >0 才开重试路径;② `RetryAllocator` 外层有 waiter 重试(`retry_allocator.cc:88-115`)。详见 `m0-report.md` §11 |
| 8 | LuaJIT GC64 | ✅ **DONE(2026-10-22)**:`tools/ci/lua_gc64_check.lua` 实测 PASS —— vcpkg LuaJIT 2.1.1774896198(x64/Windows)`16-byte GC64 layout confirmed`。判据:源码 `lj_arch.h:168/216-217/303/459` x64 默认 `LJ_TARGET_GC64=1` + 运行时 userdata 地址间隙 16 倍数 + `ffi.sizeof("void*")==8`。绑定层按 GC64 布局 |
| 9 | `__close` 对 userdata(5.4) | ✅ **DONE(2026-10-22)**:`tools/ci/lua_close_check.lua` 在 Lua 5.4.7(源码 MSVC 编译)与 5.1.5(choco)上实测 PASS,已登记 l0.yml lua-matrix 五版本。**Case A**:`<close>` 变量作用域退出触发 `__close`(表也触发 —— to-be-closed 机制对任何有 `__close` 的对象生效)。**Case B 关键发现**:GC 只走 `__gc`,**不调 `__close`** —— 用 `io.tmpfile()` 真实 userdata + `debug.setmetatable` 同挂 `__gc`/`__close`,实测 GC 后 `closed_by=="gc"`。**绑定层结论:RAII usertype 必须同时挂 `__close`(5.4 `<close>` 确定性释放)与 `__gc`(兜底释放),两者需幂等**。详见 `m0-report.md` §13 |
| 10 | 堆追踪与 Lanes 分配器冲突 | ✅ **DONE(2026-10-22)**:`tools/smoke/lanes_alloc_check.cc` 实测 PASS(`LANES ALLOC PASS`)。计数 `lua_Alloc` 装在 master state(模拟堆追踪),`on_state_create` 钩子里 `lua_getallocf` 实测 3 个 lane state 分配器**全 != counting_alloc**(inherited=0)。**① LuaJIT 下 master 的自定义 allocator 完全到不了 lane**(`state.c:253-255`,lane state 一律 `luaL_newstate()`);PUC 才复用(`state.c:257-271`)。**② 实测陷阱:`lua_setallocf` 不能裸 `realloc/free` 替换**(vcpkg LuaJIT 是 `lua51.dll` 自有 CRT 堆,跨堆 free = 0xC0000374),**必须链式转发原分配器**(先 `lua_getallocf` 再换,同 Lanes `protected` 模式 `tools.c:173-190`)。**落地:堆追踪做张量字节计数器(14-gc.md §3.1),与 Lua allocator 无关;LuaJIT 上 allocator 钩子看不到 lane 分配**。详见 `m0-report.md` §14 |
| 11 | 纯 Lua pickle 读 `.pdparams` | ✅ **DONE(2026-10-23)**:实现在独立仓库 `E:\code\lua-pickle`(rock/模块名 `pickle`)。unpickler 核心读通真实样本 `m0-11-sample.pdparams`(326B, protocol 4),`tools/check.lua` 在 Lua 5.1.5 上与 Python `pickle.load` 权威基准逐字节对拍 PASS。详见 `m0-report.md` §15 |
| 12 | ~~sol2 + Lua 5.5~~ | ✅ 已完成:不支持,5.5 出局 |
| 13 | ~~Lanes 可选依赖~~ | ✅ 已决策:强制依赖 |
| 14 | `x["::-1"]` 走 `strided_slice` 是否 view | ✅ **DONE(2026-10-23)**:`tools/smoke/smoke_strided_slice.cc` 实测 PASS —— **是 view,不是 copy**(旧记录"拷贝语义"是错的,那是 fallback 路径)。`FLAGS_use_stride_kernel` 默认 true(`kernel_factory.cc:31`),STRIDED kernel `out->ResetHolder(input.Holder())`(`stride/strided_slice_kernel.cc:95-99`);实测 `py==px+12B` + 写穿透 `px[3]` 随 `py[0]` 变。autograd 正常(seed=[1,2,3,4] → `x.grad=[4,3,2,1]`,`egr::Backward`)。详见 `m0-report.md` §16 |
| 15 | `WITH_PYTHON=OFF` 下 `OpProtoHolder` 链 | ✅ **DONE(2026-10-23)**:**可用,不依赖 pybind**。Python 侧 `OpProtoHolder`(`framework.py:3202`)只是 pybind 薄壳(`pybind.cc:2591` 遍历 `OpInfoMap`),C++ 侧 `OpInfo`/`OpInfoMap` 纯 C++(`op_info.h:42/139`);上游无 Python 环境路径自己就用它(`custom_operator.cc:1270`);build-nopy dumpbin 实测 `libop_info.lib`/`libop_registry.lib` 均导出 External 定义。Lua 侧写等价 C ABI 即可替代,无需补丁。详见 `m0-report.md` §17 |
| 16 | `WITH_PYTHON=OFF` 下 `new_executor`/`jit` 可链接 | ⬜ 与 #1 合并做 |
| 17 | `jit::Layer Load()` 加载 `jit.save` 产物 | ✅ **DONE(2026-10-23,部分验证)**:`tools/smoke/smoke_jit_load.cc` 实测(上游 CI 产物 `multi_program_load_with_property.tar.gz`)。**`jit::Load` 在 `WITH_PYTHON=OFF` 下可链接、可加载**——`libfluid_jit.lib` 导出符号、`.pdmodel/.pdiparams/.meta` 全读入、`FLAGS_jit_engine_type="New"` 时 InterpreterEngine 启动(`New Executor is Running`,`program_interpreter.cc:255`)。**两个发现**:① 链接缺口——`PirFunctionInfo` 构造(`function_schema.cc:112/115`)引用 `GetFeedTargetNames/GetFetchTargetNames`(`pybind/pir_utils.cc:20/36`),pybind 目录在无 Python 下不产出 → 冒烟自带等价 shim;② forward 撞上游 fc+oneDNN bug(`onednn_context.cc:347`;New 引擎 0xC0000005),非无 Python 特有,记 Q-24。不阻塞 G1。详见 `m0-report.md` §18 |
| 18 | 手工构造 `AttributeMap` 喂 `run_program_ad_func` | ✅ **DONE(2026-10-23)**:`tools/smoke/smoke_run_program.cc` 实测 PASS(exit 0)。入口签名纯 C++(`run_program_func.h:24-28`),`AttributeMap`/`Attribute` 可不经 pybind 手工构造(`type_defs.h:51/73`,variant 含 `pir::Value` 向量/`shared_ptr<pir::Program>`);冒烟按 `RunProgramImpl` 消费键全量填充(`run_program_impl.cc:449-490`,外层 scope `461-466`),空 program 触发预期 C++ Traceback、无 pybind 泄漏。**意义**:P16 静态图运行时参数可纯 C++ 构造,绑定层只需薄 C ABI 透传。详见 `m0-report.md` §19 |
| 19 | `SaveTensor`/`LoadTensor` 格式稳定性 | ✅ **DONE(2026-10-23)**:`tools/smoke/smoke_save_load.cc` 实测 PASS(exit 0)。链 = `framework::SaveTensor/LoadTensor`(`save_load_tensor.h:20-24`)→ `phi::SerializeToStream/DeserializeFromStream`(`save_load_tensor.cc:24-45`);格式 = uint32 版本 0 + LoD + TensorDesc proto + 原始数据(`dense_tensor_serialize.cc:69-81`/`dense_tensor_tostream.cc:97-228`),重存字节级幂等(file_size=50 两次逐字节相同)。Python 互通确认:`pybind/io.cc:140` 就是 `framework::LoadTensor`(D19 成立)。**Lua 侧 `paddle.save/load`(单张量)直接以它为线格式,不用 pickle;view 保存自动实体化**。详见 `m0-report.md` §20 |
| 20 | `string.dump` 往返比对(服务 M4) | ⬜ 可跳过 |
| 21 | luacheck parser 独立跑 5.1(服务 M4) | ⬜ 可跳过 |
| 22 | **LuaJIT 上 `debug.setupvalue` 注入 upvalue 是否可用**(`_args` 的命脉,Q-17) | ✅ **已实测通过(2026-10-21):Lua 5.1.5 与 LuaJIT 2.1 均可注入,含表 upvalue;Q-17 关闭** |
| 23 | **`loadstring`/`load` 在 5 个 Lua × 3 OS 上行为一致**(chunkname 与错误行号) | 🟡 **部分完成(2026-10-21):5.1 与 LuaJIT 的 `[string "name"]:行号` 格式、命名 chunk 错误行号、运行时错误携带 chunkname 全部一致。5.2/5.3/5.4 待补(走 CI L0)** |
| 24 | **三道墙的实测值在 5 个 Lua 上各是多少**(upvalue 上限 / 局部变量上限 / 寄存器上限)。5.1 已测 = 60 / 200 / N=122 | 🟡 **部分完成(2026-10-21):5.1 与 LuaJIT 的 upvalue=60 / locals=200 两墙实测一致且与文档吻合;寄存器墙构造依赖 —— 嵌套调用 5.1=194 / LuaJIT=124,宽参数列表 5.1=248 / LuaJIT=247(文档 N=122 来自 argrule 原型形态,量级一致)。5.2/5.3/5.4 三版待补(本机无解释器,走 CI L0 或装版本)** |

**必做 20 项(#20/#21 可跳过),预估 3 周。**

> **#22/#23 不阻塞于 G0** —— 它们是纯 Lua 验证,`WITH_PYTHON=OFF` 编不编得出来与它们无关。
> 挂了的后果是明确的:`_args` 走**解释式降级路径**(`plan/foundations.md` §4.7)。
> 越早知道越好,因为 P5 之后再改就要重写所有已写的构造期签名。

---

## 5. 文档完成度

> 路径相对 `docs/migration/`;`AGENTS.md` 与三个 README 在仓库根目录。

### 5.1 顶层

| 文档 | 行数 | 状态 |
|---|---|---|
| **`WORKPLAN.md`(总工程树)** | 248 | ✅ **新增** |
| `/AGENTS.md` | 502 | ✅ |
| `/README.md`(英文,默认) | 192 | ✅ |
| `/README.zh-CN.md` | 186 | ✅ |
| `/README.zh-TW.md` | 186 | ✅ |
| `README.md`(计划总索引) | 148 | ✅ |

### 5.2 `plan/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `plan/overview.md` | 935 | ✅ |
| `plan/roadmap.md` | 341 | ✅ |
| `plan/foundations.md` | 1105 | ✅ **+参数检查(§4)+ 基座边界与解析器项目(§5)** |
| `plan/argrule.md` | 694 | ✅ **新增**(孵化说明书,建仓后迁走;原 `argsig.md`)|
| `plan/layout.md` | 261 | ✅ **新增** |
| `plan/ci.md` | 280 | ✅ **新增** |
| `plan/api/README.md` | 361 | ✅ **+新增`top.md` 页** |
| `plan/api/io.md` | 236 | ✅ **新增**(样板)|
| `plan/api/top.md` | — | ✅ **新增(2026-10-23)**:顶层 `paddle` 命名空间契约,第一批约 80 个核心成员(创建/算子/归约/索引/逻辑/随机/激活/复数),素材 = `inventory.md` §3.1 抽查表;含 §3 index 语义参数标注表与 §6 本模块坑 |
| `plan/api/tensor.md` | — | ✅ **新增(2026-10-23)**:`paddle.Tensor` 类型契约(核心方法/属性/魔法方法/算子方法挂载机制/字符串索引 D14);上游来源 `tensor_patch_methods.py`/`generated_tensor_methods_patch.py`/`variable_index.py` 全部带 file:line |
| `plan/api/dtype-device.md` | — | ✅ **新增(2026-10-23)**:`paddle.dtype`/`paddle.device` 契约(dtype 常量一套枚举、place 值类型、默认 dtype 全局态、finfo/iinfo);与 Insight7 命名对齐(`foundations.md` §3.2/§3.6.1) |
| `plan/api/linalg.md` | — | ✅ **新增(2026-10-23)**:`paddle.linalg` 子模块契约(分解/范数/转置族);核实 `python/paddle/linalg` 目录不存在、`paddle.linalg` = `tensor/linalg.py` 再导出(`__init__.py:278`);`svd/lstsq/slogdet` 仅在子模块 |
| `plan/api/autograd.md` | — | ✅ **新增(2026-10-23)**:`paddle.autograd` 契约(backward/grad/no_grad 等);导出 re-export 自 `base/dygraph/base.py`(`__init__.py:15-21`),backward 在 `backward_mode.py:34`;no_grad 上下文语义待 P6 拍板 |
| `plan/api/scope.md` | — | ✅ **新增(2026-10-23)**:`paddle.scope` 契约——**上游无此 API,纯 Lua 设计**(`gc.md` §4 第 6 层,唯一用户可见的 gc 层);逃逸保守策略 + 默认阈值待 P6 实测 |
| `plan/api/nn.md` | — | ✅ **新增(2026-10-23)**:`paddle.nn` 契约(Layer 基类/容器/核心层 Linear·Conv·Embedding·损失);Embedding/CrossEntropyLoss 的 index 1-based 是 C4 对齐关键;`Module` 新框架待 P3 拍板 |
| `plan/api/nn-functional.md` | — | ✅ **新增(2026-10-23)**:`paddle.nn.functional` 契约(37 激活 + dropout/池化/归一化/损失函数);`_` in-place 变体随生成器;`dropout.training` 显式传参、`normalized_shape` 轴范围 1-based |
| `plan/api/nn-initializer.md` | — | ✅ **新增(2026-10-23)**:`paddle.nn.initializer` 契约(可调用对象 + Normal/Uniform/Xavier/Kaiming/Constant 等);随机数必须走 Paddle 生成器;旧 `*Initializer` 别名不搬 |
| `plan/api/optimizer.md` | — | ✅ **新增(2026-10-23)**:`paddle.optimizer` 契约(Optimizer 基类 + SGD/Momentum/Adam/AdamW + lr 调度器);`step()` 在 `backward()` 后 + `clear_grad()` 四行循环写死;weight_decay 语义 AdamW 解耦 vs Adam 耦合 |
| `plan/api/jit.md` | — | ✅ **新增(2026-10-23)**:`paddle.jit` 契约(load P8 / trace P15 / script P18);`load` 链 T-M0-17 实证(serializer.h:78);链接缺口 shim + Q-24(fc/oneDNN)已记录;save 不做(Lua 导出走 SaveTensor) |
| `plan/api/dataset.md` | — | ✅ **新增(2026-10-23)**:`paddle.dataset` 契约(12 子模块 + mnist/cifar 首批);上游 reader 生成器(`__all__=[]`)适配成 `io.Dataset`;MNIST label 1-based 是 M1 验收对齐点;IDX 解析需逐字节对拍 |
| `plan/api/vision.md` | — | ✅ **新增(2026-10-23)**:`paddle.vision` 契约(datasets 类/MNIST·FashionMNIST 首批 + models/transforms 分批);验证 `vision.datasets.MNIST` 是 `io.Dataset` 类(非 reader);图像后端按 §9.1 决策;ToTensor 是 Insight→Tensor 边界 |
| `plan/api/utils.md` | — | ✅ **新增(2026-10-23)**:`paddle.utils` 契约(download/dlpack/deprecated/run_check/require_version/try_import + **新增 fs**);**纠正 `api/README.md` 旧说法**(上游确有此模块,`utils/__init__.py:13-30`);`decorator_utils` 整层不移植(D35) |
| `plan/api/distributed.md` | — | ✅ **新增(2026-10-23)**:`paddle.distributed` 契约级文档(集合通信/parallel 环境/数据并行形态;⛔ 无环境,实现与验收挂起);rank 语义是 C4 唯一冲突候选,P17 拍板 |
| `plan/api/<其余 0 个模块>` | — | ⬜ 全部完成(除 amp 未排期无文档) |
| `plan/api/inventory.md` | 883 | ✅ **抽查表扩容(2026-10-24)**:§3.1 从 14 批(约 110 行)扩到 **34 批 / 268 行 `paddle.*` 成员**(第十五~三十四批,提交 8b1ea3a→3c3b6ea)。PYTHON 桶 328 个中核心成员**已全部覆盖**(创建/算子/归约/索引/逻辑/随机/激活/复数/三角/位运算/统计/线性代数/linalg 子模块/全局态/工具);剩余约 60 个 = in-place `_` 变体(随生成器批量覆盖)+ dtype 常量(dtype-device.md 已覆盖)+ 少量内部函数(标注放放),无独立抽查价值。**主线①核心抽查完成** |
| `plan/modules/README.md` | 70 | ✅ |
| `plan/modules/00-build.md` | 115 | ✅ |
| `plan/modules/01-c-abi.md` | 150 | ✅ |
| `plan/modules/02-binding.md` | 173 | ✅ |
| `plan/modules/03-codegen.md` | 168 | ✅ |
| `plan/modules/04-packaging.md` | 96 | ✅ |
| `plan/modules/05-tensor.md` | 156 | ✅ |
| `plan/modules/06-autograd.md` | 147 | ✅ |
| `plan/modules/07-serialization.md` | 126 | ✅ |
| `plan/modules/08-jit-infer.md` | 115 | ✅ |
| `plan/modules/09-nn.md` | 360 | ✅ |
| `plan/modules/10-optimizer.md` | 135 | ✅ |
| `plan/modules/11-io.md` | 131 | ✅ |
| `plan/modules/12-dataset-vision.md` | 129 | ✅ |
| `plan/modules/13-lanes.md` | 163 | ✅ |
| `plan/modules/14-gc.md` | 145 | ✅ |
| `plan/modules/15-to-static.md` | 131 | ✅ |
| `plan/modules/16-cpp-extension.md` | 141 | ✅ |
| `plan/modules/17-distributed.md` | 98 | ✅ |
| `plan/modules/18-script.md` | 178 | ✅ |

### 5.3 `process/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `process/status.md` | 本文件 | ✅ |
| `process/tasks.md` | 137 | ✅ |
| `process/conventions.md` | 314 | ✅ |
| `process/decisions.md` | 250 | ✅ |
| `process/open-questions.md` | 224 | ✅ |

### 5.4 `research/`

| 文档 | 行数 | 状态 |
|---|---|---|
| `research/feasibility.md` | 464 | ✅ |
| `research/architecture.md` | 251 | ✅ |
| `research/gc.md` | 417 | ✅ |
| `research/dataloader.md` | 380 | ✅ |
| `research/reuse.md` | 307 | ✅ |
| `research/to-static.md` | 239 | ✅ |
| `research/ecosystem.md` | 213 | ✅ **新增**(生态构想与顺序;构想记录,非承诺)|

**论证阶段文档已齐(约 8800 行)。G0 未通过之前不新增论证文档,只修正已有的。**

---

## 5.5 阶段进度(P0–P18)

> 顺序、开工条件与完工判据见 `plan/roadmap.md`;每阶段怎么做见 `plan/modules/`。

| 阶段 | 模块 | 状态 |
|---|---|---|
| P0 | 无 Python 构建 | ✅ **已验证通过(2026-10-22)** 编译成功 + `egr::Backward` 冒烟数值与手算一致(见 `process/m0-report.md`)。注:冒烟与手算对拍,Python 端逐位对拍未做(本构建即无 Python;如需可另建 WITH_PYTHON=ON 构建对拍,列为可选) |
| P1 | C ABI 中间层 | ✅ **已完成(2026-10-24)** 2.1 全段(paddle_c.h/status/tensor/autograd/迷你训练/device/jit),T-M1-05 DONE |
| P2 | sol2 绑定层 | ✅ **已完成(2026-10-24)** 2.2 全段(module/error/tensor_binding/slice),SOL_SMOKE PASS,T-M1-06/07 DONE |
| P3 | 算子代码生成 | ✅ **已完成(2026-10-25)** 2.3 全段(yaml_subset/三算子/全量 529 ops/regen-diff CI/emit_sol),coverage 98.0% ≥ 95%,T-M1-08/09 DONE |
| P4 | 构建与打包 | ✅ **已完成** 2.4(CMakeLists/rockspecs),见 plan/modules/04-packaging.md |
| P5 | `paddle.Tensor` | ✅ **已完成** 2.5(tensor.lua: METHODS/切片/工厂),TENSOR_SMOKE PASS |
| P6 | 自动微分与生命周期 | ✅ **已完成(2026-10-25)** 2.6 全段(backward/grad/no_grad/scope/memory_allocated),06-autograd.md §5 七项验收全过,T-M1-10 DONE |
| P7 | 序列化 | ✅ **已完成(2026-10-25)** 2.7 单张量 save/load(D19),SERIALIZATION_SMOKE PASS,T-M1-15 DONE;state_dict 接线归 M2 |
| P8 | `jit::Layer` 推理 ★ | ✅ **已完成(2026-10-25)** 2.8 全链路(paddle.jit.load),JIT_SMOKE PASS,Q-24 翻案 + patch 0002,T-M1-17 DONE;残留 Q-25(宿主退出竞态) |
| P9 | `paddle.nn` | ✅ **已完成(2026-10-26)** 2.9 全段(Layer 基类/Linear/functional 49/层类 40+/容器),NN_* 5 smoke 全绿,T-M1-11/12 DONE |
| P10 | `paddle.optimizer` | ✅ **已完成(2026-10-26)** 2.10(SGD/Adam,组合基础算子 + 核公式直抄),OPTIMIZER_SMOKE PASS(18 断言),T-M1-13 DONE |
| P11 | `paddle.io`(单 worker) | ✅ **已完成(2026-10-26)** 2.11(Dataset/Sampler/DataLoader,Insight choice shuffle),IO_SMOKE PASS(55 断言),T-M1-14 DONE |
| — | **M1 验收:MNIST** | ✅ **已通过(2026-10-27)** `lua examples/mnist.lua` 2 epoch test accuracy **0.9869 > 0.95**,G2 闸门打开,WORKPLAN 蓝点 → 3.M2 |
| P12 | dataset + vision | ⬜ |
| P13 | Lanes 多 worker | ⬜ |
| P14 | GC 九层机制 | ⬜ |
| P15 | 静态图 trace 模式 | ⬜ |
| P16 | cpp_extension | ⬜ |
| P17 | distributed | ⬜ ⚠️ 当前环境无法验收 |
| P18 | script 模式 | ⬜ |

---

## 6. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-10-27 | **3.0 前置:Insight7 axis 改 1-based 完成(跨仓库,insight7 `647af07` 已推送)** —— 开工前核实发现 foundations §3.4(2026-08-03)现状表已过时:native 层 `lua_axis_to_cpp`(insight_lua.cpp:408)早已统一转换,reduction/manipulation/fft/argsort/sort/topk/slice 全族用户侧**已 1-based**;实际漏网 3 处已全修:**① gather/scatter/scatter_add/scatter_reduce**(indexing.lua `dim_to_lua` **+1** 与 native **-1** 互相抵消 → 净 0-based,删 Lua 侧移位直通 native);**② take**(native 直绑 `&ins::take` 绕过转换 → 包 `lua_optional_axis_to_cpp`);**③ gradient**(`axis.value()` 未转换 → 包 `lua_axis_to_cpp`,insight_lua.cpp:1487)。测试补固(harnch 39 断言全绿):`sum(a,1)=[5,7,9]`(第 1 维,与 Julia `sum(a,dims=1)` 一致)、`axis=0` 报错(D13 兑现)、负数轴 `-1` 保持、gather 值序证明 `dim=1`=axis 0。**P12 transforms 前置条件满足。下一步 = 3.1 dataset+vision 本体(Q-08 已查:Paddle 无现成 HTTP 客户端可绑,走 §9.1 决策②)**,详见 WORKPLAN 3.0 节点与 foundations §3.4 |
| 2026-10-27 | **✅ G2 闸门通过,M1 验收完成(2.12,蓝点 → 3.M2)** —— `examples/mnist.lua`(验收载体)2 epoch test accuracy **0.9869(9869/10000)> 0.95**(epoch 1 loss 0.1716 / epoch 2 loss 0.0454)。**数据管线**:`from_host_bytes` 新增 sol 绑定(tensor_binding.cc,C ABI `pd_tensor_from_host_buffer` paddle_c.h:128)一次性整块导入 60000x784 UINT8 → cast F32 → scale(2/255,-1,1)(mnist.py:73-75);label 0-9 → +1 得 1-10(C4)。**三个根因全实证修复**:① **生成器 in-place predefined_out 缺口**(emit_capi.lua):yaml `inplace:` 标注 AND 单输出 AND **op 名 `_inplace$` 结尾**(gaussian_inplace/uniform_inplace)才追加 `paddle::make_optional(&x_ip) /* predefined_out */`——不加则内核分配新输出、原参数恒 Constant(0)(loss 钉在 ln10,探针 8);**asin 等一元算子的 inplace 标注只是内核 buffer 复用,绝不能加**(扩大化会静默改写输入→0xC0000005,探针);② **初始化须 no_grad**(layer.lua):gaussian_inplace 带自身反向节点(ops.yaml:2564 backward: gaussian_inplace_grad),has_grad 下挂进 autograd 图,backward 崩 "TensorWrappers do not exist"(backward.cc:215,探针 9→探针 10 loss 下降实证);③ **显式 GC**:LuaJIT 默认 GC 阈值不回收逐 batch 累积的 LuaTensor shell → C 侧 ~300 batch 堆崩 0xC0000005(探针 11),每 100 batch `collectgarbage("collect")` 后 938 batch 全跑完(探针 12)。**遗留(不阻塞)**:smoke_capi EXE 目标 LNK2001(test_c_only.c 无 main,全量构建时处理);nn.initializer 未落地(MNIST 用用户侧显式 gaussian 初始化,文件头注释)。**下一步 = 3. M2:3.1.0 API 设计 + 3.0 前置 Insight7 axis 1-based** |
| 2026-10-26 | **2.11 paddle.io(单 worker)完成(蓝点 → 2.12)** —— `lua/paddle/io.lua`(723 行)Dataset(Dataset/IterableDataset/TensorDataset/ConcatDataset/Subset/random_split)+ Sampler(Sequence/Random/WeightedRandom/Batch)+ DataLoader + default_collate,全部 argrule 包装。**shuffle RNG = Insight7 choice(用户定案:固定种子与 numpy 2.4.6 逐位一致),播种 = 构造期一次、epoch 间继续流(numpy 全局流语义)**。**1-based 全链路(C4)**:Dataset:get/采样器输出/Subset.indices/weights 全 1-based,Insight choice 0-based 值在采样层 +1;`pd_op_randperm` 有 `returns_index={1}` → C ABI 层 pd_idx_plus1,random_split 直接可用。**构造互斥前置 _init**(reader.py 同款);**default_collate**:数字→Tensor、Tensor→pd_op_stack 沿新轴 0(**直接调 C ABI axis=0,不经过 _ops/stack.lua wrapper**)、多返回→pl.List、map 保键、string 裸表。**IO_SMOKE PASS(55 断言)**:seed=42 diag 逐位对拍(seq/rand/replace/weighted/batch/drop_last)+ 1-based 迭代 + break 重开 + num_workers>0 报错 + IterableDataset + ConcatDataset/Subset/BatchSampler + REQUIRE/OPTIMIZER 回归 + lint 全绿。**踩坑(实测 4)**:① pl.class `super` 是 call_ctor 临时注入(继承链有 `_init` 才注入,用后即清)——Dataset/Sampler/DataLoader 基类无 `_init`,8 处 `self:super()` 全报 nil → 删;② collate 前向引用(collate_sample 引用后定义的 default_collate → 全局 nil,删包装直接自递归);③ **Insight7 本地 Lua 绑定链陈旧**(dll 副本陷阱的 Lua 镜像):`build/bindings/lua/_insight.dll`(10-19)+ `build/backends/cpu/insight_cpu_backend.dll`(10-18)都在 numpy 对齐前,Python 侧 10-26 新 backend 只 POST_BUILD 到 bindings/python/insight/ → 重配 `-DINSIGHT_BUILD_LUA_BINDING=ON` 重建 `insight_lua` 目标后 choice 对齐恢复;④ smoke 必须 x64 LuaJIT(32 位 lua.exe 加载 x64 paddle_core.dll 报 Win32 错误),backend 依赖 fftw3/libopenblas 须 `build\backends\cpu` 在 PATH。**下一步 = 2.12 M1 验收:MNIST > 95%** |
| 2026-10-26 | **2.10 paddle.optimizer(SGD/Adam 首批)完成(蓝点 → 2.11)** —— `lua/paddle/optimizer.lua` Optimizer 基类 + SGD + Adam 全部 argrule 包装。**M1 无融合算子(10-optimizer.md §2 前提改写)**:`sgd_`/`adam_` 全是被 skip 的 in-place 变体(skip_rules.lua:45)→ **组合基础算子 + 核公式直抄**(SGD `param -= lr*grad` cpu/sgd_kernel.cc:39;Adam adam_functors.h:325-345:lr_eff 含 bias correction、mom1/mom2 递推、denom=√m2+ε·√(1-b2p);βN_pow **初始=βN**(adam.py:311-335,非 1)且每步 `*=βN` cpu/adam_kernel.cc:118-119)。**新增 C ABI 原语 `pd_tensor_copy_from`**(paddle_c.h Group 3 + tensor.cc,内部 `Tensor::copy_` tensor.h:510,深拷贝进参数本体、保持 autograd meta/leaf 身份——Python `_C_ops.sgd_/adam_` ParamOut in-place 语义)+ sol 绑定 `t:copy_from(src)`。**step 包 no_grad**(optimizer.py:2060 的 `@no_grad`,组合算子不得在 leaf 上建图,pcall 恢复)。**clear_grad 置零**(grad shell 共享 autograd 存储 autograd.cc:167,copy_from(zeros_like) 写本体,eager_method.cc:1066 set_to_zero=true 语义)。**OPTIMIZER_SMOKE PASS(18 断言)**:SGD/Adam 一步手算定值(含 β_pow 初值 0.9/0.999、第二步 bias correction 推进)、weight_decay 耦合 L2、clear_grad 后 grad=0、参数写回保身份、5 步 Linear+MSELoss+Adam loss 下降 + 8 smoke 回归全绿(NN_CONTAINERS 43/NN_LAYERS 87/NN_FUNCTIONAL 107/NN_LINEAR/NN_LAYER/OPS/AUTOGRAD/TENSOR)+ lint 全绿。**踩坑(实测 3)**:① pl.class 必须 `class()` 而非 `class("Optimizer")`(字符串名报 "must derive from a table type");② `parameters:len()` 只对 pl.List 存在,普通 Lua 数组报 nil → 谓词与镜像循环统一用 `#v`;③ smoke 第二步 backward 会累积梯度(autograd_smoke 同款语义),复算前必须先 clear_grad。**剩余(随 2.10 后续)**:Momentum/AdamW、lr 调度器、参数组、grad_clip、state_dict 接线、set_to_zero=false。**下一步 = 2.11 paddle.io(单 worker)** |
| 2026-10-26 | **2.9.5 容器完成(蓝点 → 2.10)** —— `lua/paddle/nn/containers.lua` 三个容器全继承 Layer(D27,is_a(nn.Layer) 为真,state_dict/parameters/train 递归零特例)。**构造 = argrule parse_whole_first 变长形态**(规则表仅 `{ name="self", type=argrule.table }`,回调收 whole,whole[1]=实例、whole[2..]=子层;`Sequential{ l1, l2 }` / 命名对 `Sequential{ {"fc1", l1}, ... }` / `LayerDict{ {"conv1", l1}, ... }`)。**方法层 1-based↔0-based 转换(D-R15 双基准)**:用户下标 ml[i] 1-based(按 order 书序取第 k 个,负数从尾数),state_dict 键名 0-based 字符串("0.weight")。LayerList append/insert/extend,Sequential forward 依序 + append/insert/extend,LayerDict update/keys/values/iter。**NN_CONTAINERS_SMOKE → 43 断言**(镜像 test_layerlist.py 空表 insert 边界 + test_imperative_container_layerlist.py 的 append/insert(-1)/extend/forward shape + LayerDict 行为 + state_dict 键名 + is_a + #ml==0)+ nn 全家 5 smoke 回归全绿 + ops/autograd/tensor 回归全绿 + lint 全绿。**踩坑(3 个 argrule 形态陷阱,全实测)**:① 空构造走 parse_position_table 直接收 **self 实例**(带子层才走 parse_whole_first 收 whole 表),回调须 `rawget(whole, FIELDS)` 区分;② 方法规则表第一规则不能叫 "self"(parse_self_call 对裸表第二参数做构造展开 → extend({l1,l2}) 被拆成 3 位置参数报错),改 `{ name="container", type=Layer }`;③ insert 直接操作 sublayers 槽绕过 __newindex 时,搬移新键 "n" 必须手动补进 order 书。FIELDS 键经 layer.lua:602 模块表导出,不在 Layer 类表上(须 `layer_module.FIELDS`)。**下一步 = 2.10 paddle.optimizer(SGD/Adam)** |
| 2026-10-25 | **2.6 全部验收完成(蓝点 → 2.7)** —— 2.6.2 `paddle.scope` + `core.memory_allocated`(ee03f65)+ autograd_smoke 补 stop_gradient nil-grad 断言(0e17fe7)。06-autograd.md §5 七项验收全过(MLP 手算梯度 1e-6 / grad==backward / no_grad 错误恢复 / no_grad stop_gradient / **scope 显存下降——probe 实证 memory_allocated 真实跟踪:+4000004B→0** / 累积到 8 / nil-grad)。**scope 实现**:纯 Lua 进入水位 + 退出 `collectgarbage("collect")` 施压(api/scope.md §6③),逃逸保守(返回值/upvalue 捕获天然存活)。**坑**:scope_smoke 初版断言 `after<=before` 太弱(before=after=0 也能过)→ 加 peak_inside 断言后才证明分配可见、回收生效(先验证数据源再断言)。OOM 回调随 T-M1-18(GC)。详见 tasks.md T-M1-10 |
| 2026-10-25 | **2.6.1 autograd 绑定完成(cd61d8e,T-M1-10 DONE)** —— `lua/paddle/autograd.lua`(backward/grad/no_grad/enable_grad/set_grad_enabled/is_grad_enabled)+ `tensor.lua METHODS.backward` + `csrc/capi/autograd.cc`(`pd_grad`/`pd_has_grad`/`pd_set_has_grad`,`pd_backward` 扩 grad_tensors)+ `csrc/sol/tensor_binding.cc`(core.backward/grad/has_grad/set_has_grad + usertype backward/grad/stop_gradient)。AUTOGRAD_SMOKE PASS(2 层 MLP 手算梯度 1e-6 / grad==backward / no_grad pcall 恢复 / 累积到 8)+ 旧 smoke 无回归 + lint 全绿。**核心踩坑:sol2 把 nil 中间实参当参数链终止符**(探针矩阵实证:`({1},nil,true)→c_has=0`、`({1},{},true)→c_has=1`),修法 = Lua 层调 core 中间参数永不传 nil、用 `{}` 空表/显式默认值。剩 06-autograd.md §5 验收 paddle.scope/OOM/x.grad nil 提示,随 T-M1-18 |
| 2026-10-24 | **主线① API 迁移标注核心抽查完成** —— `inventory.md` §3.1 从 14 批扩到 **34 批 / 268 行 `paddle.*` 成员**(提交 8b1ea3a→3c3b6ea),PYTHON 桶核心成员全部覆盖(创建/算子/归约/索引/逻辑/随机/激活/复数/三角/位运算/统计/linalg 子模块/全局态/工具),每个成员带上游 `file:line` + 三态去向 + C4 标注;剩余为 in-place 变体/常量,随生成器覆盖。主线②(M0 验证)此前已全部完成并开 G1(59e12ff)。**下一步待用户定:开 M1(建 csrc/ lua/ 骨架)/ 继续抽查剩余成员 / 其他** |
| 2026-10-23 | **✅ G1 闸门通过,M0 全部 17 项必做完成** —— 补完 T-M0-14/15/17/18/19 五项(#14 strided_slice=view、#15 OpProtoHolder 无 pybind 可用、#17 jit::Load 可链接可加载、#18 手工 AttributeMap 喂 run_program_ad_func、#19 SaveTensor/LoadTensor 格式稳定),前 11 项此前已 DONE。产出 `m0-report.md` §0 G1 验收总表。**当前允许 M1 全部工作**(建 `csrc/`/`lua/`、rockspec、CMakeLists)。遗留(不阻塞):Q-24(fc+oneDNN forward bug)、#20/#21(服务 M4 可跳过)。详见 `process/m0-report.md` §0 |
| 2026-10-23 | **T-M0-19 DONE:SaveTensor/LoadTensor 格式稳定性**(#19 ✅)。`tools/smoke/smoke_save_load.cc` 实测 PASS(exit 0):2x3 float32 往返无损、version=0、重存字节一致(file_size=50)。格式 = uint32 版本 + LoD + TensorDesc proto + 原始数据(`dense_tensor_serialize.cc:69-81`/`dense_tensor_tostream.cc:97-228`);Python 侧同链(D19 成立)。Lua `paddle.save/load`(单张量)直接以它为线格式。详见 `m0-report.md` §20 |
| 2026-10-23 | **T-M0-18 DONE:手工构造 AttributeMap 喂 `run_program_ad_func`**(#18 ✅)。`tools/smoke/smoke_run_program.cc` 实测 PASS(exit 0):`AttributeMap` 纯 C++ 构造(`type_defs.h:51/73`),按 `RunProgramImpl` 键全量填充(`run_program_impl.cc:449-490`),空 program 触发预期 C++ Traceback、无 pybind 泄漏。P16 静态图运行时参数可纯 C++ 构造。详见 `m0-report.md` §19 |
| 2026-10-23 | **T-M0-17 DONE:`jit::Load` 可链接可加载**(#17 ✅)。`tools/smoke/smoke_jit_load.cc` 实测:`libfluid_jit.lib` 导出符号、`.pdmodel/.pdiparams/.meta` 全读入、New 引擎启动。发现 ① 链接缺口(GetFeedTargetNames 在 pybind 目录,冒烟自带 shim);② forward 撞上游 fc+oneDNN bug(记 **Q-24**,非无 Python 特有,不阻塞)。详见 `m0-report.md` §18 |
| 2026-10-23 | **R42:生成器全 Lua 重写,开发期工具链也零 Python**(人拍板)。原 C9"生成器可用 Python"升级为硬约束:用户要求「代码生成全部用 lua 重新复刻,用户的电脑上面不装 python 也可以用 Paddle」。实测 `ops.yaml` 无锚点/多行标量/合并键 → 纯 Lua 子集解析器 `yaml_subset.lua`(~300 行)可行;Paddle 的 Python 生成器降级为行为参考与 CI 对拍。翻转落地:`AGENTS.md` C9、`03-codegen.md` §1/§2/§3.1/§6、`ci.md` §6 ③④/§7、`layout.md`、`overview.md`、`feasibility.md`、`conventions.md`、`WORKPLAN.md` 2.3.1/2.3.2。唯一保留 Python:`tests/parity/` 对拍基准(CI 校验手段,非用户工具链)。详见 `process/decisions.md` R42 |
| 2026-10-23 | **T-M0-11 DONE:纯 Lua pickle 读真实 `.pdparams`**(#11 ✅)。unpickler 实现在**独立仓库 `E:\code\lua-pickle`**(rock/模块名 `pickle`,双向可逆,依赖全可选 pcall 探测;分层:认识上游 penlight/Insight7、不认识下游 paddle)。`tools/check.lua` 在本机 Lua 5.1.5 上读通 `m0-11-sample.pdparams`(326B, protocol 4)并与 Python `pickle.load` 权威基准逐字节对拍 PASS。关键实现:68-opcode 栈机对照 CPython `Lib/pickle.py`;numpy 还原对照 numpy 2.4.6 实测(`dtype` REDUCE+BUILD 拼 `<f4`、ndarray `_reconstruct`+5 元组 BUILD);**踩坑:Lua 5.1 `#` 对洞表未定义** —— TUPLE2/3 初版 `stack[#]=nil` 收缩栈致后续 REDUCE 索引错位,改 `table.remove` 修复。CI 框架定案 **busted**(Lua 侧)+ **pytest**(开发期 Python 工具)。详见 `m0-report.md` §15,仓库 `lua-pickle/docs/design.md` |
| 2026-10-22 | **T-M0-10 DONE:堆追踪 vs Lanes 分配器实测**(#10 ✅)。`tools/smoke/lanes_alloc_check.cc` 输出 `LANES ALLOC PASS`:3 个 lane state 的 allocator 全 != counting_alloc(**LuaJIT 下 master 的自定义 allocator 到不了 lane**,`state.c:253-255`);**陷阱:`lua_setallocf` 必须链式转发原分配器,裸 realloc/free 替换 = 0xC0000374 堆损坏**(vcpkg LuaJIT 是 lua51.dll 自有 CRT 堆)。落地:堆追踪做张量字节计数器与 Lua allocator 无关(详见 `process/m0-report.md` §14)。WORKPLAN **1.5 ✅,** 🔵 移至 **1.6**(T-M0-11..19) |
| 2026-10-22 | **T-M0-09 DONE:Lua 5.4 `__close`/`__gc` 语义实测**(#9 ✅)。`tools/ci/lua_close_check.lua` 在 5.4.7 + 5.1.5 上 PASS,已登记 l0.yml lua-matrix。关键发现:**GC 只调 `__gc`,不调 `__close`** —— RAII usertype 必须双挂且幂等(详见 `process/m0-report.md` §13) |
| 2026-10-22 | **✅ G0 闸门通过** —— `WITH_PYTHON=OFF + ON_INFER=OFF`(无 Python 训练态)编译成功:5195 files / 9.90 GB(`build-nopy`,Ninja Release,`-DWITH_SHARED_IR=OFF`,`-j6`),5 个验收符号全部确认导出;`tools/smoke/smoke.exe` 冒烟输出 `SMOKE PASS: a.grad == 4.0, b.grad == 3.0, max_err=0`(T-M0-01/T-M0-02 DONE)。Paddle 补丁 **0001**(`legacy_eager_codegen` no-op target)应用且与 `patches/paddle/` 逐字节一致(C2 满足)。记录 `process/m0-report.md` |
| 2026-10-21 | 根目录作业手册 `CLAUDE.md` 更名 `AGENTS.md`(`git mv`,历史保留),全仓库 16 个文件 43 处引用同步替换,无残留 |
| 2026-08-03 | 建立 `AGENTS.md` / `process/status.md` / `process/tasks.md` / `process/conventions.md` / `process/decisions.md` / `process/open-questions.md` / `README.md` |
| 2026-08-03 | `research/reuse.md`:执行器复用调研,推翻"自研静态图=独立项目"的评估 |
| 2026-08-03 | `research/to-static.md`:AST 选型 luacheck parser;源码契约可验证化 |
| 2026-08-03 | `plan/overview.md`:索引改全 1-based;排除清单 9→5 项;Ocean/Metrics 移至 M3 |
| 2026-08-03 | 目录重构:全部文档收进 `docs/migration/{plan,process,research}/`;`docs/` 下只留这一个子目录,未来的 API 说明另开目录 |
| 2026-08-03 | 建立根目录三语 README(英文默认 + zh-CN + zh-TW)与 `LICENSE`(Apache 2.0) |
| 2026-08-03 | `plan/roadmap.md`:新增总计划,P0–P18 依赖图 + 每阶段开工条件与完工判据 |
| 2026-08-03 | `plan/modules/`:新增 19 份模块详细设计(约 2700 行);`plan.md` 更名 `overview.md` |
| 2026-08-03 | `09-nn.md`:恢复**自动注册**(R17);`13-lanes.md`:Tensor 跨 lane 改用 `__lanesclone`,放弃 deep userdata(R18);Q-11 关闭 |
| 2026-08-03 | **新增 `plan/api/`(按 Lua 模块的接口设计)** —— 与 `plan/modules/`(按阶段的实现设计)分离:一个是「造出来长什么样」,一个是「怎么造」。立了跨模块硬规则(命名映射、index 语义必须逐模块标注、返回 `pl.List` vs 裸表的边界、`require "paddle"` 惰性加载)与固定骨架;`api/io.md` 作为样板写完。工程树里每个模块新增 `x.y.0 API 设计` 子节点 —— **先接口后实现** |
| 2026-08-03 | **新增 `WORKPLAN.md`(总工程树)** —— 智能体按它 DFS 遍历,遍历完 = 工程完成;新增 `plan/layout.md`(权威目录树 + 文件级落地顺序 + 生成代码进版本库的决策)与 `plan/ci.md`(四层 CI、阶段解锁表、五条机器红线)。关键设计:**兄弟节点排成拓扑序**,使朴素 DFS 自动满足 DAG 依赖,`前置` 字段退化为断言 |
| 2026-08-03 | 人的三条决定落盘:**① `nn.LayerList` 改为继承 `Layer`**(R22,Layer 优先;连带发现 5.1 的 `ipairs` 用 `lua_rawgeti`,`ipairs(ml)` 会静默跑空 -> 改用 `ml:iter()`,新增 Q-16);**② 「不引入新的强制 C 依赖」禁令取消**(R23,判据改为边际成本,`AGENTS.md` §9.1;解禁 HTTP 下载与图像解码,Q-08 风险下降);**③ Insight7 的 `axis` 按 bug 修成 1-based**(R24,成员索引与维度索引都要 1-based,顺手做、P12 前完成;待拍板 P1 关闭、Q-12 转已决)。新增 D27–D29 |
| 2026-08-03 | **新增 `plan/foundations.md`(生态基座)**:Penlight 定为一等公民(R19/R21),Insight7 顶替 numpy 的位置(R20)。新增硬约束 C11、决策 D23–D26、未解问题 Q-12–Q-15;`conventions.md` 章节重编号(§2 生态基座、§3 与 Python 的已知差异) |
| 2026-08-03 | **argcheck 评估落盘,当天翻案两次(R25 -> R26)** —— ① 先推翻人的两个前提:「硬编码 Torch7」不成立(耦合共 32 行 2 处,`env.istype` 原注释就是 `-- user configurable function`);「过时」也不准,真实病灶是 `graph.lua:13` 拿 `tostring(t):match(0x…)` 当标识符,**MSVC 的 `%p` 不带 `0x` -> 带默认值的规则集在 Windows 上必崩**(改 2 行后上游全套测试通过)。② 据性能(2597 ns/call)给出「冷路径 vendored argcheck」= **R25,当天即被自己推翻**。③ 补测「能不能表达」:argcheck 对每条规则枚举三态共 **3^N** 路径,9 个可选参数 1.37 MB / 840 ms,**10 个就 `control structure too long`** —— 而 `Conv2D` 11 个、`Adam` 12 个、`DataLoader` 16 个(3^16,**挂死**),**R25 说的「冷路径」正好就是这张表**。④ 定案 **R26:不 vendor,取 schema + `usage.lua`,自写 ~150 行 O(N) 生成器 `_args.lua`** —— 原型实测 3 参 403 ns(快 6.4 倍)、17 参 3.6 KB / <1 ms。教训写进 `decisions.md` §2.11:**找到第一个可接受的缺点会让人停止寻找第二个;「多慢」是程度问题,「能不能表达」是有无问题,应该先问有无** |
| 2026-08-03 | **参数解析器独立成项目(R27/D31),并按人的两条追加约束定型** —— ① **不轻易造轮子**:普查了 luarocks 全部候选(`tableshape` / `typecheck` / `checks` / `ltypekit` / `geoffleyland/argcheck`),**没有一个提供「同一份签名同时吃位置调用与具名表调用 + 默认值 + usage」** —— 因为那是 **Python kwargs 的形状**,而 Lua 自己没有 kwargs。所以只自写「调用约定层」(~200 行),**类型判定层借现成的**:`type` 槽定为可调用契约,`tableshape` 的类型对象本身就是 callable,塞进去就能用(R28/D32)。② **50 个参数**不是假想需求:Paddle 真实最大签名 `ResNetBasicBlock.__init__` **43 参数 / 33 可选**(`resnet_block.py:434`),≥12 参数的有 156 个。实测 Lua 5.1 三道墙 = **60 upvalue / 200 局部变量 / N=122 寄存器**,而 argcheck 是**一规则若干 upvalue** -> 43 参数 ≈ 86 个 upvalue,**即使没有 3^N 也编不出 Paddle 最大的签名**(R29/D33)。改成单表 upvalue 后实测 50 参数 7.4 KB / ~1 ms / 3.75 µs,N>100 走表形态(1000 参数仍可编译)。③ 名字不能叫 `argcheck`(已被 `geoffleyland/argcheck` 占用),建议 `argsig`(待拍板 P10);④ 「基于 Penlight」与 §1.3 的 vendor 决定冲突(Penlight rockspec 依赖 `lfs`),**新增待拍板 P9**,拍板前按「两边都改声明依赖」写。CI 红线 ①b 从 2 条扩到 4 条,新增 43 参数基准 |
| 2026-08-03 | **人拍板 P9:Penlight 不再 vendor,改为全生态地基级 rock 依赖(R30/D34)** —— 原话「pl 为默认依赖项,pl 在咱们所有的项目里面为地基级别的东西」。vendor 的四条理由里第 1 条(避 `lfs`)在 R23 已作废、第 3/4 条是便利性,只剩版本锁定,而它靠 rockspec 锁 minor + CI 语义测试就能达成。决定性的是 vendor 自带的代价:**两份 `pl.class` 的实例互不 `is_a`** —— 生态里有 paddle-lua / Insight7 / `argrule` / metrics / ocean 五个消费者,这个坑会在每两个库之间各出现一次。连带:`_vendor/pl/`(5374 行)从 `layout.md` / `04-packaging.md` / `overview.md` 删除;`luafilesystem` 进入传递依赖(但我们自己不 `require "lfs"`);待拍板 P7 改写为「rock 版本区间怎么定」;CI 新增红线 `grep -rn "_vendor/pl" -> 失败`。同时新增 `plan/argrule.md`(176 行)——「新时代 argcheck」的孵化说明书,对着 argcheck 列出继承/丢弃/新增三列账,每行带上游 `file:line` |
| 2026-08-03 | **签名层的用户端表面定型(`plan/argrule.md` §2 §2.5,384 行)** —— 人给的形状:`local rule = require "argrule.rule"`;规则里 `name`/`type` 可位置写(`{"x","Tensor"}`),**规则表自己也遵守它自己的调用约定**;`help` 砍掉只留 `doc`(argcheck 两个都有还要 assert 二选一,`init.lua:80`);装饰器直接吃 `_C_ops.softmax`,不包 lambda。落地时补了四条:① **位置槽只到 2**(第三个位置是 `default` 还是 `doc` 说不清);② 类型短名 `"Tensor"` 走 `alias`,**全名是规范,短名重复注册直接报错**(一个进程里 paddle 和 Insight7 同时在);③ 公开 API 一律全具名(要拿来生成文档,且会被当范例抄)—— 写进 conventions,不进 CI;④ ★ **`paddle.zeros{2,3}` 的真歧义**:第一个参数就是 table 时,表形式调用无法与「一个 table 实参」区分,**必须显式 `nonamed = true`**(逃生舱从 argcheck 继承,`init.lua:53-57`),判据机械可查已进 CI 红线 ①b —— 不用启发式,因为猜错的那次是静默的错误结果 |
| 2026-08-03 | **新增跨模块硬规则「枚举参数一律不接受裸数字」(`api/README.md` §2.1.1)** —— 人的原话:「dtype 必须是 paddle.dtype 类型或者是 string,数字不应该代表类型」。三条理由里第二条是意外收获:**它把 `paddle.zeros{2,3}` 的调用歧义消掉了** —— `{2,3}` 当「表内位置」解释时 `dtype = 3` 过不了类型检查,于是确定性地落到「整个表是第一个实参」,**不需要写 `nonamed`**。因此 `argrule.md` §2.6 从「必须写 `nonamed`」改成**四步定死的解析顺序**,只有「表内位置」与「整表实参」两种解释**同时**成立时(如 `full{{2,3},1.0}`)才要求显式声明。两条规则相互依赖,**要改一起改**。同时定下短名原则:**类型短名 = 用户能 `local Tensor = paddle.Tensor` 拿到的那个导出名**(对应 Python 的 `from paddle import Tensor`),`Tensor`/`Place`/`DType` 归 paddle、`Array` 归 Insight7,短名重复注册直接报错。CI 红线 ①b 增至 6 条 |
| 2026-08-03 | **消歧判据补上「必填参数」这一半,并新增 `extend` 处理共有类型** —— ① 人指出 `full{{2,3},1.0}` 其实不歧义:`fill_value` 必填无默认,所以「整表当 shape」这条路本身走不通。判据因此从「类型全过」改成**「类型全过 **且** 必填参数都有值」** —— 少了后半句会凭空造出一堆假歧义。真歧义的条件变得很苛刻:规则 #1 接受 table **且其余参数全可选**,典型是 `concat` / `stack` / `meshgrid` 这批**第一个参数是列表**的函数,它们一律写 `nonamed = true`(写了之后 `concat{a,b}` 反而天经地义)。② 人指出 **Insight7 也有 `Place` / `DType`** —— 因为 Insight7 的 API 本来就照着 Paddle 设计。所以这不是撞名是同一概念:新增 `argrule.extend(name, pred)`(谓词取或),`register` 撞名仍报错,两者区分开。顺带记一笔长期方向:`DType`/`Place` 最好在 C 层就是同一个值类型,零拷贝互操作已要求布局对齐,dtype 却还要在边界转换一次 |
| 2026-08-03 | **`shape` 一类参数改用 `IntList`,且判据是「是个容器 + 装的是整数」而不是类名单(R31)** —— 人的原话:「paddle 里面 shape 必须是 tuple or list or np.ndarray 不能是数字」「**反正需要是个容器**」「**而且装的是整数**」。我先写成 `{"table","pl.List","insight.Array"}`,**当天即被自己推翻**:那是把「容器」硬编码成框架名单,用户自己的容器类会被无理由挡住,正是 §4 零框架硬编码要挡的。改为组合子 `argrule.list_of(elem)`(探容器协议 + 逐元素套 `elem`),`IntList` / `TensorList` 只是宿主注册的别名。容器协议:长度 `#o` -> `:len()` -> `:size()`(⚠️ **5.1 的 `__len` 对 table 无效**,进 M0 #24 实测)、取元素 `o[i]` 1-based、自报 dtype 走 O(1) 快路否则逐元素 O(n)。**意外收获:元素逐个查把 `concat` 的歧义也消掉了** —— `concat{a,b}` 的②里 `x=a` 是 Tensor 不是 TensorList、`concat{{a,b},2}` 的③里元素 `2` 不是 Tensor,各自出局。**所以上一版「第一个参数是列表的函数必须写 `nonamed`」作废**,真歧义要求 `list_of(E)` 且 `E` 自己接受容器,现有 API 里一个都没有。教训与「必填参数是天然消歧器」同源:**判据越弱,假歧义越多** |
| 2026-08-03 | **`Tensor` 也满足容器协议(人的决定),并纠正我自己的两处断言** —— 人:「Tensor 也应该满足容器协议,毕竟只要是个 int 容器按说都应该行」。① 上游核实通过:`ShapeLike = Sequence[int|Tensor|None] | Tensor`(`_typing/shape.py:22-33`),"若 shape 是 Tensor,须是一维"(`creation.py:1832`),**且 list 的元素可以是 0-D Tensor**(`:1831`)。② 因此容器协议分 **plain / opaque** 两级 —— 区别不是"谁写的",是**逐元素访问贵不贵**:对 GPU 上的 Tensor 逐元素查类型 = n 次设备同步,**红线,禁止**。选路机械:`E` 是标量类型就用容器自报的 dtype+ndim 做 O(1) 证明,否则逐元素;没有 O(1) 证明又不许逐元素 -> 判否。**这条判据自己算出了上游的行为:一维 int Tensor 是合法 `shape`(收),二维 Tensor 不是 `TensorList`(`concat` 的 `x` 是 `Sequence[Tensor]`,`manipulation.py:1482,1507`,不收)—— 没写任何特例。** ③ ⚠️ **纠正**:我昨天写的「`paddle.zeros(5)` 本来就是错的」在当前 develop 上**不成立** —— 上游有 `@size_args_decorator`(`decorator_utils.py:406-437`),`ones(1,2,3)` / `ones(5)` / `ones(size=[...])` 全合法。但注意上游的做法:**没有把 `int` 加进 `ShapeLike`,而是用装饰器在签名外面归一化**。我们照抄这个分层 —— 表级选项 `sizeargs = "shape"`,`type` 仍是 `IntList`(`argrule.md` §2.4)。连带:第一个位置实参是整数时全部位置实参归 `shape`、`dtype` 只能具名(上游同此) |
| 2026-08-03 | **P10 拍板:签名层定名 `argrule`** —— `plan/argsig.md` -> `plan/argrule.md`,全仓库引用、rockspec 依赖项、WORKPLAN 节点 1.7.0 同步改名;`foundations.md` §5.4.7 与 `decisions.md` P10 记下"我建议的是 `argsig`,人选了 `argrule`"。**剩下的动作:去 luarocks 占位** |
| 2026-08-03 | **`sizeargs` 降级出签名层,`IntList` 的元素检查下放转换层(R32)** —— 人的两问:「`sizeargs` 之前有讨论过吗」**没有,是我上一轮临时加的**;「我想的是在函数里面有个 if 判断如果有小数直接 error」**对,而且比放类型层更好**。① `sizeargs` 全 Paddle 只用在 5 个函数(`ones`/`zeros`/`empty`/`randn`/`rand`,`creation.py:1644,1807,3081`、`random.py:961,2343`),而上游自己就是**装饰器在签名外面** —— 照抄分层比照抄行为更重要,做成 paddle-lua 侧 ~10 行,签名层零改动。② 元素检查下放的判据是机械的:**去掉它之后调用仍唯一解释 -> 可下放**。`IntList` 满足 -> 转换层 `if` + `error`(那步反正要遍历 n 次,类型层再扫一遍是查两遍;0-D Tensor 元素也只有那层处理得了;报错还能带下标 `shape[3] must be an integer, got 2.5`);`TensorList` 不满足(`concat{{a,b},2}` 靠元素类型出局)-> 留类型层。**红线:必须 error 不许静默取整、必须指向调用点、C++ 异常不得穿过 Lua(C7)** |
| 2026-08-03 | **`sizeargs` 整个砍掉,并升成一条通则:上游的兼容糖一律不移植(R33)** —— 人:「用得少的东西,为了 API 简洁似乎就该去掉」。决定性的不是"用得少",是**上游加糖的原因在 Lua 侧不存在**:Python 的 `zeros([2,3])` 又是括号又是方括号才需要 `zeros(2,3)`;Lua 的表调用本来就省括号,**`zeros{2,3}` 一样短** —— 收益 0,成本照付。同一条理由挡住整层兼容装饰器:**291 处参数别名**(`concat(tensors=,dim=)`)、5 处变长 size、`reshape(2,3)` 等。别名还额外带两个坑:`concat{x=a, tensors=b}` 要复制 291 次「不能同时给」的检查(上游 `decorator_utils.py:181-184` 抛 ValueError);`dim`/`axis` 两个名字 = index 语义标注表两条记录,漏标一条就是静默 off-by-one。**代替方案:教学式报错 `did you mean: paddle.zeros{2, 3}`**,只在错误路径上,零运行时成本。新增 `api/README.md` §2.1.3,例外判据「这个写法在 Lua 里比规范写法更短或更清楚吗」,**理由不能是"上游有"** |
| 2026-08-03 | **`decorator_utils.py` 整层不移植(R34/D35)** —— 人的原话:「这东西就是 Paddle 为了 PyTorch 用户用得惯才搞的,lua 里面不用考虑这些,**我们 Paddle 框架就应该用自己的语法和规范**」。范围从「逐个判断哪些糖要移植」升成「整层跳过」:1451 行 / 30+ 装饰器 / 68 个模块 import 它,全文 **51 处** `PyTorch`/`torch.`(docstring 直写 `PyTorch: torch.block_diag(x,y,z)` / `Paddle: paddle.block_diag([x,y,z])`,`:923-940`),甚至有专门劝返 torch 关键字的 `forbid_keywords`(`:517-542`)。**关键证据:35 处 `wrapper.__signature__ = inspect.signature(func)`** —— 上游自己把规范签名保留在内层,**所以我们的生成器读到的本来就是规范签名,忽略这层是默认行为而不是额外工作量**。⚠️ 唯一要逐个确认的:少数装饰器可能不只改参数(`legacy_reduction_decorator` / `view_decorator`),判据「壳只动参数、内层签名不变」,发现改了行为的记 OPEN_QUESTION |
| 2026-08-03 | **新增硬约束 C12:用户可见的 API 一律用 Paddle 自己的规范,不引入 PyTorch 的命名与惯例(R35/D35)** —— 人:「所以我们的 paddle-lua 也不要那些 PyTorch 的东西」。**光跳过 `decorator_utils.py` 挡不住** —— 一批 torch 风格参数**已经进了上游的规范签名本身**:`zeros(shape, dtype, name, *, out, device, requires_grad, pin_memory)`(`creation.py:1807`),而 Paddle 自己的名字在 `to_tensor` 里原样留着(`place` / `stop_gradient`,`:1124-1129`)。取 Paddle 那套:`stop_gradient` / `place` / `axis` / `x` / `shape` / `place=CUDAPinnedPlace()`。**`requires_grad` 必须挡住的真正理由:它与 `stop_gradient` 取反、默认值也反**(`False` vs `True`)—— 两个名字并存时搞混一次就是**静默地训不动**,没有报错只有 loss 不降。⚠️ 例外:`Tensor.size`(元素总数)是 Paddle 原生属性,不在禁用之列。**边界写死:约束用户看得见的表面,实现层不受限**(GC 抄 Torch7 九层 D5、schema 抄 argcheck D30 照做,因为用户不会写到它们)。CI 判据 `grep -rnE 'name *= *"(requires_grad|device|dim|input|tensors|pin_memory)"' lua/`。新增 Q-19(`out=` 是能力不是命名,倾向 v1 不做)、Q-20(`decorator_utils.py` 里是否有"壳改了语义"的例外,P3 前查完)|
| 2026-08-03 | **「尾随选项表」这个写法不存在,一张表就是全部参数(R36)** —— 人指着 README 首屏说:「DataLoader 不应有额外的 `{}` 进行包装」「**你还不如直接 `{ds, batch_size=...}`**」。查下来**不是风格问题,那一行按我们自己的规则直接报错**:`DataLoader(ds, {batch_size=64})` 是两个实参的**位置调用**,规则 #2 `batch_size:number` 收到了一张 table。修法不是逐处订正,是把**混合表**写进 `argrule` 解析顺序的第 ① 步 —— 一张表里数组部分按位置填前 k 条规则、具名键按名字填、同一参数给两次即 `error`,语义与 Python「位置在前、关键字在后」一致。于是每个 API 只有三种合法写法(位置 / 全具名 / 混合表),**没有第四种**。⚠️ **教训:`f(x, {…})` 与 `f{x, …}` 只差一个字符,而前者是绝大多数 Lua 库的习惯** —— 我自己写的 README 错了,还反复读过多次都没发现,人一眼看出来。这类「读着完全正常、却跑不通」的错误只能靠机器:CI 红线 ①b 新增 `args_no_opts_table.lua`,**且必须扫 `docs/`**(文档是用户抄走的第一份代码)。连带订正 3 份 README + `overview.md`(4)+ `03-codegen.md` + `11-io.md`(2)+ `roadmap.md` + `07-serialization.md` + `research/reuse.md`;新增 `api/README.md` §2.1.5 |
| 2026-08-03 | **一次自由讨论的落盘:新增 `research/ecosystem.md`(生态构想与顺序)** —— 内容不是承诺,是"为什么这么排序"。① 反复验证出一条判据:**「语言无关」= C ABI + 一份内存描述协议**,在 PIL(重活本来就在 libjpeg/FreeType,迁移=写绑定)、numpy(护城河是 buffer protocol/DLPack 不是 API)、设想中的 InsightPlot(资产是中间那层 display list)上都成立 —— 与 `argrule` 的「容器是协议不是类名单」(R31)是同一条教训。② **R38:索引基准与内存序属于绑定层**,Insight7 已是 `src/` + `bindings/{python,julia,lua}` 架构,Python 0-based / Lua 1-based / Julia 1-based+column-major,**同一个核心** —— Julia 那列是决定性证据(连内存序都得在绑定层处理)。这让 R24 从"针对 Lua 的修补"升为通则,我们依赖的 1-based 不会被回退。③ 实测 `python/paddle` 全部 `.py` 只用到 **132 个 `np.*` 符号**(numpy 公开 API 约 600),前 34 名一半是 dtype 与常量,**真正的函数只有十五个** —— 所以"先做 numpy 功能"的判据应改为"paddle-lua 用得到的子集",且三步不该串行。④ 新增 **Q-22**(phi 层的 DLPack 通道,决定 Insight7 协议)、**Q-23**(Array API Standard 是否如我记忆所述 —— **整段凭记忆,未核实**)、待拍板 **P11**(上游若把 `axis` 改名 `dim` 跟不跟,建议不跟)。⑤ `ci.md` 新增 §7.1:**数值对拍测不出「视图 vs 拷贝」** —— 两边输出值完全一样,差别只在改了返回值后原张量变没变,必须专门设计别名测试;它与 `requires_grad`/`stop_gradient` 同性质:**不报错,只是结果错**。⑥ 记下两条方法论:**没有 KPI 的红利是"能推翻自己"而不是"能做更全"**(numpy 的包袱正是来自它推翻不了自己);**Agent 最擅长帮人做的事恰恰是扩大范围**,而 60% 的框架等于 0 |
