# M0-REPORT.md · M0 验证报告(#1 编译 / #2 反向冒烟)

> 结论先行:**G0 闸门通过(2026-10-22)。**
> `WITH_PYTHON=OFF + ON_INFER=OFF`(无 Python 训练态)在 Windows/MSVC 上编译成功,
> `egr::Backward` 反向传播数值与手算一致,5 个验收符号全部确认导出。
> 唯一 Paddle 改动 = 补丁 0001,与 `patches/paddle/` 逐字节一致(C2 满足)。

---

## 1. 任务与验收

对应 `process/tasks.md` 的 T-M0-01(生死判定编译)与 T-M0-02(反向冒烟)。

| 验收项 | 结果 |
|---|---|
| 编译成功(或失败原因完整记录) | ✅ **成功**(5195 files / 9.90 GB) |
| 5 个验收符号导出情况 | ✅ 全部确认(见 §3) |
| `$PADDLE_ROOT` git status 干净 | ✅ 仅补丁 0001 的 +9 行(见 §5) |
| `x.grad()` 数值与手算一致 | ✅ `SMOKE PASS: a.grad == 4.0, b.grad == 3.0, max_err=0` |

---

## 2. 构建记录

| 项 | 值 |
|---|---|
| 构建目录 | `E:\code\Paddle\build-nopy`(Paddle 3.5.0,分支 `windows_dataloader_multiprocess`,git `cd` 干净) |
| cmake 命令行 | `cmake .. -GNinja -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_SHARED_IR=OFF` |
| 编译并行度 | `cmake --build . -j6`(12 核吃一半,用户 2026-10-21 指令) |
| 编译器 | MSVC 14.50.35730(Visual Studio 18.x,vcvars64) |
| 耗时 | 跨 2026-10-21 至 2026-10-22,含 3 次失败与修复(见 §6 错误史) |
| 产物规模 | **5195 files / 9.90 GB**(含全量静态库与测试二进制) |
| 冒烟产物 | `tools/smoke/smoke.exe`(链接 483 库的 rsp,`/FORCE:MULTIPLE`) |

**`WITH_SHARED_IR=OFF` 的原因**:该配置与旧构建一致,解决 186 个 dllimport 链接错误。
上游默认 `WITH_SHARED_IR=ON` 会把 IR 相关符号做成 dllimport 形式,与静态库链接冲突;
关闭后改走静态链接,编译与链接均正常。

**没有聚合 `libpaddle`**:无 Python 训练态下上游不产出单一聚合库,产出的是组件静态库
(`libbackward.lib` / `libfinal_dygraph_function.lib` / `libphi.lib` 等)+ `phi.dll`。
冒烟程序直接以这些组件库为链接基底(复用 `paddle_inference_c.dll` 的 483 库清单)。

---

## 3. 五个验收符号的导出确认

| # | 符号 | 载体 | 确认方式 |
|---|---|---|---|
| 1 | `egr::Backward` | `libbackward.lib` | `dumpbin /symbols`: `?Backward@egr...` External 定义 |
| 2 | `paddle::Tensor` | `phi.dll`(导出)+ 静态库(内联) | `dumpbin /exports`;⚠️ 见 §4 的 PHI_INNER 双重定义 |
| 3 | `jit::Layer::Load`(实为 `paddle::jit::Load` free function) | `libfluid_jit.lib` | `dumpbin /symbols`: `?Load@jit@paddle...` External 定义 |
| 4 | `framework::LoadOpMetaInfoAndRegisterOp` | `libcustom_operator.lib` | `dumpbin /symbols`: External 定义 |
| 5 | `RegisterOOMCallback` | `phi.dll` | `dumpbin /exports` |

每处引用均对应源码出处,已在 `tools/smoke/smoke.cc` 头部注释逐条标注 `文件:行号`。

---

## 4. 已知怪癖(对后续阶段有直接影响)

### 4.1 PHI_INNER 双重定义 → EXE 链接必须 `/FORCE:MULTIPLE`

`libbackward.lib` / `libinit_phi.lib` 编译时带 `-DPHI_INNER`,把 `paddle::Tensor` 等
phi 符号**定义内联进静态库**;同时链 `phi.lib`(phi.dll 导入库)即产生 **LNK2005 重复定义**。
冒烟 EXE 因此加 `/FORCE:MULTIPLE`(产生 LNK4088 警告,映像实测可正常运行)。

⚠️ **对绑定层的影响**:sol2 绑定层最终产出的是自己的 `.dll`,`paddle_inference_c.dll`
这类按 DLL 链接的路径不受此影响(实测无 LNK2005)。此怪癖**只影响「独立 EXE 直链组件库」**
的形态,如本冒烟程序与未来的 C++ 测试工具。

### 4.2 必须先 `InitMemoryMethod()`

冒烟开头必须显式调用 `paddle::framework::InitMemoryMethod();`(`paddle_gtest_main.cc:73`,
`init.cc:443`),否则 phi 分配直接 fail-fast(0xC0000409,`memory_method_ not initialized`)。
普通(带 Python)跑测试时这个调用被 gtest main 包掉,用户侧不可见 —— 无 Python 环境要自己补。

### 4.3 无 kernel 注册声明则运行时 "operator is not registered"

eager 冒烟(及未来的绑定)需要 `PD_DECLARE_KERNEL(full, ...)` 等声明把 kernel 链接进来,
范例是 `test/cpp/eager/task_tests/backward_test.cc`。漏写则编译通过、运行时报
"operator is not registered"。这是**静默型陷阱**,绑定层生成器必须自动产出这一组声明。

---

## 5. 补丁历史(C2)

只有一个补丁,应用后工作区 diff 与 patch 文件逐字节一致。

| patch | 内容 | 原因 |
|---|---|---|
| `patches/paddle/0001-cmake-nopy-training-legacy_eager_codegen.patch` | `paddle/fluid/pybind/CMakeLists.txt` 末尾加 `if(NOT WITH_PYTHON AND NOT ON_INFER) add_custom_target(legacy_eager_codegen) endif()`(+9 行) | 无 Python 训练态下 `fluid_generated` 仍会被挂载(`eager/api/generated/CMakeLists.txt:4`),其生成 CMakeLists 无条件 DEPENDS `legacy_eager_codegen`,而该 target 只在 `if(WITH_PYTHON)` 内定义。补一个 no-op target 使依赖图成立;生成文件已随源码树存在,无需实际运行 eager_generator |

验证命令:

```bash
git -C E:\code\paddle status        # 仅 M paddle/fluid/pybind/CMakeLists.txt + 未跟踪 build-nopy/
git -C E:\code\paddle diff          # 与 0001 patch 逐字节一致
git apply -R                        # 可逆还原(未执行,保持工作区现状)
```

---

## 6. 错误史(T-M0-01 过程中的 3 次失败与修复)

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 186 个 dllimport 链接错误 | `WITH_SHARED_IR=ON` 默认态与静态库冲突 | `-DWITH_SHARED_IR=OFF` 重配 |
| 2 | LNK2005:`paddle::Tensor` 重复定义(phi.lib vs libbackward/libinit_phi) | PHI_INNER 内联 + phi.dll 导入库 | 链接加 `/FORCE:MULTIPLE`(见 §4.1) |
| 3 | LNK1181 打不开 `libcblas.lib` | 它在 build-nopy 根部而非 third_party | rsp 写绝对路径 `E:\code\Paddle\build-nopy\libcblas.lib` |
| 4 | 运行时 0xC0000409 无输出 | 缺 `InitMemoryMethod()` | smoke.cc 开头补上(见 §4.2) |

另有一个**编译期**发现:冒烟 include 的 `paddle/phi/api/include/all.h` 不存在,
正确头是 `paddle/phi/api/all.h`(诊断文件已删,不影响产物)。

---

## 7. 产物落盘

| 文件 | 说明 |
|---|---|
| `tools/smoke/smoke.cc` | 冒烟源,永久保留(含 `文件:行号` 出处注释) |
| `patches/paddle/0001-*.patch` | 补丁,永久保留 |

已清理:smoke 的编译中间产物(`smoke.exe`/`.exp`/`.lib`/`.obj`)与链接响应文件
(`smoke.rsp`,483 库清单)均已删除 —— 链接基线信息已记录在 §2(复用
`paddle_inference_c.dll` 的 LINK_LIBRARIES);重编冒烟程序的方式见 §2 与
`tools/smoke/smoke.cc` 头部注释。诊断中间文件与 `%TEMP%\sym_*.txt` 等符号证据也已清理。

---

## 8. 遗留与下一步

- ✅ 冒烟与**手算**逐位一致;与 **Python 端**逐位对拍未做(本构建即无 Python)。如需,可
  另建 `WITH_PYTHON=ON` 构建对拍 —— 列为可选,不阻塞。
- ✅ **T-M0-03 已加验(2026-10-22)**:`tools/smoke/smoke_mt.cc` 多线程冒烟 ——
  4 线程×50 轮 + 8 线程×200 轮(共 1800 次 full→multiply→sum→Backward 完整图),
  全部线程**共享框架 DeviceContextPool**,每轮 `a.grad==4.0 / b.grad==3.0 / max_err=0`。
  **结论:CPU 上多线程共享 DeviceContext 安全,不需要 per-thread context。**
  依据:① `CPUContext::Impl` 仅含 `place_` + `Eigen::DefaultDevice*`(无 per-thread
  可变缓冲,`phi/backends/cpu/cpu_context.cc:27-54`);② 分配器链 `IsAllocThreadSafe`
  强制为真(`phi/core/memory/allocation/retry_allocator.h:48`)。这直接支撑 R41
  (DataLoader 多线程方案)与绑定层多线程安全。编译/链接复用 smoke.cc 的完整 483 库
  基线(从 `paddle_inference_c` 的 CMake `LINK_LIBRARIES` 提取)。
- ✅ **T-M0-04 已加验(2026-10-22)**:Lanes v3.17.2 与 4.x(master@4aa54c9)双 clone
  对比,见 §9。
- M0 其余验证项(#5–#21)按 `status.md` §4 顺序继续,下一项 = T-M0-05(Lanes 与
  绑定的 C++ ABI 一致性,MSVC)。
- 生成器必须把 `PD_DECLARE_KERNEL` 组自动写进绑定(§4.3),记入 `plan/modules/03-codegen.md`。

---

## 9. Lanes 版本差异核对(T-M0-04)

**结论:`__lanesclone`(D24 的接口面)在 v3.17 与 4.x 上调用约定完全一致;deep
userdata 的 C API 形态两版不同,但我们已不依赖它(R18 改走 `__lanesclone`)。**

| 项 | v3.17.2 | 4.x(master@4aa54c9) | 影响 |
|---|---|---|---|
| `__lanesclone` 元方法 | ✅ `src/tools.c:1542-1618, 1752-1758` | ✅ `src/intercopycontext.cpp:807-871, 1007-1016` | **两版一致**。调用:push lightuserdata(clone/source) + integer(size),`lua_call(L, 3, 0)` —— 即 `clone:__lanesclone(dest, source, size)`。D24 绑定层写一份即可跨版本 |
| deep userdata API | C API:`deep.h` —— `luaG_newdeepuserdata` / `luaG_todeep` / `DeepPrelude` struct + `luaG_IdFunction` 函数指针 + `DEEP_VERSION`(fnv164) | C++ class:`deep.hpp` —— `DeepFactory`(NVI 虚函数)/ `DeepPrelude` class + `std::atomic<int> refcount` + `kDeepVersion`(xxh64) | **结构不同,但无影响** —— R18/D24 已弃用 deep userdata,Tensor 走 sol2 usertype + `__lanesclone` |
| `on_state_create` 钩子 | ✅ `src/state.h:19-20`(`initialize_on_state_create`/`call_on_state_create`)、`src/universe.h:60`(`on_state_create_func`) | ✅ `src/state.cpp:212`、`src/keeper.cpp:1126`、`src/lanes.cpp:840` | **两版都存在**(T-M0-06 顺带确认);4.x 明确注释 "users should provide an on_state_create function to setup custom loaders"(intercopycontext.cpp:1363) |
| 版本 | `lanes-3.17.2-0.rockspec`,release 2025-10-23 | `lanes-4.0.0-0.rockspec`,release 2026-02-27 | 仍按 dataloader.md §9.5(b) 建议:**锁 v3.17.2 开发**,4.x 观望 |

**对绑定的两个结论**:
1. `__lanesclone` 的实参是 `(dest_clone_ud, source_ud, size)` 三个,不是两个 ——
   sol2 里写这个 metamethod 时**第三个参数是 userdata 字节数**,用于 `*dest = *source`
   式 memcpy;Tensor 的 `shared_ptr` 拷贝即完成,size 参数可忽略但**必须接受**。
2. 两版 `__lanesclone` 都在**源 state** 里调用(注释 "call cloning function in source
   state"),所以绑定层写它时拿到的 self 是**源 userdata** —— 语义是"在源侧产出一个
   可跨 state 的拷贝",不是"在目标侧填充"。

---

## 10. MSVC ABI 一致性(T-M0-05)与 on_state_create(T-M0-06)

**T-M0-05 实测通过**:`tools/smoke/lanes_abi_check.cc` 链接 Lanes v3.17.2 的 10 个 C
obj + Lua 5.1(导入库),从 C++ TU 调用 `luaopen_lanes_core` 创建模块表,运行输出
`LANES ABI OK: v3.17.2 C core linked from C++ TU under MSVC`(exit 0)。

| 结论 | 说明 |
|---|---|
| **v3.17.2 是纯 C** | 10 个 `.c` / 0 个 `.cpp`,`CMakeLists.txt` `PROJECT(lanes C)`。§9.5(c) 的「虚函数 ABI」担忧**只适用于 4.x**(`DeepFactory`/`DeepPrelude` C++ class,NVI 虚函数),不适用于 v3.17 |
| **导出符号是 C 链接** | `lanes.h:8` `LANES_API __declspec(dllexport)`;导出 `luaopen_lanes_core` / `luaopen_lanes_embedded`;`deep.h:63-64` 导出 `luaG_newdeepuserdata` / `luaG_todeep`。从 C++ TU 链接无需 `extern "C"` 包裹(头文件本身是 C,`.cc` 里包了更没有错) |
| **⚠️ 实测陷阱:Lua 头必须统一** | 第一次链接失败(LNK4272):本机 `Lua\5.1` 是 **x86** 库,x64 目标冲突。第二次改用 vcpkg x64 `lua51.lib`,但**头文件混用**了 —— Lanes 用 PUC 5.1 头编、调用方用 LuaJIT 头编,**链接过、运行即崩**(负数退出码,`lua_State` 布局不同)。统一用 `vcpkg/installed/x64-windows/include/luajit`(LUA_VERSION_NUM=501)后通过。**绑定层三条纪律:Lanes 与绑定同编译器、同 CRT(/MT)、同一份 Lua 头** |
| **on_state_create(T-M0-06)** | 两版都有。v3.17.2:`state.h:19-20`(`initialize_on_state_create` / `call_on_state_create`)、`universe.h:60`(`on_state_create_func` 字段);4.x:`state.cpp:212` / `keeper.cpp:1126`。语义:每个新 lane 由 keeper 调一次,`lua_CFunction` 签名 —— **正好承载 13-lanes.md §4.4 的「每个 worker lane 重新 require "paddle"」**,Q-03 关闭,Detail 在 §9 对比表 |

**对绑定层的落地**:P13 绑定时把 Lanes 源码(C 文件)编译进我们的 `.dll`(`luaopen_lanes_embedded`
正是为嵌入场景设计的入口),与绑定同 toolset / 同 CRT / 同 Lua 头,即不存在 ABI 问题。
本机开发用 vcpkg 的 LuaJIT(Lua 5.1 ABI)作为链接目标,与 sol2 无冲突。

---

## 11. OOM 回调救回验证(T-M0-07)

**实测通过**:`tools/smoke/oom_smoke.cc` 输出 `OOM SMOKE PASS`(三用例全过)。
用 `MockOomAllocator`(前 N 次 `AllocateImpl` 抛 `BadAlloc`,之后 malloc 成功)
包进 `RetryAllocator`,验证 `RegisterOOMCallback` 的重试语义
(`retry_allocator.cc:60-82`):

| 用例 | 回调返回 | 实测 | 结论 |
|---|---|---|---|
| ① 能救回 | 1024(模拟 `lua_gc` 释放的字节) | 回调被调 **2 次**,第 3 次分配成功 | `>0` 即重试,直到成功 —— **Torch7 的 1.3 方案在 Paddle 侧成立** |
| ② 救不回 | 0(GC 没释放任何东西) | 回调被调 **1 次**,`BadAlloc` 逃逸 | `>0` 判据正确,不会无限重试 |
| ③ 无回调 | —(未注册) | 无回调重试,`BadAlloc` 逃逸 | 与源码 `g_oom_callback == nullptr` 短路一致 |

**两个必须先设的前提(绑定层启动时照抄)**:

1. **`FLAGS_offload_retry_times` 必须 > 0**,默认是 **-1**(`common/flags.cc:2363`)——
   `retry_allocator.cc:67` 的 `if (FLAGS_offload_retry_times <= 0 || g_oom_callback == nullptr)`
   直接短路,**回调一次都不会被调用**(第一版冒烟就是这么失败的)。
2. **CPU 分配链默认不包 `RetryAllocator`** —— `allocator_facade.cc:401-403` 只在
   `FLAGS_gpu_allocator_retry_time > 0` 时包 **CUDA/XPU** 路径。CPU 上要手动包,
   或依赖 GPU 路径自动包。这是实现细节,不影响 API 形状。

**`RetryAllocator` 外层还有一层 waiter 重试**(`retry_allocator.cc:88-115`):
OOM 回调放弃后,它会等 `retry_time_` 毫秒让别的线程释放内存,期间反复重试,
超时后再试最后一次。所以"无回调"不等于"立即失败",冒烟里 `retry_ms=0` 才能
立刻得到确定性行为 —— 绑定时默认值要留意(0 = 不等)。

---

## 12. LuaJIT GC64 确认(T-M0-08)

**实测通过**:`tools/ci/lua_gc64_check.lua` 在 vcpkg LuaJIT 2.1.1774896198
(x64/Windows)上输出 `lua-gc64 OK: x64 LuaJIT, 16-byte GC64 layout confirmed`。

三层判据(源码 + 运行时双重验证,不靠猜):

1. **源码**:`lj_arch.h:168/181/216-217/303/459` —— x64 目标默认
   `LJ_TARGET_GC64=1`(`#ifndef LUAJIT_DISABLE_GC64`,macOS 甚至强制);
   `:597-604` 把它投影为 `LJ_GC64`。
2. **运行时布局**:GC64 下 GCobj 头 16 字节,连续分配的 userdata 地址差
   必须为 16 的倍数 —— 脚本断言 8 个连续 `newproxy(false)` 的地址差 `%16==0`。
3. **架构锚定**:`ffi.sizeof("void*")==8` 确认 x64,让第 2 条判据有意义。

**结论**:本机与 CI 的 LuaJIT 一律是 GC64。绑定层(sol2 usertype + `__lanesclone`)
按 GC64 布局写,userdata 头 16 字节,不需要兼容非 GC64 构建 —— 若要兼容,
那是 `LUAJIT_DISABLE_GC64` 的编译期决定,与我们无关。

---

## 13. Lua 5.4 `__close` / `__gc` 语义(T-M0-09)

**实测通过**:`tools/ci/lua_close_check.lua` 在 Lua 5.4.7(从 lua.org 源码 MSVC 编译)与
Lua 5.1.5(choco)上输出 `lua-close OK`。已登记 l0.yml 的 lua-matrix job(5 版本全跑)。

### 13.1 结论先行

| Case | 语义 | 实测 |
|---|---|---|
| A:`<close>` 变量作用域退出 | **触发 `__close`**(任何对象,含纯表 —— to-be-closed 机制只看对象有没有 `__close`) | ✅ 5.4 上 `n==1` |
| B:普通 userdata 被 GC | **只调 `__gc`,不调 `__close`** —— 没有 GC 兜底调用 `__close` 的路径 | ✅ 同挂两元方法,GC 后 `closed_by=="gc"` |

**对绑定层(R6 起)的落地规则**:sol2 的 RAII usertype 必须**同时挂 `__close` 与 `__gc`**
两个元方法 —— `__close` 给 5.4 用户用 `<close> x = tensor` 做确定性释放,`__gc` 做
用户未声明 `<close>` 时的兜底释放;**两者必须幂等**(shared_ptr 自减天然幂等;若是
裸指针/手动标志,两个 handler 都要检查同一个"已释放"标志,否则双释放)。

### 13.2 方法(为什么改了两版)

第一版 Case B 用 `setmetatable({}, mt2)` 造"userdata" —— 那是**表**,不是 userdata;
而 5.4 的 GC 只对 userdata 调 `__close`…… 实测打脸:**GC 根本不调 `__close`**。
所以脚本改为拿真实 userdata(`io.tmpfile()` 句柄),用 `debug.setmetatable` 把它的
metatable 换成同时带 `__gc` 与 `__close` 的版本 —— 谁被调就是谁的路径,不靠猜:

```lua
local tmp = assert(io.tmpfile())
local closed_by = nil
debug.setmetatable(tmp, {
  __gc    = function() closed_by = "gc" end,
  __close = function() closed_by = "close" end,
})
tmp = nil
collectgarbage("collect"); collectgarbage("collect")
assert(closed_by == "gc")   -- GC 走 __gc,__close 不触发
```

### 13.3 为什么这对 paddle-lua 是"验证"而非"兴趣研究"

5.4 是目标版本之一(C5)。若绑定层只挂 `__close` 不挂 `__gc`:
用户在 5.4 上不用 `<close>` 声明时,`Tensor` 显存**永不释放**(GC 不管 `__close`)
—— 这是静默泄漏,和 M0 #7 的 OOM 回调正好接上(泄漏多了才触发 OOM 救回)。
脚本 Case B 的断言把"GC 不调 `__close`"钉死,防止未来实现想当然地只挂一个。

### 13.4 副产品

本机从 lua.org 用 MSVC 编出了 `lua54.exe`(解决 luac.obj 与 lua.obj 的 main 重复,
链接清单排除 luac.obj 即可);Temp 下 `lua-5.4.7/` 保留至 M0 收尾(5.2/5.3 同理
可随时编译,避免为每个版本装全局解释器)。

---

## 14. 堆追踪 vs Lanes 自定义分配器(T-M0-10)

**实测通过**:`tools/smoke/lanes_alloc_check.cc` 输出 `LANES ALLOC PASS`。
在 master state 上装计数 `lua_Alloc`(模拟堆追踪探针),加载**真实 `lanes.lua`**,
用 `on_state_create` 钩子在**新 lane state 自己身上**调 `lua_getallocf` 回答
"我的分配器是谁?",然后跑一个分配 ~1.2MB 的 lane(20000 × 64B 字符串)。

### 14.1 结论先行

| 问题 | 实测结果 | 结论 |
|---|---|---|
| master 上的自定义 allocator 会到 lane 吗? | **3 个 lane state 全部 `!= counting_alloc`**(inherited=0) | **LuaJIT 下不会** —— `state.c:253-255` `#if LUAJIT_FLAVOR()==64` 分支 lane 一律 `luaL_newstate()`,分配器只从 `lua_newstate(allocf)` 参数来,`lua_setallocf` 改的是**单个 state 自己** |
| PUC Lua 下呢? | (源码判定)`state.c:257-271`:`create_state` 非 LuaJIT 分支走 `lua_newstate(U->protected_allocator.definition.allocF, ...)`——**复用 master 的 allocator** | PUC 下我们的计数 allocator 会被 lane 线程调用 → **必须线程安全**;本机是 LuaJIT,此分支留给 CI 矩阵 |
| 堆追踪与 Lanes 冲突吗? | 不冲突,但有一个**非显而易见的陷阱**(§14.3) | 堆追踪应做**张量字节计数器**(`14-gc.md` §3.1),**与 Lua allocator 无关**;LuaJIT 上靠 allocator 钩子根本看不到 lane 分配 |

### 14.2 方法(为什么用 on_state_create 而不是猜)

`lua_getallocf` 只能从某个 `lua_State*` 上查。查 master 容易,查 **lane 自己**
需要在 lane state 上执行代码 —— 正好用 `on_state_create` 钩子(它在**新 state 上**
被调用,`state.c:283+` `call_on_state_create`):钩子里 `lua_getallocf(L)` 的 `L`
就是刚创建的 lane state。三个 lane(定时器 + 我们的 worker + 一个内部 state)
全部返回非 counting_alloc,直接证实 LuaJIT 分支。

### 14.3 实测陷阱:`lua_setallocf` 不能裸 `realloc/free` 替换 ⚠️

第一版冒烟在 `lua_setallocf(L, counting_alloc, NULL)` 里直接调 `realloc/free`,
**运行即 0xC0000374 堆损坏**。根因:vcpkg 的 LuaJIT 是 **`lua51.dll`** —— Lua state
的内存由 DLL 自己的 CRT 堆分配;我们 exe 用 `/MT` 静态 CRT,`free(ptr)` 把 DLL
堆的指针交给 exe 堆 = 堆损坏。

**正确姿势(照抄 Lanes `protected` 模式)**:替换前先 `lua_getallocf(L, &orig_ud)`
存下原分配器,新分配器**链式转发** `orig_alloc(orig_ud, ptr, osize, nsize)`,
自己只做计数。Lanes 的 `protected_lua_Alloc`(`tools.c:173-190`)就是这么干的
(mutex + 转发)—— 我们的计数探针与 Lanes 的 mutex 包装天然可叠加,互不干扰。

### 14.4 对绑定层的落地规则(P14 前置)

1. **堆追踪 = 张量字节计数器,彻底不碰 Lua allocator**(`14-gc.md` §3.1 的
   `pd_gc_account` 设计成立,且这是 LuaJIT 下**唯一**能看到 lane 分配压力的
   途径 —— allocator 钩子看不到)。
2. 计数要 `__thread`(每个 lane 线程各一份),触发 `lua_gc` 时用**当前线程的
   `lua_State*`**(由 `on_state_create` 钩子写入 thread_local)。
3. PUC 分支若未来要装 Lua 侧钩子:必须线程安全 + 链式转发,照 §14.3。

### 14.5 构建方式(dev-only,与 lanes_abi_check 相同纪律)

```bat
REM 10 个 Lanes .c + 本 TU 全部用 vcpkg LuaJIT 头(include\luajit)、/MT、同编译器
cl /c /O2 /MT src\*.c && cl /c /O2 /MT lanes_alloc_check.cc
cl lanes_alloc_check.obj *.obj E:\code\vcpkg\...\lua51.lib /MT
REM 运行:lanes_alloc_check.exe <lanes.lua 路径> [noalloc]
REM noalloc = 对照组(不装 allocator),验证 lane 机制本身正常
```

---

## 15. 纯 Lua pickle 读真实 `.pdparams`(T-M0-11)

**验收**:不依赖 Python,纯 Lua 读出 Paddle 保存的权重文件结构,
与 Python `pickle.load` 的结果逐项一致。

**结论:通过。** 实现落在**独立仓库 `E:\code\lua-pickle`**(rock/模块名
`pickle`;双向可逆 picker+unpickler,见 `lua-pickle/docs/design.md`),
不在本仓库。`tools/check.lua` 在本机 **Lua 5.1.5** 上读通真实样本
`m0-11-sample.pdparams`(326 B, protocol 4),并与 Python `pickle.load`
生成的权威基准 `m0-11-expected.lua` 逐字节对拍 PASS:
`linear.weight`(2x3 f4)、`linear.bias`(2 f4)、`StructuredToParameterName@@`
(dict[str,str])键、shape、dtype、raw bytes 全对。

### 15.1 对拍基准的教训(反幻觉规程)

**必须用 `pickle.load` 生成基准,不能用 `paddle.load`** —— 文件里存的
本来就是 numpy(上游 `io.py:184` `np.array(value.cpu())` 先把 Tensor 转
numpy 再 pickle),而 `paddle.load` 返回的是 `paddle.Tensor`,与纯 Lua 读
出的 numpy 结构不匹配,对拍会报 `TypeError: unsupported`。

### 15.2 numpy 还原语义(对照 numpy 2.4.6 实测)

| 对象 | pickle 形态 | Lua 侧还原 |
|---|---|---|
| `dtype` | `REDUCE(numpy.dtype, ('f4', False, True))` 后接 `BUILD (3, '<', None, None, None, -1, -1, 0)` | `dtype_new` 记 code,`dtype_setstate` 取 `state[2]` 拼 `str='<f4'` |
| `ndarray` | `REDUCE(numpy._core.multiarray._reconstruct, (numpy.ndarray, (0,), b'b'))` 造骨架,再 `BUILD (1, shape, dtype, is_fortran, rawdata)` | `reconstruct` 造骨架表,`ndarray_setstate` 解 5 元组填 shape/dtype/data |
| numpy 标量 | `REDUCE(numpy._core.multiarray.scalar, (dtype, rawbytes))` | `scalar` 记 dtype+bytes(R3 接 Insight7 0-d) |

兼容两条 reducer 路径:numpy 1.x `numpy.core.multiarray` 与 2.x
`numpy._core.multiarray` 都注册(`registry.lua`)。Paddle 自己的类
(`io_utils._reconstruct_dense_tensor_data` 等)由 paddle-lua 通过
`pickle.register("module.Name", fn)` 注入 —— pickle 不认识下游(设计 §2.1)。

### 15.3 踩坑:Lua 5.1 的 `#` 对洞表未定义 ⚠️

`load_tuple2/3` 初版用 `stack[#stack]=nil` 收缩栈 → 栈表末尾留 nil 洞 →
**后续任何 `#stack` 索引都错位**(REDUCE 把 func 取成了字符串)。
修法:`table.remove(stack)` 收缩(Lua 会重排,不留洞)。
同一原则适用于一切自建栈机:栈表必须时刻保持密集,否则 `#` 不可信。

### 15.4 已定事项

- **CI 框架(2026-10-23 人拍板)**:Lua 侧用 **busted**(spec/ 权威,五版本
  lua-matrix)。生成器已定全 Lua(R42),pytest 仅保留给 `tests/parity/` 对拍基准
  (CI 校验手段,非用户工具链),不进发布包。
  `tools/check.lua` 保留为手动 smoke 前端。
- **分层定位**:pickle 认识上游(pl.List / insight.Array,pcall 探测可选),
  不认识下游(不内置 paddle)—— 下游 register 注入(设计 §2.1)。
- 后续路线(roadmap R2–R5):writer(picker)→ numpy 类型接 Insight7 →
  registry 白名单/security → rockspec 打包 + CI。Insight7 的 0-d Lua
  入口(`ins.scalar` + `item()` 严格 ndim 检查)延后建设,不阻塞。

---

## 16. T-M0-14:`x[::-1]` 走 strided_slice 是 view 不是 copy(auto grad 可用)

### 16.1 任务与结论

对应 `process/tasks.md` 的 T-M0-14。问题是:Python `__getitem__` 在
`step != 1` 时选择 `strided_slice`(`variable_index.py:409-415`),eager 层这个
算子是 **view(共享 buffer)** 还是 **copy(新分配)**?以及 `egr::Backward` 能否
穿透它。

| 验收项 | 结果 |
|---|---|
| 是 view 还是 copy | ✅ **view** —— **旧记录"copy 语义"是错的**,那是 fallback 路径 |
| autograd 梯度是否流通 | ✅ `seed=[1,2,3,4]` → `x.grad=[4,3,2,1]` |
| 实测程序 | `tools/smoke/smoke_strided_slice.cc`,输出 `SMOKE PASS` |

### 16.2 证据链(全部 file:line)

**为什么是 view:**

1. `FLAGS_use_stride_kernel` **默认 true**(`paddle/phi/core/kernel_factory.cc:31`
   `PHI_DEFINE_EXPORTED_bool(use_stride_kernel, true, ...)`)—— eager 层默认启用
   STRIDED 系 kernel。
2. STRIDED kernel 的实现就是 view:
   ```cpp
   // paddle/phi/kernels/stride/strided_slice_kernel.cc:95-99
   out->set_meta(meta);                    // 只改 offset/dims/strides
   out->ResetHolder(input.Holder());       // 共享 input 的 buffer!
   out->ShareInplaceVersionCounterWith(input);
   ```
3. 实测三个证据:① `py == px + 12 字节`(offset 恰为 3 个 float32);② **写穿透**:
   `py[0] = 99` 后 `px[3] == 99`(同一块内存);③ 逻辑值 `y[i] == x[3-i]`。

**为什么旧记录错了:** `funcs/strided_slice.h:266-281` 的 `StridedSliceCompute`
(`Alloc<T>(out)` + Eigen `stridedSlice` 拷贝)确实是 copy,但那是
`FLAGS_use_stride_kernel=false` 时的 fallback;默认路径被 STRIDED kernel 顶替。

**autograd 可用:** `strided_slice_ad_func`(`dygraph_functions.h:1038`)与
`strided_slice_grad_ad_func`(`dygraph_grad_functions.h:656`)都存在;
`egr::Backward({y}, {seed})`(`backward.h:26-31`)实测正确。

### 16.3 对绑定层/API 设计的影响

- **`Tensor.__getitem__` 的切片是 view**—— 与 numpy 语义一致,用户在 Lua 侧
  `x["::-1"]` 之后修改 view 会改到原张量。API 文档必须写明,不能当 copy 迁移。
- 不需要手工区分 kernel:绑定走 `strided_slice_ad_func` 即可,view 语义自动继承。
- 迁移 Python `x[::-1]` 的 1-based 转换:参数起点/终点传 `axis-1` 后的负步长,
  具体映射仍由 `index_semantics.lua` 标注表负责。

### 16.4 冒烟构建备注(供重编)

- 链接基线:483 库清单从 `build.ninja` 的 `paddle_inference_c_shared`
  LINK_LIBRARIES 提取(见本报告 §2);**必须补 `/LIBPATH:E:\code\Paddle\build-nopy`**
  (`libonednn.lib`/`libcblas.lib` 在 build-nopy 根部)与 `/FORCE:MULTIPLE`。
- **缺 DEFINES 会编译失败**(C2236):`interface_support.h:27` 需要
  `-DPHI_INNER -DSTATIC_IR -DPADDLE_NO_PYTHON` 等 35 个宏。
- **运行时 DLL 路径**:`paddle\phi`、`paddle\common`、`third_party\install\onednn\bin`
  (mkldnn.dll)、`third_party\install\mklml\lib`(libiomp5md.dll)—— 缺则 0xC0000005。

---

## 17. T-M0-15:`OpProtoHolder` 链在 `WITH_PYTHON=OFF` 下可用

### 17.1 任务与结论

对应 `process/tasks.md` 的 T-M0-15。问题是:Python 侧 `OpProtoHolder`
(算子原型注册表,`paddle.base.framework`)这条链在无 Python 构建里是否可用——
即 Lua 侧能否拿到每个算子的 `OpProto`(参数/输入/输出 schema)。

| 验收项 | 结果 |
|---|---|
| C++ 底层链(`OpInfoMap`)在 build-nopy 有无定义 | ✅ **有**,`libop_info.lib`/`libop_registry.lib` 均导出 External 定义 |
| 是否依赖 pybind | ❌ 不依赖。Python 的 `OpProtoHolder` 只是 pybind 薄壳,核心是纯 C++ |
| Lua 侧等价实现路径 | ✅ 直接调 `OpInfoMap::Instance()` 遍历(见 §17.2) |

### 17.2 证据链(全部 file:line)

1. Python 侧 `OpProtoHolder` 类定义于 `python/paddle/base/framework.py:3202`;其
   `__init__` 调 `get_all_op_protos()`(`framework.py:3187`),该函数调
   `core.get_all_op_protos()`(pybind 绑定,`pybind.cc:2591`)。
2. pybind 绑定实现(`pybind.cc:2591-2603`)= 遍历 `OpInfoMap::Instance().map()`,
   对每个 `info.HasOpProtoAndChecker()` 的算子做 `Proto().SerializeToString()`。
   **整段逻辑不涉及 Python 运行时的特有能力,纯 C++ 可等价复制。**
3. C++ 侧类型:`OpInfo`(`op_info.h:42`,含 `proto_`/`checker_` 与
   `HasOpProtoAndChecker()` :64、`Proto()` :71)、`OpInfoMap`
   (`op_info.h:139`,singleton + `map()`/`Get()`/`Has()`);
   `OpInfoMap::Instance()` 定义于 `op_info.cc:22`。
4. **无 Python 环境同链已被上游自己使用**:`custom_operator.cc:1270-1275`
   (自定义算子注册)直接 `OpInfoMap::Instance().Insert(...)`——这是
   `WITH_PYTHON=OFF` 也编译的路径,证明该链无 pybind 依赖。
5. build-nopy 实测(dumpbin /symbols):
   - `libop_info.lib`:`?Instance@OpInfoMap@framework@paddle@@SAAEAV123@XZ`
     **SECT244 External 定义**
   - `libop_registry.lib`:`?Get@OpInfoMap@...` SECT614 External 定义

### 17.3 对绑定层/生成器的意义

- **生成器不需要运行期 OpProto**:schema 已在构建期从 yaml 拿到(C8/R42)。
- **需要运行期查询的场景**(自定义算子、`jit::TranslatedLayer` 反向翻译 op 时拿
  proto 校验,上游 Python 侧 `translated_layer.py:557-559` 就是这么用的):
  Lua 侧写一个等价小 C ABI 函数(遍历 `OpInfoMap` + SerializeToString),
  即可替代 Python 的 `OpProtoHolder`。这条链已被本项验证,无需补丁。

---

## 18. T-M0-17:`jit::Layer Load()` 在 `WITH_PYTHON=OFF` 下可链接、可加载

### 18.1 任务与结论

对应 `process/tasks.md` 的 T-M0-17。问题是:「Python 训练 + Lua 推理」的标准
互通路径上,`jit::Load`(读 `paddle.jit.save` 的 `.pdmodel`/`.pdiparams`/`.meta`)
在无 Python 构建里能不能用。

| 验收项 | 结果 |
|---|---|
| `jit::Load` 符号可链接 | ✅ `libfluid_jit.lib` 导出 `?Load@jit@paddle`(T-M0-01 已核);但需要 **2 个 shim**(见 §18.2 ①) |
| `jit::Load` 可加载产物 | ✅ 实测:`.pdmodel` ProgramDesc / `.pdiparams` 参数 / `.meta` 属性全部读入,Layer + 引擎构造成功 |
| InterpreterEngine 可用 | ✅ `FLAGS_jit_engine_type="New"` 时 `New Executor is Running.`(`program_interpreter.cc:255`) |
| forward 跑通 | ⚠️ **撞上游 fc 算子 + oneDNN bug**(Predictor 抛 `NotFoundError`;New 引擎 0xC0000005)—— 非无 Python 特有,记 Q-24 |

### 18.2 发现与证据链(全部 file:line)

**① 链接缺口:上游把两个 PIR 辅助函数放进了 pybind 目录。**
`libfluid_jit` 的 `PirFunctionInfo` 构造函数(`function_schema.cc:112/115`)无条件
调用 `paddle::GetFeedTargetNames/GetFetchTargetNames`,但它们定义在
`paddle/fluid/pybind/pir_utils.cc:20/36` —— **pybind 目录,`WITH_PYTHON=OFF`
不产出该库**。这不是我们的缺口,是上游无 Python 训练态从未链接过 libfluid_jit
的证据(上游测试 `test/cpp/jit/CMakeLists.txt:2` 仅 `NOT WIN32` 且 CI 带 Python)。
冒烟自带等价实现(纯 PIR,零 Python)验证链路,已写入 `smoke_jit_load.cc` 头部。

**② Load 本体成功。** 测试产物来自上游 CI(`test/cpp/jit/CMakeLists.txt:6`
`multi_program_load_with_property.tar.gz`),4 个文件
(`export.pdmodel` / `export.infer.pdmodel` / `export.pdiparams` / `export.meta`);
`Deserializer::operator()`(`serializer.cc:37-98`)读入 ProgramDesc + params + attrs
均无异常;`Layer` 构造与两种引擎(`InterpreterEngine` 于 `interpreter_engine.cc:72`
创建 `InterpreterCore`)都成功。

**③ forward 撞上游 fc+oneDNN 问题(记 Q-24)。** sample 网络含 fc 算子
(pdmodel 二进制扫到 `fc` 4 次);fc 的 oneDNN kernel 查输入名 `W` 失败
(`onednn_context.cc:347`,`operator < fc >`);`FLAGS_use_mkldnn=false` 后
`oneDNN v3.6.2` 仍被初始化(`onednn_context.cc:81`)。**判定:非无 Python 构建
特有**(fc+oneDNN 组合在新版 Paddle 上未被 jit 路径验证过),不阻塞 G1;P8 落地时
用新导出产物或打补丁(见 open-questions Q-24)。

### 18.3 对绑定层的意义

- `jit.load{path}` 的 C 侧基座已确认可用:**符号、加载、InterpreterEngine 全通**。
- 链接时记得带 `GetFeedTargetNames/GetFetchTargetNames` 的等价实现(或修上游
  目录归属)—— 这属于 P8 jit 模块的已知前置。
- forward 的 fc/oneDNN 兼容问题与 `paddle.jit.save` 的导出模型构成强相关:
  绑定层文档要提示「新导出产物优先」。

---

## 19. T-M0-18:手工构造 `AttributeMap` 喂 `run_program_ad_func`(WITH_PYTHON=OFF)

### 19.1 任务与结论

对应 `process/tasks.md` 的 T-M0-18。问题是:P15/P16 静态图路径(D16/D17)的
入口 `egr::to_static::run_program_ad_func` 输入是纯 C++ 类型(`vector<Tensor>`,
`vector<Scope*>`, `AttributeMap`)。Lua 侧没有 Python,能不能**手工构造**
`AttributeMap` 把这个函数喂起来?`AttributeMap` 里的值类型
(`Attribute = paddle::variant<...>`)能否不经 pybind 直接填?

| 验收项 | 结果 |
|---|---|
| `run_program_ad_func` 符号可链接 | ✅ libeager_to_static.lib 解析,冒烟链接成功(483 库基线) |
| 手工构造 `AttributeMap`(含 `pir::Value` 向量 / `shared_ptr<pir::Program>`)可编译 | ✅ 直接 `prog_attrs["program_id"] = ...` 赋值,无 pybind 参与 |
| 调用到达 C++ 逻辑、不泄漏 pybind 错误 | ✅ 实测:空 program 触发 C++ Traceback(预期错误),输出非 pybind/python 字样 |

**结论:链可用。** 手工拼出的 `AttributeMap` 一路走到 `run_program_impl.cc`
的 C++ 逻辑(在空 program 上按预期失败),证明 `WITH_PYTHON=OFF` 构建里
这个入口对纯 C++ 调用方完全敞开 —— P16 静态图运行时不需要 Python 参与构造
运行时参数。

### 19.2 证据链(全部 file:line)

- **入口签名**:`run_program_ad_func(const std::vector<egr::Tensor>& inputs,
  const std::vector<egr::Tensor>& params, std::vector<paddle::framework::Scope*>
  step_scope, const paddle::framework::AttributeMap& prog_attrs,
  const paddle::framework::AttributeMap& cuda_graph_attrs)`(`run_program_func.h:24-28`)。
- **`AttributeMap` / `Attribute` 定义**:
  `Attribute = variant<bool, int, int64_t, float, double, std::string, std::vector<bool>,
  std::vector<int>, std::vector<int64_t>, std::vector<float>, std::vector<double>,
  std::vector<std::string>, std::vector<pir::Value>, std::vector<BlockDesc*>,
  int8_t, int16_t, BlockDesc*, std::shared_ptr<pir::Program>>`(`type_defs.h:51`);
  `AttributeMap = unordered_map<std::string, Attribute>`(`type_defs.h:73`)。
- **`RunProgramImpl` 消费的键**(冒烟按此构造,冒烟源码头注释完整列出):
  `is_test`/`program_id`/`in_sot_mode`/`fx_names`/`fp_names`/`fo_names`/
  `no_need_buffers_names`/`fo_values`/`forward_program`/`backward_program` +
  `cuda_graph_state`/`cuda_graph_dispatch_key`(`run_program_impl.cc:449-490`)。
- **外层 scope 要求**:`step_scope` 非空且索引 0 是事件 scope
  (`run_program_impl.cc:461-466`)。
- **实测输出**:`SMOKE PASS: hand-built AttributeMap reached C++ logic
  (expected error for empty program: C++ Traceback ...)` —— exit 0。

### 19.3 对绑定层的意义

- P16 静态图模块的 C ABI 面可以很薄:透传字符串键 + 十几种原子值
  (variant 的成员类型都是可序列化/可构造的标量与容器),绑定层写一个
  `capi_set_prog_attr(prog, "is_test", bool)` 家族即可,无需 pybind。
- 与 T-M0-17 互补:那边证明**读**静态产物可用,这边证明**手搓运行时参数**
  可用 —— 静态图执行的两半都在无 Python 侧立住了。
- 已知边界:空 program 走到的是预期失败路径;真实 `pir::Program` 的构造
  与执行仍依赖 model 文件(见 T-M0-17),`forward_program` 非空时的完整链路
  留待 P16 落地时再验。