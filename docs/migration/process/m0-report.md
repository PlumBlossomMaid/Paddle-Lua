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