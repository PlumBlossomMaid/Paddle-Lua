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
- M0 其余验证项(#3–#21)按 `status.md` §4 顺序继续,下一项 = T-M0-03(phi kernel /
  DeviceContext 多线程安全)。
- 生成器必须把 `PD_DECLARE_KERNEL` 组自动写进绑定(§4.3),记入 `plan/modules/03-codegen.md`。