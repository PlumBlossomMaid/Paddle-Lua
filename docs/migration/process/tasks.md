# TASKS.md · 任务板

> 状态:`TODO` / `DOING` / `DONE` / `BLOCKED` / `BLOCKED-ENV`
> **只做前置条件已满足、且在当前闸门允许范围内的任务(见 `process/status.md` §1)。**
> 完成后更新状态与实际耗时。

---

## 闸门 G0 前 · 只允许这些

### T-M0-01 · WITH_PYTHON=OFF + ON_INFER=OFF 编译 🔴 生死判定

| | |
|---|---|
| 状态 | TODO |
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
| 状态 | TODO |
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

---

### T-DOC-01 · 文档一致性巡检

| | |
|---|---|
| 状态 | TODO |
| 前置 | 无(闸门 G0 前允许) |
| 预估 | 0.5 天 |

- [ ] 所有文档里的 `文件:行号` 出处在当前 Paddle 版本上仍然成立
- [ ] `CLAUDE.md` §2 决策台账与各专题文档结论一致
- [ ] `process/status.md` §4 与 `plan/overview.md` §8 的 M0 清单条目一一对应

---

## 闸门 G0 后 · M0 其余验证

> 每项产出写入 `process/m0-report.md`,并更新 `process/status.md` §4。

| ID | 任务 | 前置 | 预估 | 状态 |
|---|---|---|---|---|
| T-M0-03 | phi kernel / DeviceContext 多线程共享安全性 | T-M0-01 | 2d | TODO |
| T-M0-04 | Lanes v3.17.x deep API 与 4.x 差异核对 | — | 1d | TODO |
| T-M0-05 | Lanes 与绑定的 C++ ABI 一致性(MSVC) | T-M0-04 | 1d | TODO |
| T-M0-06 | Lanes `on_state_create` 钩子是否存在 | T-M0-04 | 0.5d | TODO |
| T-M0-07 | `RegisterOOMCallback` + `lua_gc` 救回 OOM | T-M0-01 | 1d | TODO |
| T-M0-08 | LuaJIT GC64 确认 | — | 0.5d | TODO |
| T-M0-09 | `__close` 对 userdata 在 5.4 上的行为 | — | 0.5d | TODO |
| T-M0-10 | 堆追踪 vs Lanes 自定义分配器 | T-M0-04 | 1d | TODO |
| T-M0-11 | 纯 Lua pickle 读真实 `.pdparams` | — | 1d | TODO |
| T-M0-14 | `x["::-1"]` 走 `strided_slice` 是否 view + autograd | T-M0-01 | 0.5d | TODO |
| T-M0-15 | `WITH_PYTHON=OFF` 下 `OpProtoHolder` 链是否可用 | T-M0-01 | 0.5d | TODO |
| T-M0-17 | `jit::Layer Load()` 加载 `jit.save` 产物并 forward | T-M0-01 | 1d | TODO |
| T-M0-18 | 手工构造 `AttributeMap` 喂 `run_program_ad_func` | T-M0-01 | 1d | TODO |
| T-M0-19 | `SaveTensor`/`LoadTensor` 格式稳定性 | T-M0-01 | 0.5d | TODO |
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
| T-M1-05 | C ABI 中间层骨架(生命周期/错误/dtype/place) | D2 | 5d |
| T-M1-06 | sol2 `usertype<Tensor>` + `__lanesclone` 槽位(D24) | D1/D8 | 4d |
| T-M1-07 | `slice.lua` + `lua_spec_to_cpp`(移植 Insight7) | D14 | 2d |
| T-M1-08 | 算子生成器原型(yaml → C ABI + sol2 + `_ops.lua`) | C8 | 10d |
| T-M1-09 | `index_semantics` 标注表 + 未标注即报错的 CI | §8.2 | 3d |
| T-M1-10 | autograd `backward`/`grad` 绑定 | — | 2d |
| T-M1-11 | `nn.Layer` 类系统 | — | 5d |
| T-M1-12 | Linear/Conv2D/BN/ReLU/Dropout/Sequential | — | 8d |
| T-M1-13 | `optimizer.SGD` / `Adam` | — | 5d |
| T-M1-14 | `io.Dataset` / `DataLoader`(单 worker) | — | 5d |
| T-M1-15 | `paddle.save/load` 走 `SaveTensor`/`LoadTensor` | D19 | 2d |
| T-M1-16 | `paddle.dataset` MNIST | `plan/overview.md` §2.1.1 | 3d |
| T-M1-17 | `jit::Layer` 绑定(可选,高性价比) | D18 | 2d |
| T-M1-18 | GC 九层机制 | D5/D6 | 3d |
| T-M1-19 | 1-based 护栏(label 范围检查) | C4 | 1d |
| T-M1-20 | **M1 验收:MNIST 训练收敛** | G2 | 3d |

---

## 完成记录

| ID | 完成日期 | 实际耗时 | 备注 |
|---|---|---|---|
| — | — | — | — |
