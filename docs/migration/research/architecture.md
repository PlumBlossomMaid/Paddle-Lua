# 调研补充 · 三个路线决策

> 针对反馈:(1) 尽量不增加 C++ 代码,pybind 绑啥我们绑啥 (2) 全部用 sol2 (3) pdparams 是 pickle 吗

---

## A. "尽量不增加 C++ 代码" —— 部分成立,需要校准

### A.1 一个必须澄清的事实

**pybind 层本身就是 273,909 行 C++。** 绑定层天然是 C++,这一点绕不开 ——
任何 Lua 绑定都必须存在一层等价的 C++。所以"不增加 C++ 代码"不可能字面成立。

但这个直觉的**强形式是完全正确的,而且我采纳**:

| 诉求 | 可行性 | 说明 |
|---|---|---|
| **不改 Paddle 上游一行代码** | ✅ 完全可行 | 见 A.2,我撤回之前的 DLPack 上游 PR 建议 |
| **不发明新抽象,1:1 镜像 pybind 语义** | ✅ 强烈推荐 | 这正是"API 对齐"能实现的前提 |
| **手写 C++ 尽量少,能生成就生成** | ✅ | 786 算子全部生成 |

### A.2 好消息:确实不需要动 Paddle 一行代码

我之前建议"给 Paddle 补一个 C++ 级 DLPack 导出并 PR 上游"。**这个建议撤回。**

复查 `paddle/phi/api/include/tensor.h`,`paddle::Tensor` 的 public API 已经够用:

```cpp
int64_t numel() const;                                  // :172
const common::DDim& dims() const;                       // :189
const common::DDim& strides() const;                    // :206
DataType dtype() const;                                 // :224
const Place& place() const;                             // :292
const void* data() const;  void* data();                // :388, :397
const std::shared_ptr<phi::TensorBase>& impl() const;   // :418
```

**DLPack / data_ptr / 零拷贝互操作全部可以在我们自己的仓库里实现,零上游改动。**

唯一仍需上游关注的是构建问题(`WITH_PYTHON=OFF + ON_INFER=OFF`),
而那大概率是 patch 而非新功能,也可以先在自己仓库打 patch。

### A.3 "pybind 绑啥我们绑啥" 的边界(重要的期望校准)

这个策略只覆盖 **op 层 + Tensor 层**,也就是主报告里的 M1。

**它覆盖不到主报告 1.4 节那 110k 行纯 Python** —— `nn.Layer` 体系、optimizer、
`paddle.tensor` 里大量由 Python 组合出来的 API,**这些在 pybind 里根本不存在**,
必须用 Lua 重写。而它们占总工作量的 60-70%。

所以准确表述是:**"pybind 绑啥我们绑啥" 解决 30-40%,剩下的是纯 Lua 移植工作。**

另外 pybind 绑了一些对 Lua 无意义的东西,应主动砍掉:
buffer protocol / `__array__` / `__reduce__` / weakref / Python capsule 版 DLPack / PyLayer。
所以实际是 **pybind 的一个子集**。

---

## B. "全部用 sol2" —— 比我原先的判断更合理,建议采纳(有一个修正)

### B.1 我修正之前的判断

主报告 3.3 节我倾向"先裸 Lua C API,sol2 可选",理由之一是 LuaJIT FFI 的性能优势。
**重新核算后,这个理由被我高估了:**

| 路径 | 单次调用开销(量级) |
|---|---|
| LuaJIT FFI(JIT 内联后) | ~几 ns |
| Lua C API(sol2 走这条) | ~50-200 ns |
| **CPU 上一个小 add kernel** | **~1-5 us** |
| **GPU kernel launch** | **~5-10 us** |

**绑定层开销比 kernel 本身小 1-2 个数量级,基本被淹没。**
只有极小张量 + 极高频标量运算(如 RL 的逐元素循环)才会显现。

而 sol2 的收益是实打实的:usertype/元表/重载分派/`shared_ptr` 生命周期/
**C++ 异常自动转 Lua error(直接解决主报告第 5 节的问题)** 全部内建。

**结论:"全部 sol2、单一前端" 对 MVP 是对的选择,我原来的偏好收回。**

### B.2 但有一个修正:仍然保留中间层

"全 sol2" 解决的是**前端统一**;它没有解决**编译成本**问题:

> sol2 是重模板库。若 sol2 的 TU 直接 include `phi/api/include/tensor.h`
> (会拉进 Eigen / gflags / glog / CUDA 头),而这个 TU 要为
> **6 个 Lua 版本各编译一次** —— CI 时间和内存都会很难看。

修正方案:**前端全 sol2(采纳你的方案),后端保留一个中间层。**

```
paddle_core.{h,cpp}     include Paddle 重头,编译【一次】→ DLL/静态库
                        输出一个轻量头(只依赖 std / 前向声明,或纯 C ABI)
        |
        v
paddle_sol2.cpp         include sol2 + paddle_core.h + lua.h
                        编译【6 次】,但每次都很轻(不碰 Eigen/CUDA)
        |
        +--> paddle_lua51.dll / lua52 / lua53 / lua54 / lua55 / luajit.dll
```

**这跟原方案的唯一区别是:上层不再分 FFI / sol2 两套,统一成 sol2 一套。**
中间层的存在几乎是免费的(它是生成的),却买到:
- 6 次重编都很轻
- 异常边界收口在一处
- **保留 LuaJIT FFI 作为后期可选的性能后端** —— 如果中间层用纯 C ABI 的话

### B.3 中间层选 "纯 C ABI" 还是 "轻量 C++ 头"?

| | 纯 C ABI | 轻量 C++ 头 |
|---|---|---|
| 写起来 | 麻烦(类型退化成句柄/指针) | 舒服(可传 `std::vector`/`std::string`) |
| 生成难度 | 都是生成的,差别不大 | 同 |
| 保留 LuaJIT FFI 可能性 | ✅ | ❌ 永久放弃 |
| ABI 稳定性 | ✅ 跨编译器 | ❌ MSVC/MinGW 混用会炸 |

**建议:纯 C ABI。** 额外成本很低(反正是生成的),但保留了退路 ——
万一将来发现某个高频场景 sol2 开销确实成了瓶颈,LuaJIT FFI 后端可以增量加上,不用重构。

### B.4 sol2 + Lua 5.5 仍是唯一的未知项

这一点不因"全 sol2"而消失,**反而变得更关键**(因为没有第二条路了)。
Lua 5.5(2025)相对 5.4 有语义与 C API 变更。

**行动项:M0 阶段就用一个 hello-world 验证 sol2 能否编过 Lua 5.5。**
如果不能,需要评估:等上游 / 自己 patch sol2 / 5.5 单独走裸 C API。
好在有 B.2 的中间层,5.5 单独降级走裸 C API 的代价是可接受的。

---

## C. pdparams 确实是 pickle —— 而且情况比预想的好得多

### C.1 确认:是纯 pickle,默认 protocol 4

```python
# python/paddle/framework/io.py:790
def save(obj, path, protocol: Literal[2, 3, 4] = 4, **configs) -> None
# io.py:1060  最终就是一句
pickle.dump(saved_obj, f, protocol=protocol)
```

`.pdparams` 就是一个**裸 pickle 文件**,没有自定义容器格式、没有 header、没有 zip
(与 PyTorch 的 `.pt` 不同,后者是 zip 套 pickle)。

### C.2 关键发现:pickle 里没有任何 Paddle 自定义类

这是最重要的一点。`_build_saved_state_dict` (io.py:166) **在 pickle 之前**
就把每个 Tensor 转成了 numpy:

```python
save_dict[key] = np.array(value.cpu())      # io.py:184
...
save_dict["StructuredToParameterName@@"] = name_table   # io.py:188
```

所以 `.pdparams` 的实际结构就是:

```
dict[str, numpy.ndarray]
  + 一个额外的 key "StructuredToParameterName@@" -> dict[str, str]
```

**Lua unpickler 完全不需要理解 Paddle 的任何类型。**

### C.3 现成的规格说明书

`python/paddle/framework/restricted_unpickler.py` 的 `_ALLOWED_CLASSES`
**就是我们要实现的完整清单**,一行不多一行不少:

| 模块 | 需要支持的符号 |
|---|---|
| `builtins` | dict / list / tuple / set / frozenset / bytes / str / int / float / bool / complex / slice / range |
| `collections` | OrderedDict / defaultdict |
| `numpy` | ndarray / dtype / float32,64,16 / int8,16,32,64 / uint8 / bool_ / complex64,128 / bfloat16 |
| `numpy.core.multiarray` | `_reconstruct` / `scalar` |
| `numpy._core.multiarray` | 同上(**numpy 2.x 的新路径**) |
| `_codecs` | `encode` |
| `copyreg` | `_reconstructor` |
| `paddle.framework.io_utils` | `_reconstruct_dense_tensor_data` |
| `paddle.base.libpaddle` | `GeneratorState`(RNG 状态,可选) |

### C.4 已知的坑(都是真的,白名单能反推出来)

1. **numpy 1.x 写 `numpy.core.multiarray`,numpy 2.x 写 `numpy._core.multiarray`**
   —— 白名单里两个都列了,证明这是真实存在的兼容问题。Lua 库必须两个都认。

2. **`_codecs.encode` 在白名单里,说明 protocol 2 的 ndarray rawdata 是
   `_codecs.encode(str, 'latin1')` 而不是裸 bytes。** protocol 3+ 才是 bytes。
   这是最容易踩的坑。

3. **ndarray 的 `__setstate__` 是 5 元组** `(version, shape, dtype, is_fortran, rawdata)`,
   配合 `numpy.core.multiarray._reconstruct(ndarray, (0,), b'b')` 使用。

4. **protocol 2/3 会切分大数组。** `io_utils.py:255 _unpack_saved_dict`:
   当 `1 < protocol < 4` 且元素数 > `(2^30-1)/itemsize` 时,
   拆成 `key@@.0 / key@@.1 / ...` 并写入 `UnpackBigParamInfor@@`。
   **默认 protocol=4 不切分**,但读旧文件必须处理。

5. **非 state_dict 路径**(`_pickle_save`, io.py:437)里 Tensor 归约为
   `(tuple, ((name, data),))` —— 需要支持 `builtins.tuple` 作为 reduce callable。

6. **protocol 4 的新 opcode**:FRAME(0x95)、SHORT_BINUNICODE、MEMOIZE、
   STACK_GLOBAL、BINBYTES8 等,必须实现。

7. **bfloat16**:numpy 无原生 bfloat16 dtype,需查 Paddle 实际写出的是什么
   (大概率是 uint16 + 元信息,待验证)。

### C.5 工作量与战略价值

**约 600-1000 行 Lua。** 只需 unpickle,不需 pickle,opcode 集不大。

战略价值比工作量高得多:

1. **用户零改动**。safetensors 路线要求用户重新保存模型;
   pickle 路线让用户拿现成的 `.pdparams` 直接跑。**用户体验差距是决定性的。**
2. **顺手支持 PyTorch**。`.pt/.pth` = zip + pickle,同一个 unpickler 加个
   zip reader(Lua 已有现成库)就能读 —— 对项目吸引力是很大的加分。
3. **可独立发布**。`lua-pickle` 本身对 Lua 社区就有价值,不依赖 Paddle,
   可以作为整个项目的第一个可交付物,**而且完全不需要编译 Paddle**。
4. **纯 Lua 实现,无 C 依赖** → 自动支持 5.1-5.5 + LuaJIT 全部引擎,
   不受 B 节任何架构决策影响,可以立刻并行开工。

### C.6 修正主报告 7.1 节

主报告写"pickle 那条路死了,走 safetensors"。**这个结论作废。**

正确的结论是:

| 路线 | 定位 |
|---|---|
| **pickle reader(纯 Lua)** | **主路线**,读用户现有的 `.pdparams`,零改动 |
| safetensors(纯 Lua) | 辅助,~200 行,新模型/大模型场景更快更安全 |
| `jit::Layer`(C++) | 推理部署路线,读 `paddle.jit.save` 产物 |

三条并存,pickle 优先级最高。

---

## D. 修正后的 M0 任务清单

原 M0 之外,新增两项可**并行、且不依赖 Paddle 构建**的任务:

| # | 任务 | 依赖 | 备注 |
|---|---|---|---|
| 1 | 验证 `WITH_PYTHON=OFF + ON_INFER=OFF` 能否构建 | Paddle 源码 | 仍是头号风险 |
| 2 | 最小 C ABI + sol2 跑通 fwd/bwd | 任务 1 | |
| 3 | **验证 sol2 能否编过 Lua 5.5** | 无 | **唯一的架构级未知项** |
| 4 | **纯 Lua pickle reader,读通一个真实 .pdparams** | 无 | 可立刻开工,可独立发布 |

任务 3 和 4 **现在就能做,不需要编译 Paddle**,建议先做 —— 尤其任务 4,
它能最快产出一个真实可用的东西。
