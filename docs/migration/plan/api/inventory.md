# API 迁移清单 · 顶层 `paddle` 命名空间(第一版,机器分桶)

> 来源:`python/paddle/__init__.py` 的 `__all__`(共 551 项)× `paddle/phi/ops/yaml/`。
> 判据(AGENTS.md §4 反幻觉,**全部来自源码,不是印象**):
> - `ops.yaml` = 生成器处理(DIRECT);`legacy/static_ops.yaml` = 需个别评估(LEGACY);
> - 两者都没有的纯 Python wrapper → 看是否依赖 Python 运行时,是则放放,否则 Lua 侧重设计;
> - 人工跳过表(SPECIAL):`Model`/`summary`/`flops`/`DataParallel`(D10→ocean)、
>   `distributed`/`fleet`(无环境)、`compat`/`MmapStorage`/`LazyGuard`/`inference_mode`。
> 
> 本清单只覆盖**顶层命名空间**。`Tensor` 方法、`paddle.nn`/`optimizer`/`io` 等子模块
> 各有一份 api 文档(`plan/api/<模块>.md`),按那套骨架逐个填。这里的三态是迁移顺序的路书。

## 0. 桶统计

| 桶 | 数量 | 含义 |
|---|---|---|
| **直接绑定** | 196 | 名字在 `ops.yaml` → P3 算子生成器(C8)直接产出 |
| **LEGACY** | 17 | 只在 `legacy/static_ops.yaml` → 需逐个评估是否移植(P3 时决定) |
| **Lua 侧重设计 / 放放** | 328 | 纯 Python wrapper → 按 Lua 特性重写;依赖 Python 运行时的先放放 |
| **跳过** | 10 | D10 / 无环境 / 兼容层,明确不做 |

## 1. DIRECT(直接绑定,生成器领地)

| 名字 | 出处 |
|---|---|
| `abs` | ops.yaml |
| `acos` | ops.yaml |
| `acosh` | ops.yaml |
| `addmm` | ops.yaml |
| `all` | ops.yaml |
| `allclose` | ops.yaml |
| `amax` | ops.yaml |
| `amin` | ops.yaml |
| `aminmax` | ops.yaml |
| `angle` | ops.yaml |
| `any` | ops.yaml |
| `argmax` | ops.yaml |
| `argmin` | ops.yaml |
| `argsort` | ops.yaml |
| `as_complex` | ops.yaml |
| `as_real` | ops.yaml |
| `as_strided` | ops.yaml |
| `asin` | ops.yaml |
| `asinh` | ops.yaml |
| `atan` | ops.yaml |
| `atan2` | ops.yaml |
| `atanh` | ops.yaml |
| `baddbmm` | ops.yaml |
| `bernoulli` | ops.yaml |
| `bincount` | ops.yaml |
| `binomial` | ops.yaml |
| `bitwise_and` | ops.yaml |
| `bitwise_left_shift` | ops.yaml |
| `bitwise_not` | ops.yaml |
| `bitwise_or` | ops.yaml |
| `bitwise_right_shift` | ops.yaml |
| `bitwise_xor` | ops.yaml |
| `bmm` | ops.yaml |
| `broadcast_tensors` | ops.yaml |
| `cast` | ops.yaml |
| `ceil` | ops.yaml |
| `clip` | ops.yaml |
| `complex` | ops.yaml |
| `concat` | ops.yaml |
| `conj` | ops.yaml |
| `conv2d` | ops.yaml |
| `conv3d` | ops.yaml |
| `copysign` | ops.yaml |
| `cos` | ops.yaml |
| `cosh` | ops.yaml |
| `crop` | ops.yaml |
| `cross` | ops.yaml |
| `cummax` | ops.yaml |
| `cummin` | ops.yaml |
| `cumprod` | ops.yaml |
| `cumsum` | ops.yaml |
| `det` | ops.yaml |
| `diag` | ops.yaml |
| `diag_embed` | ops.yaml |
| `diagonal` | ops.yaml |
| `digamma` | ops.yaml |
| `dist` | ops.yaml |
| `dot` | ops.yaml |
| `empty` | ops.yaml |
| `empty_like` | ops.yaml |
| `equal_all` | ops.yaml |
| `erf` | ops.yaml |
| `erfinv` | ops.yaml |
| `exp` | ops.yaml |
| `expand` | ops.yaml |
| `expand_as` | ops.yaml |
| `expm1` | ops.yaml |
| `eye` | ops.yaml |
| `flatten` | ops.yaml |
| `flip` | ops.yaml |
| `floor` | ops.yaml |
| `fmax` | ops.yaml |
| `fmin` | ops.yaml |
| `full` | ops.yaml |
| `full_like` | ops.yaml |
| `gammaincc` | ops.yaml |
| `gammaln` | ops.yaml |
| `gather` | ops.yaml |
| `gather_nd` | ops.yaml |
| `group_norm` | ops.yaml |
| `heaviside` | ops.yaml |
| `histogram` | ops.yaml |
| `i0` | ops.yaml |
| `i0e` | ops.yaml |
| `i1` | ops.yaml |
| `i1e` | ops.yaml |
| `imag` | ops.yaml |
| `increment` | ops.yaml |
| `index_add` | ops.yaml |
| `index_fill` | ops.yaml |
| `index_put` | ops.yaml |
| `index_sample` | ops.yaml |
| `index_select` | ops.yaml |
| `inverse` | ops.yaml |
| `is_empty` | ops.yaml |
| `isclose` | ops.yaml |
| `isfinite` | ops.yaml |
| `isinf` | ops.yaml |
| `isnan` | ops.yaml |
| `kron` | ops.yaml |
| `kthvalue` | ops.yaml |
| `layer_norm` | ops.yaml |
| `lerp` | ops.yaml |
| `lgamma` | ops.yaml |
| `linspace` | ops.yaml |
| `log` | ops.yaml |
| `log10` | ops.yaml |
| `log1p` | ops.yaml |
| `log2` | ops.yaml |
| `log_softmax` | ops.yaml |
| `logcumsumexp` | ops.yaml |
| `logical_and` | ops.yaml |
| `logical_not` | ops.yaml |
| `logical_or` | ops.yaml |
| `logical_xor` | ops.yaml |
| `logit` | ops.yaml |
| `logspace` | ops.yaml |
| `logsumexp` | ops.yaml |
| `masked_fill` | ops.yaml |
| `masked_scatter` | ops.yaml |
| `masked_select` | ops.yaml |
| `max` | ops.yaml |
| `mean` | ops.yaml |
| `median` | ops.yaml |
| `meshgrid` | ops.yaml |
| `mode` | ops.yaml |
| `multinomial` | ops.yaml |
| `multiplex` | ops.yaml |
| `mv` | ops.yaml |
| `nanmedian` | ops.yaml |
| `nansum` | ops.yaml |
| `nextafter` | ops.yaml |
| `nonzero` | ops.yaml |
| `numel` | ops.yaml |
| `ones` | ops.yaml |
| `ones_like` | ops.yaml |
| `poisson` | ops.yaml |
| `polygamma` | ops.yaml |
| `pow` | ops.yaml |
| `prod` | ops.yaml |
| `put_along_axis` | ops.yaml |
| `qr` | ops.yaml |
| `randint` | ops.yaml |
| `randperm` | ops.yaml |
| `real` | ops.yaml |
| `reciprocal` | ops.yaml |
| `reduce_as` | ops.yaml |
| `relu` | ops.yaml |
| `renorm` | ops.yaml |
| `repeat_interleave` | ops.yaml |
| `reshape` | ops.yaml |
| `reverse` | ops.yaml |
| `roll` | ops.yaml |
| `round` | ops.yaml |
| `rsqrt` | ops.yaml |
| `scale` | ops.yaml |
| `scatter` | ops.yaml |
| `scatter_nd_add` | ops.yaml |
| `searchsorted` | ops.yaml |
| `shape` | ops.yaml |
| `shard_index` | ops.yaml |
| `sign` | ops.yaml |
| `sin` | ops.yaml |
| `sinh` | ops.yaml |
| `slice` | ops.yaml |
| `split` | ops.yaml |
| `sqrt` | ops.yaml |
| `square` | ops.yaml |
| `squeeze` | ops.yaml |
| `stack` | ops.yaml |
| `standard_gamma` | ops.yaml |
| `stanh` | ops.yaml |
| `std` | ops.yaml |
| `strided_slice` | ops.yaml |
| `sum` | ops.yaml |
| `take_along_axis` | ops.yaml |
| `tan` | ops.yaml |
| `tanh` | ops.yaml |
| `topk` | ops.yaml |
| `trace` | ops.yaml |
| `transpose` | ops.yaml |
| `tril` | ops.yaml |
| `tril_indices` | ops.yaml |
| `triu` | ops.yaml |
| `triu_indices` | ops.yaml |
| `trunc` | ops.yaml |
| `unbind` | ops.yaml |
| `unfold` | ops.yaml |
| `uniform` | ops.yaml |
| `unique_consecutive` | ops.yaml |
| `unsqueeze` | ops.yaml |
| `unstack` | ops.yaml |
| `var` | ops.yaml |
| `where` | ops.yaml |
| `zeros` | ops.yaml |
| `zeros_like` | ops.yaml |

## 2. LEGACY(需逐个评估)

| 名字 | 出处 |
|---|---|
| `arange` | legacy/static_ops.yaml |
| `assign` | legacy/static_ops.yaml |
| `einsum` | legacy/static_ops.yaml |
| `equal` | legacy/static_ops.yaml |
| `floor_divide` | legacy/static_ops.yaml |
| `greater_equal` | legacy/static_ops.yaml |
| `greater_than` | legacy/static_ops.yaml |
| `less_equal` | legacy/static_ops.yaml |
| `less_than` | legacy/static_ops.yaml |
| `matmul` | legacy/static_ops.yaml |
| `maximum` | legacy/static_ops.yaml |
| `min` | legacy/static_ops.yaml |
| `minimum` | legacy/static_ops.yaml |
| `not_equal` | legacy/static_ops.yaml |
| `remainder` | legacy/static_ops.yaml |
| `softmax` | legacy/static_ops.yaml |
| `unique` | legacy/static_ops.yaml |

## 3. PYTHON(Lua 侧重设计 / 先放放)

> 这桶最大,是审阅重点。迁移时逐个看实现:`python/paddle/**/*.py` 里它是什么。
> 三类去向:① 底层是几个算子组合 → Lua 侧用 `_ops` 组合;② 纯 Python 逻辑
> (索引解析/形状推断/装饰器) → 按 Lua 特性重写(**先做出来再谈优化**,D35 兼容糖不移植);
> ③ 依赖 Python 运行时(pybind 壳、`paddle.static` 程序、DLPack capsule)→ 先放放。

### 3.1 已抽查核心成员的三态去向(2026-10-23,修订)

> 抽查判据:`既然底层是 X,Lua 侧就抄 X`。出处均为上游源码,不是印象。
>
> **★ 命名铁律(用户 2026-10-23 拍板,两段式)**:
> ① **一般 API 忠实迁移**:Python 里叫什么,Lua 里就叫什么——`paddle.to_tensor` → Lua `paddle.to_tensor`,
> `paddle.zeros` → `paddle.zeros`。**不改名、不合并、不发明新名字**(如 `paddle.tensor` 收敛之类一律不做),api 名字不一样会让用户困惑。
> ② **numpy 特例换 Insight**:**凡参数类型就是 numpy 对象的,入口名换成 Insight 对应物** ——
> 唯一典型 `paddle.from_numpy` → Lua **`paddle.from_insight`**(numpy 在 Lua 生态的位置被 Insight7 顶替,D26;
> 参数从 `numpy.ndarray` 换成 `insight.Array`)。这是"语义槽位对齐",不是发明新 API。
> 每个导出函数/类方法的签名都走 `argrule` 规则表(D31);Python 侧为兼容 PyTorch 加的
> 装饰器(别名、`size_args`、`variadic` 等)不移植(D35/R33)。

| 名字 | 上游实现体 | 去向 |
|---|---|---|
| `paddle.tensor` (`creation.py:1009`) | 动态模式走 `_to_tensor_non_static`(`:1121`),静态模式走 `_to_tensor_static`(`:1125`) | **Lua 同名忠实迁移** `paddle.tensor{data, dtype, place, requires_grad}`;签名走 argrule;数据→Tensor 的构造逻辑在 Lua 侧实现(数组→`full` 或 DLPack,标量→0-d) |
| `paddle.to_tensor` (`creation.py:1201`) | 实现体就一行:`return tensor(data, dtype=dtype, device=place, requires_grad=not stop_gradient)`(`:1205`) | **Lua 同名独立入口** `paddle.to_tensor{data, dtype, place, stop_gradient}`——名字照搬,不合并 |
| `paddle.from_numpy` (`creation.py:1208`) | 实现体:`return tensor(ndarray)`(`:1230`) | **numpy 特例换 Insight** → Lua `paddle.from_insight{array}`——参数从 `numpy.ndarray` 换 `insight.Array`,底层走 DLPack 桥(Insight7/lua-pickle 已就绪) |
| `paddle.zeros`/`ones`/`full`/`full_like` | 装饰 `@size_args_decorator`(`creation.py:1807/1621/1378/1380`),内核走 `_C_ops.zeros/ones/full/full_like` | **直接绑定**(已在 DIRECT 桶);`size_args` 装饰器不移植(D33),Lua 侧 `zeros{2,3}` 直接生成绑定 |
| `paddle.add` (`math.py:723`) | 动态模式一行 `_C_ops.add(x, scaled_y, out=out)`(`:802`);静态模式走 `LayerHelper` + `append_op` | **直接绑定**;`alpha` 参数 → Lua 侧 `add{x,y,alpha}` 生成绑定处理,`out` 暂不做(Q-19) |
| `paddle.arange` (`creation.py:2206`) | 有真实实现体:`end=None` 折叠 `start`/`end`(`:2265-2266`)+ dtype 推断(`:2268-2276`)+ 动态模式 `_C_ops.arange(start, end, step, dtype, device, out=out)`(`:2321-2328`) | **LEGACY 桶**(底层 op 在 `legacy/static_ops.yaml:55`);dtype 推断与 `end=None` 折叠是纯 Python 逻辑 → Lua 侧重写(用 Lua 型检查替代 `isinstance`),边界仍调生成绑定;`requires_grad` → `stop_gradient` 取反,(C12) |
| `paddle.matmul` (`linalg.py:24-34`) | **纯 re-export**:`from paddle._C_ops import (... matmul ...)`,本体就是绑定算子 | **LEGACY 桶**(`legacy/static_ops.yaml:669`);Lua 侧 `paddle.matmul{x, y, transpose_x, transpose_y}` 生成绑定,签名走 argrule |
| `paddle.sum` (`math.py:116`) | **纯 re-export**:`from paddle._C_ops import (... sum ...)`,本体就是绑定算子 | **直接绑定**(`sum` 在 `ops.yaml`,见 DIRECT 桶 §1);Lua 侧 `paddle.sum{x, axis, dtype, keepdim}` 生成绑定(`axis` 标 1-based,§2.2) |
| `paddle.add_n` (`math.py:1994`) | 动态模式一行 `_C_ops.add_n(inputs)`(`:2052`);静态模式走 `LayerHelper` + `append_op`,对列表逐个 check | **直接绑定 + 一个 Lua 适配**:底层 `add_n` 算子对接(不在顶层 yaml,核实 `_C_ops.add_n` 为 pybind 手工绑定,P3 时评估);`inputs` 参数用 `TensorList` 类型(`api/README.md` §2.1 首个参数为列表规则),Lua 侧 `paddle.add_n{ {a,b,c} }` |
| `paddle.concat` (`manipulation.py:1481`) | 动态模式 `_C_ops.concat(input, axis, out=out)`(`:1556-1558`);静态模式先 AMP 检查再逐元素 check | **直接绑定**(DIRECT 桶);`axis` 标 1-based;⚠️ 别名 `tensors`/`dim` 在 docstring 里明确写着(`:1494-1496`),D35 不移植——Lua 侧只有 `x`/`axis` |
| `paddle.stack` (`manipulation.py:2170`) | 动态模式 `_C_ops.stack(x, axis, out=out)`(`:2327-2328`);静态模式走 DENSE_TENSOR_ARRAY 特判 | **直接绑定**(DIRECT 桶);`axis` 标 1-based;同名别名 `tensors`/`dim` 不移植(D35) |
| `paddle.transpose` (`manipulation.py:636`) | 动态模式 `_C_ops.transpose(x, perm)`(`:697`);静态模式 check 后 append_op | **直接绑定**(DIRECT 桶);参数名 `perm`(Paddle 原生,非 PyTorch `dims`),Lua 侧 `paddle.transpose{x, perm}` |
| `paddle.reshape` (`manipulation.py:5513/5517/5521`) | **三重签名**:`reshape(x, shape)` + `reshape(input, *shape)`(变长)+ 实体 `@reshape_decorator`;动态模式 `_C_ops.reshape(x, new_shape)`(`:5648-5649`),`new_shape == x.shape` 时直接返回 x | **直接绑定 + 一个 Lua 适配**:变长签名 `reshape(2,3)` 是 `decorator_utils` 兼容糖(D35 不移植)——Lua 侧只有 `reshape{x, shape}`;`shape` 用 `IntList`(`api/README.md` §2.1.2);返回值与 Python 一样**共享内存**(docstring `:5524-5527` 明说 "output Tensor will share data with origin Tensor") |
| `paddle.squeeze` (`manipulation.py:3393`) | 动态模式 `_C_ops.squeeze(input, axes)`(`:3491/3502`,两个分支:axis 为 list 与 Tensor 各一);`axis=None` 时 squeeze 全部 size-1 维 | **直接绑定**(DIRECT 桶);`axis` 标 1-based(可传 list);**共享内存**(docstring `:3396-3398` "output Tensor will share data with origin Tensor")——与 reshape 同族,view 语义 |
| `paddle.unsqueeze` (`manipulation.py:4038`) | 动态模式 `_C_ops.unsqueeze(input, axes)`(`:4116/4127`);axis 可为 int/list/Tensor,负数 `axis+R+1` | **直接绑定**(DIRECT 桶);`axis` 标 1-based;**共享内存**(docstring `:4043-4045`);⚠️ 别名 `input`/`dim`(`:4047-4049`)D35 不移植,Lua 侧只有 `x`/`axis` |
| `paddle.split` (`manipulation.py:2785`) | 动态模式 `_C_ops.split(input, num_or_sections, dim)`(`:2873/2906`);返回 `list[Tensor]` | **直接绑定**(DIRECT 桶);返回 `pl.List`(用户要遍历的集合,`api/README.md` §2.3);`axis` 标 1-based;⚠️ 参数名 `num_or_sections`(Paddle 原生),无 PyTorch 对应 |
| `paddle.flatten` (`manipulation.py:1945`) | 动态模式 `_C_ops.flatten(x, start_axis, stop_axis)`(`:2059`);start/stop 轴闭区间拍平 | **直接绑定**(DIRECT 桶);`start_axis`/`stop_axis` **都是索引语义,标 1-based**(§2.2);**共享内存**(docstring `:1949-1950`);⚠️ 别名 `input`/`start_dim`/`end_dim`(`:1966-1968`)D35 不移植 |
| `paddle.max` (`math.py:2726`) | 动态模式一行 `_C_ops.max(x, axis, keepdim)`(`:2851`);`@forbid_keywords` 禁 input/dim/other | **直接绑定**(DIRECT 桶);`axis` 标 1-based;⚠️ `@forbid_keywords(illegal_keys={"input","dim","other"})`(`:2721-2724`)就是 D35 要挡的兼容壳,不移植——Lua 侧只有 `x`/`axis`/`keepdim`;语义差异 `max` 传播梯度到所有相等极大值(amax 才均匀分摊),绑定层 doc 注明 |
| `paddle.min` (`math.py:2893`) | 动态模式一行 `_C_ops.min(x, axis, keepdim)`;结构同 max(镜像) | **直接绑定**(LEGACY 桶,`legacy/static_ops.yaml`);`axis` 标 1-based;别名 `input`/`dim` 不移植(D35);doc 注明与 `amin` 的梯度差异 |
| `paddle.mean` (`stat.py:53`) | `@param_two_alias(["x","input"],["axis","dim"])`;`dtype` 非 None 时先 `cast(x, dtype)`(`:143-147`)再 `_C_ops.mean(x, axis, keepdim, out=out)`(`:149`) | **直接绑定 + 一个 Lua 适配**:`dtype` 参数 = 先 cast 再 mean(Lua 侧 `_ops.cast` 组合);`axis` 标 1-based;别名 `input`/`dim` 不移植(D35);`out` 暂不做(Q-19) |
| `paddle.var` (`stat.py:186`) | **两路实现**:CUDA+动态走 `_C_ops.var(x, axis, keepdim, unbiased, correction, out=out)`(`:215-219`);CPU 是**纯 Python 组合**:`u=mean(x,axis,True)` + `sum(pow(x-u,2))` + 除以 `n-correction`(`:222-280`,含 0 形状 `_replace_nan` 与 0-d 特判) | **直接绑定(CUDA 路)+ Lua 侧重写(CPU 路)**:`unbiased`/`correction` 互斥校验(`:198-201`)与 CPU 组合逻辑(mean+pow+sum+max 除)照抄为 Lua 组合;`axis` 标 1-based;别名 `input`/`dim` 不移植 |
| `paddle.std` (`stat.py:304/317/328`) | `@overload`×2 + `@use_first_signature`(`:328`);CUDA 走 `_C_ops.std`,CPU 组合(见 var) | **直接绑定(CUDA 路)+ Lua 侧重写(CPU 路)**:多重签名是 `@overload`+`@use_first_signature` 装饰器(D35 不移植)——Lua 侧只有规范签名 `std{x, axis, unbiased, keepdim, correction, out}`;CPU = `var` 组合 + `sqrt`;`axis` 标 1-based |
| `paddle.norm` (`linalg.py:1091`) | **纯 Python 分发器**:按 `p`/`axis` 形态分发到 `vector_norm`/`matrix_norm`(`:1238-1271`);`p` 支持 fro/nuc/0/±1/±2/±inf/任意数 | **Lua 侧重写**:分发表(矩阵/向量范数判定、`axis` 单元素→向量、双元素→矩阵)照抄;底层 `vector_norm`/`matrix_norm` 是绑定算子(Lua 用 `_ops` 组合);`axis` 标 1-based;别名 `input`/`dim` 不移植 |
| `paddle.where` (`search.py:746`) | `@param_two_alias(["x","input"],["y","other"])`;标量 `x`/`y` 先 `paddle.to_tensor`(`:797-798`);x/y 都 None → `nonzero(as_tuple=True)`(`:800`);动态模式 broadcast_shape+broadcast_to 后 `_C_ops.where(cond, x, y, out=out)`(`:832-847`) | **直接绑定 + 一个 Lua 适配**:三态广播是显式的(broadcast_shape/broadcast_to),Lua 侧照抄;`x`/`y` 可为标量(Tensor|float)走 `to_tensor`;`where{cond}` 单参形式 = `nonzero`(Lua 侧映射);别名 `input`/`other` 不移植 |
| `paddle.topk` (`search.py:1166`) | 动态模式 `_C_ops.topk(x, k, axis, largest, sorted)`(`:1236-1240`),`axis=None` → -1(`:1234`);返回 `TopKRetType(values, indices)` | **直接绑定**(DIRECT 桶);**返回双值 → Lua 返回 `values, indices` 两个 Tensor**(索引 tensor 是 index 语义,1-based,§2.2);`axis` 标 1-based;别名 `input`/`dim` 不移植 |
| `paddle.argmax`/`argmin` (`search.py:24`) | **纯 re-export**:`from paddle._C_ops import argmax, argmin`(`search.py:24`)——本体就是绑定算子 | **直接绑定**(DIRECT 桶);**返回 index tensor,标 1-based**(C4:吐 index 的 tensor +1);签名 `argmax{x, axis, keepdim, dtype}` 走 argrule;顶层经 `tensor/__init__.py:479-480` 再导出,名字照搬 |
| `paddle.exp`/`log`/`log2`/`log10`/`sqrt`/`square`/`abs` | **纯 re-export**:`from paddle._C_ops import (…exp, log, log2, log10, sqrt, square, abs…)`(`math.py:27/85/87/88/112/114`,import 块即全部) | **直接绑定**(DIRECT 桶);一元逐元素算子无 Python 逻辑,生成器直接产出;`abs`/`exp` 等 `_` 后缀 in-place 变体(`abs_` 等)同列 DIRECT |
| `paddle.pow` (`math.py:544`) | 动态模式双路:`y` 为标量 → `_C_ops.pow(x, y, out=out)`,`y` 为 Tensor → `_C_ops.elementwise_pow(x, y, out=out)`(`:601-603`);类型分派是纯 Python | **直接绑定 + 一个 Lua 适配**:标量/Tensor 双路分派(Lua 侧 `type(y)` 判断)照抄;`y` 可为 Tensor|number;别名 `input`/`exponent` 不移植(D35) |
| `paddle.cumsum` (`math.py:3401`) | dtype 推断(整数输入默认 cast int64,`:3462-3469`)+ 动态模式 `_C_ops.cumsum(x, axis, flatten, False, False, out=out)`(`:3476-3480`) | **直接绑定 + 一个 Lua 适配**:dtype 推断表(整数→int64)是纯 Python 逻辑 → Lua 侧重写;`axis=None` → flatten=true → axis=-1;别名 `input`/`dim` 不移植 |
| `paddle.cumprod` (`math.py:3886`) | dtype 推断(同 cumsum)+ 动态模式 `_C_ops.cumprod(x, dim, False, False, out=out)`(`:3965`) | **直接绑定 + 一个 Lua 适配**:同 cumsum——dtype 推断 Lua 侧重写;`dim` 参数名是 Paddle 原生;别名 `input` 不移植 |
| `paddle.prod` (`math.py:4020`) | `dtype` 先 cast(`:4127-4132`);axis 为 0 尺寸 tensor 直接返回 x(`:4135-4136`);动态模式 `_C_ops.prod(x, axis, keepdim, reduce_all, out=out)`(`:4138-4139`) | **直接绑定 + 一个 Lua 适配**:dtype cast 与 0-size axis 特判是纯 Python 逻辑 → Lua 侧重写;`axis` 标 1-based;别名 `input`/`dim` 不移植 |
| `paddle.sort` (`search.py:509`) | 动态模式一行 `_C_ops.argsort(x, axis, descending, stable)` 取 outs(`:571-573`) | **直接绑定**(DIRECT 桶,底层 argsort);`axis` 标 1-based;参数 `descending`/`stable` 走 argrule |
| `paddle.unique` (`manipulation.py:3850`) | `@overload` 重载族(`:3733-3848`,9 个按 return_index/inverse/counts 组合);实体:空张量特判(`:3905-3926`,手工构造空 out)+ 动态模式 `_C_ops.unique(x, ri, rin, rc, axis, attr_dtype)`(`:3942-3945`) | **直接绑定 + Lua 适配**:重载族是 `@overload` 签名糖(D35 不移植)——Lua 侧只有 `unique{x, return_index, return_inverse, return_counts, axis, dtype, sorted}`;空张量特判照抄;返回值按 flag 组合(Lua 返回 1–4 个值);`axis` 标 1-based |
| `paddle.softmax` (`compat_softmax.py:31`,顶层经 `__init__.py:386`) | **PyTorch 兼容壳**:`input`/`dim` 命名(PyTorch 风格),内部 `_C_ops.softmax(outs_cast, dim, out=out)`(`:174`),前置 transpose 归一逻辑 | ⚠️ **命名冲突待拍板**:`compat_softmax.py` 的参数名是 PyTorch 的 `input`/`dim`,而 C12 要求 Paddle 规范——但 `ops.yaml` 里 softmax 在 LEGACY 桶(`legacy/static_ops.yaml`)。Lua 侧建议 `softmax{x, axis, dtype}`(Paddle 规范名),与 PyTorch 兼容壳脱钩;axis 1-based。P3 时定 |
| `paddle.clip` (`math.py:3082`) | 默认边界按 dtype 查表(int32 → `np.iinfo` min/max-2^7,`:3137-3138`)+ 动态模式 `_C_ops.clip(x, min, max, out=out)`(`:3161/3177`) | **直接绑定 + 一个 Lua 适配**:dtype 默认边界表(int/float 各查上下限)是纯 Python 逻辑 → Lua 侧重写;`min`/`max` 可为 float|int|0-d Tensor;⚠️ 顶层有 `clamp = clip` 别名(`__init__.py:1093`)——按命名铁律是 Paddle 自带别名不是 PyTorch 糖,是否同步提供 `paddle.clamp` 别名待 P3 定(倾向:同 `clip` 一样忠实迁移,零成本) |
| `paddle.diag`/`tril`/`triu` (`creation.py:29`) | **纯 re-export**:`from paddle._C_ops import diag, tril, triu`——本体就是绑定算子 | **直接绑定**(DIRECT 桶);`paddle.diag` 返回对角索引语义(无 index tensor,纯提取,不标 1-based);`tril/triu` 的 `k` 参数**不是索引语义,是偏移量,不标 1-based**(§2.2 例外) |
| `paddle.roll`/`flip` (`manipulation.py:28`) | **纯 re-export**:`from paddle._C_ops import flip, roll`——本体就是绑定算子 | **直接绑定**(DIRECT 桶);`roll` 的 `axis`/`shifts` 是**索引语义,标 1-based**;`flip` 的 `axis` 同理 1-based;无 Python 逻辑 |
| `paddle.qr` (`linalg.py:2795-2824`,顶层经 `__init__.py:456`) | `@overload` 双签名:`mode="reduced"`(Paddle 规范)+ `some=True`(PyTorch 兼容,`:2825` note 明示);动态模式 `_C_ops.qr(x, mode)`(`:2888`);返回 `QrRetType(Q, R)` | **直接绑定 + Lua 适配**:双签名是 `@overload`(D35 不移植)——Lua 侧只有 `qr{x, mode, out}`;`mode="r"` 时只返回 R 单值(返回形态随 mode,Lua 侧按 mode 返回 1 或 2 值);⚠️ docstring 注明 **backward 不支持**(`:2828`),绑定层 doc 注明 |
| `paddle.svd`/`lstsq`/`slogdet` | **⚠️ 不在顶层命名空间**:仅 `tensor/__init__.py:91/107/109` 导出,顶层 `paddle/__init__.py` 无(已 grep 核实)——属 `paddle.linalg` 子模块 | **不在本清单范围**:由 `plan/api/linalg.md`(paddle.linalg 子模块文档)覆盖;此处仅记:`svd` 动态模式 `_C_ops.svd(x, full_matrices, out=out)`(`linalg.py:2402`)、`lstsq` `_C_ops.lstsq(...)`(`:4265`)、`slogdet` `_C_ops.slogdet(x)`(`:2288`),均为直接绑定 |
| `paddle.empty` (`creation.py:3058/3071/3082`) | `@overload` 双签名 + `@size_args_decorator`(`:3082`);未初始化数据 | **直接绑定 + Lua 适配**:@overload/`size_args` 是 D35 兼容糖不移植——Lua 侧 `empty{shape, dtype}`;⚠️ 签名含 `device`/`requires_grad`/`pin_memory` PyTorch 风格参数(`:3067-3071`)C12 不引入,只保留 `shape`/`dtype`/`out` |
| `paddle.eye` (`creation.py:1957`) | 动态模式 `_C_ops.eye(num_rows, num_columns, dtype)`(`:2038`) | **直接绑定**(DIRECT 桶);`eye{n, m, dtype}`;⚠️ `paddle.eye` 在 DIRECT 桶(`eye` 在 ops.yaml,Lua 侧直接生成绑定) |
| `paddle.linspace` (`creation.py:337`) | 动态模式 `_C_ops.linspace(start, stop, num, dtype, device)`(`:428/486`,两分支) | **直接绑定**(DIRECT 桶);`linspace{start, stop, num, dtype}`;dtype 默认 float32 推断是纯 Python 逻辑 → Lua 侧重写 |
| `paddle.rand`/`randn` (`random.py:2320/938`) | `@overload` + `@size_args_decorator`;randn 内核 `_C_ops.gaussian`(`:716`),rand 内核 `_C_ops.uniform`(`:1599/1630`) | **直接绑定 + Lua 适配**:`size_args` 不移植(D35)→ Lua 侧 `rand{shape, dtype}`/`randn{shape, dtype}`;⚠️ CUDA/CPU 分支:`rand` 走 uniform、`randn` 走 gaussian 生成器(Lua 侧按 place 分派或统一走 CPU 路径) |
| `paddle.randint` (`random.py:1719/1734/1748`) | `@overload` + `@size_args_decorator`;动态模式 `_C_ops.randint(low, high, shape, dtype, place, out=out)`(`:1891/1902`) | **直接绑定 + Lua 适配**:`size_args` 不移植→ `randint{low, high, shape, dtype}`;`shape` 为 0 元素特判 Lua 侧重写 |
| `paddle.randperm` (`random.py:~2240`) | 动态模式 `_C_ops.randperm(n, dtype, device, out=out)`(`:2294`) | **直接绑定**(DIRECT 桶);`randperm{n, dtype}`;**返回 index tensor,标 1-based**(C4:randperm 的返回值是排列索引,吐 index +1);⚠️ 注意上游返回 `0..n-1`(0-based),Lua 侧要么 +1 要么 doc 特别注明 |
| `paddle.broadcast_to` (`manipulation.py:5240`) | **一行转发**:`return expand(x, shape, name)`(`:5277`)——本体就是 `expand` | **直接绑定 + 零 Lua 适配**:与 `expand` 同一底层,直接复用 Lua `_ops.expand`;`shape` 用 `IntList`,值 -1 保持原维;⚠️ 别名 `input`/`size` 不移植(D35) |
| `paddle.tile` (`manipulation.py:5000/5008/5015`) | `@overload` 三签名;动态模式 `_C_ops.tile(x, repeat_times)`(`:5130/5136`) | **直接绑定 + Lua 适配**:@overload 不移植(`tile{x, repeat_times}`);`repeat_times` 类型走 `IntList`;与 `broadcast_to` 区分:repeat 是逐维复制,不是广播到目标形状 |
| `paddle.gather` (`manipulation.py:4276/4286/4294`) | `@overload` + `@param_two_alias`;动态模式 `_C_ops.gather(x, index, axis)`(`:4226`) | **直接绑定 + Lua 适配**:@overload 不移植(`gather{x, index, axis}`);**`index` 是索引语义,标 1-based**(C4:吃 index 的 tensor -1)——Lua 侧调用 `_ops.gather` 前 index 减 1;⚠️ 与 `take_along_axis` 区分(同底层不同名,D35 别名 `input`/`dim` 不移植) |
| `paddle.take` (`math.py:5743`) | **纯 Python 组合**:`index` 先 reshape 到 1-D + `mode` 处理(raise/wrap/clip),最后 `input_1d.index_select(index_1d).reshape(index.shape)`(`:5855`) | **Lua 侧重写**:`mode` 三态(raise 报错/wrap 取模/clip 截断)是纯 Python 逻辑照抄;底层 `index_select` 绑定;**`index` 是索引语义,1-based**(C4);返回 shape 同 `index` |
| `paddle.add` (`math.py:723`) | 动态模式 `_C_ops.add(x, scaled_y, out=out)`(`:802`);`alpha` 缩放 y;`@param_two_alias` 别名 input/other | **直接绑定**(DIRECT 桶,已抽查第一批);`add{x, y, alpha}`;`alpha` 用 Lua 侧 scale 组合;别名不移植 |
| `paddle.subtract` (`math.py:921`) | 动态模式 `_C_ops.subtract(x, scaled_y, out=out)`(`:988`);结构同 add(alpha 缩放) | **直接绑定 + 一个 Lua 适配**:同 add 模式——`subtract{x, y, alpha}`;⚠️ `divide`/`multiply`/`floor_divide`/`remainder` 同一家族(`math.py:1040/1176/1247`,`_C_ops.divide` `:1093`、`floor_divide` `:1118/1240-1241`、`remainder` `:1305`),均 **直接绑定**(LEGACY 桶);`x`/`y` 可为 Tensor|标量 |
| `paddle.multiply`/`maximum`/`minimum` (`math.py:90-92`) | **纯 re-export**:`from paddle._C_ops import (…maximum, minimum, multiply…)`(`math.py:90-92`)——本体就是绑定算子 | **直接绑定**(DIRECT 桶 multiply / LEGACY 桶 maximum/minimum);`maximum{x, y}`/`minimum{x, y}` 无 Python 逻辑,生成器直接产出;⚠️ `multiply` 与 `add` 同族,`x`/`y` 可为标量 |
| `paddle.equal` (`logic.py:206`) | 标量 `y` 先 `full` 升 0-d(`:221-224`),complex 走 to_tensor(`:225-227`);动态模式 `_C_ops.equal(x, y, out=out)`(`:255`) | **直接绑定 + 一个 Lua 适配**:标量 y 升 0-d 是纯 Python 逻辑 → Lua 侧重写;比较族输出 bool 且 **stop_gradient**;别名 `input`/`other` 不移植 |
| `paddle.greater_equal`/`less_equal`/`less_than`/`not_equal` (`logic.py:322/440/541/654`) | 结构同 equal:标量升 0-d + 动态模式 `_C_ops.greater_equal/less_equal/less_than/not_equal(x, y, out=out)`(`:355/474/575/688`) | **直接绑定 + 一个 Lua 适配**:同一套标量升 0-d 逻辑 Lua 侧重写一次,四函数共用;比较族输出 bool 且 stop_gradient;LEGACY 桶 |
| `paddle.einsum` (`einsum.py:954`) | 动态模式 `_C_ops.einsum(operands, equation)[0]`(`:925`);equation 解析在 C++ 层 | **直接绑定**(LEGACY 桶);`einsum{equation, *operands}`——equation 字符串原样透传,无 Python 逻辑;⚠️ 返回取 `[0]`(单输出),Lua 侧直接返回绑定结果 |
| `paddle.assign` (`creation.py:3397`) | 动态模式 `_C_ops.assign(input)`(`:3473`);拷贝语义(不是 view) | **直接绑定**(LEGACY 桶);`assign{x, output}`;⚠️ 注意与 `Tensor.copy_` 区分:assign 是浅拷贝换 holder,绑定层按上游语义对齐 |
| `paddle.scatter` (`manipulation.py:4661/4672/4683`) | `@overload` 三签名;动态模式 `_C_ops.scatter(x, index, updates, overwrite)`(`:4604`) | **直接绑定 + Lua 适配**:@overload 不移植(`scatter{x, index, updates, overwrite}`);**`index` 是索引语义,1-based**(C4);⚠️ 注意参数名 `updates`/`overwrite`(Paddle 原生),无 PyTorch 对应 |
| `paddle.gather_nd` (`manipulation.py:6221`) | 动态模式 `_C_ops.gather_nd(x, index)`(`:6302`);ND 索引按最后一个维度展开 | **直接绑定**(DIRECT 桶);**`index` 是索引语义,1-based**(C4);`gather_nd{x, index}` 无 Python 逻辑 |
| `paddle.index_select` (`search.py:198/209/219`) | `@overload` + `@index_select_decorator`;动态模式 `_C_ops.index_select(x, index, axis, out=out)`(`:281`) | **直接绑定 + Lua 适配**:`@index_select_decorator` 处理 PyTorch 参数顺序(`(input, dim, index)` → Paddle `(x, index, axis)`,docstring `:220-225` 明示)——D35 不移植,Lua 侧只有 `index_select{x, index, axis}`;**`index` 是索引语义,1-based**(C4);`axis` 标 1-based |
| `paddle.take_along_axis` (`manipulation.py:7453`) | 动态模式 `_C_ops.take_along_axis(arr, indices, axis, out=out)`(`:7527`);与 gather 同底层 | **直接绑定**(DIRECT 桶);**`indices` 是索引语义,1-based**(C4);`axis` 标 1-based;⚠️ 与 `gather` 是同一底层两个入口(名字照搬,D35 不合并) |
| `paddle.nonzero` (`search.py:319/325/331/337`) | `@overload` 族(`as_tuple` 变体);动态模式 `_C_ops.nonzero(x, out=out)`(`:403`);`as_tuple=True` 时返回坐标元组 | **直接绑定 + Lua 适配**:`as_tuple` 变体是纯 Python 壳(D35 不移植)——Lua 侧 `nonzero{x, as_tuple}` 按 bool 返回单 Tensor 或 `pl.List`;返回坐标 index tensor,**1-based**(C4:吐 index +1) |
| `paddle.expand` (`manipulation.py:5289/5297/5304`) | `@overload` 族;动态模式 `_C_ops.expand(x, shape)`(`:5356/5372`);`expand` 是 `broadcast_to` 的实现体(反向引用 `broadcast_to → expand`,`:5277`) | **直接绑定**(DIRECT 桶);`expand{x, shape}`;`shape` 用 `IntList`,-1 保持原维;**共享内存**(docstring 同族);⚠️ `expand_as` 是同一底层另一入口 |
| `paddle.logical_and`/`or`/`not` (`logic.py:33-35`) | **纯 re-export**:`from paddle._C_ops import (…logical_and, logical_not, logical_or…)`(`logic.py:33-35`)——本体就是绑定算子;另有 in-place `logical_*_`(`logic.py:62/77/107`) | **直接绑定**(DIRECT 桶);二元 `logical_and{x, y}`/`logical_or{x, y}`,一元 `logical_not{x}`;输入须 bool 张量,Lua 侧 doc 注明;in-place 变体随生成器一起出 |
| `paddle.isfinite`/`isinf`/`isnan` (`math.py:80-82`) | **纯 re-export**:`from paddle._C_ops import (…isfinite, isinf, isnan…)`(`math.py:80-82`)——本体就是绑定算子 | **直接绑定**(DIRECT 桶);`isfinite{x}` 等一元判空;输出 bool 张量且 stop_gradient;无 Python 逻辑 |
| `paddle.cast` (`manipulation.py:225`) | `dtype` 非 `VarType/DataType` 时先 `convert_nptype_to_datatype_or_vartype`(`:251-252`)+ 动态模式 `_C_ops.cast(x, dtype)`(`:259`) | **直接绑定 + 一个 Lua 适配**:dtype 字符串→枚举归一化是纯 Python 逻辑 → Lua 侧 `_ops` 也要做同样映射(Lua 侧 dtype 表示与枚举的转换表共用,见 `api/README.md` §2 约定);`cast{x, dtype}` |
| `paddle.rank` (`attribute.py:38`) | **纯 Python 实现**:`ndims = len(input.shape); out = assign(np.array(ndims, 'int32'))`(`:69-70`)——不调算子,直接取 shape 长度 + `assign` 打包 | **Lua 侧重写**:`rank{x}` = Lua 侧 `#x.shape` + 0-d int32 Tensor 构造(DLPack/`zeros_like` 方案);⚠️ 返回值是 0-d int32 Tensor,不是 Lua number |
| `paddle.is_tensor` (`logic.py:756`) | **纯 Python 类型检查**(`isinstance` 家族,`TypeGuard` 标注 `:756`) | **Lua 侧重写**:`is_tensor{x}` = `type(x) == "userdata"` 且 metatable 是 Tensor(D32 类型判定层);一行逻辑,无绑定 |
| `paddle.argwhere` (`search.py:439`) | **一行转发**:`return nonzero(input, as_tuple=False)`(`:449-450`)——本体就是 `nonzero` | **直接绑定 + 零 Lua 适配**:与 `nonzero{x}` 同一底层,Lua 侧 `_ops` 复用;返回坐标 tensor,**1-based**(C4);顶层经 `tensor/__init__.py:482` 再导出 |
| `paddle.nan_to_num` (`math.py:1531`) | **纯 Python 组合**:`full_like` 构造 nan/+inf/-inf 替换值(`:1587-1594`,posinf/neginf None 时查 `np.finfo` 边界 `:1596-1603`)+ 三个 `where` 依次替换 nan/inf(`:1605-1608`) | **Lua 侧重写**:`nan_to_num{x, nan, posinf, neginf}` 组合(`isnan`+`equal`+`where`)照抄;`np.finfo` 边界表 → Lua 侧 dtype 边界常量表;⚠️ Python float 构造的 NOTE(`:1581-1584`)说明要用 `full_like` 而不是裸数,照抄 |
| `paddle.ceil`/`floor`/`round`/`sign` | **纯 re-export**(`math.py:27` import 块;`ops.py:31/43/47` 与 `tensor/__init__.py:293/325/417/426` 再导出) | **直接绑定**(DIRECT 桶);一元逐元素,无 Python 逻辑;`sign` 返回 -1/0/1 同 dtype;均在生成器范围 |
| `paddle.rot90` (`manipulation.py:1836`) | **纯 Python 组合**(无 `_C_ops.rot90`):`k %= 4`,`k==0` 直接返回 x,`k==2` 是 `flip(flip(x))`,`k==1/3` 是 `transpose(flip(x))`(`:1919-1932`);参数校验(axes 必须 2 维、范围检查)也是纯 Python | **Lua 侧重写**:`rot90{x, k, axes}` = Lua 侧用 `_ops.flip`+`_ops.transpose` 组合照抄;`k %= 4` 与 axes 校验 Lua 重写;`axes` 是索引语义,**1-based**(C4,且作为 flip/transpose 的轴参数传给底层前要转 0-based) |
| `paddle.masked_fill` (`manipulation.py:7228`) | 动态模式 `_C_ops.masked_fill(x, mask, value)`(`:7278`) | **直接绑定**(DIRECT 桶);`masked_fill{x, mask, value}`;`mask` 是 bool 张量同 shape;无 Python 逻辑;⚠️ `masked_fill_` in-place 变体同族 |
| `paddle.masked_select` (`search.py:1078`) | 动态模式 `_C_ops.masked_select(x, mask, out=out)`(`:1139`) | **直接绑定**(DIRECT 桶);`masked_select{x, mask}`;返回一维 bool 挑选结果;`mask` 广播规则由算子处理;无 Python 逻辑 |
| `paddle.view` (`manipulation.py:8293/8301/8308/8315/8323`) | `@overload` 五签名 + `@dygraph_only` + `@view_decorator`;实体:list/tuple → `_C_ops.view_shape(x, shape_or_dtype)`,dtype → `_C_ops.view_dtype(...)`(同 dtype 直接返回 x)(`:8383-8391`) | **直接绑定 + Lua 适配**:@overload/`view_decorator`(size/dtype 别名 + 变长)是 D35 兼容糖不移植——Lua 侧 `view{x, shape_or_dtype}` 分形状/类型两路;**共享内存**(docstring 明示 "share data with origin Tensor");⚠️ `view_shape` 走 stride kernel(T-M0-14 已验证 view 语义),`view_dtype` 是 reinterpret |
| `paddle.isclose`/`allclose` (`logic.py:22/32`) | **纯 re-export**:`from paddle._C_ops import (…allclose, isclose…)`(`logic.py:22/32`)——本体就是绑定算子 | **直接绑定**(DIRECT 桶);`isclose{x, y, rtol, atol, equal_nan}`/`allclose{...}`;`isclose` 返回逐元素 bool 张量、`allclose` 返回单个 bool Tensor;C++ 层处理 rtol/atol;无 Python 逻辑 |
| `paddle.relu` (`nn/functional/activation.py:803`,顶层经 `__init__.py:370` 从 `.nn.functional` 导入) | 动态模式 `_C_ops.relu_(x) if inplace else _C_ops.relu(x)`(`:836`);`inplace` 参数切 in-place 算子 | **直接绑定 + 一个 Lua 适配**:`relu{x, inplace}`——`inplace` 参数 Lua 侧按 bool 选 `_ops.relu`/`_ops.relu_`;⚠️ 顶层 `paddle.relu` 与 `paddle.nn.functional.relu` 是同一函数(名字照搬,不合并),Lua 侧两个入口共用一张规则表 |
| `paddle.sigmoid`/`tanh` (`math.py:104/119`) | **纯 re-export**:`from paddle._C_ops import (…sigmoid, tanh…)`(`math.py:104/119`)——本体就是绑定算子 | **直接绑定**(DIRECT 桶);`sigmoid{x}`/`tanh{x}` 一元逐元素,无 Python 逻辑;⚠️ 与 `nn.functional.sigmoid/tanh` 是同一绑定(多入口共用,命名铁律照搬),顶层经 `tensor/__init__.py:424/446` 再导出 |
| `paddle.angle`/`conj`/`real`/`imag` (`math.py:39/58`、`attribute.py:23`) | **纯 re-export**:`from paddle._C_ops import (…angle, conj…)`(`math.py:39/58`)、`from paddle._C_ops import imag, real`(`attribute.py:23`)——本体都是绑定算子 | **直接绑定**(DIRECT 桶);复数族一元操作,无 Python 逻辑;`real`/`imag` 顶层经 `tensor/__init__.py` 再导出;⚠️ `complex(real, imag)` 构造(`creation.py:3730`)与 `as_complex` 是同族另一入口,名字照搬 |
| `paddle.lerp` (`math.py:4797`) | 动态模式 `_C_ops.lerp(x, y, weight, out=out)`(`:4859`) | **直接绑定**(DIRECT 桶);`lerp{x, y, weight}`——公式 `x + weight*(y-x)` 由 C++ 层执行;`weight` 可为 float|Tensor;别名 `input`/`end` 不移植 |
| `paddle.median` (`stat.py:604/616/626`) | `@overload` 三签名 + `@param_two_alias_one_default(["x","input"],["axis","dim"],["mode",'min'])`;动态模式 `_C_ops.median(x, axis, keepdim, mode, out=out)`(`:783`) | **直接绑定 + Lua 适配**:@overload/别名装饰器 D35 不移植——Lua 侧 `median{x, axis, keepdim, mode}`;**`mode` 默认值语义诡异:别名替换时默认变 'min'**(`stat.py:626` 装饰器)——Lua 侧无别名,`mode` 默认固定 'avg';`axis` 标 1-based;返回 (values, indices) 带 `mode='min'` 双值 |
| `paddle.mode` (`search.py:669`) | 动态模式 `_C_ops.mode(x, axis, keepdim)`(`:720-723`);返回 `ModeRetType(values, indices)` | **直接绑定**(DIRECT 桶);**返回双值 → Lua 返回 `values, indices` 两个 Tensor**(indices 是索引语义,1-based);`axis` 标 1-based;别名 `input`/`dim` 不移植 |
| `paddle.logaddexp` (`math.py:845`) | **纯 Python 组合(无 `_C_ops.logaddexp`)**:`log_1p = log1p(exp(-abs(x-y)))` + `maximum(x,y)` 相加(`:906-913`);int 输入先 cast 到浮点(`:908-909`) | **Lua 侧重写**:组合(`log1p`+`exp`+`abs`+`maximum` 全在 Lua `_ops` 层)照抄;int→float cast 特判 Lua 侧重写;`out` 参数走 `assign` |
| `paddle.nn.functional.dropout` (`common.py:1234`,`nn.functional` 子模块,非顶层) | 动态模式 `_C_ops.dropout(x, p, is_test, mode, seed, fix_seed, out=out)`(`:1425`);`p` 丢弃概率语义 | **直接绑定 + Lua 适配**(`nn.functional` 模块文档覆盖);⚠️ 注意 `p` 是"丢弃概率"(与 `keep_prob` 相反);`mode` 支持 'upscale_in_train'/'downscale_in_infer';`is_test` 由框架在 eval 模式置真,Lua 侧 `paddle.is_grad_enabled` 对位 |
| `paddle.nn.functional.leaky_relu` (`activation.py:472`) | 动态模式 `_C_ops.leaky_relu_(x, negative_slope) if inplace else _C_ops.leaky_relu(x, negative_slope)`(`:516-518`) | **直接绑定 + 一个 Lua 适配**:同 relu——`inplace` 参数 Lua 侧按 bool 选 `_ops.leaky_relu_`/`_ops.leaky_relu`;`negative_slope` 默认 0.01 |
| `paddle.nn.functional.silu` (`activation.py:1121`) | 动态模式 `_C_ops.silu(x)`(`:1169`) | **直接绑定**(DIRECT 桶,silu 在 ops.yaml);`silu{x}` = `x*sigmoid(x)` C++ 层执行;`inplace` 参数同上 |
| `paddle.nn.functional.avg_pool2d`/`max_pool2d` (`pooling.py:322/1172`) | 动态模式 `_C_ops.pool2d(x, kernel_size, stride, padding, ceil_mode, exclusive, divisor_override, adaptive, ...)`(`:398/1294`);同算子不同参数形态 | **直接绑定 + Lua 适配**:avg/max 同走 `_C_ops.pool2d`(mode 参数区分);`kernel_size`/`stride`/`padding` 三态(int/tuple/'same'/'valid')解析是纯 Python → Lua 侧重写;**padding 是索引语义,标 1-based**?⚠️ 否——padding 是空间偏移不是索引,不标(§2.2 例外);`nn.functional` 模块文档覆盖 |
| `paddle.sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`sinh`/`cosh`/`tanh`/`asinh`/`acosh`/`atanh` | **纯 re-export**:`from paddle._C_ops import (…acos, acosh, asin, asinh, atan, atanh, cos, cosh, sin, sinh, tan, tanh…)`(`math.py:30/32/41/43/45/48/59/61/108/110/117/119`,import 块即全部) | **直接绑定**(DIRECT 桶);三角函数族一元逐元素,无 Python 逻辑;`tanh` 与激活章节重复标注(同一绑定多入口);`_` 后缀 in-place 变体(`sin_` 等)同列 DIRECT |
| `paddle.argsort` (`search.py:55`) | 动态模式 `_C_ops.argsort(x, axis, descending, stable)` 取 ids(`:162`);返回 (sorted_values, indices) | **直接绑定**(DIRECT 桶);**返回双值 → Lua 返回 `values, indices` 两个 Tensor**(indices 是索引语义,1-based,C4);`axis` 标 1-based;⚠️ `sort` 就是取 argsort 的 values(`search.py:584`,`sort` 包装 `_C_ops.argsort`),两入口同底层 |
| `paddle.erf`/`expm1`/`log1p`/`reciprocal`/`rsqrt` (`math.py:63/67/86/95/102`) | **纯 re-export**:`from paddle._C_ops import (…erf, expm1, log1p, reciprocal, rsqrt…)`(`math.py:63/67/86/95/102`,import 块即全部) | **直接绑定**(DIRECT 桶);一元逐元素数学函数,无 Python 逻辑;`erfinv`/`erfc` 等未在 import 块(需确认是否独立 def),随批补 |
| `paddle.bitwise_and`/`or`/`xor`/`not` (`logic.py:23-29`) | **纯 re-export**:`from paddle._C_ops import (…bitwise_and, bitwise_not, bitwise_or, bitwise_xor…)`(`logic.py:23-29`)——本体就是绑定算子 | **直接绑定**(DIRECT 桶);位运算二元/一元;⚠️ 输入须整数/布尔张量,Lua 侧 doc 注明;`_` in-place 变体(`bitwise_and_` 等)随生成器 |
| `paddle.deg2rad` (`math.py:5047`) | **纯 Python 组合**:`deg2rad_scale = pi/180` + int 输入先 cast float32 + `_C_ops.scale(x, deg2rad_scale, 0.0, True, out=out)`(`:5078-5079`) | **Lua 侧重写**:`deg2rad{x}` = `_ops.scale(x, pi/180)` 组合(int 先 cast);与 `rad2deg`(`math.py:4968`,scale `180/pi`)同族,一起写 |
| `paddle.frac` (`math.py:5584`) | **纯 Python 组合**:dtype 白名单检查(`:5616-5622`)+ `y = _C_ops.trunc(x); return _C_ops.subtract(x, y, out=out)`(`:5628-5631`) | **Lua 侧重写**:`frac{x}` = `x - trunc(x)`(`_ops.trunc`+`_ops.subtract` 组合);dtype 白名单 Lua 重写 |
| `paddle.trunc` (`math.py:2108`) | 动态模式 `_C_ops.trunc(input, out=out)`(`:2137`) | **直接绑定**(DIRECT 桶);`trunc{x}` 一元截断;C++ 层执行 |
| `paddle.trace` (`math.py:3297`) | 动态模式 `_C_ops.trace(x, offset, axis1, axis2)`(`:3384`) | **直接绑定**(DIRECT 桶);`trace{x, offset, axis1, axis2}`;**`offset` 是对角偏移量不是索引语义**(§2.2 例外);`axis1/axis2` 是维度索引,1-based |
| `paddle.multiplex` (`math.py:429`) | 动态模式 `_C_ops.multiplex(inputs, index)`(`:485`) | **直接绑定**(LEGACY 桶);`multiplex{inputs, index}`;**`index` 是张量索引语义,1-based**(C4);`inputs` 是 TensorList(pl.List) |
| `paddle.logit` (`math.py:4697`) | 动态模式 `_C_ops.logit(x, eps, out=out)`(`:4753`);`eps` 截断区间 | **直接绑定**(DIRECT 桶);`logit{x, eps}`;⚠️ `eps` 语义(输入截断而非分母 eps)Lua doc 注明 |
| `paddle.asarray` (`creation.py:1241`) | 一行转发 `tensor(data=obj, dtype, device, requires_grad)`(`:1390`);`copy` 参数被忽略(docstring 明说) | **直接绑定 + Lua 适配**:与 `to_tensor` 同底层(名字照搬);`device`/`requires_grad` 是 PyTorch 风格参数(C12)→ Lua 侧 `asarray{obj, dtype, place, stop_gradient}`;`copy` 参数是 numpy 兼容糖(D35 不移植) |
| `paddle.as_tensor` (`__init__.py:423`) | **纯别名**:`tensor as as_tensor`——就是 `paddle.tensor` 本身,无独立实现 | **直接绑定 + 零 Lua 适配**:名字照搬,与 `paddle.tensor` 同一入口两张规则表(D35 不合并);签名同 `tensor{data, dtype, place, stop_gradient}` |
| `paddle.clone` (`creation.py:3606`) | 一行 `return x.clone()`(`:3642`)——Tensor 方法转发 | **直接绑定**(Tensor 方法,`x:clone()`);Lua 侧绑定 Tensor 方法 `clone`,非顶层函数;拷贝语义可导 |
| `paddle.ravel` (`manipulation.py:2089`) | **一行转发**:`return flatten(input)`(`:2130`) | **直接绑定 + 零 Lua 适配**:与 `flatten{x}` 同底层(**共享内存 view**,docstring 明示);`ravel{x}` 无 Python 逻辑 |
| `paddle.chunk` (`manipulation.py:4929`) | chunks 校验(`:4966-4977`)+ 一行转发 `split(x, num_or_sections=chunks, axis, name)`(`:4999-5000`) | **直接绑定 + 一个 Lua 适配**:chunks 非法值校验(≤0 或 >x.shape[axis])照抄;底层 `split`;`axis` 标 1-based;返回 `pl.List`;别名 `input`/`dim` 不移植(D35) |
| `paddle.count_nonzero` (`math.py:1906`) | **纯 Python 组合**:`cast(x,'bool')` + `cast(...,'int64')` + `paddle.sum`(`:1987-1991`) | **Lua 侧重写**:`count_nonzero{x, axis, keepdim}` = `_ops.cast`→`_ops.cast`→`_ops.sum` 组合;`axis` 标 1-based;输出 int64;别名 `input`/`dim` 不移植 |
| `paddle.inner` (`math.py:2559`) | 标量特判 `_C_ops.multiply`(`:2601`);否则 reshape 后 `_C_ops.matmul`(`:2636`) | **Lua 侧重写**:`inner{x, y}` = 标量特判(multiply)+ reshape + matmul 组合照抄;`out` 参数暂不做(Q-19);别名 `input`/`other` 不移植 |
| `paddle.clamp_max`/`clamp_min` (`math.py:3233/3252`) | **一行转发**:`return clip(input, min=None, max=max, out=out)`(`:3249/3262`);`clamp = clip` 顶层别名(`tensor/__init__.py:517`) | **直接绑定 + 零 Lua 适配**:与 `clip` 同底层,三个入口共用(`clamp_max`/`clamp_min`/`clamp` 别名都照搬,Paddle 自带别名非 PyTorch 糖);`clamp_max{x, max}` 单边界 |
| `paddle.gcd` (`math.py:5117`) | **纯 Python 欧几里得算法**:broadcast_shape + abs + while 循环(`:5172-5206`) | **Lua 侧重写**:`gcd{x, y}` = broadcast + abs + while 循环(逐轮 mod/swap)照抄;⚠️ 无 `_C_ops.gcd` 算子,组合用 `_ops.mod`/`_ops.where`;整数输入 |
| `paddle.lcm` (`math.py:5254`) | **纯 Python 组合**:`gcd` + where 防零 + `abs(x*y)//d_safe`(`:5306-5312`) | **Lua 侧重写**:`lcm{x, y}` = `gcd` + where + `abs`+`floor_divide` 组合照抄;⚠️ 无算子,底层全用 `_ops` 组合 |
| `paddle.diff` (`math.py:5340`) | **纯 Python 组合**:`_diff_handler`——axis 归一 + prepend/append 拼接(`_C_ops.concat`)+ slice + subtract 递归(`:5444-5468`) | **Lua 侧重写**:`diff{x, n, axis, prepend, append}` = concat + slice(索引 1-based)+ subtract 递归照抄;`axis` 标 1-based;⚠️ `n>1` 递归,n 次迭代用 Lua 循环 |
| `paddle.frexp` (`math.py:5861`) | **纯 Python 组合**:`abs`+`log2`+`floor`+`where`+`divide`(`:5901-5932`) | **Lua 侧重写**:`frexp{x}` = 组合照抄(log2/floor/where/divide 全在 `_ops` 层);**返回双值 → Lua 返回 `mantissa, exponent`**(docstring 明示 tuple);dtype 白名单 float32/float64 |
| `paddle.ldexp` (`math.py:6377`) | dtype 提升(任一 float64 → float64)+ `paddle.multiply(x, paddle.pow(2, y))`(`:6441-6443`) | **Lua 侧重写**:`ldexp{x, y}` = dtype 提升 + `multiply(x, pow(2,y))` 组合;⚠️ 无独立算子,`pow`/`multiply` 在 `_ops` 层 |
| `paddle.isin` (`math.py:7108`) | **纯 Python 组合**:size 启发式——小 test_x 用 brute-force(`reshape`+`==`+`any`,`:7229-7236`),大 test_x 走 `flatten`+`argsort` 哈希(`:7240+`) | **Lua 侧重写**:`isin{x, test_x, assume_unique, invert}` 双路照抄;⚠️ 启发式阈值 `size_t < size_x^0.145 * 10`(`:7224`)照搬;输出 bool;别名 `elements`/`test_elements` 不移植 |
| `paddle.quantile`/`nanquantile` (`stat.py:984/1075`) | 都委托 `_compute_quantile`(`:792`;调用点 `:1063`/`:1159`)——排序 + 插值纯 Python | **Lua 侧重写**:`quantile{x, q, axis, keepdim, interpolation}` 组合照抄;`axis` 标 1-based;⚠️ `interpolation` 五态枚举('linear'/'higher'/'lower'/'midpoint'/'nearest');`nanquantile` 是 `ignore_nan=True` 变体,共享底层 |
| `paddle.nanmedian` (`stat.py:456/466/475`) | `@overload` 三签名;动态模式 `_C_ops.nanmedian(x, axis, keepdim, mode)`(`:577-578`) | **直接绑定 + Lua 适配**:@overload 不移植(`nanmedian{x, axis, keepdim, mode}`);`mode='min'` 且 `axis` 为 int 时**返回双值 (values, indices)**(`:522-525`);`axis` 标 1-based;indices 是索引语义 1-based |
| `paddle.bucketize` (`search.py:1282`) | 校验 sorted_sequence 1-D(`:1350-1353`)+ 一行转发 `searchsorted(sorted_sequence, x, out_int32, right, name, out=out)`(`:1354`) | **直接绑定 + 零 Lua 适配**:与 `searchsorted` 同底层(名字照搬,同一绑定多入口);**返回 index tensor,1-based**(C4:吐 index +1,注意上游示例 0-based);`out_int32` 控制输出 dtype |
| `paddle.is_complex`/`is_floating_point`/`is_integer` (`attribute.py:150/197/241`) | **纯 Python dtype 比较**:读 `x.dtype` 与 VarType/DataType 枚举比对(`:162-171/207-217/258-268`) | **Lua 侧重写**:一行逻辑 = Lua 侧 dtype 枚举比对;返回 Lua boolean(不是 Tensor);与 `is_tensor`(`logic.py:756`,已抽查)同族 |
| `paddle.range` (`creation.py:2421`) | `end=None` 折叠 + dtype 默认 + 动态 `_C_ops.range_v2`(`:2512`) + 静态 fill_constant 组合 | **LEGACY 桶**(底层 `range_v2` 在 legacy yaml);`end=None` 折叠(`:2495-2496`)、dtype 推断(`:2498-2499`)是纯 Python 逻辑 → Lua 侧重写;⚠️ 参数 `device`/`requires_grad` PyTorch 风格(C12)→ Lua 侧 `range{start, end, step, dtype, place, stop_gradient}` |
| `paddle.hstack` (`manipulation.py:2409`) | `paddle.atleast_1d(*x)` 升维 + `paddle.concat(arrays, axis=0/1)`(`:2472-2480`);ndim 分路决定 axis | **Lua 侧重写**:`hstack{x}` = `_ops.concat` 组合(atleast_1d 语义 = 0-d 升 1-d);`x` 用 `TensorList`;axis 选择逻辑照抄;别名 `tensors` 不移植(D35) |
| `paddle.vstack` (`manipulation.py:2491`) | `paddle.atleast_2d(*x)` + `paddle.concat(arrays, axis=0)`(`:2565-2570`) | **Lua 侧重写**:同 hstack 模式——升 2-d + concat axis=0;别名 `tensors` 不移植 |
| `paddle.dstack` (`manipulation.py:2574`) | `paddle.atleast_3d(*x)` + `paddle.concat(arrays, axis=2)`(`:2646-2650`) | **Lua 侧重写**:升 3-d + concat axis=2(索引 1-based → 底层 2);别名 `tensors` 不移植 |
| `paddle.column_stack` (`manipulation.py:2641`) | ndim<2 先 `reshape((numel,1))`(`:2720-2721`)+ `paddle.concat(arrays, axis=1)`(`:2724`) | **Lua 侧重写**:`column_stack{x}` = 逐元素 ndim 判断 + reshape + concat;别名 `tensors` 不移植 |
| `paddle.row_stack` (`manipulation.py:2711`) | **vstack 的别名**(docstring 明说 "Alias of `paddle.vstack()`") | **直接绑定 + 零 Lua 适配**:与 `vstack` 同一实现(名字照搬,不合并);同 `vstack{x}` 组合 |
| `paddle.mm` (`math.py:2164`) | `out_dtype` 特判走 `_C_ops.mm_out_dtype`(`:2279`);普通动态模式一行 `_C_ops.matmul(input, mat2, False, False, out=out)`(`:2284`);out_dtype 校验是纯 Python | **直接绑定 + 一个 Lua 适配**:`mm{x, mat2, out_dtype, out}`;⚠️ `out_dtype`(float16/bfloat16 → float32)校验与 `_C_ops.mm_out_dtype` 分支照抄;参数名 `input`/`mat2` 是 Paddle 原生(非 PyTorch) |
| `paddle.outer` (`math.py:2654`) | reshape((-1,1)/(1,-1)) 后 `_C_ops.multiply(nx, ny)`(`:2710-2711`);0-size 特判(`:2700-2705`) | **Lua 侧重写**:`outer{x, y}` = reshape + `_ops.multiply` 组合;0-size 特判照抄;别名 `input`/`vec2` 不移植(D35) |
| `paddle.isposinf`/`isneginf` (`math.py:6845/6906`) | **纯 Python 组合**:`isinf(x)` + `~signbit(x)` + `logical_and`(`:6888-6892`);neginf 是 signbit 不取反 | **Lua 侧重写**:`isposinf{x}`/`isneginf{x}` = `_ops.isinf`+`_ops.signbit`+`_ops.logical_not`+`_ops.logical_and` 组合;输出 bool;dtype 白名单(`:6880-6887`)照抄 |
| `paddle.isreal` (`math.py:6967`) | dtype 非复数 → `ones_like(x, dtype='bool')`,复数 → `equal(imag(x), 0)`(`:7008-7011`) | **Lua 侧重写**:`isreal{x}` = dtype 判断 + `_ops.imag`+`_ops.equal` 组合(或 `ones_like`);返回 bool 张量 |
| `paddle.standard_normal` (`random.py:808`) | 动态模式 `_C_ops.gaussian(shape, 0.0, 1.0, seed, dtype, ...)`(`:716`);complex dtype 走 `gaussian` 组合(`:871-879`) | **直接绑定 + Lua 适配**:`standard_normal{shape, dtype}` = `_ops.gaussian`(均值 0 方差 1);⚠️ 参数 `device`/`requires_grad` PyTorch 风格(C12 不引入);随机数必须走 Paddle 生成器(seed 全局态) |
| `paddle.normal` (`random.py:1267`) | **纯 Python 形态分派器**:mean/std 标量/Tensor/complex 分派到 `gaussian`(`:1390-1422+`);Tensor mean/std 时 shape 取 mean/std | **Lua 侧重写**:`normal{mean, std, shape}` 形态分派照抄;底层 `_ops.gaussian`;⚠️ shape=None 且 mean/std 为 Tensor 时形状取 Tensor;complex 分派(`:1394-1406`)照抄 |
| `paddle.log_normal` (`random.py:316`) | **一行组合**:`return paddle.exp(paddle.normal(mean, std, shape))`(`:377-378`) | **Lua 侧重写**:`log_normal{mean, std, shape}` = `_ops.exp(normal(...))` 组合照抄 |
| `paddle.expand_copy` (`manipulation.py:5465`) | **一行组合**:`return expand(x, shape).clone()`(`:5510`) | **直接绑定 + 零 Lua 适配**:`expand_copy{x, shape}` = `_ops.expand` + `:clone()`;**拷贝语义**(与 `expand` view 区分,docstring 明示 "without memory sharing") |
| `paddle.moveaxis` (`manipulation.py:7103`) | **纯 Python perm 计算**:source/destination 唯一性校验 + 负数归一 + perm 构造(`:7148-7205`);动态 `_C_ops.transpose(x, perm)`(`:7206`) | **Lua 侧重写**:`moveaxis{x, source, destination}` = perm 构造逻辑照抄 + `_ops.transpose`;**source/destination 是索引语义,1-based**(C4);⚠️ 与 `transpose` 的 perm 不同:这里是"移动轴",不是重排 |
| `paddle.to_dlpack`/`from_dlpack` (`utils/dlpack.py:81/144`) | 动态 `x.value().get_tensor()._to_dlpack()`(`:118`)/ `core.from_dlpack(dlpack_)`(`:264`);pybind 胶囊 + `__dlpack__` 协议 + CUDA stream 传递 | **Lua 侧重写/放放**:⚠️ 属 `paddle.utils` 子模块(顶层也导出);底层是 pybind 胶囊(依赖 Python 运行时)→ **先放放**;Lua 侧 DLPack 桥已由 Insight7 落地(记忆:insight7_dlpack_bridge),届时直接对接 C API,此入口语义对齐 |
| `paddle.atleast_1d`/`2d`/`3d` (`manipulation.py:5939/6030/6135`) | **纯 Python 升维**:非 Tensor 先 `to_tensor`;0-d → `reshape((1,)/(1,1)/(1,1,1))`,1-d/2-d → `unsqueeze(axis=0)` 补齐;多输入返回 list | **Lua 侧重写**:`atleast_1d{x, ...}` 升维逻辑照抄(reshape/unsqueeze 全在 `_ops` 层);多输入返回 `pl.List`;**共享内存**(reshape/unsqueeze 均 view 语义);hstack/vstack/dstack 依赖它(已抽查) |
| `paddle.narrow` (`manipulation.py:547`) | 纯 Python 校验(dim 负数归一、start/length 范围)+ `paddle.as_strided(input, shape=new_shape, stride=stride, offset=offset)`(`:628-632`) | **Lua 侧重写**:`narrow{x, dim, start, length}` = 校验 + `_ops.as_strided` 组合;`dim`/`start` 是索引语义,**1-based**(C4);返回 view 共享内存;⚠️ start 可为 0-d int Tensor(`:592-595` item() 取出) |
| `paddle.true_divide` (`math.py:1162`) | **一行转发**:`return divide(input, other, out=out)`(`:1169`) | **直接绑定 + 零 Lua 适配**:与 `divide` 同底层(名字照搬,不合并);`true_divide{x, y}` 即 `_ops.divide`;别名 `input`/`other` 不移植(D35) |
| `paddle.nanmean` (`math.py:1792`) | **纯 Python 组合**:`cnt = sum(~isnan(x))` + `nansum(x)` + `divide(nansum, cnt)`(`:1889-1896`) | **Lua 侧重写**:`nanmean{x, axis, keepdim, dtype}` = `_ops.isnan`+`_ops.sum`+`_ops.nansum`+`_ops.divide` 组合照抄;`axis` 标 1-based;别名 `input`/`dim` 不移植 |
| `paddle.multigammaln` (`math.py:4526`) | **纯 Python 组合**:`c = 0.25*p*(p-1)*ln(pi)` + `sum(lgamma(x.unsqueeze(-1)+b), axis=-1) + c`(`:4548-4551`);`b = 0.5*arange(1-p, 1)` | **Lua 侧重写**:`multigammaln{x, p}` = `_ops.lgamma`+`_ops.sum`+`_ops.unsqueeze` 组合照抄;`p` 校验(`:4547`)照抄;⚠️ 属 `paddle.tensor.math`(顶层经 `__init__.py` 导出),名字照搬 |
| `paddle.positive` (`math.py:4629`) | bool 检查后**返回 x 本身**(`Tensor.__pos__` 的 `+` 一元运算)(`:4660-4663`) | **Lua 侧重写**:`positive{x}` = bool 检查 + 返回原 Tensor;一行逻辑;⚠️ 不是算子,是恒等映射(与 `paddle.neg`/`negative` 对称) |
| `paddle.signbit` (`math.py:6773`) | **纯 Python 组合**:`copysign(ones, x)` + `sign(neg_zero_x)` + `cast(x_sign < 0, 'bool')`(`:6835-6840`) | **Lua 侧重写**:`signbit{x}` = `_ops.copysign`+`_ops.sign`+`_ops.cast`+`_ops.less_than` 组合;输出 bool;dtype 白名单(`:6826-6834`)照抄 |
| `paddle.sinc` (`math.py:7017`) | **纯 Python 组合**:`x==0` 特判(1e-20 替换)+ `multiply(pi)` + `sin/tmp` + `isnan` 特判补 1(`:7067-7074`) | **Lua 侧重写**:`sinc{x}` = `_ops.where`+`_ops.multiply`+`_ops.sin`+`_ops.divide` 组合照抄;⚠️ 无 `_C_ops.sinc` 算子,公式 `sin(πx)/(πx)` 手工组合 |
| `paddle.scatter_nd` (`manipulation.py:4887`) | **一行转发**:`return scatter_nd_add(zeros(shape, updates.dtype), index, updates)`(`:4928-4929`) | **直接绑定 + 零 Lua 适配**:`scatter_nd{index, updates, shape}` = `_ops.zeros` + `_ops.scatter_nd_add` 组合;**`index` 是索引语义,1-based**(C4);`shape` 用 `IntList`;重复 index 累积(`scatter_nd_add` 语义) |
| `paddle.tensordot` (`manipulation.py:6571`) | **纯 Python 收缩规划**:axes 解析(标量/单列表/双列表/0-size 特判)+ 1-size 维提前 `sum` 收缩(`:6763-6768`)+ perm/reshape 规划 + `x.matmul(y).reshape(shape_out)`(`:6797-6807`) | **Lua 侧重写**:`tensordot{x, y, axes}` = axes 解析 + perm 构造 + `_ops.transpose`+`_ops.reshape`+`_ops.matmul` 组合照抄;`axes` 是索引语义,**1-based**(C4,作 perm 前转 0-based);⚠️ 无 `_C_ops.tensordot`,纯算子组合 |
| `paddle.vecdot` (`linalg.py:1667`) | **一行组合**:`return (x.conj() * y).sum(axis=axis)`(`:1700-1701`) | **Lua 侧重写**:`vecdot{x, y, axis}` = `_ops.conj`+`_ops.multiply`+`_ops.sum` 组合;`axis` 标 1-based;⚠️ 属 `paddle.linalg` 子模块(非顶层),`linalg.md` 覆盖 |
| `paddle.logdet` (`linalg.py:2318`) | **一行组合**:`return det(input).log()`(`:2345`) | **直接绑定 + 零 Lua 适配**:`logdet{x}` = `_ops.det` + `:log()` 组合;⚠️ 属 `paddle.linalg` 子模块;负行列式 → nan、零 → -inf(docstring `:2322-2323` 明示) |
| `paddle.rand_like` (`random.py:1172`) | dtype/device 默认取 input,`shape = paddle.shape(input)`,转发 `rand(shape, dtype, device, requires_grad)`(`:1248-1253`) | **直接绑定 + Lua 适配**:`rand_like{x, dtype, place, stop_gradient}` = `_ops.rand(shape(x))` 组合;dtype 默认同 x、place 默认同 x;⚠️ 参数 `device`/`requires_grad` PyTorch 风格(C12 不引入) |
| `paddle.tensor_split` (`manipulation.py:3016`) | **纯 Python 分派**:int → 均分(divmod 计算前段 +1,`_tensor_split_sections` `:3162-3167` 走 `split`);list → 逐段 `paddle.slice`(`_tensor_split_indices` `:3150-3160`);负数索引归一(`:3154-3156`) | **Lua 侧重写**:`tensor_split{x, num_or_indices, axis}` 双路照抄;`axis` 标 1-based;`split`/`slice` 底层;**返回 `pl.List`**;别名 `input`/`indices_or_sections`/`dim` 不移植(D35) |
| `paddle.hsplit`/`dsplit`/`vsplit` (`manipulation.py:3198/3271/3332`) | 各一行分派到 `tensor_split`:`hsplit` ndim>1 → axis=1 否则 axis=0(`:3260-3265`)、`dsplit` → axis=2、`vsplit` → axis=0 | **Lua 侧重写**:三个入口 = `tensor_split` + 固定 axis 的组合(已抽查 tensor_split);**返回 `pl.List`**;alias `input`/`indices`/`sections`(`ParamAliasDecorator` `:3245-3249`)不移植(D35) |
| `paddle.unflatten` (`manipulation.py:8156`) | `non_negative_axis` 轴归一(`:8220`)+ 新 shape 拼接 `list(x.shape[:axis])+list(shape)+list(x.shape[axis+1:])`(`:8222-8223`)+ `reshape`(尾部) | **直接绑定 + Lua 适配**:`unflatten{x, axis, shape}` = 轴校验 + `_ops.reshape`;**`axis` 是索引语义,1-based**(C4);`shape` 用 `IntList`(至多一个 -1);别名 `input`/`dim`(`ParamAliasDecorator`)不移植;⚠️ 与 `flatten` 互逆 |
| `paddle.diagflat` (`creation.py:2920`) | 动态模式:`len(shape)<=1` → `_C_ops.diag(x, offset, 0)`(`:3010-3011`),否则 `flatten` 后 `_C_ops.diag`(`:3013-3014`) | **直接绑定 + 零 Lua 适配**:`diagflat{x, offset}` = `_ops.flatten` + `_ops.diag` 组合;`offset` 是对角偏移量**不是索引语义**(§2.2 例外);⚠️ 与 `diag` 的区别:diagflat 永远展开成一维再铺对角线,多维也展平 |
| `paddle.matrix_power` (`linalg.py:2707`) | 动态模式一行 `_C_ops.matrix_power(x, n)`(`:2771`) | **直接绑定**(属 `paddle.linalg` 子模块,`linalg.md` 覆盖);`matrix_power{x, n}`;⚠️ n<0 走逆矩阵幂、n=0 返单位阵(`:2715-2720` docstring),由 C++ 层处理 |
| `paddle.cdist` (`linalg.py:4393`) | **纯 Python 分派**:p==2.0 且 mm 模式 → `pow/sum/transpose/matmul` 组合(`:4498-4514`);否则 `norm(x[...,None,:] - y[...,None,:,:], p=p, axis=-1)`(`:4517-4518`);0-size 特判(`:4489-4497`) | **Lua 侧重写**(属 `paddle.linalg`):`cdist{x, y, p, compute_mode}` 分派照抄;两路都用 `_ops` 组合(mode 三值校验 `:4462-4468` 照抄);`compute_mode` 枚举参数只收字符串(§2.1.1) |
| `paddle.broadcast_shape`/`broadcast_shapes` (`math.py:4269/4222`) | `broadcast_shape` 一行 `core.broadcast_shape(x_shape, y_shape)`(`:4303`,C++ 原生);`broadcast_shapes` 纯 Python 折叠循环(`:4249-4256`) | **直接绑定 + Lua 适配**:`broadcast_shape{x_shape, y_shape}` → Lua 侧 C ABI 透传 C++ `BroadcastShape`;`broadcast_shapes{...}` = Lua 循环逐对折叠;**返回 `pl.List`/裸表**(`x:shape()` 同族,用裸表);纯函数无 Tensor 参与——⚠️ 但 `core.broadcast_shape` 是 C++ 符号,须先核实无 Python 依赖(M0 反幻觉) |
| `paddle.vander` (`math.py:6173`) | **纯 Python 组合**:`empty` 占位 + 首列置 1 + `x[:,None]` + `cumprod`(`:6249-6263`);`n=None` → `x.shape[0]`(`:6241`);降序时末尾 `flip`(`:6266`) | **Lua 侧重写**:`vander{x, n, increasing}` = `_ops.empty`+`_ops.setitem`+`_ops.cumprod`+`_ops.flip` 组合照抄;`n` 校验(`:6243-6245`)照抄;dtype 白名单(:6220-6226)照抄 |
| `paddle.combinations` (`math.py:6705`) | **纯 Python 组合**:r==0 → `empty`;`meshgrid` 生成下标网格 + `full(...,True)` mask + `where`/`masked_select` 逐对(`:6731-6760+`) | **Lua 侧重写**:`combinations{x, r, with_replacement}` = `_ops.meshgrid`+`_ops.masked_select` 组合照抄;1-D 校验(`:6725-6726`)照抄;`with_replacement` 决定 mask 构造;别名 `input` 不移植 |
| `paddle.cartesian_prod` (`math.py:7303/7307/7311`) | **纯 Python 组合**:1-D 校验 + `meshgrid` + `stack(axis=-1)` + `reshape(-1, len(x))`(`:7366-7371`) | **Lua 侧重写**:`cartesian_prod{x}` = `_ops.meshgrid`+`_ops.stack`+`_ops.reshape` 组合;**`x` 用 `TensorList`**(api/README §2.1);单输入直接返回原张量(`:7363-7364`) |
| `paddle.trapezoid` (`math.py:6023`) | **一行转发**:`return _trapezoid(y, x=x, dx=dx, axis=axis)`(`_trapezoid` 在 `:5935`,纯 Python:差分 + `cumsum`/`sum` 组合) | **Lua 侧重写**:`trapezoid{y, x, dx, axis}` = `_trapezoid` 组合照抄(avg_pool/`diff`+`sum`);`axis` 标 1-based;x/dx 互斥校验(`:5966-5967`)照抄;⚠️ `_trapezoid` 的 `mode` 参数('sum'/'cumsum')对应 `trapezoid` 与 `cumulative_trapezoid` 两入口 |
| `paddle.polar` (`creation.py:3984`) | **一行组合**:`return paddle.complex(abs*cos(angle), abs*sin(angle), out=out)`(`:4026-4027`) | **Lua 侧重写**:`polar{abs, angle}` = `_ops.multiply`+`_ops.cos`+`_ops.sin`+`_ops.complex` 组合;dtype 白名单 float32/float64(`:4022-4025`)照抄;⚠️ 输出 complex dtype(实部/虚部同精度) |
| `paddle.seed` (`framework/random.py:29`) | **纯 Python 全局态**:按编译能力遍历 CUDA/XPU/Custom 设备生成器 `manual_seed`(`:42-61`)+ 返回 `default_cpu_generator().manual_seed(seed)`(`:62`) | **Lua 侧重写**:`seed{n}` = 遍历设备生成器设种子;⚠️ 无 Tensor 参与,纯 C ABI 调 `Generator::manual_seed`;与 `paddle.rand*` 的种子语义绑定(Lua 侧随机数必须走 Paddle 生成器) |
| `paddle.set_default_dtype`/`get_default_dtype` (`framework/framework.py:31/127`) | **纯 Python 全局态**:dtype 归一化校验(`:61-80`)+ `LayerHelperBase.set_default_dtype(d)`(`:81`);getter 一行 `LayerHelperBase.get_default_dtype()`(`:141`) | **Lua 侧重写**:`set_default_dtype{d}` = dtype 字符串校验 + 全局态设置;C++ 侧 `LayerHelperBase` 有对应静态态(M0 反幻觉:须先核实);⚠️ 影响 `paddle.zeros` 等无 dtype 推断(creation 已抽查) |
| `paddle.kron` (`base/dygraph/generated_tensor_methods_patch.py:276`) | **pybind 算子包装**:`def _kron: return _C_ops.kron(*args, **kwargs)`(`:276-277`);顶层 `__init__.py:697` re-export | **直接绑定**(需 P3 核实 `kron` 是否在 `ops.yaml`;若否 → 手工绑定 `_C_ops.kron` 同款);`kron{x, y}` 克罗内克积;⚠️ 真实源码在 `generated_tensor_methods_patch.py`,`_paddle_docs.py:5195` 是文档占位不是实现 |
| `paddle.pinverse` (`__init__.py:1119`) | **纯别名**:`pinverse = pinv`(`__init__.py:1119`)——就是 `pinv` 本体 | **直接绑定 + 零 Lua 适配**:与 `pinv` 同一入口(名字照搬,不合并);`pinv` 属 `paddle.linalg` 子模块(`linalg.md` 覆盖);零独立实现 |
| `paddle.repeat` (`manipulation.py:5177`) | **一行转发**:`return tile(input, repeat_times=repeats)`(`:5240-5241`)——就是 `tile` | **直接绑定 + 零 Lua 适配**:与 `tile` 同底层(名字照搬,tile 已抽查);`repeat{x, repeats}`;repeats 单 int 只重复第一维、数组逐维(`:5199-5203` docstring) |
| `paddle.repeat_interleave` (`manipulation.py:6993`) | `axis=None` → `flatten` + axis=0(`:7066-7067`);动态模式 `_C_ops.repeat_interleave(x, repeats, axis, output_size)`(`:7074-7075`);repeats 为 Tensor 走 `_C_ops.repeat_interleave_with_tensor_index`(`:7070-7071`) | **直接绑定 + 一个 Lua 适配**:`repeat_interleave{x, repeats, axis, output_size}`;axis=None → flatten 特判照抄;`repeats` 可为 int|Tensor;`output_size` 仅为性能提示走 argrule;别名 `input`/`dim` 不移植(D35) |
| `paddle.masked_scatter` (`manipulation.py:5758`) | CUDA 走 `_C_ops.masked_scatter`(`:5823-5824`);**CPU 是纯 Python 组合**:cast mask + cumsum 前缀 + `value.flatten()[mask_prefix]` + `where`(`:5826-5840`) | **直接绑定 + Lua 适配**:`masked_scatter{x, mask, value}`;⚠️ CPU 组合(cumsum/where/masked_select)照抄,**CUDA 直接算子**——Lua 侧 CPU 用 `_ops` 组合、GPU 走绑定,mask 元素数校验(`:5833-5835`)照抄;`masked_scatter_` in-place 变体同族 |
| `paddle.scatter_add` (`manipulation.py:7411`) | **一行转发**:`return put_along_axis(input, index, src, dim, 'add', include_self=True, broadcast=False)`(`:7455-7458`) | **直接绑定 + 零 Lua 适配**:与 `put_along_axis` 同底层('add' 模式);**`index` 是索引语义,1-based**(C4);参数名 `dim` 是 Paddle 原生 |
| `paddle.put_along_axis` (`manipulation.py:7642`) | 校验(arr/indices 同维数 `:7729-7731`)+ `non_negative_axis` 归一(`:7732`)+ 0-size index 特判返回 `assign`(`:7735-7736`)+ broadcast 分派(`:7738-7767`)+ reduce 分派(assign/add/mul/mean/amin/amax) | **直接绑定 + Lua 适配**:`put_along_axis{arr, indices, values, axis, reduce, include_self, broadcast}`;`reduce` 是枚举参数只收字符串(§2.1.1);**`indices`/`axis` 索引语义,1-based**(C4);0-size 特判照抄;⚠️ 与 `take_along_axis` 互逆(已抽查) |
| `paddle.diagonal_scatter` (`manipulation.py:8659`) | **一行转发**:`return fill_diagonal_tensor(x, y, offset, axis1, axis2, name)`(`:8716-8717`) | **直接绑定 + 零 Lua 适配**:底层 `fill_diagonal_tensor`(需 P3 核实是否独立算子);`diagonal_scatter{x, y, offset, axis1, axis2}`;**`axis1/axis2` 索引语义 1-based**;`offset` 是偏移量非索引(§2.2);⚠️ y 形状须与 `diagonal` 输出一致(docstring `:8674-8675`) |
| `paddle.select_scatter` (`manipulation.py:8726`) | index/axis 负数归一(`:8752-8756`)+ 形状校验(`:8758-8772`)+ setitem 语义 slice 构造(`:8780-8789`) | **Lua 侧重写**:`select_scatter{x, values, axis, index}` = setitem(slice)组合;`axis`/`index` **索引语义,1-based**(C4);⚠️ 是 `__setitem__` 的具名版本(与 `x[axis, index] = values` 等价) |
| `paddle.slice_scatter` (`manipulation.py:8854`) | 双签名(Paddle `axes/starts/ends/strides` + PyTorch `dim/start/end/step`);同 `select_scatter` slice 构造 | **Lua 侧重写 + D35 例证**:PyTorch 签名(`dim`/`src`/`step`)不移植(D35)——Lua 侧只有 `slice_scatter{x, value, axes, starts, ends, strides}`;**`axes`/`starts` 索引语义,1-based**(C4);值广播宽容(shape 不完全匹配也放行,docstring 例 3) |
| `paddle.sgn` (`math.py:5671`) | complex → `as_real` + `unsqueeze` + `as_complex`(`:5726-5741`);real → `paddle.sign` | **直接绑定 + Lua 适配**:`sgn{x}` = complex 是 `_ops.as_real`+`_ops.unsqueeze`+`_ops.as_complex` 组合,real 是 `_ops.sign`;dtype 白名单(float16/32/64 + complex64/128,`:5715-5724`)照抄 |
| `paddle.polygamma` (`math.py:6285`) | `n==0` → `digamma(x)`(`:6318`);否则动态 `_C_ops.polygamma(x, n)`(`:6319-6320`);n 校验(`:6312-6316`) | **直接绑定 + 一个 Lua 适配**:`polygamma{x, n}`;n==0 特判 = `_ops.digamma`,否则直接绑定;n 类型/非负校验照抄 |
| `paddle.gammainc` (`math.py:4474`) | **一行组合**:`return 1 - paddle.gammaincc(x, y)`(`:4502`) | **直接绑定 + 零 Lua 适配**:`gammainc{x, y}` = `_ops.subtract(1, gammaincc(x,y))` 组合;gammaincc 已直接绑定(下行) |
| `paddle.gammaincc` (`math.py:4413`) | 动态模式 `_C_ops.gammaincc(x, y)`(`:4450-4451`);x/y 非负校验(`:4441-4448`) | **直接绑定**:`gammaincc{x, y}`;非负校验纯 Python 照抄;输出与输入同 dtype |
| `paddle.matrix_rank` (`linalg.py:1938`) | **纯 Python 分派**:hermitian 真 → `eigh` 取特征值绝对值,假 → `svd` 取奇异值;tol/atol+rtol 组合排序比较(`:2020+`);⚠️ 需 P3 确认尾部细节 | **Lua 侧重写**(属 `paddle.linalg`):`matrix_rank{x, tol, hermitian, atol, rtol}` = `_ops.svd`/`_ops.eigh` + 比较计数;⚠️ 容差矩阵运算(`max(atol, sigma_1*rtol)`)docstring `:1957-1962` 明示,照抄;**返回值是 int64 张量** |
| `paddle.save` (`framework/io.py:790`) | **纯 Python 壳**:dispatch 到 `_save/state_dict/save_vars` 等(`:846+`);单 Tensor 走 `Tensor.save`(`SaveTensor` 线格式,D19/T-M0-19 已实测);state_dict 走 pickle(protocol 4,`lua-pickle` 已就绪) | **Lua 侧重写 + 已有基座**:`save{obj, path, protocol}` 分派照抄;**单 Tensor → `SaveTensor` 线格式**(D19);**state_dict → lua-pickle 写**(`E:\code\lua-pickle` R1-R5 已落盘);静态图 Program 分支放放 |
| `paddle.load` (`framework/io.py:1064`) | **纯 Python 壳**:按文件头 magic 分派 pickle / binary / jit 产物(`:1155+`);`return_numpy` 配置(`:1112-1113`) | **Lua 侧重写 + 已有基座**:`load{path, model_filename, params_filename, return_numpy}` 分派照抄;**单 Tensor → `LoadTensor`**(D19);**pickle → lua-pickle 读**;⚠️ 支持 jit.save 前缀路径(`:1076-1083` docstring)——与 `jit.load`(T-M0-17 实测)配合 |
| `paddle.finfo` (`framework/dtype.py:305`) | **一行**:`convert_to_datatype(dtype)` + `core_finfo(dtype)`(`:357-358`) | **直接绑定 + 零 Lua 适配**:`finfo{dtype}` → C ABI 透传 `core_finfo`(纯 C++ 查表);**返回 finfo 对象**(min/max/eps/tiny/bits 等 8 属性,docstring `:326-334`);⚠️ 参数名别名 `type`(`:314-315` 注释)不移植(D35);属 dtype 工具非算子 |
| `paddle.msort` (`search.py:603`) | **一行转发**:`sort(input, axis=0) if out is None else sort(input, axis=0, out=out)`(`:654-656`)——就是 `sort` 固定 axis=0 | **直接绑定 + 零 Lua 适配**:与 `sort` 同底层(名字照搬);`msort{x, out}` = `_ops.sort(x, axis=1-based 0 调整)`;无新逻辑 |
| `paddle.searchsorted` (`search.py:1352`) | 动态模式:`sorter` 给定先 `take_along_axis`(`:1411-1413`)+ `_C_ops.searchsorted(sorted_sequence, values, out_int32, right, out=out)`(`:1414-1415`);`side` 参数改写 `right`(`:1409-1410`) | **直接绑定 + 一个 Lua 适配**:`searchsorted{sorted_sequence, values, out_int32, right, side, out, sorter}`;`side`(left/right)是 np 兼容糖(D35 是否移植待定:上游是显式参数不是装饰器,倾向照搬);**返回 index tensor,1-based**(C4:吐 index +1);`sorter` 特判照抄 |
| `paddle.matrix_transpose` (`linalg.py:270`) | **一行**:`return x.mT`(`:305`)——Tensor 属性 `mT` 的封装 | **直接绑定 + 零 Lua 适配**:`matrix_transpose{x}` = `x:mt()`(Lua 侧 Tensor 绑定同 `mT` 属性);**共享内存 view**(transpose 语义);属 `paddle.linalg` 子模块 |
| `paddle.create_parameter` (`creation.py:206`) | 校验(shape/dtype/attr/initializer,`:225-266`)+ `LayerHelper.create_parameter(attr, shape, dtype, is_bias, default_initializer)`(`:283`,C++ 侧) | **放放(P9 nn 层)**:`create_parameter{shape, dtype, attr, is_bias, default_initializer}`;⚠️ 底层是 `LayerHelper.create_parameter`——**静态图/参数注册语义**,动态图下由 `nn.Layer` 的 `create_parameter` 方法承载(`nn.md` 覆盖);顶层入口等 P9 一并定 |
| `paddle.no_grad` (`base/dygraph/base.py:328`) | 无参时返回 `_switch_tracer_mode_guard_(is_train=False)`(`:382`);有参时包装函数(`:384-392`)——**上下文管理器 + 装饰器双形态** | **直接绑定 + Lua 适配**:`no_grad{}` 返回上下文对象;Lua 侧用 `pairs` 式 `with` 模拟(C 侧 `__enter__/__exit__` 语义 = `_set_has_grad(False)`);⚠️ 装饰器形态在 Lua 无等价(表调用)——只留上下文形态;**梯度开关全局态**(`core._has_grad`,与 `stop_gradient` 关系 P6 定) |
| `paddle.set_grad_enabled`/`enable_grad` (`base/dygraph/base.py:473/581`) | 都是 `_DecoratorContextManager` 子类;`__enter__` 调 `_set_grad_enabled(mode)`/`_set_grad_enabled(True)`(`:516-517`/`:619-621`),退出恢复 `self.prev`(`:519-520`/`:623-624`);`_set_grad_enabled` 一行 `core._set_has_grad(mode)`(`:469-470`) | **直接绑定 + Lua 适配**:`set_grad_enabled{mode}`/`enable_grad{}` 上下文形态(同 no_grad);⚠️ `set_grad_enabled` **接收 bool 参数**(与 no_grad/enable_grad 无参不同)——Lua 侧 `set_grad_enabled{flag}` 规则表显式区分;核心 `core._set_has_grad` C ABI 透传 |
| `paddle.is_grad_enabled` (`base/dygraph/base.py:442`) | **一行**:`return core._has_grad()`(`:459`) | **直接绑定 + 零 Lua 适配**:`is_grad_enabled{}` = C ABI 读全局态;返回 Lua boolean;⚠️ 与 `stop_gradient`(Tensor 级)是两层概念,不做混淆 |
| `paddle.digamma` (`math.py:4350`) | 动态模式一行 `_C_ops.digamma(x)`(`:4370`) | **直接绑定**(DIRECT 桶);`digamma{x}` 一元;整数输入自动 cast float32(docstring `:4358-4359`);`digamma_` in-place 变体同族 |
| `paddle.gammaln` (`math.py:4305`) | 动态模式一行 `_C_ops.gammaln(x)`(`:4324`);**与 `lgamma` 是同一函数**(gammaln 即 log-gamma;顶层 `lgamma` 在 `tensor/__init__.py:368` 经 `.math` 导入) | **直接绑定**(DIRECT 桶);`gammaln{x}`/`lgamma{x}` 同一绑定两入口(名字照搬,不合并);dtype 白名单 float16/32/64/bfloat16(`:4319-4320`)照抄 |
| `paddle.erfinv` (`math.py:4903`) | 动态模式一行 `_C_ops.erfinv(x)`(`:4933`) | **直接绑定**(DIRECT 桶);`erfinv{x}` 逆误差函数;**⚠️ 输入超 [-1,1] 输出 nan/inf**(docstring `:4920-4921` 示例 -inf);`erfinv_` 变体同族 |
| `paddle.cumulative_trapezoid` (`math.py:6097`) | **一行转发**:`_trapezoid(y, x=x, dx=dx, axis=axis, mode='cumsum')`(与 `trapezoid` 同底层,已抽查)——cumsum 累积积分 | **Lua 侧重写**:`cumulative_trapezoid{y, x, dx, axis}` = `_trapezoid` 组合照抄(cumsum 变体);`axis` 标 1-based;⚠️ 输出 N-D(与 trapezoid 的标量/降维不同) |
| `paddle.binomial` (`random.py:198`) | 动态模式:先 `broadcast_tensors([cast(count, prob.dtype), prob])` 再 `_C_ops.binomial(count, prob)`(`:245-246`) | **直接绑定 + 一个 Lua 适配**:`binomial{count, prob}`;⚠️ count 是 int 张量、prob 是 float 张量,先广播再绑定(`:245` 照抄);输出 int64 |
| `paddle.uniform` (`random.py:1486`) | 动态模式 `_C_ops.uniform(shape, min, max, seed, dtype, place, out=out)`(`:1599/1630`);`seed=0` 用全局生成器(`:1515-1516` docstring) | **直接绑定 + Lua 适配**:`uniform{shape, dtype, min, max, seed}` = `_ops.uniform`;⚠️ 参数 `device`/`requires_grad` PyTorch 风格(C12 不引入);`shape` 用 `IntList`;**随机数必须走 Paddle 生成器**(seed 全局态) |
| `paddle.randn_like`/`randint_like` (`random.py:1078/1988`) | 都是 `rand_like` 模式(dtype/device 默认取 input,`shape = paddle.shape(x)`,转发 `randn`/`randint`)(`rand_like` 已抽查 `:1248-1253` 同款) | **直接绑定 + Lua 适配**:`randn_like{x, dtype, place, stop_gradient}`/`randint_like{x, low, high, dtype, place}` = `_ops.randn/randint(shape(x))` 组合;⚠️ `randint_like` 的 `pin_memory` 参数 C12 不引入 |
| `paddle.fmod`/`floor_mod`/`fix` (`tensor/__init__.py:530-533`) | **纯别名**:`fmod = remainder`、`floor_mod = remainder`、`fix = trunc`(`tensor/__init__.py:530-533`,顶层 `__init__.py:1113-1114` 同步)——不是独立实现 | **直接绑定 + 零 Lua 适配**:三个名字照搬,**各对应 `remainder`/`trunc` 同一绑定**(D35 不合并,别名也忠实迁移);`fmod{x,y}` = `_ops.remainder`,`fix{x}` = `_ops.trunc` |
| `paddle.mvlgamma` (`tensor/__init__.py:534`) | **纯别名**:`mvlgamma = multigammaln`(`tensor/__init__.py:534-535`)——同一函数 | **直接绑定 + 零 Lua 适配**:`mvlgamma{x, p}` 与 `multigammaln{x, p}` 同一绑定(已抽查 multigammaln);名字照搬 |
| `paddle.logsumexp`/`heaviside`/`renorm` (`base/dygraph/generated_tensor_methods_patch.py:291/234/108`) | **pybind 算子包装**:`def _logsumexp: return _C_ops.logsumexp(...)`(`:291-292`)、`_heaviside`(`:234-235`)、`_renorm`(`:108-109`)——同 `kron` 模式(已抽查) | **直接绑定**(P3 核实 yaml;若否手工绑定 `_C_ops.*` 同款);`logsumexp{x, axis, keepdim}`(`axis` 1-based)/`heaviside{x, y}`/`renorm{x, p, axis, max_norm}`(`axis` 1-based);⚠️ `_paddle_docs.py:1253/3836/5054` 是文档 stub 不是实现 |
| `paddle.addmv` (`math.py:2336`) | **组合**:`addmm(input.unsqueeze(-1), mat, vec.unsqueeze(-1), beta, alpha).squeeze(-1)`(`:2368-2370`)——矩阵乘向量后加权加 | **Lua 侧重写**:`addmv{input, mat, vec, beta, alpha}` = `_ops.unsqueeze`+`_ops.addmm`+`_ops.squeeze` 组合照抄;公式 `beta*input + alpha*(mat@vec)`(`:2345-2346` docstring) |
| `paddle.addr` (`math.py:2398`) | **组合**:`addmm(input, vec1.unsqueeze(-1), vec2.unsqueeze(0), beta, alpha)`(`:2435-2437`)——外积加权加 | **Lua 侧重写**:`addr{input, vec1, vec2, beta, alpha}` = `_ops.unsqueeze`+`_ops.addmm` 组合照抄;⚠️ 与 `outer`(已抽查)同源:outer 是 `multiply(reshape(1,n),(m,1))`,addr 是 `addmm` |
| `paddle.addcdiv` (`math.py:2457`) | **组合**:`paddle.add(input, value * paddle.divide(tensor1, tensor2))`(`:2475-2477`)——除后加权加 | **Lua 侧重写**:`addcdiv{input, tensor1, tensor2, value}` = `_ops.divide`+`_ops.multiply`+`_ops.add` 组合照抄;⚠️ `addcmul` 在 3.5.0 无顶层导出(grep 无结果),不列 |
| `paddle.histc` (`math.py:2514`) | **转发**:`_histogram(input, bins, min, max).astype('float32')`(`:2537-2538`)——linalg 的 `histogram` + 强转 float32 | **直接绑定 + 一个 Lua 适配**:`histc{input, bins, min, max}` = `_ops.histogram` + `_ops.astype(float32)` 组合;**min/max 双 0 → 取数据极值**(docstring `:2522-2523`);⚠️ `histogram` 是另一入口(名字照搬) |
| `paddle.iinfo` (`framework/dtype.py:251`) | dtype 归一化(字符串/uint16 特判)+ `core_iinfo(dtype)`(`:304`)——与 `finfo`(已抽查)同构 | **直接绑定 + 零 Lua 适配**:`iinfo{dtype}` → C ABI 透传 `core_iinfo`;**返回 iinfo 对象**(min/max/bits/dtype 4 属性,docstring `:268-271`);⚠️ 与 finfo 对称(dtype 整数界),纯 C++ 查表 |
| `paddle.concatenate` (`__init__.py:1091`) | **纯别名**:`concatenate = concat`(`__init__.py:1091`)——numpy 风格名 | **直接绑定 + 零 Lua 适配**:与 `concat`/`paddle.concat` 同一绑定(名字照搬,**numpy 别名也忠实迁移**;D35 只挡 PyTorch 糖,不挡 numpy 名);⚠️ `tensor/__init__.py:354` 另有 `cat = concat` 同款 |
| `paddle.set_flags` (`base/framework.py:133`) | **纯 Python 全局 flags**:遍历 dict 校验 `_global_flags().is_public(key)` 后赋值(`:147-157`);GFlags 注册表 | **Lua 侧重写**:`set_flags{flags}` = C ABI 遍历设置 GFlags;**⚠️ 这是运行时引擎配置**(如 `FLAGS_eager_delete_tensor_gb`),非 Tensor API;Lua 侧等 M1 引擎跑起来再定,C ABI 透传 `UpdateFlags` |
| `paddle.set_cuda_rng_state` (`framework/random.py:222`) | **纯 Python 全局态**:校验 state_list 长度 = CUDA 设备数,遍历 `default_cuda_generator(i).set_state`(`:235-241`) | **Lua 侧重写**:`set_cuda_rng_state{state_list}` = 遍历 CUDA 生成器 set_state;**与 `get_rng_state`/`set_rng_state` 同族**(已抽查 set_rng_state `:155`);⚠️ 无 GPU 环境分支(`:229-231`)照抄 |
| `paddle.set_default_tensor_type` (`framework/framework.py:80`) | **deprecated 壳**:dtype_map(FloatTensor→float32 等)+ 转发 `set_default_dtype`(`:96-115`) | **Lua 侧重写 + 弃用处理**:v1 直接不提供 + 报错指向 `set_default_dtype{v}`(docstring `:82-83` 明示已弃用)**;⚠️ 或照搬 dtype_map 别名表(4 个 Tensor 类名 → dtype),倾向不搬、只留报错** |
| `paddle.manual_seed` (`device/cpu.py:110` 等 4 处) | **纯 Python 全局态**:`core.default_cpu_generator().manual_seed(seed)`(`:134`);CUDA/XPU/Custom 各一份(`device/cuda/__init__.py:785` 等) | **Lua 侧重写**:`manual_seed{n}` = C ABI 调对应设备生成器设种子;⚠️ `paddle.seed`(已抽查 framework/random.py:29)是**全部设备**遍历版,`manual_seed` 是**单设备**版,行为差异 `:112-113` docstring(warning 明示);名字照搬 |
| `paddle.linalg.eig` (`linalg.py:3251`) | 动态模式 `_C_ops.eig(x)`(`:3300`);返回 (w, v) 双值 | **直接绑定 + Lua 适配**(属 `paddle.linalg` 子模块,`linalg.md` 覆盖):`eig{x}` 返回**双值** → Lua 多返回值 `w, v`;**⚠️ 仅 CPU**(docstring `:3256-3257`);输出恒为 complex(docstring `:3257-3258`);梯度不支持?eig 有梯度但 eigvals 无(`:3320-3321` warning) |
| `paddle.linalg.eigvals` (`linalg.py:3318`) | 校验(方阵 `:3352-3358`)+ 动态 `_C_ops.eigvals(x)`(`:3364-3365`) | **直接绑定 + 零 Lua 适配**(属 `paddle.linalg`);`eigvals{x}` 只返回特征值;**⚠️ 无梯度 kernel**(docstring `:3320-3321` warning,需 P6 反向核对);方阵校验照抄 |
| `paddle.linalg.eigvalsh` (`linalg.py:4060`) | 动态 `_C_ops.eigvalsh(x, UPLO, x.stop_gradient)`(`:4079-4080`)丢弃 vectors 只返回 values;UPLO 校验(`:4089-4091`) | **直接绑定 + 一个 Lua 适配**(属 `paddle.linalg`);`eigvalsh{x, UPLO}`;⚠️ `UPLO` 枚举参数只收字符串 `'L'/'U'`(§2.1.1);对称/Hermitian 矩阵专用(**比 eig 快**,docstring `:3253-3254` 推荐);输出升序 |
| `paddle.linalg.lu` (`linalg.py:2915`) | 动态 `_C_ops.lu(x, pivot)`(`:3019-3020`);`get_infos` 决定返回二元/三元(`:3034-3037`) | **直接绑定 + Lua 适配**(属 `paddle.linalg`);`lu{x, pivot, get_infos}`;⚠️ **`get_infos=true` 返回三元 (lu, pivots, infos)**,否则二元——返回数量可变,Lua 侧用 `select('#')` 或固定三元+可空处理;`pivot` 换行主元开关 |
| `paddle.linalg.lu_unpack` (`linalg.py:3136`) | 校验(ndim `:3218-3225`)+ 动态 `_C_ops.lu_unpack(x, y, unpack_ludata, unpack_pivots)`(`:3226-3227`) | **直接绑定 + 零 Lua 适配**(属 `paddle.linalg`);`lu_unpack{x, y, unpack_ludata, unpack_pivots}` 返回 (P, L, U) 三元;⚠️ 与 `lu` 配套(已抽查),`y` 是 pivots int32 张量 |
| `paddle.linalg.svdvals` (`linalg.py:2423`) | **一行**:`return _C_ops.svdvals(x)`(`:2461-2462`) | **直接绑定**(属 `paddle.linalg`);`svdvals{x}` = `_ops.svdvals`;**只返回奇异值**(svd 的 U/S/Vt 中的 S);输出形状 `[..., min(M,N)]`(docstring `:2438`) |
| `paddle.linalg.pinv` (`linalg.py:3463`) | **纯 Python 组合**:非 hermitian 走 `svd`(`:3550`)+ max 奇异值 + where(cutoff 阈值 `:3554-3580`)+ `1/s` 截断 + `unsqueeze/transpose/matmul` 组合(`:3581-3590`);hermitian 走 `eigh` | **Lua 侧重写**(属 `paddle.linalg`):`pinv{x, rcond, hermitian, atol, rtol}` = `_ops.svd`+`_ops.max`+`_ops.where`+`_ops.matmul` 组合照抄;⚠️ **rcond 阈值语义**(`rcond * max_singular_val`,`:3579-3581`)与 atol/rtol 分支(docstring `:3496-3501` 三态)照抄;`pinverse = pinv` 顶层别名(已抽查 `__init__.py:1119`) |
| `paddle.tolist` (`manipulation.py:1447`) | **一行**:`return x.numpy(False).tolist()`(`:1476`)——Tensor 转嵌套 Lua table | **Lua 侧重写 + 关键设施**:`tolist{x}` = Tensor → 嵌套 Lua table;**⚠️ 只动图可用**(docstring `:1450-1451` Note);⚠️ 依赖 `x.numpy()`(DLPack 桥 → Insight,已抽查 from_dlpack/to_dlpack);返回深层嵌套 table(不是 `pl.List`,用户要裸表可写操作) |
| `paddle.pdist` (`nn/functional/distance.py:127`) | **纯 Python 组合**:`norm(x[...,None,:] - x[...,None,:,:], p=p, axis=-1)` + `~tril(ones)` mask + `masked_select`(`:161-164`) | **Lua 侧重写**(属 `paddle.nn.functional`,`nn-functional.md` 覆盖):`pdist{x, p}` = `_ops.norm`+`_ops.tril`+`_ops.masked_select` 组合照抄;⚠️ 输入 2-D 校验(`:160` assert);输出一维 `N(N-1)/2`(docstring `:132-133`) |
| `paddle.unstack` (`manipulation.py:743`) | 校验(axis 范围 `:770-772`、num 范围 `:773-775`)+ 动态 `_C_ops.unstack(x, axis, num)`(`:777`);num=None 取 `x.shape[axis]`(`:776`) | **直接绑定 + Lua 适配**:`unstack{x, axis, num}` = `_ops.unstack`;**`axis` 索引语义,1-based**(C4);**返回 `pl.List`**;num==0 → 空列表特判(`:778`);⚠️ 与 `stack` 互逆(已抽查) |
| `paddle.logspace` (`creation.py:560`) | dtype 默认 `get_default_dtype`(`:624`)+ 标量经 `fill_constant` 打包成 0-d 张量(`:634-647`)+ 动态 `_C_ops.logspace(start, stop, num, base, dtype, place, out=out)`(`:648-653`) | **直接绑定 + Lua 适配**:`logspace{start, stop, num, base, dtype}` = 标量打包(fill_constant)+ `_ops.logspace` 组合;⚠️ 参数 `device`/`requires_grad` PyTorch 风格(C12 不引入);⚠️ **无梯度**(docstring `:568` Notes) |
| `paddle.linalg.slogdet` (`linalg.py:2249`) | 动态模式一行 `_C_ops.slogdet(x)`(`:2270`) | **直接绑定**(属 `paddle.linalg`);`slogdet{x}`;**返回双值 (sign, logabsdet)** → Lua 多返回值;**det=0 → (0, -inf)**(docstring `:2261-2262`);复数输入 sign=det/abs(det)(`:2264-2265`);⚠️ 输出形状 `(2, *)`(docstring `:2272`) |
| `paddle.linalg.lstsq` (`linalg.py:4137`) | **device/driver 分派**:CPU 五驱动、CUDA 仅 gels(`:4196-4215`);rcond 默认 `eps*max(M,N)`(`:4248-4256`);动态 `_C_ops.lstsq(x, y, rcond, driver)`(`:4263-4264`);gels/gelsy 后置空 rank/singular_values(`:4265-4269`) | **直接绑定 + Lua 适配**(属 `paddle.linalg`);`lstsq{x, y, rcond, driver}`;**返回四元 (solution, residuals, rank, singular_values)** → Lua 4 返回值;driver 枚举只收字符串(§2.1.1);rcond 默认计算照抄;⚠️ driver 影响返回语义(gels 无 rank) |
| `paddle.block_diag` (`manipulation.py:8980/8984/8988`) | **纯 Python 组合**:逐元素 `to_2d` 升维(0-d/1-d → 2-d,`:9038-9047`)+ 列块构造(非对位补 `zeros`,`:9028-9036`)+ `concat(axis=0)` 拼行 + `concat(axis=1)` 拼列(`:9053-9059`) | **Lua 侧重写**:`block_diag{inputs}` = 升维 + zeros 补块 + `_ops.concat` 组合照抄;**`inputs` 用 `TensorList`**;⚠️ 每元素维度必须 0/1/2(`:9044-9046` 校验);输出 `sum(rows) x sum(cols)` |
| `paddle.addmm`/`baddbmm` (`base/dygraph/generated_tensor_methods_patch.py:81/87`) | **pybind 算子包装**:`def _addmm: return _C_ops.addmm(...)`(`:81-82`)、`_baddbmm`(`:87-88`)——同 `kron` 模式(已抽查) | **直接绑定**(P3 核实 yaml;若否手工绑定 `_C_ops.*` 同款);`addmm{input, x, y, beta, alpha}`(公式 `beta*input + alpha*x@y`)/`baddbmm{x, y, beta, alpha}`(批外积加);⚠️ `_paddle_docs.py:4705/4774` 是文档 stub 不是实现;sparse/multiary.py:28 是稀疏版不在此列 |
| `paddle.linalg.cholesky` (`base/dygraph/generated_tensor_methods_patch.py:159`) | **pybind 算子包装**:`def _cholesky: return _C_ops.cholesky(...)`(`:159-160`) | **直接绑定**(属 `paddle.linalg`;P3 核实 yaml);`cholesky{x, upper=false}` 下三角分解;⚠️ 输出下三角(**upper=false** 默认,`_paddle_docs.py:5661` 文档 stub);⚠️ 输入须正定(docstring 注明),非正定返回 nan 不报错 |
| `paddle.linalg.lu_solve` (`linalg.py:3044`) | 校验(b 与 lu 行数/方阵/pivots 长度 `:3075-3100`)+ batch 广播(`:3101-3111`)+ `pivots.stop_gradient = True`(`:3130`)+ 动态 `_C_ops.lu_solve(b, lu, pivots, trans)`(`:3131-3132`) | **直接绑定 + Lua 适配**(属 `paddle.linalg`);`lu_solve{b, lu, pivots, trans}` = 广播 + `_ops.lu_solve` 组合;`trans` 枚举只收 `'N'/'T'/'C'`(§2.1.1);⚠️ `trans` 非 'N' 一律转 'T'(`:3112` 简化),Lua 照抄;与 `lu`/`lu_unpack` 配套(已抽查) |
| `paddle.hypot` (`math.py:6633`) | **纯 Python 组合**:`(pow(x,2) + pow(y,2)).sqrt()`(`:6675-6678`)——斜边 √(x²+y²) | **Lua 侧重写**:`hypot{x, y}` = `_ops.pow`+`_ops.add`+`_ops.sqrt` 组合照抄;⚠️ 无 `_C_ops.hypot` 算子;dtype 白名单(两种输入须同 dtype,docstring `:6639-6640`)照抄;别名 `input`/`other` 不移植 |
| `paddle.get_flags` (`base/framework.py:163`) | **纯 Python 全局态**:遍历 list/tuple 校验 `_global_flags().is_public(key)` 后读取(`:183-192`)——`set_flags`(已抽查)的反向读取 | **Lua 侧重写**:`get_flags{flags}` = C ABI 遍历读 GFlags;⚠️ 返回 Lua table(key→value,值类型 string/number/boolean 取决于 flag);与 `set_flags` 同族(已抽查 `:133`) |
| `paddle.get_cuda_rng_state` (`framework/random.py:128`) | **纯 Python 全局态**:有 CUDA 时遍历 `default_cuda_generator(i).get_state()`(`:143-151`)——`set_cuda_rng_state`(已抽查 `:222`)的 getter | **Lua 侧重写**:`get_cuda_rng_state{}` = 遍历 CUDA 生成器 get_state;⚠️ 返回 `pl.List`(GeneratorState 对象序列);与 `set_cuda_rng_state`/`get_rng_state`(已抽查)同族 |
| `paddle.histogram_bin_edges` (`linalg.py:2192`) | **纯 Python 组合**:min/max 双 0 → 取数据极值(`:2234-2236`)+ min==max → 扩 0.5(`:2239-2241`)+ `linspace(min, max, bins+1)`(`:2243`) | **Lua 侧重写**(属 `paddle.linalg`):`histogram_bin_edges{input, bins, min, max}` = 极值推断 + `_ops.linspace` 组合照抄;**返回 bins+1 个边界值**;dtype 白名单 float32/64/int32/64(`:2228-2232`)照抄 |
| `paddle.histogramdd` (`linalg.py:5047`) | **纯 Python 组合**:reshape 展平(`:5191-5192`)+ 逐维 `linspace`/Tensor bins(`:5217-5242`)+ `searchsorted` 映射 + `bincount`(`:5261-5276`)+ density 归一(`:5282-5287`) | **Lua 侧重写**(属 `paddle.linalg`):`histogramdd{x, bins, ranges, density, weights}` 全组合照抄;**返回双值 (hist, bin_edges)** → Lua 多返回值;**⚠️ 依赖 `bincount`**(DIRECT 桶)与 `searchsorted`(已抽查);3 组校验函数(`__check_x/__check_bins/__check_weights` `:5111-5166`)照抄 |
| `paddle.index_add_` (`manipulation.py:8054`) | **一行绑定**:`scaled_value = value * alpha if alpha != 1 else value` + `_C_ops.index_add_(x, index, scaled_value, axis)`(`:8063-8064`) | **直接绑定 + 一个 Lua 适配**:`index_add_{x, index, axis, value, alpha}` = `_ops.index_add_`;**`index`/`axis` 索引语义,1-based**(C4);⚠️ in-place 变体(`_` 后缀)只动图可用,`value*alpha` 特判照抄 |
| `paddle.index_fill_` (`manipulation.py:8646` → `_index_fill_impl` `:8498`) | **转发** `_index_fill_impl(x, index, axis, value, True)`(`:8652`);impl 校验(index 1-D `:8520-8522`、axis 范围 `:8513-8517`)+ `_C_ops.index_fill_`(`:8528-8529`) | **直接绑定 + Lua 适配**:`index_fill_{x, index, axis, value}`;**`index`/`axis` 索引语义,1-based**(C4);⚠️ 0-size index 特判返回拷贝(`:8526-8527`);value 非 Tensor → 0-d 转换(`:8507-8510`) |
| `paddle.ger` (`__init__.py:1096`) | **纯别名**:`ger = outer`(`__init__.py:1096`)——numpy 名,就是 `outer` | **直接绑定 + 零 Lua 适配**:与 `outer{x, y}`(已抽查 math.py:2654)同一绑定;名字照搬(numpy 别名忠实迁移);⚠️ 与 `outer` 的 reshape+multiply 组合同源 |
| `paddle.movedim` (`__init__.py:1110` + `tensor/__init__.py:528`) | **纯别名**:`movedim = moveaxis`(两处 `__init__.py:1110`/`tensor/__init__.py:528`)——PyTorch 名 | **直接绑定 + 零 Lua 适配**:与 `moveaxis{x, source, destination}`(已抽查 manipulation.py:7103)同一绑定;**⚠️ 这是 PyTorch 名**(C12 命名规则针对 `requires_grad`/`device` 等语义参数,别名本身忠实迁移——上游两者并存,照搬不合并);source/destination 索引语义 1-based |
| `paddle.disable_static`/`enable_static` (`base/dygraph/base.py:236/280`) | **纯 Python 全局态**:`disable_static` 置 `_functional_dygraph_context_manager = guard(place)` 并 `__enter__`(`:259-263`);`enable_static` 置 None(`:304` 附近);`in_dynamic_mode()` 读该态 | **Lua 侧重写 + 放放**:⚠️ 静态图入口已收敛到 `paddle.jit`(api/README §2 明确不做 `paddle.static`)——**这两个开关在 Lua 侧无对应物**;动图是唯一模式,不需要切换;记入 §7 明确不提供 |
| `paddle.logcumsumexp` (`math.py:3778`) | 动态模式一行 `_C_ops.logcumsumexp(x, axis, flatten, False, False, out=out)`(`:3848-3850`);`axis=None` → flatten 语义(`:3841-3846`);dtype 预转换(`:3847-3848`) | **直接绑定 + 一个 Lua 适配**:`logcumsumexp{x, axis, dtype}`;`axis=None` → flatten 特判照抄;`axis` 标 1-based;整数输入 autocast float32(docstring `:3792`);dtype 预转调 `_ops.cast` |
| `paddle.rad2deg` (`math.py:4968`) | **组合**:int32/64 先 `cast(x, 'float32')`(`:5009-5010`)+ `_C_ops.scale(x, 180/pi, 0.0, True, out=out)`(`:5011-5012`)——就是 scale | **直接绑定 + 一个 Lua 适配**:`rad2deg{x}` = int 先 cast + `_ops.scale(180/π)` 组合;⚠️ 与 `deg2rad`(已抽查)对称:deg2rad 用 scale,rad2deg 也是 scale(180/π);整数输入自动转 float32(docstring `:4978-4979`) |
| `paddle.unbind` (`manipulation.py:4379`) | 校验(axis 类型 `:4417-4420`、范围 `:4421-4423`)+ 动态 `_C_ops.unbind(input, axis)`(`:4426`);静态模式按 num 拆(`:4428-4449`) | **直接绑定 + Lua 适配**:`unbind{x, axis}` = `_ops.unbind`;**`axis` 索引语义,1-based**(C4);**返回 `pl.List`**(元素数 = x.shape[axis]);⚠️ 与 `stack` 互逆(已抽查);别名 `dim` 不移植 |
| `paddle.view_as_complex`/`view_as_real` (`manipulation.py:6913/6955`) | **一行转发**:`return as_complex(x=input)`(`:6952`)`/ return as_real(x=input)`(`:6988`)——接口文档独立、实现复用 as_complex/as_real | **直接绑定 + 零 Lua 适配**:`view_as_complex{x}` = `_ops.as_complex`(DIRECT 桶,已抽查 sgn 时提过)、`view_as_real{x}` = `_ops.as_real`;**⚠️ view 语义共享内存**(docstring `:6919`/`:6963` 明示);形状 `(*,2)` ↔ `(*)` |
| `paddle.as_strided` (`manipulation.py:8247`) | **一行绑定**:`_C_ops.as_strided(x, shape, stride, offset)`(`:8297`) | **直接绑定**:`as_strided{x, shape, stride, offset}`;**⚠️ view 语义共享内存**(docstring `:8256-8258`),⚠️ 需 `FLAGS_use_stride_kernel`(`:8261`)——Lua 侧同依赖引擎 flag;`shape`/`stride` 用 `IntList`;别名 `size`/`storage_offset` 不移植 |
| `paddle.cummax`/`cummin` (`math.py:3543/3661`) | `axis=None` → flatten + axis=-1(`:3624-3625`);动态 `_C_ops.cummax(x, axis, dtype)`(`:3633-3634`)/`_C_ops.cummin`(`:3752-3753`);返回 (values, indices) 双值;dtype 白名单 int32/64(`:3627-3629`) | **直接绑定 + Lua 适配**:`cummax{x, axis, dtype}`/`cummin{x, axis, dtype}`;**返回双值 (values, indices)** → Lua 多返回值;**indices 是索引语义,1-based**(C4);`axis=None` → flatten 特判照抄;⚠️ `dtype` 默认 'int64' 控制 indices 类型;out 参数暂不做(Q-19) |
| `paddle.histogram` (`linalg.py:2110`) | 动态模式一行 `_C_ops.histogram(input, weight, bins, min, max, density)`(`:2154`);min/max int→float 归一(`:2150-2151`) | **直接绑定**:`histogram{input, bins, min, max, weight, density}`;**输出 int64 计数**(docstring `:2122`);⚠️ 与 `histc`(已抽查,输出 float32)是不同入口——histc 转发 histogram 后 astype;`min/max` 双 0 → 数据极值(与 histc 同);`density` 归一语义(`:2121-2123`) |
| `paddle.grad` (`base/dygraph/base.py:706`) | **纯 Python**:计算 outputs 对 inputs 的梯度之和;仅动图可用(docstring `:715-718`);`only_inputs`/`create_graph`/`retain_graph` 等反向选项 | **直接绑定 + Lua 适配**(autograd.md 已覆盖 `grad` 契约):`grad{outputs, inputs, grad_outputs, retain_graph, create_graph, only_inputs, allow_unused}`;⚠️ 底层 `_C_ops` 反向引擎(P6 `backward` 同源);`allow_unused=False` 时无梯度输入报错 |
| `paddle.erf`/`erfc`/`expm1`/`i0`/`i1`/`nextafter` (`base/dygraph/generated_tensor_methods_patch.py:192/204/237/243/327`) | **pybind 算子包装**:`def _erf: return _C_ops.erf(...)`(`:192-193`)、`_expm1`(`:204-205`)、`_i0`(`:237-238`)、`_i1`(`:243-244`)、`_nextafter`(`:327-328`)——同 `kron` 模式(已抽查) | **直接绑定**(P3 核实 yaml;若否手工绑定 `_C_ops.*` 同款);`erf{x}` 误差函数/`erfc{x}` 余误差函数/`expm1{x}`/`i0{x}` 零阶贝塞尔/`i1{x}` 一阶贝塞尔/`nextafter{x, y}` 相邻浮点;⚠️ `_paddle_docs.py:1354/4643/3643` 等是文档 stub 不是实现 |
| `paddle.fft` 族 (`fft.py:173/235/308/1467/1614`) | **独立子模块 `paddle.fft`**(非 tensor 目录):`fft`/`ifft`(`:173/235`)、`rfft`(`:308`)、`fftfreq`(`:1467`)、`fftshift`(`:1614`);纯 Python 封装 + `_C_ops.fft_*` | **放放/子模块**(不属于顶层命名空间,是 `paddle.fft` 子模块):⚠️ 顶层 `__init__.py` 有装配;fft 属信号处理,训练路径几乎不用——**P17 前后再定**;记 OPEN_QUESTION 或 §7 延后;`norm` 枚举('forward'/'backward'/'ortho')只收字符串 |
| `paddle.eq`/`ne`/`lt`/`less`/`le`/`ge` (`tensor/__init__.py:518-523`) | **纯别名族**:`eq = equal`、`ne = not_equal`、`lt = less_than`、`less = less_than`、`le = less_equal`、`ge = greater_equal`(`:518-523`)——numpy 风格比较运算符名 | **直接绑定 + 零 Lua 适配**:六个名字照搬,各对应 `_ops.equal/not_equal/less_than/less_equal/greater_equal`(已抽查逻辑族);⚠️ 与 `==`/`~=`/`<`/`<=`/`>=` 运算符同底层;**与 Python 的 `paddle.eq(x,y)` 逐字对应**;`gt` 别名在 `logic.py` 内(`tensor/__init__.py` 未见,归逻辑族) |
| `paddle.neg`/`negative` (`math.py:4576/4663`) | `neg` = `scale(x, -1.0, bias=0.0, bias_after_scale=True)`(`:4606-4613`);`negative` = bool 检查 + `-x`(`:4692-4693`)——一元取负 | **直接绑定 + 零 Lua 适配**:`neg{x}`/`negative{x}` 同一绑定(`_ops.scale(-1)`);⚠️ bool 输入报错(`:4693`);与 `x:neg()` 方法同源;`-x` 运算符(Lua 无一元 `-` 的取负元方法语义——C 侧挂 `__unm`) |
| `paddle.get_rng_state`/`set_rng_state` (`framework/random.py:73/155`) | **纯 Python 全局态**:`get_rng_state` 按 place 遍历对应设备生成器 `get_state`(`:100-114`);`set_rng_state` 反向遍历 `set_state`(`:170-200`)——`seed`/`manual_seed` 的成对序列化接口 | **Lua 侧重写**:`get_rng_state{device}`/`set_rng_state{state_list, device}` = C ABI 遍历设备生成器 get/set_state;**返回 `pl.List`**(GeneratorState 对象);⚠️ 与 `get_cuda_rng_state`(已抽查 `:128`)同族;device 枚举('cpu'/'gpu'/'xpu' 等,`:159-161`)只收字符串 |
| `paddle.index_put_` (`base/dygraph/generated_tensor_methods_patch.py:252`) | **pybind 算子包装**:`def _index_put_: return _C_ops.index_put_(...)`(`:252-253`);`index_put`(`:249-250`)同款 | **直接绑定**(P3 核实 yaml;若否手工绑定 `_C_ops.*` 同款);`index_put_{x, indices, value, accumulate}`;**indices 是索引语义,1-based**(C4);⚠️ `_paddle_docs.py:1773/1823` 是文档 stub 不是实现;`accumulate`(加还是覆盖)语义照抄 |

**抽样结论:创建类在 Python 侧是"装饰器 + 一行转发",真实现全在 `_C_ops`/`_to_tensor_non_static` 里。** 名字按命名铁律处理(一般忠实迁移、numpy 特例换 Insight);同一底层可以供多个入口共用。Lua 侧每个导出函数/方法都有一张 `argrule` 规则表,签名与 Python 规范签名逐字对齐(`plan/api/README.md` §2.1);PyTorch 兼容装饰器相关参数(别名、`size_args`、`variadic`)不移植(D35)。

| 名字 |
|---|
| `BFloat16Tensor` |
| `BoolTensor` |
| `ByteTensor` |
| `CPUPlace` |
| `CUDAPinnedPlace` |
| `CUDAPlace` |
| `CharTensor` |
| `DoubleTensor` |
| `FloatTensor` |
| `Generator` |
| `HalfTensor` |
| `IntTensor` |
| `LongTensor` |
| `ParamAttr` |
| `ShortTensor` |
| `Size` |
| `Tensor` |
| `XPUPinnedPlace` |
| `abs_` |
| `acos_` |
| `adaptive_avg_pool1d` |
| `add` |
| `add_n` |
| `addcdiv` |
| `addcdiv_` |
| `addmm_` |
| `addmv` |
| `addmv_` |
| `addr` |
| `addr_` |
| `argwhere` |
| `as_tensor` |
| `asarray` |
| `atan_` |
| `atleast_1d` |
| `atleast_2d` |
| `atleast_3d` |
| `autocast` |
| `baddbmm_` |
| `batch` |
| `bernoulli_` |
| `bfloat16` |
| `bitwise_and_` |
| `bitwise_invert` |
| `bitwise_invert_` |
| `bitwise_left_shift_` |
| `bitwise_not_` |
| `bitwise_or_` |
| `bitwise_right_shift_` |
| `bitwise_xor_` |
| `block_diag` |
| `bool` |
| `broadcast_shape` |
| `broadcast_shapes` |
| `broadcast_to` |
| `bucketize` |
| `cartesian_prod` |
| `cast_` |
| `cat` |
| `cauchy_` |
| `cdist` |
| `cdouble` |
| `cfloat` |
| `check_shape` |
| `chunk` |
| `clamp` |
| `clamp_` |
| `clamp_max` |
| `clamp_min` |
| `clip_` |
| `clone` |
| `column_stack` |
| `combinations` |
| `complex128` |
| `complex64` |
| `concatenate` |
| `conv1d` |
| `copysign_` |
| `cos_` |
| `count_nonzero` |
| `create_parameter` |
| `cumprod_` |
| `cumsum_` |
| `cumulative_trapezoid` |
| `deg2rad` |
| `diagflat` |
| `diagonal_scatter` |
| `diff` |
| `digamma_` |
| `disable_signal_handler` |
| `disable_static` |
| `div` |
| `div_` |
| `divide` |
| `divide_` |
| `double` |
| `dsplit` |
| `dstack` |
| `dtype` |
| `e` |
| `enable_grad` |
| `enable_static` |
| `eq` |
| `equal_` |
| `erf_` |
| `expand_copy` |
| `expm1_` |
| `finfo` |
| `fix` |
| `fix_` |
| `flatten_` |
| `float` |
| `float16` |
| `float32` |
| `float64` |
| `float8_e4m3fn` |
| `float8_e5m2` |
| `floor_divide_` |
| `floor_mod` |
| `floor_mod_` |
| `fmod` |
| `fmod_` |
| `frac` |
| `frac_` |
| `frexp` |
| `from_dlpack` |
| `from_numpy` |
| `gammainc` |
| `gammainc_` |
| `gammaincc_` |
| `gammaln_` |
| `gcd` |
| `gcd_` |
| `ge` |
| `geometric_` |
| `ger` |
| `get_autocast_cpu_dtype` |
| `get_autocast_dtype` |
| `get_autocast_gpu_dtype` |
| `get_cuda_rng_state` |
| `get_default_dtype` |
| `get_flags` |
| `get_rng_state` |
| `grad` |
| `greater_equal_` |
| `greater_than_` |
| `gt` |
| `half` |
| `histc` |
| `histogram_bin_edges` |
| `histogramdd` |
| `hsplit` |
| `hstack` |
| `hypot` |
| `hypot_` |
| `i0_` |
| `iinfo` |
| `in_dynamic_mode` |
| `index_add_` |
| `index_fill_` |
| `index_put_` |
| `inf` |
| `initial_seed` |
| `inner` |
| `int16` |
| `int32` |
| `int64` |
| `int8` |
| `is_autocast_enabled` |
| `is_complex` |
| `is_floating_point` |
| `is_grad_enabled` |
| `is_integer` |
| `is_tensor` |
| `isin` |
| `isneginf` |
| `isposinf` |
| `isreal` |
| `lcm` |
| `lcm_` |
| `ldexp` |
| `ldexp_` |
| `le` |
| `less` |
| `less_` |
| `less_equal_` |
| `less_than_` |
| `lgamma_` |
| `load` |
| `log10_` |
| `log2_` |
| `log_` |
| `log_normal` |
| `log_normal_` |
| `logaddexp` |
| `logdet` |
| `logical_and_` |
| `logical_not_` |
| `logical_or_` |
| `logit_` |
| `lt` |
| `manual_seed` |
| `masked_fill_` |
| `masked_scatter_` |
| `matrix_transpose` |
| `mm` |
| `mod` |
| `mod_` |
| `moveaxis` |
| `movedim` |
| `msort` |
| `mul` |
| `multigammaln` |
| `multigammaln_` |
| `multiply` |
| `multiply_` |
| `mvlgamma` |
| `mvlgamma_` |
| `nan` |
| `nan_to_num` |
| `nan_to_num_` |
| `nanmean` |
| `nanquantile` |
| `narrow` |
| `ne` |
| `neg` |
| `neg_` |
| `negative` |
| `negative_` |
| `newaxis` |
| `no_grad` |
| `normal` |
| `normal_` |
| `outer` |
| `pdist` |
| `permute` |
| `pi` |
| `pinverse` |
| `polar` |
| `polygamma_` |
| `positive` |
| `pow_` |
| `pstring` |
| `put_along_axis_` |
| `quantile` |
| `rad2deg` |
| `rand` |
| `rand_like` |
| `randint_like` |
| `randn` |
| `randn_like` |
| `range` |
| `rank` |
| `ravel` |
| `raw` |
| `remainder_` |
| `renorm_` |
| `reshape_` |
| `resize_as_` |
| `rot90` |
| `row_stack` |
| `save` |
| `scatter_` |
| `scatter_add` |
| `scatter_add_` |
| `scatter_nd` |
| `scatter_reduce` |
| `scatter_reduce_` |
| `seed` |
| `select_scatter` |
| `set_cuda_rng_state` |
| `set_default_dtype` |
| `set_default_tensor_type` |
| `set_flags` |
| `set_grad_enabled` |
| `set_printoptions` |
| `set_rng_state` |
| `sgn` |
| `signbit` |
| `sin_` |
| `sinc` |
| `sinc_` |
| `sinh_` |
| `slice_scatter` |
| `sort` |
| `square_` |
| `squeeze_` |
| `standard_normal` |
| `sub` |
| `sub_` |
| `subtract` |
| `subtract_` |
| `swapaxes` |
| `swapdims` |
| `t` |
| `t_` |
| `take` |
| `take_along_dim` |
| `tan_` |
| `tanh_` |
| `tensor_split` |
| `tensordot` |
| `tile` |
| `to_dlpack` |
| `to_tensor` |
| `tolist` |
| `transpose_` |
| `trapezoid` |
| `tril_` |
| `triu_` |
| `true_divide` |
| `true_divide_` |
| `trunc_` |
| `uint16` |
| `uint32` |
| `uint64` |
| `uint8` |
| `unflatten` |
| `unsqueeze_` |
| `vander` |
| `vecdot` |
| `view` |
| `view_as` |
| `view_as_complex` |
| `view_as_real` |
| `vsplit` |
| `vstack` |
| `where_` |

## 4. SPECIAL(明确跳过)

| 名字 | 理由 |
|---|---|
| `DataParallel` | 见 §0 跳过表 |
| `LazyGuard` | 见 §0 跳过表 |
| `MmapStorage` | 见 §0 跳过表 |
| `Model` | 见 §0 跳过表 |
| `disable_compat` | 见 §0 跳过表 |
| `enable_compat` | 见 §0 跳过表 |
| `flops` | 见 §0 跳过表 |
| `inference_mode` | 见 §0 跳过表 |
| `summary` | 见 §0 跳过表 |
| `use_compat_guard` | 见 §0 跳过表 |
