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
