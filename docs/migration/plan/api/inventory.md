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
