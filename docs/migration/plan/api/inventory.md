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
