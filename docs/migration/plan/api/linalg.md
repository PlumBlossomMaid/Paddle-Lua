# paddle.linalg

| | |
|---|---|
| 阶段 | P5 |
| 依赖模块 | `paddle`(顶层)、`paddle.Tensor` |
| 稳定性 | **稳定**(核心分解/范数)/ 实验(近似算法 svd_lowrank 等) |
| 实现文档 | `plan/modules/03-codegen.md`(算子生成)、`plan/modules/05-tensor.md` |
| 上游来源 | `python/paddle/tensor/linalg.py`(实体);顶层 `paddle.linalg` = 该模块再导出(`paddle/__init__.py:278` `linalg as linalg`) |

> 📌 `paddle.linalg` 是**顶层函数的子集再导出 + 少量模块专属函数**。
> 大部分成员与顶层同名函数共用同一 `_C_ops` 绑定(如 `matmul`/`qr`/`norm`),
> 名字照搬、不合并(`api/README.md` §2.1 命名铁律)。**注意:`python/paddle/linalg`
> 目录不存在**,`paddle.linalg` 是 `tensor/linalg.py` 的模块对象 —— 这是上游
> 的既有结构,不是我们发明的。

---

## 1. 导出清单

### 1.1 `_C_ops` 纯 re-export(直接绑定)

上游 `tensor/linalg.py:27-36` 的 `from paddle._C_ops import (...)` 块:

| 导出 | 备注 |
|---|---|
| `paddle.linalg.matmul` | LEGACY 桶算子,`x, y, transpose_x, transpose_y` |
| `paddle.linalg.bmm` / `dot` / `mv` / `cross` | 基础矩阵/向量运算 |
| `paddle.linalg.cholesky` | `upper` 参数 |
| `paddle.linalg.det` / `diagonal` / `dist` / `eigh` | 分解与提取 |
| `paddle.linalg.bincount` | 计数 |

### 1.2 有 Python 实体的成员(逐批标注)

`tensor/linalg.py` 的 `def` 清单(58 个 def,含 @overload 变体):

| 导出 | 去向 | 备注 |
|---|---|---|
| `paddle.linalg.norm` (`:1091`) | **Lua 侧重写(分发表)** | 分发给 vector_norm/matrix_norm(`:1238-1271`),`p` 支持 fro/nuc/0/±1/±2/±inf |
| `paddle.linalg.vector_norm` (`:404`) | 直接绑定 | `_C_ops.vector_norm` 族 |
| `paddle.linalg.matrix_norm` (`:663`) | 直接绑定 | 同上 |
| `paddle.linalg.qr` (`:2795-2824`) | 直接绑定 + Lua 适配 | @overload 双签名(mode/some,D35 不移植);`_C_ops.qr(x, mode)`(`:2888`);返回 (Q, R) 双值;**backward 不支持**(`docstring :2828`) |
| `paddle.linalg.svd` (`:2337`) | 直接绑定 | `_C_ops.svd(x, full_matrices, out=out)`(`:2402`);返回 (U, S, VH) 三值;⚠️ 不在顶层命名空间,只有 `paddle.linalg.svd` |
| `paddle.linalg.svdvals` (`:2423`) | 直接绑定 | `_C_ops.svdvals(x)`(`:2456`) |
| `paddle.linalg.slogdet` (`:2249`) | 直接绑定 | `_C_ops.slogdet(x)`(`:2288`);返回 (sign, logabsdet) 双值 |
| `paddle.linalg.logdet` (`:2318`) | 直接绑定 | `_C_ops.logdet` |
| `paddle.linalg.lstsq` (`:4137`) | 直接绑定 | `_C_ops.lstsq(...)`(`:4265`);返回 (solution, residuals, rank, singular_values) 多值 |
| `paddle.linalg.lu` / `lu_solve` / `lu_unpack` (`:2915/3044/3136`) | 直接绑定 | 分解族 |
| `paddle.linalg.eig` (`:3251`) | 直接绑定 | 返回 (values, vectors) 双值 |
| `paddle.linalg.inv` / `pinv` / `matrix_power` / `matrix_rank` (`:2707/1938`) | 直接绑定 | 逆/伪逆/幂/秩 |
| `paddle.linalg.cond` (`:1275`) | 直接绑定 | 条件数 |
| `paddle.linalg.vecdot` (`:1667`) | 直接绑定 | 向量点积(广播) |
| `paddle.linalg.cov` (`:1703`) | 直接绑定 | 协方差 |
| `paddle.linalg.transpose` / `permute` / `matrix_transpose` / `t` (`:79/241/270/1830`) | 直接绑定 | 转置族;**perm 是索引语义,1-based**;⚠️ `transpose` 的别名装饰器(`transpose_decorator`)D35 不移植 |
| `paddle.linalg.histogram` / `histogram_bin_edges` (`:2110/2192`) | 直接绑定 | 直方图(顶层也导出,共用绑定) |
| `paddle.linalg.svd_lowrank` / `pca_lowrank` (`:2500/2622`) | **Lua 侧重写(组合)** | 近似算法 = svd + 若干组合;P5 延后(实验) |

---

## 2. 详细签名

### 2.1 分解族(返回多值)

```lua
local u, s, vh = paddle.linalg.svd{x, full_matrices=false}
local q, r     = paddle.linalg.qr{x, mode="reduced"}     -- mode="r" 只返回 r
local sign, la = paddle.linalg.slogdet{x}
local w, v     = paddle.linalg.eig{x}
local sol, res, rank, sv = paddle.linalg.lstsq{x, y, ...}
```

**返回形态随函数定**,Lua 侧按上游返回值个数直接返回多值(C++ 侧已是
多 Tensor 返回,见 `01-c-abi.md` 的 multi-return 约定)。

### 2.2 范数族

```lua
paddle.linalg.norm{x, p=nil, axis=nil, keepdim=false}
paddle.linalg.vector_norm{x, p=2, axis=nil, keepdim=false}
paddle.linalg.matrix_norm{x, p="fro", axis={1,2}, keepdim=false}
```

`norm` 的分发表 Lua 侧重写(`linalg.py:1238-1271`):`p` 为 'fro'/int/float,
`axis` 形态决定向量/矩阵范数。⚠️ `axis` 是索引语义,1-based(§3)。

### 2.3 转置族

```lua
paddle.linalg.transpose{x, perm={2, 1}}     -- perm 1-based
paddle.linalg.t{x}                          -- 2-D 转置
paddle.linalg.permute{x, dims={2, 1}}       -- ⚠️ 参数名 dims(PyTorch),见 §6
```

---

## 3. index 语义参数

| 参数 | 语义 | 基准 |
|---|---|---|
| `transpose.perm` / `permute.dims` | 维度排列 | **1-based**(逐元素 -1) |
| `norm.axis` / `vector_norm.axis` / `matrix_norm.axis` | 维度索引 | **1-based** |
| `histogram` 的 bins 边界 | 数值区间(非索引) | 不标 |
| `diagonal.offset` / `tril.k` / `triu.k` | **偏移量,不是索引** | 不标 1-based(§2.2 例外) |

**没有 index 参数的成员**:`matmul/bmm/dot/mv/cross/cholesky/det/eigh/svd/
slogdet/logdet/lstsq/lu/inv/pinv/matrix_power/matrix_rank/cond/vecdot/cov`
—— 均无维度或张量索引参数(显式列出,证明"看过")。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.linalg.svd(x)` 返回 tuple | Lua 返回 **u, s, vh 三值** | 无 namedtuple,直接多返回 |
| 2 | `paddle.linalg.qr(x, mode='reduced')` 与 `qr(input, some=True)` 双签名 | **只有 `qr{x, mode}`** | @overload(PyTorch some 签名)是 D35 兼容糖不移植 |
| 3 | `paddle.linalg.transpose(x, perm)` | `transpose{x, perm}` | 一张参数表 |
| 4 | `paddle.linalg.permute(input, dims)` | **⚠️ 参数名待定** | `dims` 是 PyTorch 名;倾向 Paddle 规范 `perm`——P3 拍板 |
| 5 | `paddle.linalg.norm` axis 从 0 | **1-based** | D-R6 |
| 6 | `svd_lowrank`/`pca_lowrank` 常用参数 `niter/q` | 照搬 | 实验性 API,按需 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"
local la = paddle.linalg

local x = paddle.to_tensor{{1, 2}, {3, 4}, {5, 6}}

-- SVD(三返回值)
local u, s, vh = la.svd{x}

-- QR(双返回值;mode="r" 只返回 r)
local q, r = la.qr{x}

-- 范数(axis 1-based)
local n = la.norm{x, p="fro", axis={1, 2}}

-- 转置(perm 1-based)
local t = la.transpose{x, perm={2, 1}}
```

---

## 6. 本模块特有的坑

**① `paddle.linalg` 不是独立目录,是模块对象。** 上游 `python/paddle/linalg`
目录不存在(`glob` 已核实),`paddle.linalg` 由 `paddle/__init__.py:278`
`linalg as linalg` 把 `tensor/linalg.py` 装成子模块。Lua 侧 `paddle.linalg`
是 `require "paddle"` 时惰性挂载的表(`api/README.md` §2.5)。

**② `qr` 的 backward 不支持。** docstring 明示(`linalg.py:2828`)。
Lua 侧绑定层 doc 必须注明,否则用户对 `qr` 结果 `backward()` 会得到静默错。
建议:绑定层在 `qr` 的规则表里加 `stop_gradient` 提示。

**③ `svd`/`lstsq`/`slogdet` 只在 `linalg` 子模块,不在顶层。** 这是与
`norm`/`qr` 相反的情形(那俩顶层也导出,`__init__.py:453/456`)。
Lua 侧按名字照搬:`paddle.linalg.svd` 只存在于 `linalg` 表下,顶层没有
`paddle.svd` —— 用户从 Python 文档抄代码时不会踩,但 Lua 用户写
`paddle.svd` 会得到 nil,教学式报错 `did you mean: paddle.linalg.svd?`。

**④ `permute` 的参数名是 `dims`(PyTorch)。** `transpose_decorator` 把
`input/dims` 别名到 `x/perm`(`linalg.py:241`)——D35 不移植。Lua 侧倾向
`permute{x, perm}`(Paddle 规范名),P3 拍板;若忠实照搬 `dims` 则违反 C12
的参数名铁律,二选一必须明确。

**⑤ 近似算法(`svd_lowrank`/`pca_lowrank`)是组合不是算子。** 上游是
svd + 若干中间运算的组合(`linalg.py:2500-2622`)。Lua 侧要么整组合重写,
要么延后(P5 不做,实验 API)。**倾向:延后**,M1 训练不需要。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `svd_lowrank` / `pca_lowrank` | **延后**(实验,组合重写成本高) |
| `permute` 参数名(dims vs perm) | P3 拍板(C12) |
| `eigvalsh` / `eigvals` | 随 eigh 一起(P5) |
| `cholesky_solve` / `solve` | P5 按需补 |
| `tensordot` | P5 按需补 |
| `diagonal` 的 offset 语义 | 偏移量,非索引;doc 注明即可 |
