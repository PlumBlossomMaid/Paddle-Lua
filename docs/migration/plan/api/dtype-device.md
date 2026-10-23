# paddle.dtype / paddle.device

| | |
|---|---|
| 阶段 | P5 |
| 依赖模块 | `paddle`(顶层)、Insight7(**dtype/place 是两边共有类型**,`foundations.md` §3.6.1) |
| 稳定性 | **稳定**(dtype 常量、place 构造器、默认 dtype)/ 实验(device 管理函数逐批) |
| 实现文档 | `plan/modules/01-c-abi.md`(dtype 映射)、`plan/modules/05-tensor.md` |

> 📌 本模块回答两件事:`paddle.float32` 这类 dtype 常量怎么定义,
> `paddle.CPUPlace()` 这类 place 怎么构造。**关键事实:Insight7 的
> dtype 常量与 Place 构造器已经是 Paddle 命名**(`foundations.md` §3.2 已核实),
> 两边是**同一个概念不是撞名**(§3.6.1)——Lua 用户 `local float32 = paddle.float32`
> 与 `local float32 = ins.float32` 拿到同一套命名,`argrule` 用 `extend`
> 把两边谓词取或。

---

## 1. 导出清单

### 1.1 dtype 常量(`paddle.float32` 等)

| 导出 | 上游值 | 来源 |
|---|---|---|
| `paddle.float32` / `paddle.float` | `VarDesc.VarType.FP32` | `framework/dtype.py:87/119` |
| `paddle.float64` / `paddle.double` | `VarDesc.VarType.FP64` | `:90/121` |
| `paddle.float16` / `paddle.half` | `VarDesc.VarType.FP16` | `:91/122` |
| `paddle.bfloat16` | `VarDesc.VarType.BF16` | `:92` |
| `paddle.float8_e4m3fn` / `float8_e5m2` | `FP8_E4M3FN` / `FP8_E5M2` | `:93-94` |
| `paddle.int8` / `int16` / `int32` / `int64` | `INT8` / `INT16` / `INT32` / `INT64` | `:80-85/111-115` |
| `paddle.uint8` / `uint16` / `uint32` / `uint64` | `UINT8` / `UINT16` / `UINT32` / `UINT64` | `:76-79/106-109` |
| `paddle.bool` | `VarType.BOOL` | `:96/124` |
| `paddle.complex64` / `complex128` | `COMPLEX64` / `COMPLEX128` | `:95/123` |
| `paddle.pstring` / `paddle.raw` | `STRING` / `RAW` | `:97-98` |
| 别名(`short`/`int`/`long`/`float`/`half`/`double`/`cfloat`/`cdouble`) | 指向主名 | `:81-95` |

**两套绑定**:动态图走 `DataType`(`dtype.py:189` 的 `float32 = DataType.FLOAT32`),
静态走 `VarDesc.VarType`(`:87`)。Lua 侧 **只有一套**:C ABI 层的 dtype 枚举
(`01-c-abi.md`),不再区分 VarType/DataType 两套。

### 1.2 place 构造器

| 导出 | 说明 | 来源 |
|---|---|---|
| `paddle.CPUPlace()` | CPU | **pybind 绑定**(`paddle/fluid/pybind/`,非 Python class;`python/paddle/device/cpu.py:29` 里 `from paddle import CPUPlace` 是消费方) |
| `paddle.CUDAPlace(dev_id)` | GPU | 同上;⚠️ 默认 `-1` 自动选卡 |
| `paddle.CUDAPinnedPlace()` | 固定内存 | 同上 |
| `paddle.XPUPlace(dev_id)` | 昆仑 | 同上 |
| `paddle.CustomPlace(kind, dev_id)` | 自定义设备 | 同上 |

**C ABI 面**:place 是 C++ `phi::Place` 值类型(见 `01-c-abi.md`),
Lua 侧 `CPUPlace()` 返回 usertype,底层是 `Place` 的 RAII 包装。
⚠️ `paddle.device.set_device('gpu')` 是**运行时全局状态**,见 §2.3。

### 1.3 查询与工具

| 导出 | 来源 |
|---|---|
| `paddle.finfo(dtype)` | `framework/dtype.py:305`(float 信息:max/min/eps 等) |
| `paddle.iinfo(dtype)` | `dtype.py:251`(int 信息) |
| `paddle.get_default_dtype()` | `framework.py:127` |
| `paddle.set_default_dtype(d)` | `framework.py:31` |
| `paddle.is_compiled_with_cuda()` | `framework.py`(编译期能力查询) |
| `paddle.device.get_device()` / `set_device(name)` | `python/paddle/device/__init__.py` |

---

## 2. 详细签名

### 2.1 dtype 常量

```lua
local paddle = require "paddle"
paddle.float32      -- dtype 值(usertype 或字符串?见 §6)
paddle.int64
paddle.bool
```

**dtype 表示**:Lua 侧 dtype 是 C ABI 层的一个值类型(`01-c-abi.md` 的枚举
+ 字符串双向映射)。`paddle.cast{x, "float32"}` 与 `paddle.cast{x,
paddle.float32}` 等价 —— `api/README.md` §2.1.1(枚举参数一律不接受裸数字)
的下半句:字符串和类型化常量都收,裸数字不收。

### 2.2 place 构造器

```lua
paddle.CPUPlace()                       -- CPU
paddle.CUDAPlace(0)                     -- 0 号卡
paddle.CUDAPlace()                      -- -1 = 自动选卡
paddle.CUDAPinnedPlace()
paddle.XPUPlace(0)
paddle.CustomPlace("gpu_custom", 0)
```

place 是**值语义**(拷贝/比较按字段)。Lua 侧 `paddle.to_tensor{data,
place = paddle.CPUPlace()}` 的 `place` 参数收 place 值(不是裸字符串,
`api/README.md` §2.1.1)。

### 2.3 device 管理(运行时全局状态)

```lua
paddle.device.set_device("gpu")         -- 设置默认设备(全局)
paddle.device.get_device()              -- 返回 "gpu:0" 之类的字符串
```

⚠️ 这是**进程级全局状态**,与 Python 行为一致(`paddle/device/__init__.py`)。

### 2.4 finfo / iinfo

```lua
local f = paddle.finfo(paddle.float32)
f.max / f.min / f.eps / f.tiny / f.resolution
local i = paddle.iinfo(paddle.int64)
i.max / i.min / i.bits
```

上游返回 namedtuple(`dtype.py:251/305`),Lua 侧返回带 `max/min/...` 字段
的表。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** dtype 常量与 place 构造器都不涉及维度或
张量索引。显式写"无"以证明看过(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.float32`(模块属性) | `paddle.float32`(表字段) | 同名照搬;`local float32 = paddle.float32` 可 local 化 |
| 2 | `paddle.CPUPlace()`(class 实例) | `paddle.CPUPlace()`(构造器) | 同名同形;返回 place usertype |
| 3 | 裸数字当 dtype(偶见 `dtype=0`) | **拒绝** | `api/README.md` §2.1.1:枚举参数一律不接受裸数字 |
| 4 | `device` 参数名(PyTorch) | **`place`** | C12;`paddle.to_tensor{place=...}` |
| 5 | `dtype` 接受 np.dtype / VarType / DataType / str | **只收 str 与 paddle.dtype 值** | 去重后只剩 Paddle 命名一套(D26 对齐 Insight7) |
| 6 | `paddle.device.set_device` | 同名 | 全局状态,照搬 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

-- dtype 常量与查询
local f32 = paddle.float32
print(paddle.finfo(f32).max)            -- 3.4028235e+38

-- place
local x = paddle.to_tensor{1, 2, 3, dtype = paddle.float64,
                           place = paddle.CPUPlace()}

-- 默认 dtype(全局)
paddle.set_default_dtype(paddle.float64)
local y = paddle.zeros{2, 3}            -- float64
print(y:dtype() == paddle.float64)      -- true
```

---

## 6. 本模块特有的坑

**① dtype 的"两套值"问题只在 Python 存在。** 上游 `VarDesc.VarType`
(静态)与 `DataType`(动态)是两套枚举(`dtype.py:87` vs `:189`),Python 靠
`bind_datatype()` 合并。Lua 侧 C ABI 层只有**一套 dtype 枚举**,字符串
`"float32"` 与 `paddle.float32` 映射到同一枚举值 —— 这是 `01-c-abi.md`
的设计前提,不是缺陷。

**② place 是值类型,不是句柄。** `paddle.CUDAPlace(0) ==
paddle.CUDAPlace(0)` 必须为真(字段比较);如果绑定层把 place 做成带
内部指针的句柄,相等性会按指针比 → 两个 `CPUPlace()` 不相等 → 用户
`if x:place() == paddle.CPUPlace()` 永远 false。**值类型 + 字段比较**写进
`01-c-abi.md` 验收。

**③ `paddle.CUDAPlace()` 默认 `-1` = 自动选卡。** 与 Python 一致(`CUDAPlace(-1)`
选当前卡)。Lua 侧 `CUDAPlace()` 无参时传 `-1`,文档写明,别默认 0。

**④ `get_default_dtype` 是全局可变状态。** `set_default_dtype` 影响后续
所有不显式给 dtype 的创建(zeros/ones/empty 等)。这与 Python 行为一致,但
**Lua 侧多线程(Lanes)下要注意**:全局状态按 lane 各自独立(PUC 每 lane
新 state)——文档提示"默认 dtype 是每 lane 一份",避免用户以为跨 lane 共享。

**⑤ `paddle.float32` 在 Lua 里是值,不是可调用类型。** Python 侧
`paddle.float32` 可作类型注解用;Lua 无类型注解,`float32` 只是值。
用户若写 `paddle.float32(x)`(想当构造器)会报错 —— 教学式报错
`did you mean: paddle.to_tensor{x, dtype=paddle.float32}`(同 `api/README.md`
§2.1.3 的报错风格)。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `paddle.device` 全量管理函数(get/set/max_memory_allocated 等) | **P5 只做 get_device/set_device**;内存查询延后(P14 GC 机制) |
| `paddle.Generator` / `manual_seed` | P6 随机数生成器一起做 |
| `paddle.float8_*` 实际计算支持 | 跟随 C++ 侧算子支持度,P3 时按 ops.yaml 核对 |
| `paddle.set_printoptions` | P5 顺手做(`tensor_to_string` 配套) |
| `paddle.is_compiled_with_*` 系列 | P5 只做 `is_compiled_with_cuda`;其余按需补 |
