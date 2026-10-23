# paddle.utils

| | |
|---|---|
| 阶段 | **P4**(工具,先于模型/算子) |
| 依赖模块 | `paddle` 核心之外的纯工具;`download` 依赖网络 |
| 稳定性 | **稳定**(`download`/`fs`——dataset 需要)/ 实验(其余) |
| 实现文档 | `plan/modules/04-packaging.md`(下载/缓存语义) |
| 上游来源 | `python/paddle/utils/__init__.py`(真实导出,见下文);`download.py`;`dlpack.py`(Q-22 相关);`image_util.py`;`unique_name.py` |

> 📌 **⚠️ 修正 `api/README.md` 模块清单的一条旧说法**:清单写
> 「`fs` / `download`,我们自己的,Python 侧无对应」——**核实后不准确**。
> 上游 `python/paddle/utils/` **确有 `download.py` 与 `dlpack.py` 等**
> (`utils/__init__.py:16-21` 挂 `cpp_extension/decorator_utils/dlpack/download/
> gpu_utils/image_util/layers_utils/unique_name`;`__all__` 为
> `deprecated/run_check/require_version/try_import`)。`paddle.utils.fs` 才是
> 我们自己的(Python 侧无 fs 模块)。本文按**上游真实导出**写。

---

## 1. 导出清单

`python/paddle/utils/__init__.py` 的真实导出:

| 导出 | 来源 | 阶段 |
|---|---|---|
| `paddle.utils.download` | `utils/__init__.py:19`(子模块) | **P4 首批**(dataset/vision 需要) |
| `paddle.utils.dlpack` | `:18`(子模块) | **P4 首批**(Q-22 相关;`to_dlpack/from_dlpack`) |
| `paddle.utils.deprecated` | `:25`(函数) | P4 |
| `paddle.utils.run_check` | `:28`(函数,安装自检) | P4 |
| `paddle.utils.require_version` | `:13`(函数) | P4 |
| `paddle.utils.try_import` | `:30`(函数,惰性 import) | P4 |
| `paddle.utils.unique_name` | `:20`(子模块) | P4 |
| `paddle.utils.image_util` | `:19`(子模块,图像工具) | P12 随 vision |
| `paddle.utils.cpp_extension` | `:17`(子模块) | 延后(P16 自定义算子) |
| `paddle.utils.data` | **惰性 `__getattr__`**(`__init__.py:38-46`) | 延后 |
| **`paddle.utils.fs`(我们自己的)** | 无上游对应 | **P4 新增**(文件系统工具,`pl.path` 包一层) |

⚠️ **`decorator_utils` 整层不移植**(`__init__.py:18` 挂了,但 D35 排除,
见 `api/README.md` §2.1.3)。

---

## 2. 详细签名

### 2.1 `paddle.utils.download`(★ P4 核心)

```lua
paddle.utils.download.download{ url = "...", path = "...", md5 = nil }
-- 下载到 path(带缓存/MD5 校验);已存在且 MD5 匹配 → 不重复下载
```

| 参数 | 说明 |
|---|---|
| `url` | 下载地址 |
| `path` | 落盘路径 |
| `md5` | 期望校验和;nil = 跳过校验 |

上游 `python/paddle/utils/download.py` 有 `download`/`_get_unique_paths`/
`_map_url_to_file_path` 等(`dataset.md` §6 ② 已引用 `common.py:74`)。
Lua 侧:
- 缓存目录固定在 `paddle.utils.cache_dir`(可配置)
- **断点续传** P4 做(temporary part 文件)
- **代理**:走系统/luarocks 既有通道(§6 ①)

### 2.2 `paddle.utils.dlpack`

```lua
paddle.utils.dlpack.to_dlpack(tensor)        -- → DLManagedTensor userdata
paddle.utils.dlpack.from_dlpack(managed)     -- → Tensor
```

**这是 Q-22 的落地通道**(内存互通协议)。Insight7 已有同名 API
(`ins.to_dlpack/from_dlpack`,P4 前已建成)——Lua 侧 `paddle.utils.dlpack`
**包装 Insight7 的桥**(`foundations.md` §3.6.1),不自己实现 DLPack。

### 2.3 `paddle.utils.fs`(我们自己的)

```lua
paddle.utils.fs.list(dir)      -- pl.List of 文件路径
paddle.utils.fs.exists(p)      -- boolean
paddle.utils.fs.mkdirs(p)      -- 递归建目录
paddle.utils.fs.remove(p)      -- 删除文件/目录
paddle.utils.fs.join(a, b)     -- 路径拼接
```

⛔ 上游没有 `paddle.utils.fs`。Lua 侧用 `pl.path`/`pl.dir`/`lfs` 包一层
(旧禁令已取消,§9.1 可引入 `lfs`)。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** `url`/`path`/`md5`/`dlpack` 均非索引。
显式写"无"以证明看过(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `paddle.utils` 存在且挂 8 子模块 | 同名同样式(子模块) | 名字照搬 |
| 2 | `paddle.utils.download.download(url, path)` | **`download{ url, path, md5 }`** | 一张参数表 |
| 3 | `paddle.utils.dlpack` 是模块 | 包装 Insight7 的 DLPack | Q-22:phi 层有非 pybind DLPack;Insight7 桥已建 |
| 4 | **无 `paddle.utils.fs`** | **有 `fs`(我们新增)** | 文件系统工具,`pl.path` 包一层 |
| 5 | `paddle.utils.data`(惰性子模块) | 延后 | 静态图数据工具,不搬 |
| 6 | `decorator_utils`(装饰器工厂) | **不移植** | D35 整层排除 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

-- 下载 MNIST 数据(dataset/vision 内部会调用)
local path = paddle.utils.download.download{
  url  = "https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz",
  path = "cache/mnist/train-images-idx3-ubyte.gz",
  md5  = "f68b3c2dcbeaaa9fbdd348bbdeb94873",
}

-- 文件系统
local files = paddle.utils.fs.list("cache/mnist")   -- pl.List

-- DLPack 互通(包装 Insight7 桥)
local managed = paddle.utils.dlpack.to_dlpack(x)
local back = paddle.utils.dlpack.from_dlpack(managed)
```

---

## 6. 本模块特有的坑

**① 下载的代理与超时是 P12 数据管线的第一道真实网络问题。**
`dataset.md` §6 ② 已列:代理、断点续传、MD5、缓存目录。Lua 侧实现时:
- 走系统代理/env(`http_proxy`,本机惯例)——不自己实现 TLS
- 超时与重试(指数退避,参照记忆 `retry_pattern_follow_existing`)
- **离线时清晰报错**,不静默返回

**② `dlpack` 不要重复造轮子。** Insight7 的 DLPack 桥
(`ins.to_dlpack/from_dlpack`)已完成(记忆 `insight7_dlpack_bridge`);
`paddle.utils.dlpack` 只是给它包 Paddle 命名。**任何"自己实现 DLPack v1.3"
的冲动都违反 D24/§9.1 边际成本判据** —— 我们自己的 `.so` 里加一个转发
函数即可。

**③ `utils.data` 惰性加载是坑源。** 上游 `paddle.utils.data` 通过
`__getattr__` 惰性 import(`__init__.py:38-46`)。Lua 侧 `require` 语义不同
(显式),**不学惰性 `__getattr__`**——`paddle.utils.data` 直接标延后,
避免"访问字段触发副作用"的意外。

**④ `deprecated` 装饰器不移植语义,只留报错形态。** 上游 `deprecated`
是 Python 装饰器(`deprecated.py`)。Lua 无装饰器语法;Paddle 侧已废弃的 API
在 Lua 侧直接**不导出**(或导出后调用时 `warn` + 跳转新名)。这是 C11
「一套基座」精神在 API 层的体现。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `fs`(我们自己的文件系统工具) | P4 新增(pl.path + lfs 包一层) |
| `download`(下载+MD5+续传) | P4 首批(dataset P12 依赖) |
| `dlpack`(包装 Insight7 桥) | P4 首批(Q-22 已定方向) |
| `run_check` / `require_version` / `try_import` | P4 顺手做 |
| `image_util` | P12 随 vision |
| `cpp_extension` | 延后(P16 自定义算子) |
| `data`(惰性子模块) | 延后(静态图工具,不搬) |