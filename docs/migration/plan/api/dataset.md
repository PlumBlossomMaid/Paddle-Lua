# paddle.dataset

| | |
|---|---|
| 阶段 | **P12**(与 `paddle.vision.datasets` 互补;`paddle.io.Dataset` 的数据源) |
| 依赖模块 | `paddle.io`(Dataset 基类)、`paddle.utils.download`(下载缓存)、Insight7(数据数组) |
| 稳定性 | **稳定**(mnist/cifar——训练必需)/ 实验(其余) |
| 实现文档 | `plan/modules/12-dataset-vision.md` |
| 上游来源 | `python/paddle/dataset/__init__.py`(12 子模块,`__all__ = []`);`mnist.py`(reader_creator,`mnist.py:47`);`cifar.py`/`imdb.py`/`imikolov.py` 等 |

> 📌 **上游 `paddle.dataset` 是静态图时代的 reader 风格**:每个数据集是
> 「reader 工厂」,返回 Python 生成器(一次 yield 一个样本)。**`__all__ = []`
> 不导出任何 API 名**,只挂 12 个子模块。Lua 侧把这批数据源适配成
> `paddle.io.Dataset`(`io.md` 的 `get/len` 形态)——reader 生成器是我们的
> **数据准备中间产物**,不是最终 API 形态。

---

## 1. 导出清单

上游 `paddle/dataset/__init__.py` 挂载的 12 个子模块:

| 子模块 | 数据集 | 来源 |
|---|---|---|
| `paddle.dataset.mnist` | MNIST(手写数字) | `mnist.py`(★ M1 训练用) |
| `paddle.dataset.cifar` | CIFAR-10/100 | `cifar.py` |
| `paddle.dataset.imdb` | IMDB(情感分析) | `imdb.py` |
| `paddle.dataset.imikolov` | 词向量语料 | `imikolov.py` |
| `paddle.dataset.uci_housing` | 房价 | `uci_housing.py` |
| `paddle.dataset.movielens` | 推荐 | `movielens.py` |
| `paddle.dataset.conll05` / `voc2012` / `flowers` / `image` / `wmt14` / `wmt16` | NLP/CV/翻译 | 各子模块 |

**Lua 侧第一批**:`mnist`(M1 验收) + `cifar`;其余随 P12 分批。

---

## 2. 详细签名

### 2.1 上游形态(reader 工厂,供参考)

```python
# 上游 python/paddle/dataset/mnist.py:47
def reader_creator(image_filename, label_filename, buffer_size):
    def reader():
        # gzip + struct 解 IDX 格式(大端)
        # yield (image, label) 元组,image 已归一化到 [-1, 1]
        ...
    return reader

paddle.dataset.mnist.train()   # 训练 reader(生成器)
paddle.dataset.mnist.test()    # 测试 reader
```

数据源:`mnist.py:30-38`(URL + MD5,从 bcebos 下载)。

### 2.2 Lua 形态(适配 `paddle.io.Dataset`)

```lua
-- Lua 侧:数据源 → io.Dataset(不是生成器)
local ds = paddle.dataset.mnist{ split = "train" }  -- 或 vision.datasets.MNIST
-- 返回 paddle.io.Dataset 实例(io.md 契约):
ds:len()                    -- 60000
local x, y = ds:get(1)      -- 1-based 样本
```

**适配层职责**(`12-dataset-vision.md`):
1. 下载 + MD5 校验(`paddle.utils.download`,上游 `common.py:74`)
2. IDX/二进制格式解析(上游 gzip + struct,`mnist.py:56-90`)
3. 样本 → `insight.Array`(图像)或 Lua 标量/表(标签)

⚠️ **不把生成器搬进 Lua**:Lua 的闭包迭代器 + `io.DataLoader` 已有
`Dataset:get(i)` 契约,生成器是 Python 的产物(静态图时代的 shim)。

---

## 3. index 语义参数

**本模块 index 语义密集**(与 `io.md` 同一约定):

| 位置 | 语义 | 基准 |
|---|---|---|
| `ds:get(i)` 的 `i` | 样本序号 | **1-based** |
| MNIST 标签值 `ds:get(i)` 返回的 label | 类别 id | **1-based**(⚠️ 上游 0..9;Lua 侧 +1,对齐 C4) |
| 数据集划分(训练/测试) | — | 非索引 |

⚠️ **MNIST label 1-based 是 M1 验收的关键对齐点**:`CrossEntropyLoss` 的
label 也是 1-based(`nn.md` §3),两头对齐,C4 在数据管线上的完整闭环。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `mnist.train()` 返回 reader(生成器) | **`dataset.mnist{split="train"}` 返回 io.Dataset** | io.md 的 get/len 契约;生成器不搬 |
| 2 | `for x, y in mnist.train():` | **`for x, y in ds:iter() do`** | 经 DataLoader 或 Dataset:iter() |
| 3 | 标签 0..9 | **1..10(1-based)** | D-R6;C4 |
| 4 | `__all__ = []`(无导出) | 显式导出 `paddle.dataset.mnist` 等 | Lua 侧按需导出,不学 `__all__=[]` |
| 5 | 图像数据 numpy | **insight.Array** | D26 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

-- M1 验收形态:MNIST 数据 → io.Dataset → DataLoader
local ds = paddle.dataset.mnist{ split = "train" }
local loader = paddle.io.DataLoader{
  dataset = ds, batch_size = 64, shuffle = true,
}

for x, y in loader:iter() do
  local loss = criterion(net(x), y)   -- x: [B,1,28,28] insight/insight 转换后 Tensor
  loss:backward()
  opt:step(); opt:clear_grad()
end
```

---

## 6. 本模块特有的坑

**① 上游 reader 生成器不要照搬。** 生成器是静态图时代
`paddle.batch(paddle.reader.shuffle(mnist.train(), 512), 128)` 的配套。
Lua 侧 `io.Dataset + DataLoader` 已有更现代的等价物(D7:DataLoader 用线程
不用进程)。**照搬 reader 生成器 = 自造第二套数据管线**,违反 C11 精神。

**② 下载是 P12 的第一道真实网络依赖。** `mnist.py:30-38` 从 bcebos 下载
`.gz`。Lua 侧 `paddle.utils.download`(P4)要处理:代理、断点续传、MD5 校验、
缓存目录。⚠️ **用户机器没网时** `dataset.mnist` 必须报清晰错误
(`did you mean: 先运行 paddle.utils.download?`),不静默降级。

**③ MNIST 的 IDX 格式解析要逐字节对齐上游。** `mnist.py:56-90`:
大端 `>IIII` magic + 数量 + 行列,归一化 `x/255*2-1`。Lua 侧解析必须
**逐字节对拍**(生成 `m0-11-sample.pdparams` 式的测试基准,CI 校验)。
偏移/字节序错一点就是静默的乱数据,训练照常收敛但结果全错。

**④ 图像是 `[N,1,28,28]` 还是 `[N,28,28]`?** 上游 mnist reader 产出的是
展平 `[784]` float32(`mnist.py:84` reshape)。Lua 侧适配层把它 reshape 成
`[1,28,28]`(CNN 输入形态)还是保持 `[784]`(MLP 输入)取决于用户模型——
**适配层提供两个 split 形态参数**,不替用户定。

**⑤ 数据集缓存与 `io.DataLoader` 的 `num_workers` 是两条线。** 缓存
(P12)在**主进程**做一次(下载 + 解压到内存);`num_workers > 0`(P13)时
worker 共享这份缓存(数据准备结果经 Lanes 传递)。不要在 worker 里重复下载。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `cifar` / `uci_housing` / `imdb` | P12 首批(mnist 后) |
| `movielens` / `voc2012` / `flowers` / `wmt14/16` / `conll05` | P12 分批 |
| `image`(通用图像 reader) | 延后(依赖 `vision.transforms` 先定) |
| 数据格式逐字节对拍基准 | P12 与 mnist 一起建 |
| 断点续传下载 | P4 `utils.download` 一起做 |
