# paddle.vision

| | |
|---|---|
| 阶段 | **P12**(与 `paddle.dataset` 同批;`paddle.nn` P9 之后) |
| 依赖模块 | `paddle`、`paddle.io`、`paddle.nn`、`paddle.dataset`(部分)、Insight7(图像数组) |
| 稳定性 | **稳定**(datasets 的 MNIST/FashionMNIST——训练必需)/ 实验(models/transforms 分批) |
| 实现文档 | `plan/modules/12-dataset-vision.md` |
| 上游来源 | `python/paddle/vision/__init__.py`(导出);`datasets/`(MNIST/FashionMNIST/Cifar10/...);`models/`(ResNet/LeNet/...);`transforms/`;`ops/`;`image.py`(get_image_backend/image_load/set_image_backend) |

> 📌 本模块四个子命名空间:**datasets**(数据,适配 `io.Dataset`)、**models**
> (预训练模型,`nn.Layer` 子类)、**transforms**(图像预处理管线)、**ops**
> (图像算子)。**MNIST/FashionMNIST 是 M1 训练直接需要的**;models/transforms
> P12 分批。

---

## 1. 导出清单

`python/paddle/vision/__init__.py` 的真实导出:

### 1.1 datasets(8 个,`:23-31`)

`MNIST` / `FashionMNIST` / `Cifar10` / `Cifar100` / `VOC2012` / `Flowers` /
`DatasetFolder` / `ImageFolder`

⚠️ **`MNIST` 与 `paddle.dataset.mnist` 的区别**:vision 的是 **`io.Dataset`
类**(`get/len` 契约),dataset 的是 reader 生成器。Lua 侧**用 vision 类**,
dataset 的 reader 降级为数据源参考(`dataset.md` §2)。

### 1.2 models(:38-49+)

`VGG` / `AlexNet` / `DenseNet` / `GoogLeNet` / `InceptionV3` / `LeNet` /
`MobileNetV1` / `MobileNetV2` / `MobileNetV3Large` / `MobileNetV3Small` /
`ResNet`(+ 变体)…

### 1.3 image(:33-36)

`get_image_backend()` / `set_image_backend(name)` / `image_load(path)`
——图像解码后端选择(PIL/OpenCV 类比,见 §6 ②)

### 1.4 transforms / ops

`transforms`(预处理组合:Resize/Normalize/ToTensor 等)与 `ops`
(算子)是子模块,导出随 P12 分批。

---

## 2. 详细签名

### 2.1 `vision.datasets.MNIST`(★ M1 训练入口)

```lua
local ds = paddle.vision.datasets.MNIST{
  image_path,      -- 或使用默认下载
  label_path,
  mode = "train",  -- "train"/"test"
  transform = nil, -- transforms 管线(§2.2)
  backend = "cv2" | "pil",   -- 图像后端
}
ds:len()                    -- 60000
local x, y = ds:get(1)      -- 1-based;x 是 insight.Array 图像,y 是 label
```

**与 `paddle.dataset.mnist` 的关系**:`vision.datasets.MNIST` 是类,
内部复用 `paddle.dataset.mnist` 的下载与 IDX 解析(`dataset.md` §6 ③),
但暴露 `io.Dataset` 形态。Lua 侧**只做这个类**,reader 生成器不搬。

### 2.2 `transforms`(图像预处理管线)

```lua
local t = paddle.vision.transforms.Compose{
  transforms = {
    paddle.vision.transforms.Resize{ size = {224, 224} },
    paddle.vision.transforms.Normalize{ mean = {0.5, 0.5, 0.5}, std = {0.5, 0.5, 0.5} },
    paddle.vision.transforms.ToTensor{},   -- 图像 → [C,H,W] Tensor
  },
}
-- t 是可调用对象:t(x) 依次应用
```

`Compose` 是管线容器;每步是可调用对象(与 `nn-initializer.md` 同一约定)。
⚠️ `Normalize` 的 mean/std 是 **数值列表不是索引**(§3)。

### 2.3 `models.ResNet` 等(预训练模型)

```lua
local net = paddle.vision.models.ResNet{ depth = 18, num_classes = 10 }
-- nn.Layer 子类:net(x) 前向,net:parameters() 取参数
```

模型 = `nn.Layer` 子类(`nn.md` §2.1 契约)。⚠️ **预训练权重加载**走
`paddle.jit.load`(P8)或 `paddle.load`(T-M0-19)——P12 随 `models` 分批定。

---

## 3. index 语义参数

| 位置 | 语义 | 基准 |
|---|---|---|
| `ds:get(i)` 的 `i` | 样本序号 | **1-based**(与 `io.md` 对齐) |
| 数据集 label | 类别 id | **1-based**(C4,与 `dataset.md` §3 对齐) |
| `Normalize.mean/std` | 数值列表 | 非索引(§2.2 注明) |
| `Resize.size` | 像素尺寸 | 非索引 |
| 模型 `num_classes` | 超参 | 非索引 |

**没有 index 参数的成员**:`models` 全族、`image_load`、`transforms` 的数值
类变换 —— 均无维度或张量索引(显式列出,证明"看过")。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | `vision.datasets.MNIST(...)` | **`MNIST{ mode="train" }`** | 一张参数表 |
| 2 | `transform` 传 callable | 传可调用对象 | 形态一致 |
| 3 | `dataset[i]` 返回 (x, y) | **`ds:get(i)`** | `io.md` 契约(`__index` 冲突) |
| 4 | 图像是 numpy/PIL | **insight.Array** | D26 |
| 5 | `transforms.Compose([...])` | **`Compose{ transforms = {...} }`** | 一张参数表 |
| 6 | `model = vision.models.resnet18()` | **`ResNet{ depth=18 }`** | 名称照搬,构造形态统一 |

---

## 5. 用法示例

```lua
local paddle = require "paddle"

local ds = paddle.vision.datasets.MNIST{
  mode = "train",
  transform = paddle.vision.transforms.ToTensor{},
}
local loader = paddle.io.DataLoader{
  dataset = ds, batch_size = 64, shuffle = true,
}

for x, y in loader:iter() do
  local loss = criterion(net(x), y)   -- x: [B,1,28,28]
  loss:backward()
  opt:step(); opt:clear_grad()
end
```

---

## 6. 本模块特有的坑

**① `vision.datasets.MNIST` 与 `paddle.dataset.mnist` 是两代 API,别混。**
vision 的是 `io.Dataset` 类(现代),dataset 的是 reader 生成器(静态图遗留,
`__all__=[]`)。Lua 侧**只做 vision 类**;`paddle.dataset.mnist` 降级为
「数据源参考」(`dataset.md` §2)。两处同时出现会让用户困惑,文档必须写清
对照关系。

**② 图像后端是 P12 的第一道 C 依赖决策。** `get_image_backend/set_image_backend`
(`image.py`)支持 cv2/pil 两套解码。Lua 侧等价物:JPEG/PNG 解码 → 按 §9.1
取舍程序(C 依赖),**倾向直接引入 image 解码 rock**(JPEG/PNG 自己写 = 重造
轮子),或先只支持 PNG(MNIST 不需要 JPEG)。P12 定。

**③ transforms 的 `ToTensor` 是「图像 → Tensor」的转换点。** 上游
`ToTensor` 把 PIL/numpy 的 HWC 图像转成 CHW Tensor(并归一化到 [0,1])。
Lua 侧 = insight.Array → `paddle.to_tensor` 的 DLPack 桥 + 转置/缩放
组合(`top.md` §2.1)。**这是 transforms 管线里唯一涉及 Tensor 的地方**,
其余(Resize/Normalize)在 Insight 数组上做 —— 分层要写死:
`transforms` 大部分在 Insight 层,`ToTensor` 是 Insight → Tensor 边界。

**④ `Normalize` 的 mean/std 是 CHW 通道顺序的数值,不是索引。** 用户容易
把它和「轴索引」混。§3 已注:数值列表,不标 1-based。但**通道顺序**
(Python 是 CHW)要照搬,不转成 Lua 习惯的 HWC —— 与上游互操作一致。

**⑤ 预训练权重加载链要复用 P8。** `models.ResNet{pretrained=true}` 的权重
是 jit 导出产物(Python 端)或 SaveTensor 格式。Lua 侧加载走
`paddle.jit.load`(`jit.md`)或 `paddle.load`(T-M0-19),**不新写一套下载**。
P12 与 `models` 一起定 `pretrained` 参数形态。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| `MNIST` / `FashionMNIST` | P12 首批(M1 训练) |
| `Cifar10/100` / `VOC2012` / `Flowers` | P12 分批 |
| `ImageFolder` / `DatasetFolder` | P12(依赖文件系统遍历) |
| `transforms` 全量(Resize/Normalize/CenterCrop/... ) | P12 首批 ToTensor + Resize + Normalize;其余分批 |
| `models`(ResNet/LeNet/... ) | P12(依赖 nn 稳定 + pretrained 加载链) |
| `ops`(图像算子) | 延后 |
| 图像后端(cv2/pil 类比) | P12 按 §9.1 决策(倾向引入解码 rock) |
