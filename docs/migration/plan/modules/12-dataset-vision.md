# P12 · dataset + vision

| | |
|---|---|
| 阶段 | P12 |
| 类别 | 语义直抄(A1) |
| 开工条件 | P11 完工 |
| 预估 | 3 周 |

---

## 1. 做什么 / 不做什么

**做:** `paddle.dataset`(MNIST / CIFAR / …的下载与解析)、
`paddle.vision.transforms`、`paddle.vision.models` 里的几个常用模型。

**不做:** 不做全部预训练模型。挑 ResNet / VGG / MobileNet 三个系列即可。

**这块的定位:难度低,但不能省。**
`process/decisions.md` R9 记录了一次翻案 —— 曾经把 dataset 当"下载脚本、与框架无关"排除掉,
这是把"难度"和"该不该做"混为一谈。**没有 dataset,M1 的 MNIST 验收就闭不了环。**

---

## 2. 上游有什么可以用

| 出处 | 用途 |
|---|---|
| `python/paddle/vision/transforms/functional*.py` | 变换的数学定义 |
| `python/paddle/vision/models/*.py` | 模型结构定义,**结构是纯声明,最容易翻译** |
| `python/paddle/dataset/*.py` | 下载 URL、md5、解析格式 |
| P3 生成的算子 | 大部分 transform 落到算子上 |

---

## 3. 设计

### 3.1 下载

Lua 没有标准 HTTP 库。方案:

> 🔄 **2026-08-03:「不引入新的强制 C 依赖」已取消(R23)。本节据此重写。**
> 原表里 `luasocket`+`luasec` 被判「又两个依赖」而否决 —— 那是数量判据,已作废。

按 `CLAUDE.md` §9.1 的三步走:

| 步骤 | 方案 | 评价 |
|---|---|---|
| ① 我们自己的 `.so` 能不能做 | 在 C ABI 层提供 `pd_http_get` | **待核实(Q-08)。** 要看 Paddle 有没有现成可绑的 HTTP 客户端。**若有,选这个** —— 边际成本 ≈ 0。若要为此把 OpenSSL 拖进 Paddle 的构建,则**不划算,直接跳到 ②** |
| ② 做不了就直接引入 | `luasocket` + `luasec` | **现在允许。** 这是 Lua 生态里 HTTP+TLS 的事实标准,rock 覆盖成熟。luasec 的 OpenSSL 版本问题是真的烦,但那是**打包问题**,不是「不该引入」的理由 |
| ③ 兜底 | 调 `curl` / `wget` 子进程 | 保留为降级路径(§9.1 第 3 步要求的"降级路径"),不作为主方案 |

**结论:先查 ①,查不到就走 ②,③ 永远保留。**
不再出现"因为不能加依赖,所以让用户手工下载"这种把成本转嫁给用户的做法。

路径拼接与目录创建用 `paddle.utils.fs`(C++17 `std::filesystem`)。
~~理由:`pl.path`/`pl.dir` 依赖 lfs,被 C11 禁~~ ——
**现在的理由是 §9.1 第 1 步:我们的 `.so` 顺手就能做,比多一个 rock 划算。**
见 `plan/foundations.md` §1.2。

**无论哪种方案,md5 校验必须做。** 半个下载文件解析出的乱七八糟的报错
会让用户浪费几个小时。

### 3.2 transforms

```lua
local T = paddle.vision.transforms
local tf = T.Compose{
  T.Resize(256),
  T.CenterCrop(224),
  T.ToTensor(),
  T.Normalize{ mean = {0.485, 0.456, 0.406}, std = {0.229, 0.224, 0.225} },
}
```

**transform 必须是纯函数**(无状态,或状态只有随机种子)。
这是 P13 的硬约束 —— transform 要在 worker 线程里执行。
有状态的 transform(如"记住上一张图")在多 worker 下行为不可预测。

#### transform 内部用 Insight7,不用 Tensor

D26 把 Insight7 定为 numpy 的位置。transform 是它最典型的用武之地:

| | 为什么 |
|---|---|
| **不需要梯度** | transform 在 `no_grad` 语境里,建 autograd 图纯属浪费 |
| **在 CPU 上** | worker 线程做数据准备,GPU 留给训练主线程(`plan/foundations.md` §3.6) |
| **要 numpy 式操作** | crop / flip / pad / astype —— 全是数组操作,不是算子 |

流水线:`解码 -> insight.Array 上做变换 -> paddle.from_insight -> 送 Linda`。

⚠️ **两个前提**(否则退回"transform 全用 `paddle.Tensor`"):
- **Q-12(已决,待实施)**:Insight7 的 `axis` 现在是 0-based,
  **已拍板改成 1-based**(R24 / D29,"顺手修")。transform 里大量用到轴
  (`transpose` HWC->CHW、`concat`),这里正是最容易中招的地方,
  所以**本阶段开工前必须已经修完**,不能一边写 transform 一边等。
  修完之前若要先写代码,每个吃 `axis` 的调用点加
  `--[[ Insight7 axis 待 1-based 化,见 foundations §3.4 ]]`,修好后统一清理。
- **Q-14**:`insight::Array` 能否跨 lane。不能的话就在 worker 里
  转成 `paddle.Tensor` 再送出去 —— 多一次拷贝,但不阻塞。

### 3.3 图像解码

JPEG/PNG 解码在哪做?

**v1 的答案:要求数据集提供已解码的原始像素,或依赖 `paddle.vision` 已有的 C++ 解码能力。**
(第三条路:Insight7 若有图像 IO 能力则直接用 —— 未核实,归入 Q-08 一并调研。)
⚠️ 上游是否有可绑定的解码接口未核实。若没有,v1 只支持 MNIST/CIFAR 这类
**二进制原始格式**的数据集,ImageNet 类的留到 M2 再解决。

**这个限制要在文档里明说**,而不是让用户试了才发现。

### 3.4 models

模型结构是纯声明,是整个项目里最机械的部分:

```lua
function resnet18(pretrained)
  local m = ResNet(BasicBlock, {2,2,2,2})
  if pretrained then m:set_state_dict(paddle.load(download(URLS.resnet18))) end
  return m
end
```

**预训练权重直接用 Python 生态的 `.pdparams`** —— 这就是 P7 的 pickle 解析器
真正有用的地方,也是它该在 M2 完成的原因。

---

## 4. 已知的坑

**① 数据集 URL 会失效。** 硬编码 URL 是长期维护负担。
应该把 URL 集中在一个 `datasets.lua` 表里,并支持环境变量覆盖
(`PADDLE_LUA_DATASET_MIRROR`),让用户能用镜像。

**② 解压。** tar.gz / zip 的解压在纯 Lua 里做很痛苦。
同 §3.1,优先看能不能在 C ABI 层解决。

**③ transform 的数值必须与 Python 一致。** `Resize` 的插值方式、`Normalize` 的
就地与否,细微差别会让预训练模型的精度掉几个点,**而且不报错**。
验收里必须有像素级对拍。

**④ 别在 transform 里用 `math.random`。** 同 `11-io.md` §3.4,
要用 Paddle 的随机源,否则多 worker 下每个 lane 的随机流不受控。

---

## 5. 验收

- [ ] MNIST / CIFAR-10 下载 + md5 校验 + 解析,样本数与官方一致
- [ ] 每个 transform 与 Python 对拍:同一张图,输出**像素级一致**
- [ ] ResNet18 结构与 Python 版 `state_dict` 键名逐字符一致
- [ ] 加载 Python 的预训练 ResNet18,ImageNet 验证集 top-1 与官方数字一致
- [ ] ResNet18 在 CIFAR-10 上 10 epoch 准确率 > 80%(**M2 验收项**)
- [ ] 下载失败时的错误信息包含手动下载指引

---

## 6. 未解问题

- Paddle 是否有可绑定的 HTTP 下载 / 解压 / 图像解码能力?**待核实**,
  结论直接决定本阶段是 3 周还是 5 周
