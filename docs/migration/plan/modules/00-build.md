# P0 · 无 Python 构建

| | |
|---|---|
| 阶段 | P0 —— **项目生死判定** |
| 类别 | 基础设施 |
| 开工条件 | 无 |
| 预估 | 3–5 天 |

---

## 1. 做什么 / 不做什么

**做:** 把 Paddle 编成一个**不含 Python 且能训练**的 `libpaddle`,并用一个 C++ 冒烟程序
证明 `egr::Backward` 真的能算出梯度。

**不做:** 任何 Lua 相关的东西。这一阶段一行 Lua 都不写。

**为什么这是第一件事:** `cmake/configure.cmake:15` 对无 Python 场景的守卫是
`(NOT WITH_PYTHON) AND ON_INFER` —— **上游预期的无 Python 场景是推理**。
我们要的是"无 Python + 训练",这个组合可能从未被任何人编译过。
在验证它之前,后面所有设计都建立在假设上。

---

## 2. 上游有什么可以用

| 出处 | 内容 |
|---|---|
| `cmake/configure.cmake:15` | `(NOT WITH_PYTHON) AND ON_INFER` -> 定义 `PADDLE_NO_PYTHON` |
| `paddle/fluid/eager/backward.h:25-29` | `TEST_API void Backward(const std::vector<paddle::Tensor>&, const std::vector<paddle::Tensor>&, bool retain_graph, bool create_graph, std::string dump_backward_graph_path)` |
| `paddle/fluid/eager/backward.h:31-40` | `TEST_API std::vector<paddle::Tensor> Grad(...)` |
| `paddle/phi/api/include/tensor.h:98-510` | `paddle::Tensor` 的完整公开接口 |
| `paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h` | 全部 `*_ad_func`,例如第 12 行 `TEST_API paddle::Tensor abs_ad_func(const paddle::Tensor& x, ...)` |

注意 `TEST_API` 宏 —— 这些符号是**显式导出**的,不是内部符号。这是好消息。

---

## 3. 设计

### 3.1 要跑的构建矩阵

按顺序试,**一个成了就停**,后面的是退路:

| # | 配置 | 说明 |
|---|---|---|
| 1 | `-DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=ON` | 目标配置 |
| 2 | `-DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=OFF` | 排除 CUDA 干扰,先证明 CPU 路径成立 |
| 3 | `-DWITH_PYTHON=OFF -DON_INFER=ON` | 上游支持的配置。**若只有这条能过,说明训练态被裁掉了** |
| 4 | `-DWITH_PYTHON=ON` + 运行时不加载 Python | 最后的退路,违反项目前提,只用于定位问题 |

### 3.2 冒烟程序

```cpp
// tools/smoke/smoke.cc  —— 约 50 行
#include "paddle/phi/api/include/tensor.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"

int main() {
  auto a = /* full({2,2}, 3.0) */;
  auto b = /* full({2,2}, 4.0) */;
  // a 与 b 需 SetStopGradient(false)
  auto c = multiply_ad_func(a, b);
  auto d = sum_ad_func(c, {}, phi::DataType::FLOAT32, false);
  egr::Backward({d}, {});
  // 期望:a.grad == 4.0,b.grad == 3.0
}
```

**判据不是"跑完不崩",而是梯度数值与 Python 端逐位一致。**
对照脚本用 Python 写同样五行,`np.testing.assert_array_equal`。

### 3.3 同时要回答的三个附带问题

| 问题 | 为什么在这一阶段问 | 影响 |
|---|---|---|
| `new_executor` / `jit` 是否编出并可链接 | 编一次的成本很高,顺手验 | 挂了则 P8/P15 全部作废 |
| `OpProtoHolder` 链是否存活 | 自定义算子依赖它 | 挂了则 P16 方案要改 |
| `TEST_API` 在 Release 下是否仍导出 | 名字里带 TEST,值得怀疑 | 挂了则要自己加导出宏 |

---

## 4. 已知的坑

**① `TEST_API` 这个名字有误导性。** 它是导出宏不是测试宏,但**不能假设 Release 构建下
它一定展开成 `__declspec(dllexport)`**。要实际 `dumpbin /exports` 或 `nm -D` 确认。

**② Windows 上符号导出比 Linux 严格得多。** Linux 默认导出所有符号,Windows 默认不导出。
在 Linux 上链接成功不代表 Windows 也行,**两边都要验**。

**③ 编译时间以小时计。** 全量 GPU 构建可能 2–6 小时。先跑配置 2(CPU-only)拿快速反馈,
再上 GPU。不要一上来就等六小时才发现一个 typo。

**④ 别改 Paddle 源码来"修"链接错误。** 这违反 C2(上游零改动)。
如果必须改才能编过,那结论就是"这条路不通",要如实记录,而不是偷偷改一行。

---

## 5. 验收

- [ ] 上表配置 1 或 2 编译成功,产出 `libpaddle.so` / `paddle.dll`
- [ ] 冒烟程序链接成功并运行
- [ ] 梯度数值与 Python 端逐位一致
- [ ] `nm -D` / `dumpbin /exports` 确认 `egr::Backward` 与至少一个 `*_ad_func` 已导出
- [ ] 记录完整 cmake 命令行、编译耗时、产物体积到 `process/status.md` §3
- [ ] 若失败:把失败点(哪个目标、哪条链接错误)写进 `process/open-questions.md`,**不要绕过**

---

## 6. 未解问题

- Q-01 `WITH_PYTHON=OFF + ON_INFER=OFF` 能否编出可用 libpaddle(**本阶段的全部意义**)
- Q-07 该配置下 `OpProtoHolder` 链是否存活
