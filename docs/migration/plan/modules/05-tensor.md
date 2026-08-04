# P5 · paddle.Tensor

| | |
|---|---|
| 阶段 | P5 |
| 类别 | 手写 C + Lua |
| 开工条件 | P2 + P3 完工 |
| 预估 | 2 周 |

---

## 1. 做什么 / 不做什么

**做:** Tensor 的构造、元信息、类型/设备转换、**索引与切片**、运算符重载、打印。

**不做:** 自动微分(P6)、序列化(P7)。

**这一阶段的核心难点只有一个:索引。** 其余都是机械劳动。

---

## 2. 上游有什么可以用

| 出处 | 用途 |
|---|---|
| `paddle/phi/api/include/tensor.h:135` | `Tensor(const Place&, const std::vector<int64_t>& shape)` |
| `paddle/phi/api/include/tensor.h:199,224,248` | `shape()` / `dtype()` / `is_dense_tensor()` |
| `paddle/phi/api/include/tensor.h:492` | `Tensor copy_to(const Place&, bool blocking) const` |
| `paddle/phi/ops/yaml/ops.yaml:5443` | `strided_slice` 算子 —— **切片的落地实现** |
| `paddle/phi/kernels/stride/strided_slice_kernel.cc` | stride kernel,配合 `FLAGS_use_stride_kernel` |
| `paddle/phi/kernels/stride/` | 共 6 个 stride kernel,决定哪些操作是零拷贝 view |

> 📌 **Insight7 在本项目里不只是"参考实现"了。** D26 已把它定为
> **numpy 的位置、一等公民**,`paddle.from_insight` / `paddle.to_insight`
> 是 P5 的产出之一。互操作设计、零拷贝的四个前提、
> **它的 `axis` 目前是 0-based —— 已拍板改成 1-based(R24 / D29,"顺手修",P12 前完成)**,
> 全在 `plan/foundations.md` §3。
> **开工前必读那一节。**

**Insight7 的参考实现**(同一套字符串索引方案的既有实现):

| 出处 | 用途 |
|---|---|
| `Insight7 include/insight/core/slice.h:67` | `Slice parse_slice(const std::string& spec)` |
| `Insight7 src/core/array.cpp:628` | `Array::operator[](const std::string& spec)` —— 逗号分割 + 逐维解析 |
| `Insight7 bindings/lua/insight_lua.cpp:212-266` | `lua_spec_to_cpp()` —— **1-based 到 0-based 的字符串重写** |

---

## 3. 设计

### 3.1 索引:统统 1-based

已定决策 D-R6,**不要重新论证**。理由与代价见 `plan/overview.md` §6.1.1。

| 用户写 | 含义 | 转换后送给 C++ |
|---|---|---|
| `x[1]` | 第一个元素 / 第一行 | `0` |
| `x[{2, 3}]` | 第 2 行第 3 列 | `(1, 2)` |
| `x[-1]` | 最后一个 | `-1`(**负数不转换**) |
| `x["1:3"]` | 第 1、2 个 | `"0:2"` |
| `x["1:5:2"]` | 第 1、3 个 | `"0:4:2"` |
| `x["::-1"]` | 反转 | `"::-1"`(**step 不转换**) |
| `x[":"]` | 全部 | `":"` |
| `paddle.sum(x, 1)` | 沿第 1 维求和 | `axis=0` |

**转换规则(照抄 Insight7 `insight_lua.cpp:212-266`):**

```
只转换 start 与 stop 两个字段;step 原样透传。
字段值 > 0  → 减 1
字段值 <= 0 → 不变
字段为空    → 不变
```

**为什么负数不转换:** `-1` 在两种约定下都表示"最后一个"。
这不是巧合,而是 Python 负索引本来就是从末尾数,与起点是 0 还是 1 无关。

### 3.2 字符串索引:为什么不用 table

Lua 的 `__index` **只接受一个键**,天然表达不了 `x[1:3, :]`。三个选项:

| 方案 | 写法 | 评价 |
|---|---|---|
| 方法调用 | `x:slice{{1,3},{}}` | 啰嗦,且 `{}` 表示"全部"很不直观 |
| 表键 | `x[{ {1,3}, ":" }]` | 嵌套表,可读性差 |
| **字符串** | `x["1:3, :"]` | **与 Python 逐字符一致,零学习成本** |

选字符串(D-R7)。代价是**语法错误只能在运行时报**,所以解析器的报错必须精确到列:

```
paddle: bad slice spec "1:3, ::" at column 7: empty step
```

### 3.3 `__index` 分派表

承接 `02-binding.md` §3.5,**属性和方法优先于切片**:

```
键类型      →  处理
─────────────────────────────────────
number      →  第一维下标(1-based)
table       →  多维下标或混合切片
string
  ├ 已知属性 →  shape / dtype / place / grad / ndim ...
  ├ 已知方法 →  sum / mean / reshape / to ...
  └ 其它     →  切片表达式
```

**必须有一条兜底诊断:** 用户写 `x.shpae`(拼错)时,不应该报
"bad slice spec",而应该报 "unknown attribute 'shpae', did you mean 'shape'?"。
判据:字符串里**不含** `:` 和 `,` 且不是数字时,当作属性名拼写错误处理。

### 3.4 运算符

| Lua 元方法 | 映射 | 5.1 可用 |
|---|---|---|
| `__add` `__sub` `__mul` `__div` `__unm` `__pow` `__mod` | 逐元素算子 | ✅ |
| `__eq` `__lt` `__le` | ⚠️ **不实现** | 见坑③ |
| `__tostring` | 打印,截断大张量 | ✅ |
| `__len` | ⚠️ **不实现** | 5.1 对 userdata 不生效 |
| `__concat` | 不实现 | 语义不清 |

逐元素比较用 `x:eq(y)` / `paddle.equal(x, y)`,返回 bool 张量。

---

## 4. 已知的坑

**① `x["::-1"]` 是不是零拷贝,取决于 stride kernel 是否启用。**
`FLAGS_use_stride_kernel` 关掉时会退化成拷贝。这影响性能不影响正确性,
但**文档必须写清楚**,否则用户会以为反转是免费的。M0 第 14 项验证此事。

**② `__eq` 在 Lua 里必须返回 boolean。** Lua 会对 `__eq` 的返回值做真假转换,
返回张量是无效的 —— 5.1/5.2 甚至要求两个操作数元表相同才调用 `__eq`。
所以**不实现 `__eq`**,让 `x == y` 走默认的引用比较(同一对象才 true),
这与 Python 不同但**至少是可预测的**。文档里明写这一条。

**③ 打印大张量会卡住。** `__tostring` 必须截断:超过 1000 个元素只打印前后各 3 个,
并显示 shape/dtype/place。**而且要先把数据搬到 host** —— GPU 张量直接读指针是错的。

**④ 0 维张量(标量)。** Paddle 有真正的 0 维张量。`x:item()` 返回 Lua number,
`x[1]` 对 0 维张量应报错而不是返回自己。

---

## 5. 验收

- [ ] 索引测试全绿,至少覆盖:`x[1]` `x[-1]` `x[{2,3}]` `x["1:3, :"]` `x["::-1"]`
      `x["::2"]` `x[":"]` `x["2:"]` `x[":3"]`
- [ ] 每一条与 Python 对应写法结果逐位一致(转换脚本自动对拍)
- [ ] 越界索引报错,信息中的下标是 **1-based**(用户看到的数字和他写的一致)
- [ ] `x.shpae` 报拼写错误而不是切片错误
- [ ] 0 维张量的 `item()` / 打印 / 索引行为正确
- [ ] 大张量打印不卡、不 OOM
- [ ] 五个 Lua 版本行为一致

---

## 6. 未解问题

- Q-05 `x["::-1"]` 是否零拷贝(M0 第 14 项)
- 是否支持布尔掩码索引 `x[mask]`?**v1 不做**,用 `paddle.masked_select(x, mask)`
