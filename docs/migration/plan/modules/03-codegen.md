# P3 · 算子代码生成

| | |
|---|---|
| 阶段 | P3 |
| 类别 | 生成器(开发期工具,不随产品分发) |
| 开工条件 | P1 完工(可与 P2 并行) |
| 预估 | 3 周 |

---

## 1. 做什么 / 不做什么

**做:** 读 Paddle 自己的 yaml,生成约 4000 个算子的 C ABI 包装 + Lua 侧签名。

**不做:** 不手写任何一个算子绑定。C8 是硬约束 —— **发现有人手写算子绑定,直接回滚**。

**生成器本身用 Python 写。** 这不违反"不需要 Python 运行时":
生成器是**开发期工具**,产物是纯 C/Lua 源码,最终用户装的包里没有 Python,
也没有 yaml,也没有生成器。这和 Paddle 自己用 Python 生成 C++ 是同一个道理。

---

## 2. 上游有什么可以用

**这是整个项目复用率最高的一块 —— 上游已经有一个功能完全同构的生成器。**

| 出处 | 内容 |
|---|---|
| `paddle/phi/ops/yaml/ops.yaml` | 前向算子定义(如 `strided_slice` 在第 5443 行) |
| `paddle/phi/ops/yaml/backward.yaml` | 反向算子定义 |
| `paddle/fluid/eager/auto_code_generator/generator/python_c_gen.py` | **1103 行,pybind 层生成器 —— 我们的直接蓝本** |
| `python_c_gen.py:53-73` | `atype2func` 映射表:`"bool" -> "CastPyArg2Boolean"` 等 20 条 |
| `python_c_gen.py:160` | `PYTHON_C_FUNCTION_TEMPLATE` —— 单个函数的代码模板 |
| `python_c_gen.py:1020` | `--api_yaml_path` 命令行参数 |
| `paddle/fluid/eager/auto_code_generator/generator/codegen_utils.py` | 664 行,yaml 解析与类型工具,**可直接复用** |
| `paddle/fluid/eager/auto_code_generator/generator/eager_gen.py` | 3918 行,生成 `*_ad_func` 本体。**我们不需要,它已经编进 libpaddle 了** |
| `.../forwards/dygraph_functions.h` | 生成器的产物,我们的调用目标 |

**关键洞察:我们要写的是 `lua_c_gen.py`,它与 `python_c_gen.py` 是同一个东西的两个后端。**
输入相同(yaml)、中间表示相同(`codegen_utils.py`)、调用目标相同(`*_ad_func`),
**只有出参入参的转换函数不同**:`CastPyArg2Boolean` -> `CastLuaArg2Boolean`。

这意味着这 3 周里有相当一部分是"照着 1103 行改后端",而不是从零设计。

---

## 3. 设计

### 3.1 数据流

```
ops.yaml + backward.yaml
        │
        │  复用 codegen_utils.py 的解析
        ▼
   中间表示(算子名 / 输入 / 属性 / 输出 / 默认值)
        │
        ├──► gen_c.py     → csrc/generated/ops_*.c      调用 xxx_ad_func
        ├──► gen_lua.py   → lua/paddle/_ops/*.lua       1-based 转换 + kwargs 分派
        └──► gen_doc.py   → docs/api/ops.md             (M2 之后)
```

### 3.2 生成的 C 侧长什么样

对 `ops.yaml` 里的 `multiply`:

```c
/* GENERATED — DO NOT EDIT. source: ops.yaml multiply */
pd_tensor pd_op_multiply(pd_tensor x, pd_tensor y) {
  PD_GUARD({
    auto out = multiply_ad_func(*AS_T(x), *AS_T(y));
    return new_handle(std::move(out));
  })
  return NULL;
}
```

### 3.3 生成的 Lua 侧长什么样

```lua
-- GENERATED — DO NOT EDIT. source: ops.yaml sum
local wrap = require "paddle._wrap"
M.sum = wrap("sum", {"x", "axis", "dtype", "keepdim"},
             { axis = "axis1based", dtype = "dtype", keepdim = false })
```

`_wrap` 提供三模式调用(参考 Insight7 `bindings/lua/insight/_wrap.lua`):

```lua
paddle.sum(x, 1)                      -- 位置
paddle.sum{ x = x, axis = 1 }         -- 关键字
paddle.sum(x, { axis = 1 })           -- 混合
```

**`axis1based` 标记是 1-based 决策(D-R6)的落地点。**
生成器根据 yaml 里的参数名判定哪些参数是轴,自动插入 `axis > 0 and axis - 1 or axis`。
**这个转换只在生成的 wrapper 里出现一处,不散落在用户代码里** —— 这是选 1-based 时
承诺的"边界在库里"。

### 3.4 哪些算子跳过

`python_c_gen.py:49` 有个 `SkipAPIGeneration(forward_api_name)`,先照抄它的跳过名单,
再加我们自己的:

| 跳过类别 | 理由 |
|---|---|
| 名字以 `_` 结尾的 in-place 变体 | 第一版不暴露,减少 API 面积。M2 再加 |
| 需要 `pir::Value` 参数的 | 静态图专用,P15 再处理 |
| `sparse_*` / `strings_*` | 有独立 yaml,不在 v1 范围 |
| 分布式相关(`c_allreduce_*` 等) | P17 单独处理 |

### 3.5 生成器必须可重入

**同样的输入跑两次,产物必须逐字节相同。** 这意味着:

- 遍历 yaml 时**排序**,不依赖 dict 迭代顺序
- 不写时间戳、不写机器名、不写绝对路径
- 生成文件头写的是 **yaml 里的算子名**,不是"generated at 2026-08-03"

理由:产物要进 git。不可重入的生成器会让每次跑生成都产生几万行虚假 diff,
review 就废了。

---

## 4. 已知的坑

**① yaml 会随 Paddle 版本变。** 生成器必须记录它是针对哪个 Paddle 版本生成的
(写进 `csrc/generated/VERSION`,内容是 Paddle 的 git commit)。
升级 Paddle 时重跑生成器,diff 就是这个版本的 API 变更清单 —— 这其实是个额外好处。

**② 4000 个函数会撑爆链接器。** Windows 上单个 `.obj` 有符号数上限,
MSVC 的 `/bigobj` 是必须的。同时要把生成的 C 文件**按首字母分片**(`ops_a.c`, `ops_b.c` …),
不要生成一个 5 万行的巨型文件。

**③ 默认值的表示是最容易出错的地方。** yaml 里 `beta1 = 0.9f` 这类默认值,
在 C、Lua、文档三处必须一致。**唯一正确的做法是从 yaml 单一来源生成,任何一处手写都会漂移。**
`dygraph_functions.h` 里 `adam__ad_func` 有 8 个带默认值的参数,这不是小概率场景。

**④ 别对 `optional<Tensor>` 想当然。** `dygraph_functions.h:12` 的
`paddle::optional<paddle::Tensor*> predefined_out = paddle::none` 说明新版 API
带了输出复用参数。生成器要能识别并**先跳过它**(传 `paddle::none`),
而不是错误地把它当成必填输入。

---

## 5. 验收

- [ ] 覆盖 `ops.yaml` 中 >= 95% 的算子,未覆盖的逐条列出理由
- [ ] 连跑两次,`git diff` 为空(可重入)
- [ ] 随机抽 20 个算子,Lua 调用结果与 Python 逐位一致
- [ ] 抽 5 个带默认值的算子,不传该参数时行为与 Python 一致
- [ ] 抽 5 个带 axis 的算子,验证 1-based 转换正确(含负数轴不转换)
- [ ] 生成物全部带 `GENERATED — DO NOT EDIT` 头,且 `csrc/generated/` 在
      `.gitattributes` 里标 `linguist-generated`
- [ ] 生成耗时 < 60 秒

---

## 6. 未解问题

- `codegen_utils.py` 能否在不改动的前提下被我们 import?
  若它 import 了 Paddle 的 Python 包,我们就要把需要的部分**复制**到 `tools/codegen/`
  并注明来源与 commit(这属于开发期工具,不违反 C2)
