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

**生成器本身用 Lua 写(R42,2026-10-23 人拍板)。** 原 C9 只约束"产物不带 Python",
允许生成器用 Python;用户要求升级:「代码生成全部用 lua 重新复刻,
用户的电脑上面不装 python 也可以用 Paddle」。所以 `tools/gen/` 是纯 Lua 5.1 子集,
**开发期工具链也零 Python** —— 生成、yaml 解析、index 语义检查,全在 Lua 侧完成。
Paddle 自己的 Python 生成器(`python_c_gen.py` / `codegen_utils.py`)**不再参与构建**,
只留作行为参考与 CI 对拍基准(R42)。

---

## 2. 上游有什么可以用

**整个项目复用率最高的一块 —— 上游已有功能完全同构的生成器,它是我们 Lua 版的「行为字典」。**

| 出处 | 内容 | 我们在 Lua 侧怎么用 |
|---|---|---|
| `paddle/phi/ops/yaml/ops.yaml` | 前向算子定义(如 `strided_slice` 在第 5443 行) | **输入,由 Lua 解析器直接读** |
| `paddle/phi/ops/yaml/backward.yaml` | 反向算子定义 | **输入,由 Lua 解析器直接读** |
| `paddle/fluid/eager/auto_code_generator/generator/python_c_gen.py` | **1103 行,pybind 层生成器** | **只作行为参考**,不 import、不运行。它的每一条逻辑都要在 Lua 里找到对应实现(R42) |
| `python_c_gen.py:53-73` | `atype2func` 映射表:`"bool" -> "CastPyArg2Boolean"` 等 20 条 | 把 20 条映射**手工翻译**成 Lua 表(静态翻译,只做一次) |
| `python_c_gen.py:160` | `PYTHON_C_FUNCTION_TEMPLATE` —— 单个函数的代码模板 | 字符串模板在 Lua 里等价改写(`string.gsub` 占位) |
| `python_c_gen.py:1020` | `--api_yaml_path` 命令行参数 | Lua 侧用 `arg` 表解析等价的 CLI |
| `paddle/fluid/eager/auto_code_generator/generator/codegen_utils.py` | 664 行,yaml 解析与类型工具 | **语义参照,R42 要求零 Python** —— 用 Lua 5.1 子集实现等价功能(见 §3.1 的 `yaml_subset.lua`) |
| `paddle/fluid/eager/auto_code_generator/generator/eager_gen.py` | 3918 行,生成 `*_ad_func` 本体。**我们不需要,它已经编进 libpaddle 了** | 不需要 |
| `.../forwards/dygraph_functions.h` | 生成器的产物,我们的调用目标 | 签名出处(§4 反幻觉规程) |

**关键洞察:我们要写的是 `emit_capi.lua` / `emit_lua.lua`,它们与 `python_c_gen.py` 是同构的
两个后端。** 输入相同(yaml)、中间表示相同(我们自己用 Lua 解析)、调用目标相同(`*_ad_func`),
**只有出参入参的转换函数不同**:`CastPyArg2Boolean` -> `CastLuaArg2Boolean`。

这意味着这 3 周里有相当一部分是"照 1103 行的行为把后端翻成 Lua",而不是从零设计。
**唯一新增的、Python 版没有的组件是 `yaml_subset.lua`**(见 §3.1)—— 上游的 yaml 解析
藏在 codegen_utils.py 里,我们要用 Lua 重新实现。

**实测(2026-10-23):`ops.yaml` 全文无 YAML 锚点(`&`/`*`)、无多行标量(`|`/`>`)、无合并键(`<<`)。
** 这份文件只用到了块级列表、块级映射、行内 `args`、注释、标量 —— 一个约 300 行的
Lua 子集解析器足够,不需要完整的 YAML 实现。**上线前用全部 519+107 个 op 块做解析对拍,
发现新语法就扩展子集(R42 验收项)。

**⚠️ 生成器输入清单(2026-10-24 实测修正):`add` / `matmul` 不在 `ops.yaml` 里!**
上游 3.5 把基础算术算子挪进了 `inconsistent/dygraph_ops.yaml`(38 块,`add` 在 :4、
`matmul` 在 :248)。`ops.yaml` 只有 529 块,不含 `add`/`matmul`/`multiply` 这类
基础算子。**上游生成器的真实输入是 9 份 yaml**(`build-nopy/build.ninja:31039` 实测):
`ops.yaml` + `inconsistent/dygraph_ops.yaml` + `sparse_ops.yaml` + `fused_ops.yaml` +
`strings_ops.yaml` + `backward.yaml` + `inconsistent/dygraph_backward.yaml` +
`sparse_backward.yaml` + `fused_backward.yaml`。我们的 `yaml_subset.lua` 已对
`inconsistent/dygraph_ops.yaml` 验证可解析(38 块)。**2.3.4 全量生成时输入必须含
`inconsistent/dygraph_ops.yaml`,否则基础算子静默缺失。**

---

## 3. 设计

### 3.1 数据流

```
ops.yaml + backward.yaml
        │
        │  yaml_subset.lua(纯 Lua 5.1 子集解析器,~300 行)
        ▼
   中间表示(算子名 / 输入 / 属性 / 输出 / 默认值)
        │
        ├──► emit_capi.lua → csrc/generated/ops_*.c      调用 xxx_ad_func
        ├──► emit_lua.lua  → lua/paddle/_ops/*.lua       1-based 转换 + kwargs 分派
        └──► gen_doc.lua   → docs/api/ops.md             (M2 之后)
```

**四个 `.lua` 全是纯 Lua 5.1 语法,跑在 `lua` / `luajit` 上,零 Python(C9/R42)。**
`yaml_subset.lua` 是唯一新增组件 —— 上游的解析藏在 codegen_utils.py 里,
我们按它的语义用 Lua 重写,输出 IR 形状与 `codegen_utils.py` 对齐,
这样 emit 逻辑可以和 Python 原版逐条对拍。

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
paddle.sum{ x, axis = 1 }             -- 混合表:数组部分 + 具名键
```

> ⚠️ 混合是**一张表内部**的事(`argrule.md` §2.5⑧)。
> `paddle.sum(x, {axis=1})` 这种「位置参数 + 尾随选项表」的写法**不成立** ——
> 那是两个实参的位置调用,规则 #2 `axis:number` 会收到一张 table 并报错。

**`axis1based` 标记是 1-based 决策(D-R6)的落地点。**
生成器根据 yaml 里的参数名判定哪些参数是轴,自动插入 `axis > 0 and axis - 1 or axis`。
**这个转换只在生成的 wrapper 里出现一处,不散落在用户代码里** —— 这是选 1-based 时
承诺的"边界在库里"。

### 3.4 哪些算子跳过

`python_c_gen.py:49` 有个 `SkipAPIGeneration(forward_api_name)`,先照它的跳过名单的**行为**翻译成 Lua,
再加我们自己的:

| 跳过类别 | 理由 |
|---|---|
| 名字以 `_` 结尾的 in-place 变体 | 第一版不暴露,减少 API 面积。M2 再加 |
| 需要 `pir::Value` 参数的 | 静态图专用,P15 再处理 |
| `sparse_*` / `strings_*` | 有独立 yaml,不在 v1 范围 |
| 分布式相关(`c_allreduce_*` 等) | P17 单独处理 |
| 图学习/树模型/分布式训练器算子(`graph_khop_sampler`/`graph_sample_neighbors`/`weighted_sample_neighbors`/`reindex_graph`/`dgc`/`tdm_sampler`) | `row`/`colptr` 是 0-based CSC 结构偏移(前缀和,`colptr[0]==0`),节点重编号必须跨输入输出保持一致,不是 index_semantics 的「逐参数 ±1」能表达的;图学习模块后续单独设计。P17 |

> **注意**:skip 名单是生成器的单一事实来源,集中在 `tools/gen/skip_rules.lua`,
> `emit_capi.lua` / `emit_lua.lua` / 覆盖统计工具共用,禁止在别处重复判断。

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

**⑤ 输入形态有 4 种,不是只有 `Tensor`。** 编译验证(2026-10-24,墙 #2)抓到的
真实缺口——`signature()` 把所有输入一律生成 `pd_tensor` 传 `->value`,但 yaml
输入的 C++ 类型经过 `yaml_types_mapping` 后有 4 种(`ops.yaml` 普查):`Tensor`(921)、
`Tensor?`(160)、`std::vector<Tensor>`(37)、`std::vector<Tensor>?`(20)。后三种的
ad_func 形参分别是 `paddle::optional<paddle::Tensor>&`、
`const std::vector<paddle::Tensor>&`、`paddle::optional<std::vector<paddle::Tensor>>&`
(证据:`dygraph_functions.h:131` bicubic_interp)。C ABI 表示(01-c-abi.md §4 数组传
(ptr,n) 的推广):`Tensor?` → `pd_tensor`,`NULL` = 无,调用点
`<name> ? paddle::make_optional(->value) : paddle::none`;`std::vector<Tensor>` →
`(const pd_tensor* p, int64_t n)` + prelude 循环构造 vector;
`std::vector<Tensor>?` → 同上,NULL = 无。**坑:`optional<Tensor>` 能从 `Tensor`
隐式构造(optional.h:482 非 explicit),所以「漏掉 optional 处理」的生成代码照样
编译过但语义是错的** —— 编译验证必须覆盖 `optional<vector<Tensor>>` 这种
不能隐式转换的形态(如 bicubic_interp 的 `size_tensor`),否则会漏。

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

- ~~`codegen_utils.py` 能否在不改动的前提下被我们 import?~~ **已被 R42 作废** ——
  生成器零 Python,不存在 import。改问:**`yaml_subset.lua` 对 4 份 yaml 的语法覆盖度**。
  **✅ 已闭环(2026-10-24,2.3.2)**:`tools/gen/yaml_subset.lua` 写完即做全量对拍
  —— `ops.yaml` 529 / `backward.yaml` 396 / `legacy/static_ops.yaml` 109 /
  `op_compat.yaml` 548 个块,**IR 逐块与 Python 原版(PyYAML + codegen_utils.py 语义)一致,
  四份全绿**。对拍中发现并支持的语法:
  - **多行 plain scalar 折叠**(ops.yaml:75 的 `args :` 跨行):续行按单空格折回
  - **`output :` 空值**(静态图算子,如 `comm_init_all` / `p_send`):returns = 空列表
  - **嵌套块**(`infer_meta :` / `kernel :` 下的 4 空格键值):按缩进跳过,
    2 空格字段行恢复采集(nesting 按层级分派,不是"一进嵌套就吞到尾")
  - **花括号默认值**(`IntArray axis={0}`):split_args 的逗号切分用
    `,(?![^{]*\})` 语义模拟(Lua 5.1 pattern 无负向先行,手写扫描)
  - **`- op :` / `- backward_op :` 两种块头**:kind 分别为 forward / backward_op
  - 上游字段全集已固化:ops.yaml 14 种(args/output/infer_meta/kernel/inplace/
    interfaces/traits/backward/optional/data_transform/intermediate/invoke/
    no_need_buffer/view)+ backward.yaml 的 forward/composite 行内签名

- `inplace: (x -> out)` 这类**行内映射**:✅ **已验证(2.3.2 对拍覆盖)** ——
  上游 `ParseYamlInplaceInfo:477` 行为 = 按逗号切分 → 去首尾括号 → 按 `->` 切左右,
  多输出形式(`(x -> out0), (y -> out2)` 已在 ops.yaml 真实出现,如 `adagrad_:94`)
  已按同语义实现并通过对拍。
