# paddle.scope

| | |
|---|---|
| 阶段 | P6(核心回收入口)/ P14(与显存分配器联动) |
| 依赖模块 | `paddle`(顶层)、GC 机制(`14-gc.md`) |
| 稳定性 | **稳定**(作为文档默认写法) |
| 设计来源 | **我们自己的设计**,非上游移植 —— `research/gc.md` §4 第 6 层 |
| 实现文档 | `plan/modules/14-gc.md` |

> 📌 **本模块与前面所有 api 文档不同:上游 Python 没有 `paddle.scope`。**
> 它是 gc.md 九层方案里**唯一用户可见的一层**(`gc.md:246/261`),
> 用纯 Lua 5.1 语法实现(~40 行),是"训练大模型 + 显存吃紧"场景下
> **文档里推荐的默认写法**(Torch7 做晚了的事,我们从第一天就做)。

---

## 1. 导出清单

| 导出 | 类型 | 阶段 | 稳定性 |
|---|---|---|---|
| `paddle.scope(fn)` | 函数 | P6 | 稳定 |
| `paddle.scope.enable(bool)` | 函数 | P6 | 稳定 |
| `paddle.scope.threshold(bytes)` | 函数 | P6 | 稳定 |

**没有其他导出。** `t:free()`(gc.md 第 5 层)仍是独立兜底,不属于 scope。

---

## 2. 详细签名

### 2.1 `paddle.scope(fn)`

```lua
paddle.scope(function()
  local x = make_big_tensor()   -- 作用域内创建的 Tensor
  local y = forward(x)
  return y                       -- y 会逃逸,不回收
end)
```

**语义**:进入函数时记住当前"live tensor 水位";函数返回时,把作用域内
新建且**未逃逸**(未作为返回值/未存入外部引用)的 Tensor 显式回收
(等效 `t:free()` + 通知 GC 施压,`gc.md` 第 4/5/6 层联动)。

**逃逸规则(写入契约)**:
1. 函数**返回值**(及其指向的对象)不回收
2. 被**外部 upvalue 修改捕获**的对象不回收(保守:不可判定就留下)
3. 其余作用域内新建的 `paddle.Tensor` 一律回收

⚠️ **逃逸判定是保守的**:Lua 无法精确做逃逸分析,规则 2 采用"被捕获就
不碰"的保守策略 —— 宁可漏回收,不可悬垂。这正是 Torch7 的教训。
它是**建议性回收**(性能优化),不是正确性依赖 —— 删错了也必须有别的
机制兜底(引用计数是第一道防线,gc.md 第 1 层)。

### 2.2 `paddle.scope.enable(value)`

```lua
paddle.scope.enable(false)    -- 关闭(默认开?见 §6 ①)
```

开关。嵌套 scope 语义:内部 `enable(false)` 只影响其所在 scope,外层不受影响。

### 2.3 `paddle.scope.threshold(bytes)`

```lua
paddle.scope.threshold(1e8)   -- 100MB
```

设置"作用域内 live bytes 超过阈值才真正跑 GC 施压"的阈值。
低于阈值直接返回(零开销);超过才 `lua_gc(L, LUA_GCSTEP, n)`(`gc.md:246`)。

---

## 3. index 语义参数

**本模块没有 index 语义参数。** scope 是纯控制流结构,无维度/张量索引。
显式写"无"以证明看过(`api/README.md` §2.2)。

---

## 4. 与 Python 的差异

| # | Python | Lua | 为什么 |
|---|---|---|---|
| 1 | **无此 API**(上游没有 `paddle.scope`) | `paddle.scope(fn)` | 我们的 gc 第 6 层,`gc.md:246` |
| 2 | `with` 语句(若有) | **函数式** `scope(fn)` | Lua 无 with;函数式与 `no_grad(fn)` 一致(见 `autograd.md` §6 ①) |
| 3 | — | `scope.enable/threshold` | 控制子项,纯 Lua 配置 |

**迁移注意**:Python 用户**不需要**从任何东西迁移过来 —— 这是纯增量的
Lua 侧 API。但**文档教程里它是默认写法**,与 Python 侧写 `del x` /
`torch.cuda.empty_cache()` 的地位对应。

---

## 5. 用法示例

```lua
local paddle = require "paddle"

function train_step(batch)
  local loss
  paddle.scope(function()
    local x = batch:to_insight()
    local y = model:forward{x}     -- 作用域内中间张量
    loss = criterion(y, batch.label)
  end)
  -- 中间张量已回收,只剩 loss
  loss:backward()
  opt:step(); opt:clear_grad()
end
```

嵌套与逃逸:

```lua
local keep
paddle.scope(function()
  local a = paddle.zeros{100, 100}
  keep = paddle.zeros{10}      -- 被外部 upvalue 捕获 → 不回收
end)
print(keep)                    -- 有效
-- a 已回收
```

---

## 6. 本模块特有的坑

**① 默认开关要尽早定。** `gc.md:261` 说"应该从第一天就把 paddle.scope 做成
文档里推荐的默认写法"。**倾向默认开**(与 D6"堆追踪默认开启"一致),但
`threshold` 默认值要与 P6 训练基准一起定 —— 阈值太低会在小模型上产生
无谓的 GC 施压,太高则大模型救不回来。P6 实测后定,文档先留 TODO。

**② 逃逸分析是保守的,不是精确的。** 规则 2"被外部捕获就不碰"意味着:
作用域内创建、被塞进外部 table 的 Tensor 会漏回收。这是**故意的成本**
(悬垂 > 泄漏)。文档必须写明,否则用户会以为 scope 是精确的 GC 作用域。
**scope 是建议性回收,不是正确性依赖** —— 这句话写进 docstring 头部。

**③ 嵌套 scope 的回收边界。** 内层 scope 返回后,外层 scope 接着跑;
内层"看到"的外层对象**不属于内层职责**。封闭原则:每个 scope 只收自己
进入后被创建的、且未逃逸的对象。实现上用"进入时水位"记录(**gc.md:246**
的 live bytes 统计),而不是"离开时全清"。

**④ 与 `t:free()`(第 5 层)的关系。** `t:free()` 是显式单对象兜底;
`paddle.scope` 是作用域批量。两者不冲突,`t:free()` 在 scope 内用也有效。
作用域回收本质上是对"忘了 free"的自动补位 —— 用户仍然应该学会
`t:free()`,scope 不是免死金牌(§6 ②)。

**⑤ 错误路径要恢复水位。** `fn` 抛异常时,离开 scope 必须先恢复水位
再传播错误 —— 否则错误后的 scope 水位错乱,后续所有 scope 都按错误水位
施压。用 Lua 的 `xpcall` 包一层,`finally` 语义手动写(5.1 无 finally)。

---

## 7. 未实现 / 延后

| 项 | 状态 |
|---|---|
| 默认阈值 | **P6 训练基准实测后定**(先不拍数字) |
| 与 CUDA 显存池的联动(`cuda_empty_cache` 类比) | P14 |
| scope 内 Tensor 数/字节数的诊断接口(`scope.stats()`) | P6 顺手做,调试用 |
| `<close>` 单对象作用域(5.4+) | 可选,`gc.md:245` 第 7 层;不做也不阻塞 |