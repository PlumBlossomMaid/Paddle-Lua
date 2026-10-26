# WORKPLAN · 总工程树

> **这不是说明文档,是执行计划。**
> 智能体按本文件做**深度优先遍历**;**树遍历完 = 工程完成**。
>
> 约束与决策在 `/AGENTS.md`;当前位置在 `process/status.md`;
> 每个叶节点的做法在它自己的文档里。

---

## 1. 遍历协议

```
从根开始,深度优先,左到右。
遇到内部节点 -> 展开它的子节点,继续下潜。
遇到叶节点   -> 读它的文档,执行,验收,回填状态,回溯。
```

**四个状态:**

| 标记 | 含义 | 谁能改 |
|---|---|---|
| ⬜ | 未开始 | — |
| 🔵 | 进行中(**同一时刻全树只能有一个**) | 智能体 |
| ✅ | 完工判据已通过 | 智能体(但判据必须是可执行的,见 §3) |
| ⛔ | 阻塞 —— 前置未满足 / 环境不具备 / 等人拍板 | 智能体标记,**必须同时写明阻塞原因** |

**恢复:** 只需要两样东西 —— 树里唯一那个 🔵,和 `status.md` 里的一行现场笔记。
不需要维护显式的栈,因为**从根重新 DFS 一遍会走到同一个位置**(状态在树上,不在内存里)。

**遇到 ⛔ 的处理:** 不停在那里等。标记原因,**回溯到父节点,走下一个兄弟**。
把阻塞的推到后面,而不是让整棵树停住。
但**如果阻塞的是硬依赖的前置**(下面所有兄弟都需要它),那就必须停 —— 例如 P0。

---

## 2. 为什么朴素 DFS 是安全的

⚠️ **这一节是本文件最容易被忽略、又最容易出事的地方。**

**树是分解结构(谁属于谁),依赖是 DAG(谁必须在谁之前)。两者形状不同。**
`roadmap.md` §0 里 P5 依赖 P2/P3/P4,P9 依赖 P5/P6 —— 这是图,不是树。

如果兄弟节点的顺序随便排,DFS 会走到一个**前置还没做**的叶子上:
先进 `nn/` 子树写 `Linear`,而 `tensor.lua` 还不存在。

**解法:把兄弟顺序排成拓扑序,让 DFS 的访问顺序 = 依赖的合法执行顺序。**

这做得到,因为 `roadmap.md` 的 P0→P18 本来就是拓扑排序过的。
于是:

| | |
|---|---|
| 主机制 | **兄弟顺序 = 拓扑序**。按顺序走就不会踩空 |
| 安全网 | 每个节点仍写 `前置:` 字段。执行前**断言**它已 ✅;不满足就是树排错了,**报错,不要将就着往下做** |

**代价:失去并行。** P8 本可以和 P6 并行,树把它们串起来了。
这是**故意的** —— 单人 + 单智能体的场景下,并行只会制造"两处都做了一半"。
真要并行,人手动挑一个 ⬜ 分支去做,协议不阻止。

---

## 3. 叶节点的规格

**叶节点是"一个智能体会话内能做完的事"。** 做不完就说明它还不是叶子,
要再挂一层指针,分裂成更小的节点。

每个叶节点文档**必须**有这四样(`ci.md` §5 的 `doctree` job 机器检查):

| 字段 | 为什么必须 |
|---|---|
| **前置** | 没有它,DFS 的安全网失效(§2) |
| **产出** | 具体到文件路径。"完善一下 X" 不是产出 |
| **完工判据** | **必须可执行** —— 一条命令 / 一个 CI job。不能是"看起来对了" |
| **踩坑记录** | 执行完回填。**这是树最大的长期价值** —— 下一个人不会再踩 |

> 判据不可执行,是这套机制唯一的真实失效模式。
> 一个写着"实现 Layer 基类"、判据是"实现完成"的叶子,
> 会被标成 ✅ 然后把问题留给三个月后的自己。**判据必须能被机器判定。**

---

## 4. 工程树

> 一行一个节点。缩进 = 层级。`->` 后面是该节点的文档(指针)。
> **状态只在本文件维护**,叶子文档里不记状态,避免两处不同步。

```
[⬜] Paddle-Lua
 |
 +-- [🔵] 0. 论证与规划                     -> docs/migration/README.md
 |    +-- [✅] 0.1 可行性论证                -> research/feasibility.md
 |    +-- [✅] 0.2 架构选型                  -> research/architecture.md
 |    +-- [✅] 0.3 GC 与显存                 -> research/gc.md
 |    +-- [✅] 0.4 DataLoader 方案           -> research/dataloader.md
 |    +-- [✅] 0.5 C++ 资产复用              -> research/reuse.md
 |    +-- [✅] 0.6 动转静路线                -> research/to-static.md
 |    +-- [✅] 0.7 范围与总账                -> plan/overview.md
 |    +-- [✅] 0.8 阶段顺序                  -> plan/roadmap.md
 |    +-- [✅] 0.9 生态基座                  -> plan/foundations.md
 |    +-- [✅] 0.10 代码目录与模块           -> plan/layout.md
 |    +-- [✅] 0.11 CI 计划                  -> plan/ci.md
 |    +-- [✅] 0.12 工程树(本文件)         -> WORKPLAN.md
 |    +-- [✅] 0.13 API 设计规范 + 样板       -> plan/api/README.md + plan/api/io.md
 |    +-- [✅] 0.14 参数检查选型(argcheck)  -> plan/foundations.md §4   前置:0.9 0.13
 |    +-- [✅] 0.15 L0 CI 落地               -> plan/ci.md §2   前置:0.11 0.14
 |
 +-- [🔵] 1. M0 可行性闸门 (G0/G1)          -> plan/modules/00-build.md
 |    +-- [✅] 1.1 无 Python 构建 ★生死判定  -> process/tasks.md T-M0-01
 |    +-- [✅] 1.2 C++ 侧反向验证            -> process/tasks.md T-M0-02
 |    +-- [✅] 1.3 线程安全验证              -> process/tasks.md T-M0-03
 |    +-- [✅] 1.4 Lanes 三项验证            -> process/tasks.md T-M0-04..06
 |    +-- [✅] 1.5 GC / OOM 验证             -> process/tasks.md T-M0-07..10
 |    +-- [✅] 1.6 序列化与执行器验证        -> process/tasks.md T-M0-11..19
 |    +-- [⬜] 1.7 参数签名层(独立项目)     -> plan/argrule.md(孵化说明书)
 |         1.7.0 ✅ 定名 argrule(P10 已拍板)-> ⬜ 去 luarocks 占位
 |         1.7.1 Penlight 分发方式拍板(待拍板 P9)
 |         1.7.2 三道墙实测(5 个 Lua × upvalue/局部/寄存器,= M0 #24)
 |         1.7.3 调用约定层(位置 <-> 具名 + 默认值 + usage,~200 行)
 |         1.7.4 Insight7 先接入(它体量小,替 paddle-lua 踩坑)
 |         ⚠️ 与 1.1-1.6 **无依赖关系**(纯 Lua,不依赖 G0),可并行
 |         ⚠️ 但**必须在 2.5(P5)之前可用** —— 它决定所有公开 API 的签名形状
 |
 +-- [🔵] 2. M1 MVP:MNIST 能训练            前置:1  ✅ G1 已开(2026-10-23)
 |    +-- [✅] 2.1 C ABI 中间层              -> plan/modules/01-c-abi.md
 |    |    +-- [✅] 2.1.1 paddle_c.h 接口定义
 |    |    +-- [✅] 2.1.2 status.cc 异常边界     ★ 早做,见 layout.md §3
 |    |    +-- [✅] 2.1.3 tensor.cc
 |    |    +-- [✅] 2.1.4 autograd.cc
 |    |    +-- [✅] 2.1.5 纯 C 迷你训练循环      ★ 不许跳过
 |    |    +-- [✅] 2.1.6 device.cc / jit.cc
 |    +-- [✅] 2.2 sol2 绑定层               -> plan/modules/02-binding.md
 |    |    +-- [✅] 2.2.1 module.cc + error.cc
 |    |    +-- [✅] 2.2.2 tensor_binding.cc
 |    |    +-- [✅] 2.2.3 require 冒烟通过     ★ P2 的真里程碑(sol_smoke)
 |    |    +-- [✅] 2.2.4 slice.cc
 |    +-- [⬜] 2.3 算子代码生成              -> plan/modules/03-codegen.md
 |    |    +-- [✅] 2.3.1 index_semantics.lua  ★ 必须先于生成器
 |    |    +-- [✅] 2.3.2 yaml_subset.lua     ★ 4 份 yaml 对拍全绿(529/396/109/548 块)
 |    |    +-- [✅] 2.3.3 三算子试生成 + 人工 review ★ 首个端到端产物(add/matmul/relu)
 |    |    +-- [✅] 2.3.4 全量生成 + regen-diff CI  ★ 529 ops 全量落库 + regen_diff.lua 门
 |    +-- [✅] 2.4 构建与打包                -> plan/modules/04-packaging.md
 |    +-- [✅] 2.5 paddle.Tensor             -> plan/modules/05-tensor.md
 |    |    +-- [✅] 2.5.0 API 设计                -> plan/api/{top,tensor,dtype-device}.md ★先接口后实现
 |    +-- [✅] 2.6 自动微分与生命周期        -> plan/modules/06-autograd.md
 |    |    +-- [✅] 2.6.0 API 设计                -> plan/api/{autograd,scope}.md
 |    |    +-- [✅] 2.6.1 backward/grad/no_grad 绑定  ★ 2026-10-25(cd61d8e):T-M1-10 DONE,autograd_smoke 全过
 |    |    +-- [✅] 2.6.2 paddle.scope + memory_allocated  ★ 2026-10-25(ee03f65):scope_smoke 显存下降 PASS;OOM 回调随 T-M1-18(GC)
 |    +-- [✅] 2.7 序列化                    -> plan/modules/07-serialization.md ★ 单张量 save/load(D19)已完成;state_dict/manifest 归 P9/M2
 |    +-- [✅] 2.8 jit::Layer 推理 ★可先交付 -> plan/modules/08-jit-infer.md ★ 2026-10-25:T-M1-17 DONE,paddle.jit.load 全链路(JIT_SMOKE PASS);翻案 Q-24(@fetch 后缀断裂是 0xC0000005 真根因,patch 0002),新记 Q-25(宿主退出竞态)
 |    +-- [🔵] 2.9 paddle.nn ★最大一块       -> plan/modules/09-nn.md
 |    |    +-- [✅] 2.9.0 API 设计                -> plan/api/{nn,nn-functional,nn-initializer}.md  ★三文档已齐(2026-10-23):nn.md 基类/容器/核心层 + index 语义、nn-functional.md 37 激活/pool/norm、nn-initializer.md 可调用对象;核对通过(2026-10-25)
 |    |    +-- [✅] 2.9.1 Layer 基类 + 它的测试   ★ 2026-10-25:nn/layer.lua(FIELDS 私有键 + _create/_class_init 钩子 + __index/__newindex 自动注册 + order 书 + pre-order DFS 迭代器 + state_dict 键名与 Python 逐字符一致)+ nn/init.lua + 顶层 lazy 挂 nn;NN_LAYER_SMOKE PASS(15 节:三层继承/order 替换保持位置/nil 摘除/state_dict 键名/set_state_dict/train-eval 递归/print 不崩/200 层不栈溢/class.cast 禁用/List 返回/forward/cpu);三个迭代器 bug 实测修复(named_* stack 重建死循环、pre-order 跳过 b、training 落 FIELDS 顶层读不到——改走 plain 槽)
 |    |    +-- [✅] 2.9.2 Linear 打通闭环   ★ 2026-10-25:nn/layers/linear.lua + 挂载 + **构造函数与正式 API 一律 argrule 包装(用户指示,D31)**:Linear._init、Layer.create_parameter/add_parameter/add_sublayer 全部规则表化(self 规则 + 表内位置/具名/位置三形态,argrule 装于本机 luarocks);NN_LINEAR_SMOKE PASS(注册/state_dict 键名/forward 数值 y=xW+b/no-bias matmul 分支/backward 梯度)+ 全部 10 smoke 回归全绿 + lint 全绿。踩坑:argrule 的 `default=nil` 不算 optional(要 `opt=true`);表内位置不能留洞(`{2,3,nil,false}` 数组部分在 nil 截断——用混合表 `{2,3,bias_attr=false}`)
 |    |    +-- [🔵] 2.9.3 functional        -> plan/api/nn-functional.md
 |    |    |    +-- [✅] 2.9.3.1 激活函数(首批 16) ★ 2026-10-25:nn/functional.lua 16 个激活全部 argrule 包装(relu/leaky_relu/silu/sigmoid/tanh/gelu/softmax/log_softmax/softplus/hardswish/elu/celu/relu6/hardtanh/hardsigmoid/thresholded_relu/selu)+ nn/init.lua 挂载 + **NN_FUNCTIONAL_SMOKE PASS(45 断言,镜像 Python test_activation_op.py 公式逻辑)** + 10 smoke 回归全绿 + lint 全绿。**发现并修复 argrule 真实 bug**(独立仓库 E:\code\argrule,spec 8/8):parse_position_table 对「无规则名键的混合表」不查多余键,未知键被静默吞掉 → 现在 string 键/洞表数字键一律报 unknown;已 luarocks make 重装。**两个生成物默认值坑(记入 tasks)**:`_ops/leaky_relu.lua` 默认 negative_slope=0(yaml 0.02 / Python 0.01)、`_ops/softplus.lua` 默认 threshold=2.0(yaml/Python 20.0)→ functional 层必须显式传 Python 契约默认值,不依赖生成物
 |    |    |    +-- [✅] 2.9.3.2 其余激活 + dropout ★ 2026-10-25:第二批 10 激活(softsign/tanhshrink/swish/log_sigmoid/mish/softshrink/maxout/prelu/rrelu/gumbel_softmax,全部 argrule 包装 + 显式传 Python 契约默认值)提交 b3d56ef + 命名修正 73e8b03(tanh_shrink→tanhshrink,Python API 名照搬 activation.py:1610)。**★ index 语义参数必须转发 `_ops` 生成物**(softmax/log_softmax/maxout/gumbel_softmax 的 axis 1-based→0-based 转换在生成物层,直调 pd_op_maxout 实测报 "axis only supported 1, -1 or 3" unary.cc:2825);**又两个生成物默认值坑**:mish lambda=20(Python 硬编码 activation.py:1599)、softshrink threshold=0.5(生成物默认 0)、prelu data_format/mode 生成物默认带引号("NCHW")。NN_FUNCTIONAL_SMOKE 62 断言(27 激活+argrule 门禁)+ 10 smoke 回归全绿 + lint 全绿。剩余:glu/swiglu(y=nil 组合,chunk 未生成)、hardshrink、dropout 系列
 |    |    |    +-- [✅] 2.9.3.3 池化(18) ★ 2026-10-25:functional.lua 池化段 **12 个 API 全部 argrule 包装**(avg/max_pool1d/2d/3d + adaptive_avg/max_pool1d/2d/3d),直调 pd_op_pool2d/pool3d(单输出,ops_gen.h:388/389)+ pd_op_max_pool2d_with_index/max_pool3d_with_index(双输出,ops_gen.h:326/327;mask 已在 C ABI 层 +1 转 1-based,emit_capi returns_index)。**三态 padding 解析 Lua 重写**(pooling.py:135-193 `_update_padding_nd` 等价:SAME/VALID 字符串 + 数字 + 对偶列表 + 2n 对称折叠 + n 列表,含 batch/channel 维校验 pooling.py:95-107);**1d 池化 = unsqueeze(0-based 轴2)→pool2d→squeeze 收回**(pooling.py:246-293);divisor_override = scale(out, k0*k1/div, 0, 1)(pooling.py:407);dilation≠1 或 return_mask → max_pool*_with_index 分支。**NN_FUNCTIONAL_SMOKE → 89 断言**(池化段 20+,镜像 test_pool2d_op.py 常量输入→公式期望→rtol 比较:avg 均值/max 全等/dilation 形状/adaptive 形状/mask 展平索引 {1,3,9,11} 1-based/divisor_override/padding SAME/smoke 门禁 4 项)+ 10 smoke 回归全绿 + lint 全绿。剩余(fractional/lp/max_unpool 6 个,算子已生成未包装)随 2.9.3.4 或延后
 |    |    |    +-- [✅] 2.9.3.4 归一化 + 损失 ★ 2026-10-25:functional.lua 归一化 + 损失段 **7 个 API 全部 argrule 包装**(batch_norm/layer_norm + mse_loss/l1_loss/smooth_l1_loss/binary_cross_entropy/binary_cross_entropy_with_logits)。直调 pd_op_batch_norm(ops_gen.h:54,6 输出取第 1,scale/bias 可空 sol gen_ops_b.cc:48)/pd_op_layer_norm(281,scale/bias 可空)/pd_op_bce_loss(56)/pd_op_sigmoid_cross_entropy_with_logits(450,pos_weight 可空)/pd_op_huber_loss(249);组合用 subtract/square/abs/multiply/scale/mean_all/sum。**normalized_shape 是 1-based 轴范围(C4,nn-functional.md sec.3)**,functional 转 C++ 0-based begin_norm_axis(起点轴-1);batch_norm data_format 归一化 NCHW/NHWC(按第 2 字符 C,norm.py:241)+ use_global_stats=None → is_test=not training(norm.py:246-252)。**NN_FUNCTIONAL_SMOKE → 107 断言**(归一化/损失段 18+,镜像 Python test_bce_loss.py 公式逻辑:bce(0.5,1)=-log0.5=ln2、bce_with_logits(0,1)=ln2、mse(2,1)=1、l1=1、smooth_l1 huber |0.5|<1→0.125/|1.5|≥1→1.0、layer_norm 全 1→全 0 + w=2 b=1→全 1、batch_norm is_test (5-0)/sqrt(4)≈2.5)+ 10 smoke 回归全绿 + lint 全绿。剩余损失(cross_entropy/nll_loss/kl_div/ctc 等)与剩余激活(glu/swiglu/hardshrink/dropout)随 2.9.4
 |    |    +-- [🔵] 2.9.4 其余 39 层
 |    |    |    +-- [✅] 2.9.4.1 层类第一批(MNIST 验收路径 8 层) ★ 2026-10-26:`lua/paddle/nn/layers/` 新建 **conv.lua(Conv2D)/activation.lua(ReLU)/pooling.lua(MaxPool2D)/norm.lua(BatchNorm2D)/loss.lua(CrossEntropyLoss)/common.lua(Flatten+Dropout+Embedding)**,全部构造 argrule 包装(self 规则);functional.lua 补 **M.dropout + M.cross_entropy(hard-label 主路径)**。Conv2D 含 `_update_padding_nd` Lua 重写(conv.py:83-136 三态:字符串 SAME/VALID + 数字 + 对称折叠 + pair-list 剔 batch/channel)+ 默认 Normal 初始化(pd_op_gaussian_inplace)+ bias reshape→add(conv.py:259-262);Flatten start/stop_axis 1-based→0-based 层内转(生成物不转,flatten=true no_index);CrossEntropyLoss 走 functional.cross_entropy,C ABI 对 label 内容 pd_idx_minus1(ops_c.cc:433),ignore_index/axis 正数 -1 在 functional 层(生成物不做标量转换);BatchNorm2D 的 _mean/_variance 走 buffers 槽(非参数)+ momentum=None 动态语义(norm.py:898-907)。**NN_LAYERS_SMOKE PASS(38 断言,MNIST 组合 Conv2D→ReLU→MaxPool2D→Flatten→CrossEntropyLoss 定值手算 + Embedding 查表(1-based id)+ argrule 门禁 7 项)** + functional/layer/linear/ops/autograd/tensor 6 smoke 回归全绿 + lint 8 文件全绿。踩坑:`pd_op_arange` sol 绑定要求三个 LuaTensor(gen_ops_a.cc:305),smoke 传数字 0xC0000005 → 非等值张量用 pd_op_concat 拼 full;dropout/cross_entropy 的 p/reduction 门禁在 forward 而非构造(F.dropout common.py:1404 / F.cross_entropy loss.py:3196)
 |    |    |    +-- [✅] 2.9.4.2a 池化层类全 12 个 ★ 2026-10-26:pooling.lua 重写为 **AvgPool1D/2D/3D + MaxPool1D/2D/3D + AdaptiveAvgPool1D/2D/3D + AdaptiveMaxPool1D/2D/3D**(全部构造 argrule 包装 + forward 一行转发 functional 2.9.3.3 已验 API),init.lua 挂 12 个。上游参数照抄 pooling.py(MaxPool1D:684 / MaxPool2D:816 / MaxPool3D:939 / AvgPool1D:115 / AvgPool2D:231 / AvgPool3D:341 / AdaptiveAvg1D:1034 / 2D:1114 / 3D:1221 / AdaptiveMax1D:1326 / 2D:1421 / 3D:1455)。**NN_LAYERS_SMOKE → 59 断言**(38+21:池化段 11 层类数值/形状 + divisor_override + return_mask 双出 + 门禁 3 项,镜像 test_pool2d_op.py 定值手算)+ functional/layer/linear/ops 4 smoke 回归全绿 + lint 全绿
 |    |    |    +-- [✅] 2.9.4.2b 其余 20 层 ★ 2026-10-26:conv.lua 泛化 **Conv1D/Conv3D/Conv1DTranspose/Conv2DTranspose/Conv3DTranspose**(_ConvNd 工厂,1D 走 unsqueeze→conv2d→squeeze 链 conv.py:653-708,transpose 归一化 NCHW/NHWC conv.py:1913);norm.lua 泛化 **BatchNorm1D/3D + LayerNorm + GroupNorm**(LayerNorm normalized_shape 层内换算轴范围、GroupNorm 直调 pd_op_group_norm);activation.lua 批量 **28 个激活层类**(CELU/ELU/GELU/Hardshrink/Hardswish/Tanh/Hardtanh/PReLU/RReLU/ReLU6/SELU/LeakyReLU/Sigmoid/Hardsigmoid/Softplus/Softshrink/Softsign/Swish/Mish/Tanhshrink/ThresholdedReLU/Silu/LogSigmoid/Softmax/LogSoftmax/Maxout/Softmax2D,act_layer 工厂 + PReLU 可学习 weight + RReLU training 透传 + Softmax2D softmax(axis=-3));loss.lua 表导出 **MSELoss/L1Loss/SmoothL1Loss/BCELoss/BCEWithLogitsLoss**(forward 转发 functional);common.lua 加 **Dropout2D/3D**(axis 分支走 functional.dropout dropout_nd 组合)。functional.lua 补 **hardshrink + dropout axis(dropout_nd 组合 common.py:1473-1520)**。**NN_LAYERS_SMOKE → 87 断言** + functional(107)/layer/linear/ops/autograd/tensor 6 smoke 回归全绿 + lint 7 文件全绿。踩坑:functional.hardswish 上游有 inplace(activation.py:400),Lua 规则表原漏 → 补;Hardsigmoid 位置调用会错位到 slope 槽 → 具名表调用
 |    |    +-- [✅] 2.9.5 容器(Sequential/LayerList/LayerDict) ★ 2026-10-26:containers.lua 三个容器全继承 Layer(D27,is_a(nn.Layer) 为真,state_dict/parameters/train 递归零特例)。**构造 = argrule parse_whole_first 变长形态**(规则表仅 `{ name="self", type=argrule.table }`,回调收 whole,whole[1]=实例、whole[2..]=子层;`Sequential{ l1, l2 }` / 命名对 `Sequential{ {"fc1", l1}, ... }` / `LayerDict{ {"conv1", l1}, ... }`)。**方法层 1-based↔0-based 转换(D-R15 双基准)**:用户下标 ml[i] 1-based(nn.md §3,按 order 书序取第 k 个,负数从尾数),state_dict 键名 0-based 字符串("0.weight",与 Python 逐字符一致)。LayerList append/insert/extend,Sequential forward 依序 + append/insert/extend,LayerDict update/keys/values/iter。**NN_CONTAINERS_SMOKE → 43 断言**(镜像 test_layerlist.py 空表 insert 边界 + test_imperative_container_layerlist.py 的 append/insert(-1)/extend/forward shape + LayerDict 行为 + state_dict 键名 + is_a + #ml==0 ipairs 跑空)+ nn 全家 5 smoke 回归全绿 + ops/autograd/tensor 回归全绿 + lint 全绿。**踩坑(3 个 argrule 形态陷阱,全实测)**:① **空构造形态不同**——`LayerList{}` 走 parse_position_table 直接把 **self 实例**传给回调(带子层才走 parse_whole_first 收 whole 表),回调须用 `rawget(whole, FIELDS)` 区分,不能无脑 `whole[1]`(空构造时那是实例[1] → 触发 __index → _len 未初始化崩);② **方法规则表第一规则不能叫 "self"**——argrule 的 parse_self_call 对「self + 裸表第二参数」一律做构造链展开,`ml:extend({l1,l2})` 的裸表会被 table_with_self 拆成 3 个位置参数 → "too many positional table arguments";方法规则改 `{ name="container", type=Layer }`;③ **insert 的 order 书维护**——直接操作 sublayers 槽绕过 __newindex 时,搬移引入的新键 "n" 必须手动补进 order 书末尾(add_sublayer 的 push_order 只处理 idx 槽)。FIELDS 键经 layer.lua:602 模块返回表导出,不在 Layer 类表上(containers.lua 须 `layer_module.FIELDS` 而非 `Layer.FIELDS`)
 |    +-- [✅] 2.10 paddle.optimizer(SGD/Adam 首批) ★ 2026-10-26:`lua/paddle/optimizer.lua` Optimizer 基类 + SGD + Adam 全部 argrule 包装(用户纪律:构造函数与正式 API 一律 argrule)。**M1 无融合算子(10-optimizer.md §2 前提改写)**:`sgd_`/`adam_` 全是被 skip 的 in-place 变体(skip_rules.lua:45)→ **组合基础算子 + 核公式直抄**(SGD `param -= lr*grad` cpu/sgd_kernel.cc:39;Adam adam_functors.h:325-345,lr_eff 含 bias correction,βN_pow 初始=βN(adam.py:311-335)且每步 `*=βN` cpu/adam_kernel.cc:118-119)。**新增 C ABI 原语 `pd_tensor_copy_from`**(paddle_c.h + tensor.cc,→ paddle::Tensor::copy_ tensor.h:510,深拷贝进参数本体、保持 autograd meta/leaf 身份——Python `_C_ops.sgd_/adam_` 的 ParamOut in-place 语义)+ sol 绑定 `t:copy_from(src)`。**step 包 no_grad**(optimizer.py:2060 的 `@no_grad`,组合算子不得在 leaf 上建图,pcall 恢复)。**clear_grad 置零**(grad shell 共享 autograd 存储 autograd.cc:167,copy_from(zeros_like) 写本体,eager_method.cc:1066 set_to_zero=true 语义)。**OPTIMIZER_SMOKE PASS(18 断言)**:SGD/Adam 一步手算定值(含 β_pow 初值 0.9/0.999、第二步 bias correction 推进)、weight_decay 耦合 L2(grad+wd*param)、clear_grad 后 grad=0、5 步 Linear+MSELoss+Adam loss 单调下降 + 8 smoke 回归全绿 + lint 全绿。**踩坑(实测 3)**:① pl.class 必须 `class()` 而非 `class("Optimizer")`(Layer 同款,Layer 若传字符串名报 "must derive from a table type");② `parameters:len()` 只对 pl.List 存在,普通 Lua 数组报 nil→谓词与镜像循环统一用 `#v`;③ smoke 第二步 backward 会累积梯度(autograd_smoke 同款语义),复算前必须先 clear_grad。**下一步 = 2.11 paddle.io(单 worker)** |
 |    |    +-- [✅] 2.10.0 API 设计               -> plan/api/optimizer.md  ★已完成(契约:Optimizer 基类 + SGD/Adam 签名、四行循环、lr 调度器形态、state_dict 续训)
 |    +-- [✅] 2.11 paddle.io(单 worker) ★ 2026-10-26:`lua/paddle/io.lua`(723 行)Dataset 全族 + Sampler 全族 + DataLoader + default_collate,全部 argrule 包装。**shuffle 随机源 = Insight7 choice(用户 2026-10-22 定案,固定种子与 numpy 2.4.6 逐位一致)+ 播种语义权威(numpy 全局流:构造期播一次、epoch 间继续,每轮 iter 重播会导致每 epoch 同序)**。**1-based 全链路(C4)**:Dataset:get(i)/采样器索引/Subset.indices/weights 下标全 1-based,Insight choice 0-based 值在采样层 +1(边界在库内);`pd_op_randperm` 经 emit_capi `returns_index={1}`(index_semantics.lua)已 C ABI 层 +1 → random_split 直接用。**构造互斥全部前置到 _init**(reader.py 同款):IterableDataset+shuffle/sampler/batch_sampler 报错、batch_sampler→batch_size==1 等、num_workers>0 报"尚未实现"(P13)。**collate**:数字→Tensor(I64/F64 判全整)、单字段→pd_op_stack 沿新轴 0 拼 batch(**直接调 C ABI 不经过 _ops/stack.lua 的 1-based→0-based wrapper,axis 传 0**)、x,y 多返回值→pl.List(D27)、map→保键、string→裸表(api/io.md §2.6)。**random_split**:绝对数/比例(余数轮流分配,dataset.py:334-344,非全给最后)、seed 走 Insight randperm。**IO_SMOKE PASS(55 断言)**:diag_io_expected.py(seed=42)逐位对拍(seq/rand/replace/weighted/batch/drop_last)+ 1-based 迭代 + break 重开 + num_workers 报错 + IterableDataset + ConcatDataset/Subset/BatchSampler + TensorDataset 多字段。**踩坑(实测 4)**:① **pl.class 的 `super` 是 call_ctor 临时注入的**(class.lua 仅在继承链上存在某个 `_init` 时注入,`c._init` 返回后清掉)→ 基类(Dataset/Sampler/DataLoader)无 `_init` 时 8 处 `self:super()` 全报 nil,删除(optimizer 能跑是因为基类 Optimizer 有 `_init`);② **collate 递归前向引用**:`collate_sample` 定义在 `default_collate` 前,体内引用全局 nil → 删包装直接递归 `default_collate`(local function 自体内可见);③ **Insight7 Lua 绑定链陈旧**(dll 副本陷阱的 Lua 镜像):`build/bindings/lua/_insight.dll`(10-19)+ `build/backends/cpu/insight_cpu_backend.dll`(10-18)都在 numpy 对齐前,Python 侧 10-26 新 backend 只 POST_BUILD 到 bindings/python/insight/ → 重建 `insight_lua` 目标(连带 insight_core+backend)后 choice 对齐恢复;④ smoke 必须 x64 LuaJIT(vcpkg `tools\luajit\luajit.exe`),32 位 lua.exe 加载 x64 paddle_core.dll 报"不是有效的 Win32 应用程序";backend 依赖 fftw3/libopenblas 须 `build\backends\cpu` 在 PATH。**下一步 = 2.12 M1 验收:MNIST > 95%** |
 |    |    +-- [✅] 2.11.0 API 设计               -> plan/api/io.md  ★已完成(样板)
 |    +-- [🔵] 2.12 M1 验收:MNIST > 95%     -> plan/roadmap.md §1
 |
 +-- [⬜] 3. M2 可用                         前置:2
 |    +-- [⬜] 3.1 dataset + vision          -> plan/modules/12-dataset-vision.md
 |    |    +-- [⬜] 3.1.0 API 设计               -> plan/api/{dataset,vision}.md
 |    |    +-- [⬜] 3.0 前置:Insight7 axis 改 1-based  ★ 跨仓库,见 foundations §3.4
 |    +-- [⬜] 3.2 Lanes 多 worker           -> plan/modules/13-lanes.md
 |    +-- [⬜] 3.3 GC 九层机制               -> plan/modules/14-gc.md
 |    +-- [⬜] 3.4 M2 验收:CIFAR + 吞吐 3×  -> plan/roadmap.md §1
 |
 +-- [⬜] 4. M3 完整                         前置:3
 |    +-- [⬜] 4.1 静态图 trace              -> plan/modules/15-to-static.md
 |    +-- [⬜] 4.2 cpp_extension             -> plan/modules/16-cpp-extension.md
 |    +-- [⛔] 4.3 distributed               -> plan/modules/17-distributed.md
 |    |         阻塞:无多卡环境(待拍板 P2),写得了测不了
 |    +-- [⬜] 4.4 metrics-lua / ocean-lua   独立仓库
 |
 +-- [⬜] 5. M4 更远                         前置:4
      +-- [⬜] 5.1 script 模式(AST)        -> plan/modules/18-script.md
```

**关于 `x.y.0 API 设计` 节点:** 每个模块的第一个子节点都是「先写 api 文档」。
这不是形式主义 —— **实现先行会让 API 长成实现的形状**,
而 api 文档一旦发布就是契约,改起来要走废弃流程。见 `plan/api/README.md`。

**关于 1.x 的任务编号缺口:** `tasks.md` 里没有 `T-M0-12/13/16` ——
12(sol2+5.5)和 13(Lanes 可选性)**已经关闭并转成决策**(D4、D-R5),
16 **并入 T-M0-01 一起做**。缺口是有意的,不是漏了。

---

## 5. 现在在哪

**✅ 0.13** —— API 设计规范已立(`plan/api/README.md`),样板已写(`plan/api/io.md`)。
其余模块的 api 文档**不在这里一次写完**,而是挂在各自模块下的 `x.y.0` 节点上,
在那个模块开工时写 —— 提前写会写成猜测。

**✅ 0.14 参数检查选型** —— 论证已完成并落盘(`plan/foundations.md` §4,D-R26):
**不 vendor argcheck**,取它的规则表 schema 与 `usage.lua`,求解器自写 `_args.lua`(~150 行,O(N))。

| 完工判据 | 状态 |
|---|---|
| `foundations.md` §4 写明否决理由(3^N,编不出 `Conv2D`/`DataLoader`)与替代形态 | ✅ |
| `decisions.md` 有 R25 -> R26 的**完整翻案痕迹**(含「为什么第一次判错」) | ✅ |
| `api/README.md` §2.1 写明「不支持位置参数中间省略」 | ✅ |
| `layout.md` / `ci.md` / `AGENTS.md` 同步 | ✅ |
| **M0 #22(LuaJIT `debug.setupvalue`)有结论** | ✅ **2026-10-21 实测:Lua 5.1.5 与 LuaJIT 2.1 均可注入(含表 upvalue);Q-17 关闭** |
| **M0 #23(`loadstring` 跨 5 Lua × 3 OS 行为一致)** | 🟡 **5.1/LuaJIT 已测一致(2026-10-21);5.2-5.4 腿并入 0.15 的 lua-matrix CI job** |

⚠️ **这个节点不能只凭文档写完就打 ✅** —— 判据里那两项是**可执行**的(§3)。
但它们不需要 libpaddle,可以和 0.15 并行跑,只须在进 **3.5(P5)之前**闭掉。

> 📌 **这个节点本身是 §7「判据注水」的一个真实样本,值得留着当例子。**
> 第一版判据只有「性能数据是否落盘」,于是 R25 顺利「达标」了 —— 而它是错的。
> 真正该问的是**「用它能不能表达 `plan/api/` 里已经写下的签名」**。
> **判据要盯产出物之间的一致性,不要盯「我是否做了调研」。**

DFS 的下一个是 **0.15(L0 CI 落地)**,然后回溯到根,进入 **1. M0**,
第一个叶子是 **1.1 无 Python 构建**(生死判定)。

⚠️ **注意 0.15 排在 M0 之前不是笔误。** L0 CI 不需要 libpaddle(`ci.md` §2),
现在就能建;而且它检查的 `doctree` 正好保护这棵树本身。
先建它,后面每个节点都受保护。

**🔵 0.15 L0 CI 落地** —— 按 `plan/ci.md` §2 建静态层。产出:

| 产出 | 完工判据 |
|---|---|
| `.github/workflows/l0.yml` | 四 job 齐活:doctree / redlines / no-crlf / lua-matrix |
| `tools/ci/check_doctree.lua` | 本机 Lua 5.1 跑通,且能抓出「指针指向不存在文件」 |
| `tools/ci/lua_matrix.lua` | 5 个 Lua 上跑通 M0 #23/#24(替代本机缺失的 5.2-5.4) |
| `plan/roadmap.md` / `AGENTS.md` §6 同步登记 `tools/ci/` | 两处都查得到 |

---

## 6. 树是活的

**执行中发现新工作是正常的,不是计划失败。** 允许:

- 叶子太大 -> **分裂**,挂一层指针(这正是"指针指向更小模块"的用法)
- 发现漏了的工作 -> **插入新节点**
- 发现顺序错了 -> **重排兄弟**,但必须重新检查 §2 的拓扑序

**唯一的要求:改树要留痕。**
结构性的改动(删节点、重排、翻案)记进 `process/decisions.md`,
和翻案记录用同一套规矩 —— **不留痕迹的计划变更,和没有计划是一样的。**

---

## 7. 已知的失效模式

| 失效 | 长什么样 | 防御 |
|---|---|---|
| **判据注水** | 叶子标了 ✅,判据是"实现完成" | §3 强制可执行判据 + `doctree` job |
| **树与现实脱节** | 代码改了,树没动 | 每个阶段的完工必须包含"回填踩坑记录" |
| **⛔ 堆积** | 一堆阻塞节点没人管,树看起来在推进 | `status.md` 必须列出全部 ⛔ 及其原因 |
| **拓扑序失效** | 插入新节点时没检查依赖 | 前置断言(§2 的安全网) |
| **多个 🔵** | 并行做两件事,两边都半成品 | 协议:全树只能有一个 🔵 |
