# `_ref/` · 第三方参考源码片段

这里放的**全是别人的代码**,抓下来是为了在论证时能引 `file:line`,
不是本项目的一部分。**本目录不适用仓库根目录的 Apache-2.0 许可证。**

## 清单与出处

| 文件 | 上游 | 上游许可证 | 抓这个干嘛 |
|---|---|---|---|
| `lua_version.hpp` | sol2 `include/sol/version.hpp` 一族 | **MIT**(文件内保留了完整版权头) | `SOL_LUA_VERSION` 的判定逻辑 —— 5.5 出局的证据(D4 / `research/architecture.md`) |
| `sol_config.hpp` | sol2 `include/sol/config.hpp` 一族 | **MIT**(文件内保留了完整版权头) | sol2 的编译期开关面 |
| `lanes_deep.hpp` | lua_lanes 的 deep userdata 公开头 | **MIT** | deep userdata 的 API 形状 —— 后来据此**放弃**了这条路(R18) |
| `THGeneral.c` | torch7 `lib/TH/THGeneral.c` | **BSD 3-Clause** | 堆追踪 / OOM 回调的实现(`research/gc.md`) |
| `torch_init.c` | torch7 `init.c` | **BSD 3-Clause** | 同上 |
| `torch_utils.c` | torch7 `utils.c` | **BSD 3-Clause** | 同上 |
| `torch_initlua.lua` | torch7 `init.lua` | **BSD 3-Clause** | `torch.setheaptracking(true)` 默认开的证据(R3) |

## 两条使用规则

1. **只读,不改,不抄进产品代码。**
   需要借鉴实现时,在 `plan/` 里写清楚思路,由我们自己重新实现;
   真要整段复用,必须先在本文件登记许可证兼容性与署名要求。
2. **上游路径是抓取时记录的,没有逐一复核。**
   要拿它当权威证据(而不是"我读过"),重新对着上游仓库核一遍行号。

## 已删除的条目

| 文件 | 为什么删 |
|---|---|
| `sol_compat.hpp` | 抓取拿到的是 **404 页面**(内容就是 `404: Not Found`),从未真正读过。`02-binding.md` 曾引用它,已改正 |
| `solver.hpp` | 同上,404 |

**教训:抓完要看一眼内容再引用。** 一个 0 行的 "404: Not Found" 文件混在证据里,
比没有证据更危险 —— 它让论证看起来是有出处的。
