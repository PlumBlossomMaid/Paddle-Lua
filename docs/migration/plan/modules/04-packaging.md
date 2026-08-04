# P4 · 构建与打包

| | |
|---|---|
| 阶段 | P4(贯穿全程,持续维护) |
| 类别 | 基础设施 |
| 开工条件 | P2 完工 |
| 预估 | 1 周(首次)|

---

## 1. 做什么 / 不做什么

**做:** 让用户能装上并 `require "paddle"`。

**不做:** 不打包 `libpaddle` 本身。Paddle 的二进制体积以 GB 计,
我们**链接**它而不是**分发**它。用户需要自己有一份 Paddle 构建产物。

---

## 2. 上游有什么可以用

| 出处 | 用途 |
|---|---|
| `cmake/` 下的 Paddle 构建脚本 | 参考它怎么找 CUDA / MKL / 依赖 |
| Paddle 安装产物的目录布局 | 决定我们怎么定位 `libpaddle` |

---

## 3. 设计

### 3.1 用户拿到的东西

```
paddle-lua/
├── paddle_core.so         C 扩展(链 libpaddle + liblua5.x)
├── paddle/                纯 Lua
│   ├── init.lua
│   ├── nn/  optimizer/  io/  vision/ ...
│   ├── _ops/              生成的 Lua wrapper
│   └── _vendor/pl/        vendored Penlight 受限子集(11 文件,纯 Lua)
└── (可选) lanes           多 worker 依赖
```

**`_vendor/pl/` 里恰好 11 个文件**(`class` `List` `compat` `pretty` `tablex`
`utils` `stringx` `lexer` `operator` `text` `types`),5374 行,157 KB。
这是我们要用的模块的**传递闭包**,已实测封闭且**不含任何 `lfs` 引用**。
`paddle.pl` 暴露这一份给用户。清单与理由见 `plan/foundations.md` §1.2 §1.3。

### 3.2 找 `libpaddle` 的顺序

和 `CLAUDE.md` §4 的 `$PADDLE_ROOT` 发现顺序保持一致:

1. 环境变量 `PADDLE_LUA_LIBPADDLE`(显式指定,优先级最高)
2. 环境变量 `PADDLE_ROOT` 下的 `build/paddle/fluid/`
3. 系统库路径(`LD_LIBRARY_PATH` / `PATH`)
4. 找不到 -> **报一条能照着做的错误信息**,不要只说 "library not found"

### 3.3 rockspec

```lua
-- paddle-lua-scm-1.rockspec
dependencies = {
  "lua >= 5.1",
  "lanes >= 3.16",
  -- 注意:**不写 "penlight"**。见下。
  -- Insight7 不写死版本,由用户自行安装(软强制)
}
```

**Lanes 是强制依赖**(D-R5),不做成 optional。
理由在 `research/dataloader.md` §9.5(a):单一 Tensor 表示、单一代码路径、少一个验证项。

**Penlight 不写进 `dependencies`,而是 vendor。** 三条理由(`plan/foundations.md` §1.3):

1. **版本锁定(主要理由)** —— `pl.class` 若某天改了 `_create` 语义,我们的 `nn.Layer`
   会**静默**坏掉(不是编译错误,是自动注册悄悄失效),
   必须锁 commit + CI 校验 sha256(待拍板 P7)
2. Penlight 的 rockspec 声明 `dependencies = {"lua >= 5.1", "luafilesystem"}` ——
   写依赖会把 C 库 `lfs` 拖进来,多 5 Lua × 3 OS = 15 个构建组合。
   ~~违反 `CLAUDE.md` §9~~ —— **该禁令已取消(R23)**,现在这只是**不划算**:
   我们一个 `pl.path`/`pl.dir`/`pl.file` 都不用,却要为它们承担构建面
3. 与 `research/to-static.md` §5 vendor luacheck parser 的做法一致,不新增机制

**Insight7 是软强制**:核心不 `require "insight"`,
只有 `paddle.np` / `paddle.from_insight` / `paddle.vision.transforms` 惰性加载,
缺失时给一条能照着做的错误信息。理由见 `plan/foundations.md` §3.7。

**文件系统需求不用 lfs**,用我们自己的 `paddle.utils.fs`
(C++17 `std::filesystem`,Paddle 本来就要求 C++17,零新依赖)。
这是 `CLAUDE.md` §9.1 第 1 步的直接应用:我们已经有一个链着 libpaddle 的 `.so`,
往里加几个 `<filesystem>` 转发函数的边际成本 ≈ 0。

⚠️ **但 §9.1 第 2 步同样适用**:HTTP+TLS 与 JPEG/PNG 解码我们的 `.so` 做不了
(要拖 OpenSSL / libjpeg 进 Paddle 的构建,那比多一个 rock 更贵)。
这两项**允许**声明为 rock 依赖(`luasocket` + `luasec`、图像解码库),
按 §9.1 第 3 步给出 5 Lua × 3 OS 的覆盖证据与降级路径。见 P12 §3.1 / §3.3。

### 3.4 五个版本怎么装

luarocks 本身是按 Lua 版本隔离安装的,所以 rockspec 只需要写一份,
`luarocks --lua-version=5.1 install` 与 `--lua-version=5.4` 各装各的。
**我们不需要做 fat binary。**

---

## 4. 已知的坑

**① Windows 上 DLL 搜索路径与 Linux 完全不同。** Linux 的 `RPATH` 在 Windows 上没有对应物,
必须靠 `PATH` 或 `SetDllDirectory`。这会成为最常见的用户问题,
**文档里要有一节专门讲它**,而不是让用户去 issue 里问。

**② CUDA 版本必须与 `libpaddle` 一致。** 我们链的是同一个 CUDA runtime。
构建时应该**主动检测并在不一致时拒绝构建**,而不是让用户在运行时撞到诡异的崩溃。

**③ 别把生成的 4000 个算子文件都编进一个动态库的同一个 TU。** 见 `03-codegen.md` §4 坑②。

---

## 5. 验收

- [ ] 干净机器(无 Lua 开发环境)上 `luarocks install` 成功
- [ ] Linux 与 Windows 各验一遍
- [ ] 五个 Lua 版本各装一遍,互不干扰
- [ ] 故意不设 `PADDLE_ROOT`,确认错误信息里写清了要设什么
- [ ] `require "paddle"` 到返回的耗时 < 500ms

---

## 6. 未解问题

- 是否提供预编译二进制?**v1 不提供** —— CUDA 版本组合爆炸,维护成本远超收益
