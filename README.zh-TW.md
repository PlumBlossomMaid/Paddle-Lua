[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Lua](https://img.shields.io/badge/Lua-5.1%20|%205.2%20|%205.3%20|%205.4%20|%20JIT-blue.svg)](https://www.lua.org/)
[![Upstream](https://img.shields.io/badge/Paddle-3.5.0%20零改動-green.svg)](https://github.com/PaddlePaddle/Paddle)
[![Status](https://img.shields.io/badge/status-論證階段-orange.svg)](docs/migration/README.md)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Paddle-Lua

**為 PaddlePaddle 打造一個 Lua 前端。`local paddle = require "paddle"`,不需要 Python 執行環境。**

> ⚠️ **目前處於論證階段,尚無程式碼。**
> 本倉庫目前是一份完整的技術方案。
> 見 **[docs/migration/](docs/migration/README.md)** —— 13 份文件,約 3000 行。

---

## 為什麼這件事做得成

Paddle 的訓練引擎是**純 C++** 的。自動微分(`egr::Backward`)、張量型別
(`paddle::Tensor`)、以及全部運算子(phi)都和 Python 沒有任何關係。Python 只是一層**外殼**:
參數解析、`nn.Layer` 的簿記、最佳化器迴圈、資料管線。

所以這個專案不是重新實作一個深度學習框架,而是**換掉那層外殼**。

```
             Python 外殼                     Lua 外殼          ← 我們寫這個
  ─────────────────────────────   ─────────────────────────
   pybind11 綁定                   sol2 + 純 C ABI 中間層     ← 我們寫這個
  ═══════════════════════════════════════════════════════════
   產生的前向 API  ·  eager 自動微分  ·  phi 運算子           ← 原樣不動,純 C++
```

**上游零改動。** Paddle 按原樣使用。

---

## 長什麼樣

```lua
local paddle = require "paddle"
local class  = require "pl.class"
local nn, F  = paddle.nn, paddle.nn.functional

local Net = class(nn.Layer)
function Net:_init()
  self:super()
  self.fc = nn.Linear(784, 10)     -- 和 Python 一樣,自動註冊
end
function Net:forward(x) return self.fc(x) end

local net = Net()
local opt = paddle.optimizer.Adam{ learning_rate = 1e-3, parameters = net:parameters() }

for _, batch in paddle.io.DataLoader{ ds, batch_size = 64, num_workers = 4 } do
  local loss = F.cross_entropy(net(batch[1]), batch[2])
  loss:backward()
  opt:step()
  opt:clear_grad()
end
```

索引**統統 1-based** —— 包括 `axis` 參數。切片借用 Python 的語法,透過字串傳入,
這個做法在 [Insight7](https://github.com/PlumBlossomMaid) 裡已經跨語言驗證過:

```lua
x["1:3, :"]     -- 第 1、2 列,全部行
x["::-1"]       -- 反轉
x[{1, 2}]       -- 取單一元素
```

---

## 設計概覽

| | |
|---|---|
| 宿主語言 | Lua 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT |
| Python 執行環境 | **不需要** |
| 綁定層 | sol2,底下墊一層純 C ABI |
| 上游改動 | **零** |
| 函式庫程式碼方言 | 只用 Lua 5.1 語法(一份原始碼同時服務五個虛擬機) |
| 多 worker 資料載入 | [Lua Lanes](https://github.com/LuaLanes/lanes),強制相依 —— 用執行緒,不用行程 |
| 索引 | 全部 1-based |
| 運算子綁定 | 從 Paddle 自己的 `ops.yaml` / `backward.yaml` 產生,不手寫 |
| 類別與集合 | [Penlight](https://github.com/lunarmodules/Penlight) —— `pl.class`、`pl.List`,vendored 純 Lua |
| numpy 的位置 | [Insight7](https://github.com/PlumBlossomMaid) —— 無梯度的陣列,用於資料準備 |

### 三個值得解釋的決策

**LuaJIT 也走 sol2,不走 FFI。** 綁定開銷是 50–200 ns,而一次 kernel 啟動是 1–10 µs。
為了省這 1–2% 去維護兩套綁定路徑,不划算。

**`num_workers` 用執行緒而不是行程。** Python 的 DataLoader 之所以 fork,是因為 GIL。
Lua 沒有 GIL —— 多個 `lua_State` 在同一行程裡是真並行的,這意味著 worker 可以把張量
**按指標**交給主執行緒,完全不需要序列化。`paddle::Tensor` 本身就用 `shared_ptr` 對緩衝做
原子參考計數,跨執行緒的代價只是一次加一。

**Penlight 是一等公民,不是隨手一提。** Lua 的標準函式庫幾乎什麼都沒有:沒有類別、
沒有 list、沒有跨版本相容層。與其自己造一套(結果是只有我們自己會除錯),不如讓
`nn.Layer` **就是**一個 `pl.class`,`parameters()` **就是**回傳 `pl.List`,
版本墊片**就是** `pl.compat`。你已有的 Penlight 知識在這裡直接可用。

**靜態圖複用 Paddle 的 C++ 執行器。** `StandaloneExecutor` / `InterpreterCore` 已經提供了
IR、排程、垃圾回收、pass 與序列化;`run_program_ad_func` 已經讓 traced 圖能參與 eager 自動微分。
Lua 側只需要寫一個 tracer。

---

## 唯一能決定專案生死的風險

`cmake/configure.cmake:15` 對無 Python 路徑的守衛是 `(NOT WITH_PYTHON) AND ON_INFER`。
**上游預期的無 Python 場景是推論。** 我們要的這個組合 —— 無 Python 但要訓練 ——
可能從來沒有人編譯過。

所以第一件事不是寫程式碼,而是:

```bash
cmake .. -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=ON
```

如果這條路編不出一個 `egr::Backward` 真能跑的 `libpaddle`,整個專案需要重新評估。
計畫裡其它所有東西都在這一項的下游。

---

## 範圍

**做:** `paddle.Tensor`、`paddle.nn`、`paddle.optimizer`、`paddle.io`、`paddle.vision`、
`paddle.jit`(先 trace 後 script)、`paddle.dataset`、`paddle.static`、
`paddle.distributed`(可以寫,但目前測不了)。

**不做:** `paddle.Model` 和 `paddle.metric.*` —— 這是故意的。它們已被
[PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean)(Lightning + Keras 風格的
trainer)和 [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics)
(torchmetrics 風格)取代,我們照後兩者的設計走。

同樣不做:`paddle.incubate`、`paddle.fluid`,以及一切只為伺候 Python 打包生態而存在的東西。

---

## 路線圖

| | 里程碑 | 交付物 |
|---|---|---|
| **M0** | 可行性閘門 | 無 Python 建置;反向傳播跑通 |
| **M1** | MVP | MNIST 端到端訓練;用 `jit::Layer` 推論 Python 訓練的模型 |
| **M2** | 可用 | `paddle.vision`、Lanes 多 worker 的 DataLoader、完整 GC 機制 |
| **M3** | 完整 | trace 模式動轉靜;`metrics-lua` |
| **M4** | 更遠 | script 模式(帶控制流,vendored luacheck parser) |

粗估:MVP 4–6 人月。

---

## 相關專案

```
paddle-lua        C++ 綁定 + 核心 Lua 層            ← 本倉庫
  ├─ metrics-lua     純 Lua,torchmetrics 風格
  └─ ocean-lua       純 Lua,Lightning/Keras 風格
```

| | |
|---|---|
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | 上游。原樣使用,**永不改動** |
| [PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean) | 取代 `paddle.Model`,我們照它的設計走 |
| [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics) | 取代 `paddle.metric.*`,我們照它的設計走 |
| [Lua Lanes](https://github.com/LuaLanes/lanes) | 多執行緒,`num_workers > 0` 的前提 |
| [sol2](https://github.com/ThePhD/sol2) | C++/Lua 綁定函式庫 |
| [Penlight](https://github.com/lunarmodules/Penlight) | 類別、集合、相容層。vendored 受限子集,MIT |
| [Insight7](https://github.com/PlumBlossomMaid) | 這套技術棧裡 numpy 的位置。字串切片與關鍵字參數約定也源自它 |

---

## 文件

完整技術方案在 **[`docs/`](docs/migration/README.md)**。

---

## License

Apache 2.0,與 PaddlePaddle 一致。
