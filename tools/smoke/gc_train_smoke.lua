-- tools/smoke/gc_train_smoke.lua -- M2 3.3 验收:「训练显存不涨」
-- (research/gc.md §6 / 14-gc.md §5:长时间训练显存曲线平稳,无阶梯式增长)
--
-- 镜像 2.12 的堆崩复现场景(根因三):M1 验收时 LuaJIT 默认 GC 阈值不回收
-- 逐 batch 累积的 LuaTensor shell → C 侧 ~300 batch 堆崩 0xC0000005,
-- 当时靠每 100 batch 手动 `collectgarbage("collect")` 压住。3.3 的目标
-- 就是把这手动 GC 变成机制:堆追踪默认开(D6)+ softmax bridge 自动在
-- 正分配 flush 越过软上限时对本线程 lua_State 跑全量 collect,死 shell
-- 的 LuaTensor dtor 立即回账,ledger 因而有界。
--
-- 本 smoke 的断言 = 全程【不调用 collectgarbage】,跑 240 步训练,
-- 每 24 步采样 core.tracked_bytes(),确认:
--   1. 训练确实在跑(loss 单调下降,镜像 optimizer_smoke 第 5 节);
--   2. 显存 ledger 有界(末段采样 <= 基线 + 数个软上限,而非随步数线性涨)。
-- 若无 bridge 自动回收,240 步 × 每步 ~300KB 临时张量 ≈ 72MB 漏账,
-- 断言 2 必然崩溃——这就是"无阶梯式增长"的可执行判据。
--
-- 判据参数:软上限调到 8MB(默认 300MB 对这个小网络永不触发,必须调小
-- 才能让 bridge 在循环内真实点火 = 镜像大网络在 300MB 上限下的行为)。
--
-- Run (from the repo root, with the paddle runtime on PATH):
--   set PATH=<paddle dll dirs>;%PATH% && luajit tools/smoke/gc_train_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"
local nn = paddle.nn
local optimizer = paddle.optimizer

local F32 = core.PD_DTYPE_FLOAT32
local cpu = core.pd_place_new(core.PD_PLACE_CPU, 0)

local n_asserts = 0
local function check(cond, msg)
  n_asserts = n_asserts + 1
  assert(cond, "GC_TRAIN assertion #" .. n_asserts .. ": " .. tostring(msg))
end

local function item(t)
  if t:numel() == 1 then return t:item() end
  return t:item(1)
end

-- --- 0: D6 前置 -- 堆追踪默认开 -------------------------------
check(paddle.heap_tracking() == true,
      "D6: heap tracking defaults ON (train context)")

-- --- 1: 训练循环(不手动 collectgarbage) --------------------------
-- 网络:Linear{128,128} -> ReLU -> Linear{128,10}(~33KB 参数,常驻 ledger
-- 计入基线)。每个 batch 新建 x(128x128 = 64KB)+ 中间 h + out + loss +
-- 反向图,全部在步末失去引用——若 bridge 不回收,这些泄漏会线性累加。
local net = nn.Sequential{
  nn.Linear{ 128, 128 },
  nn.ReLU(),
  nn.Linear{ 128, 10 } }
local mse = nn.MSELoss()
local opt = optimizer.Adam{ parameters = net:parameters(),
                            learning_rate = 0.01 }
local y = core.pd_op_full({ 32, 10 }, 0.1, F32, cpu)   -- 常驻 target

core.set_heap_softmax(8 * 1024 * 1024)   -- 8MB 软上限,让 bridge 点火
local busylevel = coroutine.running() and "coro" or "main"

local losses = {}
local samples = {}
local steps, sample_every = 240, 24
for i = 1, steps do
  local x = core.pd_op_full({ 32, 128 }, 0.7, F32, cpu)
  local out = net(x)
  local loss = mse(out, y)
  losses[i] = item(loss)
  loss:backward()
  opt:step()
  opt:clear_grad()
  if i % sample_every == 0 then
    samples[#samples + 1] = core.tracked_bytes()
  end
end

-- --- 2: 训练确实在跑 -------------------------------------------
-- 非空断言:loss 末段(后 1/3 均值)严格小于首段(前 1/3 均值)。
local n3 = math.floor(steps / 3)
local head, tail = 0.0, 0.0
for i = 1, n3 do head = head + losses[i] end
for i = steps - n3 + 1, steps do tail = tail + losses[i] end
head, tail = head / n3, tail / n3
check(tail < head,
      ("training runs: loss head=%.6g tail=%.6g (must decrease)")
        :format(head, tail))

-- --- 3: 显存 ledger 有界(无阶梯式增长) --------------------------
-- 基线 = 首个采样点(24 步时 bridge 已点火过,包含网络参数 + 一点噪声)。
-- 判据:全程最大采样 <= 基线 + 2×软上限。240 步无回收 = 漏账 ~72MB,
-- 远超 16MB 限;bridge 正常 = 每次越过 8MB 被 collect 压回,有界。
local baseline = samples[1]
local peak = 0
for _, s in ipairs(samples) do
  if s > peak then peak = s end
end
check(peak <= baseline + 2 * 8 * 1024 * 1024,
      ("tracked bounded: baseline=%d peak=%d (leak would climb ~72MB)")
        :format(baseline, peak))
check(peak <= 64 * 1024 * 1024,
      ("tracked sane absolute: peak=%d"):format(peak))

-- --- 4: 显式释放路径在训练循环里也工作 ---------------------------
-- y 只有 32x10x4 = 1280B,free() 只减线程 delta,未达层2 的 ±1MB 批量
-- 阈值不会立即写全局 ledger(gc.cc pd_gc_account;gc_smoke 第 8 节用
-- 1MB 张量测的才是"立即回落")。这里验证的是路径安全:free 幂等不崩、
-- 之后 collect 把 delta 入账后 ledger 不增(1280 + 其他死壳只减不增)。
local before_free = core.tracked_bytes()
y:free()
y:free()                      -- 幂等,不崩
collectgarbage("collect")     -- flush 线程 delta 入账 + 回收死壳
check(core.tracked_bytes() <= before_free,
      ("free path safe post-train (before=%d now=%d, must not grow)")
        :format(before_free, core.tracked_bytes()))

core.set_heap_softmax(300 * 1024 * 1024)   -- 恢复默认

print(("GC_TRAIN_SMOKE PASS (%d assertions, %s thread)"):format(
  n_asserts, busylevel))