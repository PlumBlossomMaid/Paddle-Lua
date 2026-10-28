-- tools/smoke/gc_smoke.lua
-- M2 3.3: GC 九层机制,层1-3 堆追踪(14-gc.md sec.3.1 / THGeneral.c 移植)
--   Covers D6 (default ON), accounting (+bytes at LuaTensor birth,
--   -bytes at death), the 1MB batch + softmax path (gc.cc), the toggle
--   (paddle.set_heap_tracking, argrule-wrapped formal API) and the
--   diagnostic surface (heap_tracking / tracked_bytes).
--
-- Run (from the repo root, with the paddle runtime on PATH):
--   set PATH=<paddle dll dirs>;%PATH% && luajit tools/smoke/gc_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

require "paddle.tensor"
local paddle = require "paddle"

local CPU = core.PD_PLACE_CPU
local F32 = core.PD_DTYPE_FLOAT32
local cpu = core.pd_place_new(CPU, 0)

local n_asserts = 0
local function check(cond, msg)
  n_asserts = n_asserts + 1
  assert(cond, msg)
end

-- --- 1: D6 -- heap tracking is ON by default -------------------------
check(paddle.heap_tracking() == true,
      "heap tracking defaults ON (D6)")
check(core.heap_tracking() == true, "core.heap_tracking mirrors paddle.heap_tracking")

-- --- 2: accounting -- live tensor bytes are tracked ------------------
-- Baseline first (eager init may have left a few live shells).
local base = core.tracked_bytes()
local big = core.pd_op_full({100, 100, 100}, 1.0, F32, cpu)  -- 1e6 * 4B = 4MB
check(big:numel() == 100 * 100 * 100, "big tensor numel")
local after_alloc = core.tracked_bytes()
-- The ledger must have grown by >= the tensor's own bytes (other live
-- shells may add a little noise, but never less than this tensor).
check(after_alloc >= base + 100 * 100 * 100 * 4,
      ("tracked grew by >= 4MB: base=%d after=%d"):format(base, after_alloc))

-- --- 3: death -- dropping the shell releases the claim --------------
big = nil
collectgarbage("collect")  -- LuaTensor dtor -> pd_gc_account(-bytes)
local after_free = core.tracked_bytes()
check(after_free <= base + 100 * 100 * 100 * 4,
      ("tracked released after collect: base=%d after=%d"):format(base, after_free))

-- --- 4: toggle -- disabled tracking stops counting -------------------
local prev = paddle.set_heap_tracking{ on = false }
check(paddle.heap_tracking() == false, "paddle.set_heap_tracking(false) works")
-- a fresh allocation while disabled must NOT move the ledger
local before_off = core.tracked_bytes()
local temp = core.pd_op_full({50, 50, 50}, 2.0, F32, cpu)
check(core.tracked_bytes() == before_off,
      ("disabled: tracked frozen (was %d)"):format(before_off))
temp = nil
collectgarbage("collect")
paddle.set_heap_tracking{ on = true }
check(paddle.heap_tracking() == true, "re-enabled")

-- --- 5: enabled again -- accounting resumes --------------------------
base = core.tracked_bytes()
local t2 = core.pd_op_full({200, 200, 200}, 3.0, F32, cpu)  -- 8e6 * 4 = 32MB
check(core.tracked_bytes() >= base + 200 * 200 * 200 * 4,
      "re-enabled: ledger grows again")
t2 = nil
collectgarbage("collect")

-- --- 6: softmax -> bridge -> full collect path ------------------------
-- Deterministic end-to-end: with a 1MB softmax, every positive flush
-- past it runs the thread-local bridge (module.cc gc_softmax_bridge ->
-- lua_gc(L, LUA_GCCOLLECT)). Observable: a dead shell that we NEVER
-- collectgarbage() manually must be reclaimed by the bridge's collect
-- during the NEXT allocation's flush.
core.set_heap_softmax(1024 * 1024)             -- 1MB cap
paddle.set_heap_tracking{ on = true }
base = core.tracked_bytes()
local dead_a = core.pd_op_full({512, 512}, 1.0, F32, cpu)  -- 1MB
local dead_b = core.pd_op_full({512, 512}, 1.0, F32, cpu)  -- 1MB
-- delta is now 2MB > the 1MB batch AND > the 1MB softmax: the flush
-- during dead_b's allocation already ran the bridge (lua_gc ran while
-- both were still referenced; nothing to reclaim, but the path fired).
local after_two = core.tracked_bytes()
check(after_two >= base + 2 * 512 * 512 * 4,
      "softmax path: 2MB ledged (was " .. base .. ")")
-- Drop both WITHOUT explicit collect: the shells stay unreachable.
dead_a, dead_b = nil, nil
-- Next 1MB allocation: delta crosses the batch again -> flush -> bridge
-- lua_gc now DOES reclaim the two dead shells (their LuaTensor dtor
-- runs pd_gc_account(-bytes)), so the ledger returns to ~base+1MB.
local alive = core.pd_op_full({512, 512}, 1.0, F32, cpu)
local after_bridge = core.tracked_bytes()
check(after_bridge <= base + 512 * 512 * 4 + 4096,
      ("bridge collect reclaimed dead shells: was %d now %d")
        :format(after_two, after_bridge))
alive = nil
core.set_heap_softmax(300 * 1024 * 1024)       -- restore default
collectgarbage("collect")

-- --- 7: layers 6-8 -- OOM rescue callback (14-gc.md sec.3.2) ----------
-- The callback is registered at luaopen (module.cc pd_gc_register_oom ->
-- RegisterOOMCallback, retry_allocator.h:32). On a CPU-only build real
-- allocation failures never route through RetryAllocator (it wraps only
-- GPU/XPU -- allocator_facade.cc:403/979/987/1408), so pd_gc_fire_oom
-- drives the SAME registered callback deterministically; the smoke proves
-- the callback -> per-thread bridge -> lua_gc(LUA_GCCOLLECT) -> LuaTensor
-- dtor -> -bytes -> freed-byte-retry plumbing end to end.
core.set_heap_softmax(300 * 1024 * 1024)
paddle.set_heap_tracking{ on = true }
collectgarbage("collect")
local base7 = core.tracked_bytes()
local oom_a = core.pd_op_full({1024, 1024}, 1.0, F32, cpu)  -- 4MB
local oom_b = core.pd_op_full({1024, 1024}, 1.0, F32, cpu)  -- 4MB
local held = core.tracked_bytes()
check(held >= base7 + 2 * 1024 * 1024 * 4,
      "oom: 8MB dead-able tensors allocated")
-- Drop both WITHOUT manual collect: shells become unreachable but the
-- ledger still holds their bytes until the callback's lua_gc runs.
oom_a, oom_b = nil, nil
local freed = core.gc_fire_oom(0)   -- drive callback; fake_bytes unused
check(freed > 0,
      ("oom: callback lua_gc freed %d bytes (dead shells reclaimed)"):format(freed))
local after_oom = core.tracked_bytes()
check(after_oom <= base7 + 1024 * 1024 * 4,
      ("oom: ledger back to ~base after callback (was %d now %d)")
        :format(held, after_oom))
-- Reentry safety + no-crash: a second fire with nothing to reclaim must
-- return a small/zero freed count without crashing or recursing
-- (thread_local g_in_oom_callback guard, 14-gc.md 坑①).
local freed2 = core.gc_fire_oom(0)
check(freed2 < 1024 * 1024,
      ("oom: second fire mostly idempotent (freed2=%d)"):format(freed2))
collectgarbage("collect")

print(("GC_SMOKE PASS: %d assertions"):format(n_asserts))