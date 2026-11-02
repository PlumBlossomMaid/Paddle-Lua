-- lua/paddle/init.lua -- require "paddle" entry point
-- Stage: P4 / M1 (WORKPLAN 2.4, 04-packaging.md). First hand-written Lua
-- file of the public API layer; everything below paddle_core (version /
-- init / tensor / slice_spec) is the C extension surface, consumed here.
--
-- Lazy loading (api/README.md sec.2.5): require "paddle" MUST NOT
-- recursively load every submodule. _ops/ is ~20k lines of generated Lua;
-- paddle.vision / paddle.dataset depend on Insight7 which is a SOFT
-- requirement (foundations.md sec.3.7). The top-level table carries a
-- __index that requires a submodule only on first access, and missing
-- optional deps surface an actionable error there -- not "module not
-- found" at require time.
--
-- Lua 5.1 subset (C3): setmetatable/__index/pcall are all 5.1; no
-- goto, no table.unpack.

local paddle_core = require "paddle_core"

-- P5 dtype constants / place constructors are tiny pure-Lua tables (a
-- string table + a pd_place_new redirect), so they load eagerly -- the
-- same way `import paddle` gives you paddle.float32 / paddle.CPUPlace
-- immediately (dtype-device.md sec.2.1). paddle.device's runtime-global
-- functions (get_device/set_device) stay on the lazy submodule below.
local dtype = require "paddle.dtype"
local device = require "paddle.device"

local M = {}

-- ---- C extension surface (module.cc + tensor_binding.cc + slice.cc) --
-- Named with the underscore prefix: these are the raw C ABI bindings,
-- not the final public API (top.md is the contract for paddle.* names).
-- The final names (paddle.tensor{data,...}, paddle.to_tensor{...}) are
-- Lua rewrites arriving with 2.5/2.6.
M._core = paddle_core
M._version = paddle_core.version()

-- Eager runtime init: phi kernel factory etc. Idempotent (pd_init_once).
-- MUST run before any tensor{} / op touches the framework; done here so
-- a plain `require "paddle"` leaves a ready runtime behind.
M.init = paddle_core.init

-- ---- P5 dtype surface (dtype-device.md sec.1.1/1.3) ------------------
-- dtype constants: paddle.float32 etc (the canonical string IS the
-- value; dtype.lua also carries the alias names on itself, but the
-- TOP-level table is what users touch -- dtype-device.md sec.2.1).
for name in pairs(dtype) do
  M[name] = dtype[name]
end
M.finfo = dtype.finfo
M.iinfo = dtype.iinfo
M.get_default_dtype = dtype.get_default_dtype
M.set_default_dtype = dtype.set_default_dtype

-- ---- 顶层算子层(top.md §2,4.4 metrics-lua 前置) ---------------
-- paddle.zeros/cast/sum/... 是公开 API —— metrics-lua 只能调公开 API
-- (overview.md §1),但生成物在 _ops/ 子包从未接线到顶层(实测顶层 nil)。
-- top_ops.lua 把生成物包装成契约(dtype 字符串→枚举、IntArray 标量→表、
-- argrule 表调用),第一批 15 个(创建/逻辑/归约/基础);其余随消费者补齐。
local top_ops = require "paddle.top_ops"
for name, fn in pairs(top_ops) do
  M[name] = fn
end

-- ---- P5 place constructors (dtype-device.md sec.1.2) -----------------
M.CPUPlace = device.CPUPlace
M.CUDAPlace = device.CUDAPlace
M.CUDAPinnedPlace = device.CUDAPinnedPlace
M.XPUPlace = device.XPUPlace
M.XPUPinnedPlace = device.XPUPinnedPlace
M.IPUPlace = device.IPUPlace
M.CustomPlace = device.CustomPlace

-- ---- P5 tensor face (layout.md P5 step 3) ---------------------------
-- tensor.lua ENHANCES the SHARED LuaTensor usertype metatable (__index
-- dispatch / __tostring / arithmetic). The enhancement must happen before
-- ANY tensor is used, so this is an eager require with no named export --
-- the LuaTensor type itself is what users touch, created via
-- paddle_core.tensor{...} (paddle.to_tensor lands later, 2.6).
require "paddle.tensor"

-- ---- P7 serialization (top-level save/load, inventory.md:439-440) ----
-- Top-level FUNCTIONS, not a submodule: paddle.save{t, path} /
-- paddle.load{path} (07-serialization.md sec.3). Single tensors go
-- through Paddle's SaveTensor/LoadTensor wire format (D19); state_dict
-- dispatch to lua-pickle lands with P9/M2.
local serialization = require "paddle.serialization"
M.save = serialization.save
M.load = serialization.load

-- ---- 3.3 GC heap tracking (top-level switch, 14-gc.md 坑④/D6) -------
-- Default ON (D6); paddle.set_heap_tracking{on=false} turns it off. The
-- nine-layer scheme drives pressure from the C-side ledger (pd_gc_account,
-- gc.cc); this is the only user-facing switch, plus diagnostics for smoke
-- and tuning. Eager like serialization: gc.lua is a tiny hard-dep module
-- (paddle_core + argrule, both already required).
local gclib = require "paddle.gc"
M.set_heap_tracking = gclib.set_heap_tracking
M.heap_tracking = gclib.heap_tracking
M.tracked_bytes = gclib.tracked_bytes

-- ---- lazy submodule skeleton ----------------------------------------
-- Keys resolve to module names; first access requires + caches. Populate
-- as each stage lands: device (P5), autograd (P6), nn (P9), optimizer
-- (P10), io (P11), vision (P12).
local lazy = { device = "paddle.device", autograd = "paddle.autograd",
               scope = "paddle.scope", jit = "paddle.jit",
               nn = "paddle.nn", optimizer = "paddle.optimizer",
               io = "paddle.io", utils = "paddle.utils",
               vision = "paddle.vision" }

setmetatable(M, {
  __index = function(t, k)
    local modname = lazy[k]
    if modname == nil then
      return nil
    end
    local ok, mod = pcall(require, modname)
    if not ok then
      error(("paddle.%s: failed to load %s -- %s"):format(
        k, modname, tostring(mod)), 2)
    end
    t[k] = mod
    return mod
  end,
})

return M
