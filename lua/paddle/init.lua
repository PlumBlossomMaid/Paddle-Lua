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

-- ---- lazy submodule skeleton ----------------------------------------
-- Keys resolve to module names; first access requires + caches. Populate
-- as each stage lands: nn (P9), optimizer (P10), io (P11), vision (P12).
local lazy = {}

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
