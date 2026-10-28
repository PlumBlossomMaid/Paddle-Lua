-- tools/smoke/lanes_smoke_ds.lua
-- Test-only Dataset modules for lanes_dataloader_smoke.lua. They live in
-- their own file so the worker lanes (fresh lua_State) can `require` them
-- via the factory function -- the multi-worker contract (dataloader.md §4)
-- requires the dataset to be constructible inside the worker through a
-- pure factory, NOT passed across the lane as an object.
--
-- Lua 5.1 subset (C3); no paddle_core usage here (worker-safe require).

local class = require "pl.class"
local io = require "paddle.io"

-- NumDS: sample i is the number i (mirror io_smoke's local NumDS).
local NumDS = class(io.Dataset)
NumDS._name = "NumDS"

function NumDS:_init(n)
  self._n = n
end

function NumDS:len()
  return self._n
end

function NumDS:get(i)
  return i
end

-- SlowDS: NumDS with a tunable per-sample busy-loop, to make through-put
-- differences between num_workers values measurable.
local SlowDS = class(io.Dataset)
SlowDS._name = "SlowDS"

function SlowDS:_init(n, iters)
  self._n = n
  self._iters = iters or 20000
end

function SlowDS:len()
  return self._n
end

function SlowDS:get(i)
  local x = 0
  for k = 1, self._iters do
    x = x + k
  end
  return i + (x % 3)   -- deterministic but CPU-heavy
end

-- BadDS: get() raises on a specific index (error-propagation test).
local BadDS = class(io.Dataset)
BadDS._name = "BadDS"

function BadDS:_init(n, bad_at)
  self._n = n
  self._bad_at = bad_at
end

function BadDS:len()
  return self._n
end

function BadDS:get(i)
  if i == self._bad_at then
    error(("BadDS.get(%d): injected dataset error"):format(i), 2)
  end
  return i
end

return {
  NumDS = NumDS,
  SlowDS = SlowDS,
  BadDS = BadDS,
}