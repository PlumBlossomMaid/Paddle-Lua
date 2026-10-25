-- lua/paddle/jit/init.lua -- paddle.jit.load (P8 prewire, M1 2.8)
-- Stage: P8 / M1 (WORKPLAN 2.8). Thin Lua face over the LuaLayer
-- usertype; the 200-line C ABI slice is csrc/capi/jit.cc, the ~100-line
-- sol slice is csrc/sol/jit_binding.cc (08-jit-infer.md sec.3.2).
--
--   paddle.jit.load{ path, skip_load_weights=false } -> TranslatedLayer
--   layer(x) / layer:forward(x)   single Tensor or multi-return
--   layer:to(place) / layer:function_names() / layer:clone()
--
-- Lua 5.1 subset (C3).

local paddle_core = require "paddle_core"

local M = {}

-- paddle.jit.load{path, skip_load_weights=false} -> layer (core.jit_load)
-- skip_load_weights=true is rejected in the binding (jit::Load always
-- reads weights, serializer.h:78 has no skip flag) -- api/jit.md sec.2.2.
function M.load(t)
  if type(t) ~= "table" then
    error("paddle.jit.load: expected a table {path}", 2)
  end
  local path = t.path or t[1]
  if type(path) ~= "string" then
    error("paddle.jit.load: path must be a string", 2)
  end
  local layer = paddle_core.jit_load{ path = path,
    skip_load_weights = t.skip_load_weights }
  -- surface as TranslatedLayer (same handling; the load face may grow
  -- methods later -- 08-jit-infer.md sec.3.1)
  return layer
end

-- P15 trace / P18 script land here later (D20; api/jit.md sec.2.3).
M.to_static = nil

return M