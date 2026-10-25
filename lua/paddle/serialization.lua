-- lua/paddle/serialization.lua -- paddle.save / paddle.load
-- Stage: P7 / M1 (WORKPLAN 2.7). After autograd/scope (same M1 run).
--
-- Design: docs/migration/plan/modules/07-serialization.md + api/inventory.md
-- :439-440. Top-level table-call API:
--   paddle.save{tensor, path}          single tensor -> SaveTensor wire format
--   paddle.load{path}                  single tensor <- LoadTensor
--   state_dict / nested structures     -> lua-pickle (soft dep; M2/P9)
-- The wire format is Paddle's own (D19; T-M0-19 verified versioned,
-- lossless, byte-identical re-save). Lua 5.1 subset (C3).

local paddle_core = require "paddle_core"

local M = {}

-- paddle.save{obj, path, overwrite=true}
-- obj = a paddle.Tensor (userdata) | state_dict-ish table (lua-pickle, M2).
-- Single-tensor path: core.save_tensor -> pd_tensor_save -> SaveTensor.
function M.save(t)
  if type(t) ~= "table" then
    error("paddle.save: expected a table {obj, path}", 2)
  end
  local obj = t.obj or t[1]
  local path = t.path or t[2]
  if type(path) ~= "string" then
    error("paddle.save: path must be a string", 2)
  end
  if type(obj) == "userdata" then
    -- Single tensor: Paddle's own wire format (D19). sol binding defaults
    -- overwrite=true, matching upstream save semantics (io.py:790).
    return paddle_core.save_tensor(obj, path,
                                   t.overwrite ~= nil and t.overwrite or true)
  end
  -- state_dict / nested structures land here in M2 via lua-pickle
  -- (inventory.md:439); reject loudly now rather than silently corrupting.
  error("paddle.save: only single tensors are supported in M1 "
        .. "(state_dict via lua-pickle lands with P9/M2)", 2)
end

-- paddle.load{path} -> paddle.Tensor
-- Single-tensor path: core.load_tensor -> pd_tensor_load -> LoadTensor.
-- pickle files (.pdparams) dispatch to lua-pickle (soft dep; M2).
function M.load(t)
  if type(t) ~= "table" then
    error("paddle.load: expected a table {path}", 2)
  end
  local path = t.path or t[1]
  if type(path) ~= "string" then
    error("paddle.load: path must be a string", 2)
  end
  return paddle_core.load_tensor(path)
end

return M