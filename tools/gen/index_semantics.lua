-- tools/gen/index_semantics.lua
-- Operator index-semantics annotation table (C4/D13 1-based conversion).
-- Stage: P3 / M1 (WORKPLAN 2.3.1). MUST be written before the generator.
--
-- Why this file exists (process/conventions.md sec.5.2):
--   ops.yaml has NO information about which parameters/outputs are
--   index semantics -- that table is maintained by hand here. The
--   generator MUST error on any operator NOT present in this table
--   instead of defaulting to pass-through: a silently skipped
--   off-by-one is the hardest bug class in this project.
--
-- Annotation forms (conventions.md sec.5.1):
--   takes_index  = { "arg", ... }   args that are indices (label, ids,
--                                   token ids): their VALUES are 1-based
--                                   in Lua and must be decremented by 1
--                                   before the C++ call.
--   returns_index = { 1, ... }      1-based OUTPUT POSITIONS that are
--                                   indices: the C++ result's values are
--                                   0-based and must be incremented by 1
--                                   before returning to Lua.
--
-- Sources read 2026-10-24 (paddle/phi/ops/yaml/ops.yaml at
-- $PADDLE_ROOT, branch windows_dataloader_multiprocess, v3.5.0):
--   argmax  :346   args (Tensor x, Scalar(int64_t) axis, ...) -> Tensor(out)
--   argmin  :358   same shape as argmax
--   argsort :370   output Tensor(out), Tensor(indices)
--   index_select :2981  args (Tensor x, Tensor index, int axis=0)
--   index_sample :2968  args (Tensor x, Tensor index)
--   index_add    :2895  args (Tensor x, Tensor index, Tensor add_value, int axis=0)
--   gather  :2516   args (Tensor x, Tensor index, Scalar axis=0)
--   gather_nd :2527 args (Tensor x, Tensor index)
--   scatter :4868   args (Tensor x, Tensor index, Tensor updates, ...)
--   topk    :5646   output Tensor(out), Tensor(indices)
--   one_hot :4191   args (Tensor x, Scalar(int) num_classes)   -- x is label
--   cross_entropy_with_softmax :1275  args (Tensor input, Tensor label, ...)
--
-- This first batch covers operators whose index semantics were VERIFIED
-- against the yaml above. Treat every other operator as UNANNOTATED: the
-- generator must refuse to emit it until someone reads its yaml and adds
-- an entry here (never guess, never default).

local M = {}

-- op -> { takes_index = {param names} }
-- Parameters whose VALUE carries index semantics (label / ids / token ids),
-- i.e. 1-based in Lua, decrement by 1 before the C++ call.
M.takes_index = {
  index_select = { "index" },
  index_sample = { "index" },
  index_add    = { "index" },
  gather       = { "index" },
  gather_nd    = { "index" },
  scatter      = { "index" },
  one_hot      = { "x" },        -- x is the class label (1..num_classes)
  cross_entropy_with_softmax = { "label" },
}

-- op -> { output positions 1-based }
-- Outputs whose VALUES are indices (0-based from C++), increment by 1
-- before returning to Lua.
M.returns_index = {
  argmax = { 1 },
  argmin = { 1 },
  argsort = { 2 },   -- second output (indices)
  topk = { 2 },      -- second output (indices)
}

-- op -> true
-- Operators whose args/outputs were READ and found to carry NO index
-- semantics at all. The generator must still see the operator in this
-- table before emitting it (conventions.md sec.5.2: never default an
-- unannotated operator through); `no_index` is the explicit "I checked,
-- nothing to convert" annotation. An operator is only allowed to skip
-- conversion if a human has read its yaml and put it HERE (or in
-- takes_index / returns_index).
--
-- Verified 2026-10-24 (2.3.3 trial generation):
--   add    : inconsistent/dygraph_ops.yaml:4   (Tensor x, Tensor y) -> Tensor(out)
--   matmul : inconsistent/dygraph_ops.yaml:248 (Tensor x, Tensor y,
--            bool transpose_x=false, bool transpose_y=false) -> Tensor
--   relu   : ops.yaml:4626                     (Tensor x) -> Tensor(out)
M.no_index = {
  add = true,
  matmul = true,
  relu = true,
}

-- ---------------------------------------------------------------------------
-- Query API for the generator
-- ---------------------------------------------------------------------------

-- Returns true if the operator is known to the table at all.
function M.has(op)
  return M.takes_index[op] ~= nil or M.returns_index[op] ~= nil or M.no_index[op] == true
end

-- Params of `op` that are 1-based indices (empty table if none annotated).
function M.index_params(op)
  return M.takes_index[op] or {}
end

-- 1-based output positions of `op` that carry index values.
function M.index_outputs(op)
  return M.returns_index[op] or {}
end

-- Hard guard for the generator (conventions.md sec.5.2): emit a fatal
-- error unless the operator is annotated. NEVER default to pass-through;
-- a skipped annotation is a silent off-by-one.
--   op_name: the yaml `op :` name
--   where:   caller context for the error message
function M.require_annotated(op_name, where)
  if not M.has(op_name) then
    error(string.format(
      "index_semantics: operator '%s' is not annotated (%s). "
      .. "Read its args/output in paddle/phi/ops/yaml/ops.yaml and add an "
      .. "entry to tools/gen/index_semantics.lua before generating. "
      .. "Never default an unannotated operator through.",
      tostring(op_name), tostring(where)))
  end
end

return M