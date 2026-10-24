-- tools/gen/skip_rules.lua
-- Operator skip rules shared by emit_capi.lua / emit_lua.lua and the
-- survey tools (03-codegen.md sec.3.4). Pure Lua 5.1 subset, zero Python
-- (C9 / R42). Single source of truth: a skip decision must not diverge
-- between the C ABI emit, the Lua wrappers emit and coverage counting.
--
-- Why these ops are skipped (03-codegen.md sec.3.4 table):
--   trailing `_`       in-place variants, M2
--   pir::Value args    static-graph only, P15
--   sparse_/strings_   separate yaml, v1 scope excludes
--   c_allreduce_*      distributed collectives, P17
--   graph-learning samplers (graph_khop/graph_sample/weighted_sample/
--   reindex_graph):    row/colptr are 0-based CSC structure offsets
--                      (prefix sums, colptr[0] == 0); node renumbering must
--                      stay consistent across all inputs & outputs, which
--                      is NOT the per-argument -1/+1 conversion that
--                      index_semantics.lua expresses. Graph module follows
--                      (P17 / later).
--   dgc                deep gradient compression, distributed trainer
--                      state (k / gather_buff), P17.
--   tdm_sampler        tree-based deep model sampler: travel/layer encode
--                      tree structure, not plain indices; out of v1 scope.
--   coalesce_tensor / distribute_fpn_proposals / einsum / rnn:
--                      runtime-length outputs -- the *_ad_func returns a
--                      std::tuple containing std::vector<Tensor> (element
--                      count only known at runtime, dygraph_functions.h
--                      coalesce_tensor_ADFUNC / rnn:907). The single-Tensor
--                      out-array C ABI (wall #3 emit_op_multi) cannot
--                      express them; a two-phase query ABI is a later
--                      extension. Verify the Python-side exposure before
--                      re-enabling: paddle.einsum returns only `out`, the
--                      vector outputs are grad-time scratch.

local M = {}

-- Returns a human-readable reason if `op_name` is skipped, nil otherwise.
function M.skip_reason(op_name)
  if string.sub(op_name, -1) == "_" then
    return "in-place variant (trailing _)"
  end
  if string.sub(op_name, 1, 7) == "sparse_" then
    return "sparse ops (separate yaml, v1 scope excludes)"
  end
  if string.sub(op_name, 1, 8) == "strings_" then
    return "strings ops (separate yaml, v1 scope excludes)"
  end
  if string.sub(op_name, 1, 11) == "c_allreduce" then
    return "distributed collective (P17)"
  end
  if op_name == "graph_khop_sampler"
  or op_name == "graph_sample_neighbors"
  or op_name == "weighted_sample_neighbors"
  or op_name == "reindex_graph"
  or op_name == "dgc"
  or op_name == "tdm_sampler" then
    return "graph-learning / tree / distributed trainer edge op (P17)"
  end
  if op_name == "coalesce_tensor"
  or op_name == "distribute_fpn_proposals"
  or op_name == "einsum"
  or op_name == "rnn" then
    return "runtime-length output (std::vector<Tensor> in ad_func tuple) -- two-phase ABI (later)"
  end
  return nil
end

return M