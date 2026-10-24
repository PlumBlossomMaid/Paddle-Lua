-- tools/gen/emit_capi.lua
-- Generates C ABI operator wrappers (csrc/capi_gen/ops_*.cc + ops_gen.h)
-- from the yaml IR produced by yaml_subset.lua. Pure Lua 5.1 subset,
-- zero Python (C9 / R42).
--
-- Output shape follows the REAL hand-written code in csrc/capi/ (the
-- example in 03-codegen.md sec.3.2 uses AS_T/new_handle which do NOT
-- exist -- never reference them):
--   - handle unwrap: reinterpret_cast<pd_tensor_t*>(h)->value  (tensor.cc)
--   - exception boundary: PD_GUARD_BEGIN ... PD_GUARD_END(nullptr)
--     (capi_impl.h)
--   - *_ad_func declared in:
--     paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h
--     (include path precedent: tools/smoke/capi_mini_train.cc:38)
--   - bool attrs ride as int (0/1): paddle_c.h has NO C99 bool for the
--     LuaJIT FFI (01-c-abi.md sec.4), C++ bool params get `!= 0`.
--   - optional<Tensor*> predefined_out: skipped (defaults to paddle::none,
--     03-codegen.md sec.4 item 4).
--
-- 1-based conversion (C4/D13): Tensor-shaped index semantics are converted
-- HERE, at the C ABI layer, because paddle::experimental::subtract/add
-- have no scalar overloads (api.h:1412/1316) -- building an all-ones
-- tensor needs libpaddle (full_like api.h:514), which Lua 5.1 cannot do.
-- Scalar axis/dim attrs and scalar takes_index attrs (index_select_strided
-- index) are converted by emit_lua.lua (arithmetic on numbers only).
-- Every op MUST pass index_semantics.require_annotated first
-- (conventions.md sec.5.2) -- add/matmul/relu were read and marked
-- no_index in index_semantics.lua (2.3.3).
--
-- Usage:
--   lua tools/gen/emit_capi.lua <out-dir> <yaml> [<yaml> ...] [--sha <git-sha>]
--     reads all yaml files, emits one ops_<letter>.cc per first letter of
--     the op name plus a single ops_gen.h with all declarations.

-- Locate sibling modules: generators always run from the repo root
-- (layout.md workflow), so tools/gen/ is on the module path. MUST be
-- set before the requires below (require runs at first use).
package.path = "tools/gen/?.lua;" .. package.path

local yaml = require "yaml_subset"
local index_sem = require "index_semantics"
local skip_rules = require "skip_rules"

-- C ABI parameter C type for a C++ type from yaml_types_mapping
-- (yaml_subset.lua YAML_TYPES). Scalar-ish types map to a single C
-- parameter; array-like types expand to TWO C parameters `(const T* data,
-- int64_t n)` per 01-c-abi.md sec.4 item 4 ("structure-free ABI: pass
-- shape as (dims, ndim), never define pd_shape") and are constructed
-- back to the C++ container at the call site. dtype -> pd_dtype enum
-- (01-c-abi.md sec.3.4, paddle_c.h), place -> pd_place handle
-- (paddle_c.h). Scalar -> double: Lua numbers are doubles only; upstream
-- kernels consume a Scalar via scalar.to<T>() and never branch on its
-- internal dtype (paddle/fluid/pybind/eager_utils.cc:2541 constructs per
-- Python type, phi kernels read with to<T>), so double is behaviourally
-- equivalent (decision 2026-10-24, see process/decisions.md).
--
-- Returns a table describing ONE attr param:
--   { params = {decl...},   -- C declarations with <name> placeholder
--     call   = {expr...},   -- call-site argument expressions
--     prelude = {...}}      -- extra statements before the call (rare)
local function capi_type(cpp_type)
  local scalar = {
    ["Tensor"]    = { "pd_tensor <name>" , "reinterpret_cast<pd_tensor_t*>(<name>)->value /* Tensor */" },
    ["bool"]      = { "int <name>"       , "<name> != 0 /* bool */" },
    ["int"]       = { "int <name>"       , "<name> /* int */" },
    ["int64_t"]   = { "int64_t <name>"   , "<name> /* int64_t */" },
    ["float"]     = { "float <name>"     , "<name> /* float */" },
    ["double"]    = { "double <name>"    , "<name> /* double */" },
    ["std::string"] = { "const char* <name>", "<name> /* std::string */" },
    ["phi::DataType"] = { "pd_dtype <name>", "static_cast<phi::DataType>(<name>) /* phi::DataType */" },
    ["paddle::Place"] = { "pd_place <name>", "reinterpret_cast<pd_place_t*>(<name>)->value /* paddle::Place */" },
    ["paddle::experimental::Scalar"] =
      { "double <name>", "paddle::experimental::Scalar(<name>) /* paddle::experimental::Scalar */" },
  }
  local s = scalar[cpp_type]
  if s then
    return { params = { s[1] }, call = { s[2] } }
  end
  local array = {
    --   elem C type      call-site expression (constructor shape differs:
    --   IntArray takes (ptr, n), std::vector takes (first, last) iterators)
    ["paddle::experimental::IntArray"] =
      { "const int64_t*", "paddle::experimental::IntArray(<name>, <name>_n)" },
    ["std::vector<int64_t>"] =
      { "const int64_t*", "std::vector<int64_t>(<name>, <name> + <name>_n)" },
    ["std::vector<int>"]     =
      { "const int*",     "std::vector<int>(<name>, <name> + <name>_n)" },
    ["std::vector<float>"]   =
      { "const float*",   "std::vector<float>(<name>, <name> + <name>_n)" },
    ["std::vector<double>"]  =
      { "const double*",  "std::vector<double>(<name>, <name> + <name>_n)" },
  }
  local a = array[cpp_type]
  if a then
    return {
      params = { a[1] .. " <name>", "int64_t <name>_n" },
      call   = { a[2] .. " /* " .. cpp_type .. " */" },
    }
  end
  if cpp_type == "std::vector<phi::Scalar>" then
    -- elementwise double -> phi::Scalar; only op is set_value (build a
    -- temp vector in a prelude, then pass it).
    return {
      params = { "const double* <name>", "int64_t <name>_n" },
      prelude = {
        "  std::vector<phi::Scalar> <name>_tmp;",
        "  <name>_tmp.reserve(<name>_n);",
        "  for (int64_t i = 0; i < <name>_n; ++i) <name>_tmp.emplace_back(<name>[i]);",
      },
      call = { "<name>_tmp /* std::vector<phi::Scalar> */" },
    }
  end
  assert(false, "emit_capi: no C ABI type for '" .. cpp_type
          .. "' (add to capi_type table in emit_capi.lua; don't guess)")
end

-- True if this C++ type needs `!= 0` at the ad_func call site (C99-int
-- ABI param back to C++ bool).
local function is_bool(cpp_type)
  return cpp_type == "bool"
end

-- Skip rules (03-codegen.md sec.3.4): in-place '_'-suffixed variants,
-- sparse/strings (separate yaml, out of v1 scope), distributed c_allreduce,
-- graph-learning / tree / distributed-trainer edge ops. Single source of
-- truth is tools/gen/skip_rules.lua.
local function is_skipped(op_name)
  return skip_rules.skip_reason(op_name)
end

-- Build the op list from yaml IR, applying skip + index-semantics gates.
-- Returns array of IR tables (same shape as yaml_subset.block_ir output).
-- `whitelist` (array of op names) limits emission to a trial set
-- (2.3.3 three-op trial); empty = all ops (2.3.4 full generation).
local function collect_ops(paths, whitelist)
  local want = {}
  for _, w in ipairs(whitelist or {}) do
    want[w] = true
  end
  local ops = {}
  for _, p in ipairs(paths) do
    local blocks = yaml.parse_file(p)
    for _, b in ipairs(blocks) do
      if b.kind == "forward" then        -- ops.yaml `- op :` blocks only
        if #whitelist == 0 or want[b.name] then
          local reason = is_skipped(b.name)
          if reason then
            io.stderr:write("  skip " .. b.name .. " (" .. reason .. ")\n")
          else
            index_sem.require_annotated(b.name, "emit_capi (" .. p .. ")")
            local ok, ir = pcall(yaml.block_ir, b)
            if not ok then
              error("emit_capi: block_ir failed for '" .. b.name .. "': "
                    .. tostring(ir))
            end
            ops[#ops + 1] = ir
          end
        end
      end
    end
  end
  -- deterministic order for re-entrancy (03-codegen.md sec.3.5): the yaml
  -- files are stable, sort by file order then op name to be safe
  table.sort(ops, function(a, b)
    if a.op ~= b.op then return a.op < b.op end
    return a.line < b.line
  end)
  return ops
end

-- yaml `optional:` names -> set (a Tensor listed there is Tensor?;
-- dygraph_functions.h:131 proves ad_func takes paddle::optional for them).
local function optional_set(ir)
  local opt = {}
  if ir.optional then
    for n in string.gmatch(ir.optional, "[^,%s]+") do
      opt[n] = true
    end
  end
  return opt
end

-- Names of this op's 1-based index parameters / output positions
-- (index_semantics.lua; empty if none annotated).
local function takes_index_set(ir)
  local s = {}
  for _, p in ipairs(index_sem.index_params(ir.op)) do
    s[p] = true
  end
  return s
end
local function returns_index_set(ir)
  local s = {}
  for _, p in ipairs(index_sem.index_outputs(ir.op)) do
    s[p] = true
  end
  return s
end

-- True if the op needs the pd_idx_* helper (any Tensor-ish takes_index
-- input, or any returns_index output). Scalar takes_index attrs (e.g.
-- index_select_strided's int64 index) are converted by the Lua wrapper
-- (emit_lua), not here -- they need no tensor arithmetic.
local function needs_index_helper(ir)
  if #index_sem.index_outputs(ir.op) > 0 then return true end
  local takes = takes_index_set(ir)
  for _, in_ in ipairs(ir.inputs or {}) do
    if takes[in_.name] then return true end
  end
  return false
end

-- C arguments list for a wrapper: inputs as pd_tensor (four shapes:
-- Tensor -> pd_tensor, Tensor? -> pd_tensor with NULL = none, and
-- std::vector<Tensor>[-?] -> (const pd_tensor* p, int64_t n) pair per
-- 01-c-abi.md sec.4 item 4), attrs expanded via capi_type (one scalar C
-- param, or (ptr, n) for array-like / extra prelude statements).
-- Returns (params, callargs, preludes) as strings.
local function signature(ir)
  local params, callargs, preludes = {}, {}, {}
  local opt = optional_set(ir)
  local takes = takes_index_set(ir)
  local function add_abi(abi, name)
    for _, decl in ipairs(abi.params or {}) do
      params[#params + 1] = string.gsub(decl, "<name>", name)
    end
    for _, pre in ipairs(abi.prelude or {}) do
      preludes[#preludes + 1] = string.gsub(pre, "<name>", name)
    end
    for _, c in ipairs(abi.call or {}) do
      callargs[#callargs + 1] = string.gsub(c, "<name>", name)
    end
  end
  for _, in_ in ipairs(ir.inputs or {}) do
    local name = in_.name
    if in_.type == "Tensor" then
      if takes[name] then
        -- 1-based index tensor: elementwise -1 via subtract(full_like(t,1)).
        -- full_like: paddle/phi/api/include/api.h:514; subtract: api.h:1412.
        if opt[name] then
          add_abi({
            params = { "pd_tensor <name>" },
            prelude = {
              "  paddle::optional<paddle::Tensor> <name>_1b = paddle::none;",
              "  if (<name>) {",
              "    const paddle::Tensor& <name>_t = "
                .. "reinterpret_cast<pd_tensor_t*>(<name>)->value;",
              "    <name>_1b = paddle::make_optional(pd_idx_minus1(<name>_t));",
              "  }",
            },
            call = { "<name>_1b /* Tensor? index 1-based -> 0-based */" },
          }, name)
        else
          add_abi({
            params = { "pd_tensor <name>" },
            prelude = {
              "  const paddle::Tensor& <name>_t = "
                .. "reinterpret_cast<pd_tensor_t*>(<name>)->value;",
              "  paddle::Tensor <name>_1b = pd_idx_minus1(<name>_t);",
            },
            call = { "<name>_1b /* Tensor index 1-based -> 0-based */" },
          }, name)
        end
      elseif opt[name] then
        add_abi({
          params = { "pd_tensor <name>" },
          call = { "<name> ? paddle::make_optional("
            .. "reinterpret_cast<pd_tensor_t*>(<name>)->value) : paddle::none"
            .. " /* Tensor? */" },
        }, name)
      else
        add_abi({
          params = { "pd_tensor <name>" },
          call = { "reinterpret_cast<pd_tensor_t*>(<name>)->value /* Tensor */" },
        }, name)
      end
    elseif in_.type == "std::vector<Tensor>" then
      if takes[name] then
        if opt[name] then
          add_abi({
            params = { "const pd_tensor* <name>", "int64_t <name>_n" },
            prelude = {
              "  std::vector<paddle::Tensor> <name>_tmp;",
              "  if (<name>) {",
              "    <name>_tmp.reserve(<name>_n);",
              "    for (int64_t i = 0; i < <name>_n; ++i)",
              "      <name>_tmp.emplace_back(pd_idx_minus1("
                .. "reinterpret_cast<pd_tensor_t*>(<name>[i])->value));",
              "  }",
            },
            call = { "<name> ? paddle::make_optional(<name>_tmp) : paddle::none"
              .. " /* std::vector<Tensor>? index */" },
          }, name)
        else
          add_abi({
            params = { "const pd_tensor* <name>", "int64_t <name>_n" },
            prelude = {
              "  std::vector<paddle::Tensor> <name>_tmp;",
              "  <name>_tmp.reserve(<name>_n);",
              "  for (int64_t i = 0; i < <name>_n; ++i)",
              "    <name>_tmp.emplace_back(pd_idx_minus1("
                .. "reinterpret_cast<pd_tensor_t*>(<name>[i])->value));",
            },
            call = { "<name>_tmp /* std::vector<Tensor> index */" },
          }, name)
        end
      elseif opt[name] then
        add_abi({
          params = { "const pd_tensor* <name>", "int64_t <name>_n" },
          prelude = {
            "  std::vector<paddle::Tensor> <name>_tmp;",
            "  if (<name>) {",
            "    <name>_tmp.reserve(<name>_n);",
            "    for (int64_t i = 0; i < <name>_n; ++i)",
            "      <name>_tmp.emplace_back("
              .. "reinterpret_cast<pd_tensor_t*>(<name>[i])->value);",
            "  }",
          },
          call = { "<name> ? paddle::make_optional(<name>_tmp) : paddle::none"
            .. " /* std::vector<Tensor>? */" },
        }, name)
      else
        add_abi({
          params = { "const pd_tensor* <name>", "int64_t <name>_n" },
          prelude = {
            "  std::vector<paddle::Tensor> <name>_tmp;",
            "  <name>_tmp.reserve(<name>_n);",
            "  for (int64_t i = 0; i < <name>_n; ++i)",
            "    <name>_tmp.emplace_back("
              .. "reinterpret_cast<pd_tensor_t*>(<name>[i])->value);",
          },
          call = { "<name>_tmp /* std::vector<Tensor> */" },
        }, name)
      end
    else
      error("emit_capi: unsupported input type '" .. in_.type .. "' for '"
            .. ir.op .. "." .. name .. "'")
    end
  end
  for _, attr in ipairs(ir.attrs or {}) do
    local abi = capi_type(attr.type)
    local name = attr.name
    for _, decl in ipairs(abi.params) do
      local d = string.gsub(decl, "<name>", name)
      params[#params + 1] = d
    end
    for _, pre in ipairs(abi.prelude or {}) do
      local p = string.gsub(pre, "<name>", name)
      preludes[#preludes + 1] = p
    end
    for _, c in ipairs(abi.call) do
      local e = string.gsub(c, "<name>", name)
      callargs[#callargs + 1] = e
    end
  end
  return table.concat(params, ", "), table.concat(callargs, ", "),
         table.concat(preludes, "\n")
end

-- True iff every output is a plain Tensor (a multi-output *_ad_func returns
-- std::tuple<Tensor, ...> for such ops; the C ABI is an out-array, matching
-- the pd_jit_layer_forward precedent in paddle_c.h). Outputs of shape
-- std::vector<Tensor> (einsum/rnn/coalesce_tensor/distribute_fpn_proposals)
-- carry a runtime length -- the tuple element is a vector, not a Tensor --
-- and need a two-phase query ABI, designed separately (wall #3 extension).
local function all_tensor_outputs(rs)
  for _, r in ipairs(rs) do
    if r.type ~= "Tensor" then return false end
  end
  return true
end

-- Effective returns: yaml `output:` with the `intermediate:` names removed.
-- Upstream *_ad_func tuple contains ONLY the non-intermediate outputs
-- (verified: lstm yaml declares 4 outputs, intermediate batch_gate/
-- batch_cell_pre_act, lstm_ad_func returns std::tuple<Tensor, Tensor> --
-- dygraph_functions.h). Counts, std::get indices and multi/single dispatch
-- must all use this filtered list, not the raw yaml returns.
local function effective_returns(ir)
  local inter = {}
  if ir.intermediate then
    for w in string.gmatch(ir.intermediate, "[^,%s]+") do
      inter[w] = true
    end
  end
  local eff = {}
  for _, r in ipairs(ir.returns or {}) do
    if not inter[r.name] then
      eff[#eff + 1] = r
    end
  end
  return eff
end

-- True iff the op returns exactly ONE std::vector<Tensor> (no tuple):
-- ap_facade / broadcast_tensors / meshgrid / split / split_with_num /
-- sync_comm_stream / unbind / unstack / ap_variadic (`output : Tensor[]{...}`
-- in the yaml; ad_func returns std::vector<paddle::Tensor> directly,
-- dygraph_functions.h split_ad_func:1013 / unstack_ad_func:1126).
-- These are NOT the tuple-with-vector case (which needs the two-phase
-- query design and lives in skip_rules) -- a single vector output can ride
-- the same out-array C ABI: `int64_t pd_op_x(pd_tensor* outs, int64_t nout,
-- ...)` returns the runtime count and fills at most nout entries, caller
-- queries-then-allocates (03-codegen.md sec.4 pit #7, single-vector form).
local function single_vec_output(ir)
  local eff = effective_returns(ir)
  return #eff == 1 and eff[1].type == "std::vector<Tensor>"
end

-- Emit one single-output wrapper body for the op.
local function emit_op(ir, out)
  -- op name -> C function name: pd_op_<name>
  local fn = "pd_op_" .. ir.op
  local params, callargs, preludes = signature(ir)
  local ret_idx = returns_index_set(ir)
  out[#out + 1] = string.format(
    "/* %s (yaml block line %d) */", ir.op, ir.line)
  out[#out + 1] = string.format(
    "pd_tensor %s(%s) {", fn, params)
  out[#out + 1] = "  PD_GUARD_BEGIN;"
  if preludes ~= "" then
    out[#out + 1] = preludes   -- already indented, ends with newline
  end
  out[#out + 1] = "  paddle::Tensor out = " .. ir.op .. "_ad_func("
  out[#out + 1] = "      " .. callargs .. ");"
  if ret_idx[1] then
    -- returns_index position 1: the output's VALUES are 0-based indices,
    -- bump to 1-based before handing to Lua (conventions.md 5.1).
    out[#out + 1] = "  out = pd_idx_plus1(out);"
  end
  out[#out + 1] = "  return reinterpret_cast<pd_tensor>("
  out[#out + 1] = "      new pd_tensor_t{std::move(out)});"
  out[#out + 1] = "  PD_GUARD_END(nullptr);"
  out[#out + 1] = "}"
  out[#out + 1] = ""
end

-- Emit one multi-output wrapper body. C ABI shape (design 01-c-abi.md
-- sec.3.1 ownership + paddle_c.h pd_jit_layer_forward precedent):
--   int64_t pd_op_x(pd_tensor* outs, int64_t nout, <params>...)
--     ret: number of outputs written
--     outs[i]: new handle, caller owns each (must pd_tensor_free)
--     nout < actual count: writes min(nout, count) and returns count
--       (caller resizes and retries -- query-then-allocate)
-- The yaml count is a compile-time lower bound for every Tensor output; only
-- vector<Tensor> outputs (rejected in emit_all) have a runtime length.
-- Output count = effective_returns (intermediate outputs are not returned
-- by the ad_func tuple; e.g. lstm yaml declares 4, tuple has 2).
local function emit_op_multi(ir, out)
  local fn = "pd_op_" .. ir.op
  local params, callargs, preludes = signature(ir)
  local eff = effective_returns(ir)
  local n = #eff
  local ret_idx = returns_index_set(ir)
  out[#out + 1] = string.format(
    "/* %s (yaml block line %d, %d outputs) */", ir.op, ir.line, n)
  out[#out + 1] = string.format(
    "int64_t %s(pd_tensor* outs, int64_t nout%s%s) {", fn,
    (params ~= "" and ", " or ""), params)
  out[#out + 1] = "  PD_GUARD_BEGIN;"
  if preludes ~= "" then
    out[#out + 1] = preludes   -- already indented, ends with newline
  end
  out[#out + 1] = "  auto r = " .. ir.op .. "_ad_func("
  out[#out + 1] = "      " .. callargs .. ");"
  -- unpack the tuple element-by-element; std::get<N> is the safe C++17 form
  -- (structured bindings on a tuple of references would over-complicate).
  for i = 1, n do
    if ret_idx[i] then
      -- returns_index position i: the output's VALUES are 0-based indices,
      -- bump to 1-based before handing to Lua (conventions.md 5.1).
      out[#out + 1] = string.format(
        "  std::get<%d>(r) = pd_idx_plus1(std::get<%d>(r));", i - 1, i - 1)
    end
    out[#out + 1] = string.format(
      "  if (nout >= %d) outs[%d] = reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(std::get<%d>(r))});",
      i, i - 1, i - 1)
  end
  out[#out + 1] = string.format("  return %d;", n)
  out[#out + 1] = "  PD_GUARD_END(0);"
  out[#out + 1] = "}"
  out[#out + 1] = ""
end

-- Emit one single-VECTOR-output wrapper body. Same out-array C ABI shape as
-- emit_op_multi (caller queries-then-allocates); the difference is the
-- ad_func returns std::vector<paddle::Tensor> directly instead of a tuple,
-- so the count is runtime-known and we iterate the vector. Ops: split,
-- unstack, unbind, broadcast_tensors, meshgrid, ap_facade, sync_comm_stream
-- ... (`output : Tensor[]{...}` in the yaml).
--   int64_t pd_op_x(pd_tensor* outs, int64_t nout, <params>...)
--     ret: number of outputs written (= vector size)
--     outs[i]: new handle, caller owns each (must pd_tensor_free)
local function emit_op_vec(ir, out)
  local fn = "pd_op_" .. ir.op
  local params, callargs, preludes = signature(ir)
  local ret_idx = returns_index_set(ir)
  out[#out + 1] = string.format(
    "/* %s (yaml block line %d, single vector output) */", ir.op, ir.line)
  out[#out + 1] = string.format(
    "int64_t %s(pd_tensor* outs, int64_t nout%s%s) {", fn,
    (params ~= "" and ", " or ""), params)
  out[#out + 1] = "  PD_GUARD_BEGIN;"
  if preludes ~= "" then
    out[#out + 1] = preludes   -- already indented, ends with newline
  end
  out[#out + 1] = "  auto r = " .. ir.op .. "_ad_func("
  out[#out + 1] = "      " .. callargs .. ");"
  out[#out + 1] = "  int64_t n = static_cast<int64_t>(r.size());"
  if ret_idx[1] then
    -- returns_index position 1: each element's VALUES are 0-based indices
    out[#out + 1] = "  for (int64_t i = 0; i < n && i < nout; ++i) {"
    out[#out + 1] = "    r[i] = pd_idx_plus1(r[i]);"
    out[#out + 1] = "    outs[i] = reinterpret_cast<pd_tensor>("
    out[#out + 1] = "        new pd_tensor_t{std::move(r[i])});"
    out[#out + 1] = "  }"
  else
    out[#out + 1] = "  for (int64_t i = 0; i < n && i < nout; ++i)"
    out[#out + 1] = "    outs[i] = reinterpret_cast<pd_tensor>("
    out[#out + 1] = "        new pd_tensor_t{std::move(r[i])});"
  end
  out[#out + 1] = "  return n;"
  out[#out + 1] = "  PD_GUARD_END(0);"
  out[#out + 1] = "}"
  out[#out + 1] = ""
end

-- Deterministic header + bucket emission.
local function emit_all(ops, outdir, sha)
  local buckets = {}   -- letter -> array of op names
  for _, ir in ipairs(ops) do
    -- multi-output ops: the *_ad_func returns a std::tuple; wrapping them
    -- as a single paddle::Tensor would silently produce uncompilable code
    -- (2.3.3 discipline: refuse to emit rather than emit something wrong).
    -- Pure-Tensor tuples are handled by emit_op_multi (wall #3); tuples
    -- containing std::vector<Tensor> keep refusing (two-phase ABI design is
    -- a separate extension). Count uses effective_returns: `intermediate:`
    -- outputs never appear in the ad_func tuple (lstm: 4 yaml outputs, 2
    -- in the tuple).
    local eff = effective_returns(ir)
    if not all_tensor_outputs(eff) and not single_vec_output(ir) then
      error("emit_capi: op '" .. ir.op .. "' has "
            .. #eff .. " returned output(s) with a std::vector<Tensor> "
            .. "element -- runtime-length outputs need a two-phase ABI, not "
            .. "implemented yet (wall #3 extension, see skip_rules.lua); "
            .. "refuse to emit")
    end
    -- guardrail: a returns_index position MUST point at a real effective
    -- output. Position 2+ on an op whose intermediate filtering collapses
    -- the tuple to 1 (e.g. sequence_pool max_index) is a dead annotation --
    -- the user never sees that output, so converting it is meaningless and
    -- silently skipping it hides an annotation bug (conventions sec.5.2).
    local rpos = index_sem.index_outputs(ir.op)
    for _, pos in ipairs(rpos) do
      if pos > #eff then
        error("emit_capi: op '" .. ir.op .. "' has returns_index position "
              .. pos .. " but only " .. #eff
              .. " effective output(s) -- the annotated output is probably "
              .. "`intermediate:` (never returned to the user); fix "
              .. "tools/gen/index_semantics.lua, don't generate")
      end
    end
    local letter = string.sub(ir.op, 1, 1)
    buckets[letter] = buckets[letter] or {}
    buckets[letter][#buckets[letter] + 1] = ir
  end

  -- declarations header
  local header = {}
  header[#header + 1] = "/* GENERATED BY tools/gen/emit_capi.lua"
  header[#header + 1] = " * FROM  paddle/phi/ops/yaml/{ops,backward,...}.yaml @ " .. sha
  header[#header + 1] = " * DO NOT EDIT"
  header[#header + 1] = " */"
  header[#header + 1] = "#ifndef PADDLE_LUA_CAPI_GEN_H"
  header[#header + 1] = "#define PADDLE_LUA_CAPI_GEN_H"
  header[#header + 1] = '#include "paddle_c.h"'
  header[#header + 1] = ""
  for _, ir in ipairs(ops) do
    local params = signature(ir)
    -- single std::vector<Tensor> output (split/unstack/unbind/...) also
    -- rides the out-array form: runtime count, caller queries-then-allocates
    if #effective_returns(ir) > 1 or single_vec_output(ir) then
      header[#header + 1] = string.format(
        "int64_t pd_op_%s(pd_tensor* outs, int64_t nout%s%s);", ir.op,
        (params ~= "" and ", " or ""), params)
    else
      header[#header + 1] = string.format(
        "pd_tensor pd_op_%s(%s);", ir.op, params)
    end
  end
  header[#header + 1] = ""
  header[#header + 1] = "#endif  /* PADDLE_LUA_CAPI_GEN_H */"
  local hf, herr = io.open(outdir .. "/ops_gen.h", "w")
  assert(hf, "emit_capi: cannot write ops_gen.h: " .. tostring(herr))
  hf:write(table.concat(header, "\n") .. "\n")
  hf:close()

  -- per-letter implementation files
  for letter in pairs(buckets) do
    table.sort(buckets[letter], function(a, b) return a.op < b.op end)
    -- 1-based conversion helpers needed by this bucket? Emit once per .cc
    -- (wrapper-local static, never exported -- paddle_c.h stays clean).
    local need_helper = false
    for _, ir in ipairs(buckets[letter]) do
      if needs_index_helper(ir) then need_helper = true break end
    end
    local cc = {}
    cc[#cc + 1] = "/* GENERATED BY tools/gen/emit_capi.lua"
    cc[#cc + 1] = " * FROM  paddle/phi/ops/yaml/{ops,backward,...}.yaml @ " .. sha
    cc[#cc + 1] = " * DO NOT EDIT"
    cc[#cc + 1] = " */"
    cc[#cc + 1] = '#include "capi_impl.h"'
    cc[#cc + 1] = "#include \"paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h\""
    if need_helper then
      cc[#cc + 1] = ""
      cc[#cc + 1] = "/* 1-based index conversion (conventions.md sec.5.1):"
      cc[#cc + 1] = " * takes_index tensors get elementwise -1 before the ad_func"
      cc[#cc + 1] = " * call; returns_index outputs get +1 before wrapping. Uses"
      cc[#cc + 1] = " * full_like to build an all-ones tensor (api.h:514) because"
      cc[#cc + 1] = " * subtract/add have no scalar overload (api.h:1412/1316)."
      cc[#cc + 1] = " * dtype/place follow the operand so GPU tensors stay on-device."
      cc[#cc + 1] = " */"
      cc[#cc + 1] = "static paddle::Tensor pd_idx_minus1(const paddle::Tensor& t) {"
      cc[#cc + 1] = "  return paddle::experimental::subtract("
      cc[#cc + 1] = "      t, paddle::experimental::full_like(t,"
      cc[#cc + 1] = "          paddle::experimental::Scalar(1), t.dtype(), t.place()));"
      cc[#cc + 1] = "}"
      cc[#cc + 1] = "static paddle::Tensor pd_idx_plus1(const paddle::Tensor& t) {"
      cc[#cc + 1] = "  return paddle::experimental::add("
      cc[#cc + 1] = "      t, paddle::experimental::full_like(t,"
      cc[#cc + 1] = "          paddle::experimental::Scalar(1), t.dtype(), t.place()));"
      cc[#cc + 1] = "}"
    end
    cc[#cc + 1] = ""
    for _, ir in ipairs(buckets[letter]) do
      -- dispatch on EFFECTIVE returns, not the raw yaml list: `intermediate:`
      -- outputs never appear in the ad_func tuple (sequence_pool yaml
      -- declares 2, ad_func returns a single Tensor -- dygraph_functions.h)
      -- so an op whose tuple is a single Tensor MUST take the single-output
      -- form or std::get<0>(r) would not compile. A single
      -- std::vector<Tensor> output (split/unstack/broadcast_tensors/...)
      -- needs emit_op_vec (out-array, runtime count) -- emitting it as a
      -- plain Tensor would not compile either.
      if single_vec_output(ir) then
        emit_op_vec(ir, cc)
      elseif #effective_returns(ir) > 1 then
        emit_op_multi(ir, cc)
      else
        emit_op(ir, cc)
      end
    end
    local f = "ops_" .. letter .. ".cc"
    local fh, err = io.open(outdir .. "/" .. f, "w")
    assert(fh, "emit_capi: cannot write " .. f .. ": " .. tostring(err))
    fh:write(table.concat(cc, "\n"))
    fh:close()
    io.stdout:write("  wrote " .. f .. " (" .. #buckets[letter] .. " ops)\n")
  end
end

-- ---------------------------------------------------------------------------
-- CLI
-- ---------------------------------------------------------------------------
local outdir, sha = nil, "unknown"
local paths, whitelist = {}, {}
local n = select("#", ...)
local i = 1
while i <= n do
  local a = select(i, ...)
  if a == "--sha" then
    i = i + 1
    sha = select(i, ...)
  elseif a == "--ops" then
    i = i + 1
    for w in string.gmatch(select(i, ...), "[^,]+") do
      whitelist[#whitelist + 1] = w
    end
  elseif a and string.sub(a, 1, 1) ~= "-" and outdir == nil then
    outdir = a
  elseif a and string.sub(a, 1, 1) ~= "-" then
    paths[#paths + 1] = a
  else
    io.stderr:write("emit_capi: unknown arg '" .. tostring(a) .. "'\n")
    os.exit(2)
  end
  i = i + 1
end

if not outdir or #paths == 0 then
  io.stderr:write(
    "usage: lua tools/gen/emit_capi.lua <out-dir> <yaml> [<yaml> ...] "
    .. "[--ops a,b,c] [--sha <git-sha>]\n")
  os.exit(2)
end

local ok, ops = pcall(collect_ops, paths, whitelist)
if not ok then
  io.stderr:write("emit_capi FAIL: " .. tostring(ops) .. "\n")
  os.exit(1)
end
if #ops == 0 then
  io.stderr:write("emit_capi: no ops to emit\n")
  os.exit(1)
end
io.stdout:write("emit_capi: " .. #ops .. " ops, output -> " .. outdir .. "\n")
local ok2, err2 = pcall(emit_all, ops, outdir, sha)
if not ok2 then
  io.stderr:write("emit_capi FAIL: " .. tostring(err2) .. "\n")
  os.exit(1)
end
os.exit(0)