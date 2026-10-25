-- tools/gen/emit_sol.lua
-- Generates the sol2 operator-binding layer (csrc/sol/gen_ops_<letter>.cc
-- + csrc/sol/gen_ops_all.cc) from the yaml IR produced by yaml_subset.lua.
-- Pure Lua 5.1 subset, zero Python (C9 / R42).
--
-- Role (AGENTS sec.8.3: csrc/sol/gen_*.cpp is GENERATED, DO NOT EDIT):
--   the generated lua/paddle/_ops/*.lua wrappers call
--   paddle_core.pd_op_<name>(...) -- but nothing on the C++ side
--   registers pd_op_* yet. This generator closes that gap: each pd_op_*
--   C ABI function (csrc/capi_gen/ops_gen.h) becomes a sol2 lambda on the
--   paddle_core module table (02-binding.md sec.3, 03-codegen.md sec.3.1
--   emit_sol branch added 2026-10-25).
--
-- Shape decisions:
--   - Lua -> C ABI conversion lives HERE (the sol2 lambda body), NOT in
--     the Lua wrapper: emit_lua passes raw Lua values (numbers, strings,
--     LuaTensor usertype instances, tables), this layer turns them into
--     the C ABI types (pd_tensor / pd_dtype / pd_place / (ptr, n) pairs).
--   - 1-based conversion is NOT this layer's job: Tensor index +/-1 is
--     done in the C ABI layer (emit_capi pd_idx_minus1/plus1), scalar
--     axis/dim in the Lua wrapper (emit_lua). The sol2 layer is a thin
--     pass-through (conventions.md sec.5.1, 03-codegen.md sec.4 item 8).
--   - bool attrs arrive as 0/1 numbers (emit_lua emits `(x and 1 or 0)`),
--     so the lambda takes `int` -- never `bool` (C ABI has no C99 bool,
--     01-c-abi.md sec.4; sol2 would lua_toboolean 0 -> false, fine, but
--     int matches the C ABI surface exactly).
--   - Multi-output ops (C ABI int64_t pd_op_x(pd_tensor* outs, ...)):
--     the effective output count is compile-time known (effective_returns
--     -- `intermediate:` outputs never appear in the ad_func tuple, so
--     this is NOT a query-then-allocate case). Emit a fixed
--     std::array<pd_tensor, N> and one call, then sol::variadic_results
--     (emit_lua contract: `y1, y2 = paddle_core.pd_op_x(...)`).
--   - Single-vector-output ops (split/unstack/... runtime length) DO use
--     query-then-allocate (C ABI returns the runtime count; the query
--     call with nout=0 computes and discards -- documented cost,
--     03-codegen.md sec.4 item 9), then sol::as_table (Python returns a
--     list; Lua gets a 1-based table -- sol2 pusher starts at index 1,
--     stack_push.hpp:420).
--   - Place attrs: Lua side passes a LuaPlace (paddle_core.pd_place_new
--     factory, hand-written enums_place.cc); the lambda takes
--     `const LuaPlace&` and passes .h. dtype attrs: Lua passes the
--     PD_DTYPE_* integer constant (hand-written enums_place.cc).
--
-- Usage:
--   lua tools/gen/emit_sol.lua <out-dir> <yaml> [<yaml> ...] [--ops a,b,c]
--       [--sha <git-sha>]
--     emits one csrc/sol/gen_ops_<letter>.cc per first letter of the op
--     name, each defining pd::register_ops_<letter>(sol::table&), plus
--     gen_ops_all.cc defining pd::register_ops(lua_State*, int).

-- Locate sibling modules: generators always run from the repo root
-- (layout.md workflow), so tools/gen/ is on the module path. MUST be
-- set before the requires below (require runs at first use).
package.path = "tools/gen/?.lua;" .. package.path

local yaml = require "yaml_subset"
local index_sem = require "index_semantics"
local skip_rules = require "skip_rules"

-- Skip rules (03-codegen.md sec.3.4): single source of truth is
-- tools/gen/skip_rules.lua -- mirrored from emit_capi (pipelines, not
-- rules: the skip REASONS themselves live in skip_rules.lua).
local function is_skipped(op_name)
  return skip_rules.skip_reason(op_name)
end

-- Build the op list from yaml IR, applying skip + index-semantics gates
-- (same shape as emit_capi.collect_ops -- deterministic order for
-- re-entrancy: sort by op name then yaml line, 03-codegen.md sec.3.5).
local function collect_ops(paths, whitelist)
  local want = {}
  for _, w in ipairs(whitelist or {}) do
    want[w] = true
  end
  local ops = {}
  for _, p in ipairs(paths) do
    local blocks = yaml.parse_file(p)
    for _, b in ipairs(blocks) do
      if b.kind == "forward" then
        if #whitelist == 0 or want[b.name] then
          local reason = is_skipped(b.name)
          if reason then
            io.stderr:write("  skip " .. b.name .. " (" .. reason .. ")\n")
          else
            index_sem.require_annotated(b.name, "emit_sol (" .. p .. ")")
            local ok, ir = pcall(yaml.block_ir, b)
            if not ok then
              error("emit_sol: block_ir failed for '" .. b.name .. "': "
                    .. tostring(ir))
            end
            ops[#ops + 1] = ir
          end
        end
      end
    end
  end
  table.sort(ops, function(a, b)
    if a.op ~= b.op then return a.op < b.op end
    return a.line < b.line
  end)
  return ops
end

-- yaml `optional:` names -> set (a Tensor listed there is Tensor?;
-- dygraph_functions.h:131 proves ad_func takes paddle::optional).
local function optional_set(ir)
  local opt = {}
  if ir.optional then
    for n in string.gmatch(ir.optional, "[^,%s]+") do
      opt[n] = true
    end
  end
  return opt
end

-- Effective returns: yaml `output:` minus `intermediate:` names (the
-- ad_func tuple contains ONLY the non-intermediate outputs, 03-codegen.md
-- sec.4 item 6). Mirrored from emit_capi -- must stay in lockstep.
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

-- True iff every output is a plain Tensor (multi-output tuple form).
local function all_tensor_outputs(rs)
  for _, r in ipairs(rs) do
    if r.type ~= "Tensor" then return false end
  end
  return true
end

-- True iff the op returns exactly ONE std::vector<Tensor> (split /
-- unstack / unbind / broadcast_tensors / meshgrid / ... runtime length;
-- rides the same out-array C ABI, query-then-allocate).
local function single_vec_output(ir)
  local eff = effective_returns(ir)
  return #eff == 1 and eff[1].type == "std::vector<Tensor>"
end

-- ---------------------------------------------------------------------
-- sol2-side parameter mapping (Lua value -> C ABI call argument)
-- ---------------------------------------------------------------------

-- One sol2 lambda parameter for one input or attr. Returns
--   { decls  = { <lambda parameter declaration>... },
--     prelude = { <statements before the call>... },   (or nil)
--     args   = { <C ABI call arguments>... } }
-- `name` is the yaml parameter name (already unique; C++ identifiers
-- match yaml here because the C ABI layer uses the same names).

local function sol_attr(attr)
  local name = attr.name
  local t = attr.type
  local scalar = {
    ["bool"]     = { "int <name>"        , "<name> /* bool rides as 0/1 */" },
    ["int"]      = { "int <name>"        , "<name>" },
    ["int64_t"]  = { "int64_t <name>"    , "<name>" },
    ["float"]    = { "float <name>"      , "<name>" },
    ["double"]   = { "double <name>"     , "<name>" },
    ["std::string"] = { "std::string <name>", "<name>.c_str()" },
    -- pd_dtype is a C enum (paddle_c.h) -- C++ forbids implicit int->enum,
    -- so the call site needs the explicit cast (mirrors emit_capi's
    -- capi_type "static_cast<phi::DataType>").
    ["phi::DataType"] = { "int <name>"   , "static_cast<pd_dtype>(<name>)" },
    ["paddle::Place"] = { "const LuaPlace& <name>", "<name>.h" },
    ["paddle::experimental::Scalar"] =
      { "double <name>", "<name> /* Scalar rides as double */" },
  }
  local s = scalar[t]
  if s then
    return {
      -- parens force single value: gsub returns (str, count), and a bare
      -- call in a table constructor expands ALL returns at its position
      -- (Lua 5.1 multi-return pitfall, same one emit_capi documents in
      -- lua_array_literal)
      decls = { (string.gsub(s[1], "<name>", name)) },
      args = { (string.gsub(s[2], "<name>", name)) },
    }
  end
  local array = {
    -- elem C type      (data(), size()) argument pair
    ["paddle::experimental::IntArray"] = "int64_t",
    ["std::vector<int64_t>"] = "int64_t",
    ["std::vector<int>"]     = "int",
    ["std::vector<float>"]   = "float",
    ["std::vector<double>"]  = "double",
  }
  local e = array[t]
  if e then
    -- BY VALUE, never const&: sol2 3.5 on this env crashes 0xC0000005 when
    -- a Lua table is converted to `const std::vector<T>&` (verified
    -- standalone + in-module 2026-10-25; by-value is safe for int64_t/
    -- double/string/usertype elements and optional containers).
    local d = "std::vector<" .. e .. "> <name>"
    local a = "<name>.data(), static_cast<int64_t>(<name>.size())"
    return {
      decls = { (string.gsub(d, "<name>", name)) },
      args = { (string.gsub(a, "<name>", name)) },
    }
  end
  if t == "std::vector<phi::Scalar>" then
    -- only op is set_value; C ABI takes (const double*, n) and builds
    -- the vector<phi::Scalar> itself (emit_capi capi_type prelude); by
    -- value for the same sol2 reason as above. Parens + gsub: force the
    -- single substitution value (Lua 5.1 multi-return, see scalar path).
    return {
      decls = { (string.gsub("std::vector<double> <name>", "<name>", name)) },
      args = { (string.gsub("<name>.data(), static_cast<int64_t>(<name>.size())", "<name>", name)) },
    }
  end
  error("emit_sol: no sol2 lambda type for attr type '" .. t
        .. "' (op " .. tostring(attr) .. "; add to sol_attr, don't guess)")
end

-- One input: Tensor / Tensor? / std::vector<Tensor> / vector<Tensor>?.
local function sol_input(in_, opt)
  local name = in_.name
  if in_.type == "Tensor" then
    if opt[name] then
      -- nil -> nullopt (sol2 optional usertype getter, stack_get_unqualified.hpp:163)
      return {
        decls = { "sol::optional<LuaTensor> " .. name },
        args = { name .. " ? " .. name .. "->h : nullptr" },
      }
    end
    return {
      decls = { "const LuaTensor& " .. name },
      args = { name .. ".h" },
    }
  end
  if in_.type == "std::vector<Tensor>" then
    -- BY VALUE / optional<vector>, NEVER `const std::vector<T>&`: sol2 3.5
    -- on this env crashes 0xC0000005 converting a Lua table to a container
    -- by const-ref (verified standalone round1/2 + in-module sum/split
    -- 2026-10-25). By-value and sol::optional<std::vector<T>> are safe.
    if opt[name] then
      -- optional vector: C ABI distinguishes none (p == NULL) from an
      -- empty list (p != NULL, n == 0). std::vector::data() on an empty
      -- vector may return nullptr (implementation-defined), which would
      -- be misread as "none" -- anchor an empty list to a local cell so
      -- the pointer is always non-null when engaged.
      return {
        decls = { "sol::optional<std::vector<LuaTensor>> " .. name },
        prelude = {
          "  std::vector<pd_tensor> " .. name .. "_h;",
          "  if (" .. name .. ") {",
          "    " .. name .. "_h.reserve(" .. name .. "->size());",
          "    for (const LuaTensor& t : *" .. name
            .. ") " .. name .. "_h.push_back(t.h);",
          "  }",
          "  pd_tensor " .. name .. "_anchor = nullptr;",
          "  const pd_tensor* " .. name .. "_p = " .. name
            .. " ? (" .. name .. "_h.empty() ? &" .. name
            .. "_anchor : " .. name .. "_h.data()) : nullptr;",
        },
        args = {
          name .. "_p",
          name .. " ? static_cast<int64_t>(" .. name .. "_h.size()) : 0",
        },
      }
    end
    return {
      decls = { "std::vector<LuaTensor> " .. name },
      prelude = {
        "  std::vector<pd_tensor> " .. name .. "_h;",
        "  " .. name .. "_h.reserve(" .. name .. ".size());",
        "  for (const LuaTensor& t : " .. name .. ") "
          .. name .. "_h.push_back(t.h);",
      },
      args = {
        name .. "_h.data()",
        "static_cast<int64_t>(" .. name .. "_h.size())",
      },
    }
  end
  error("emit_sol: unsupported input type '" .. in_.type .. "' for '"
        .. tostring(in_) .. "'")
end

-- ---------------------------------------------------------------------
-- Emission
-- ---------------------------------------------------------------------

local function gen_header(sha)
  return {
    "// GENERATED BY tools/gen/emit_sol.lua",
    "// FROM  paddle/phi/ops/yaml/{ops,backward,...}.yaml @ " .. sha,
    "// DO NOT EDIT",
  }
end

-- Emit the sol2 lambda body for ONE op into `out` (a table of lines).
local function emit_op(ir, out)
  local fn = "pd_op_" .. ir.op
  local opt = optional_set(ir)
  local eff = effective_returns(ir)
  local is_vec_out = single_vec_output(ir)
  local multi = #eff > 1 or is_vec_out

  local decls, preludes, args = {}, {}, {}
  for _, in_ in ipairs(ir.inputs or {}) do
    local p = sol_input(in_, opt)
    for _, d in ipairs(p.decls) do decls[#decls + 1] = d end
    for _, l in ipairs(p.prelude or {}) do preludes[#preludes + 1] = l end
    for _, a in ipairs(p.args) do args[#args + 1] = a end
  end
  for _, attr in ipairs(ir.attrs or {}) do
    local p = sol_attr(attr)
    for _, d in ipairs(p.decls) do decls[#decls + 1] = d end
    for _, l in ipairs(p.prelude or {}) do preludes[#preludes + 1] = l end
    for _, a in ipairs(p.args) do args[#args + 1] = a end
  end

  local callargs = table.concat(args, ", ")
  -- sol::this_state is only needed by variadic_results (make_object);
  -- single-output and single-vector-output lambdas never touch L. sol2
  -- injects the state automatically, so keep the parameter list minimal.
  local need_state = multi and not is_vec_out
  local state = need_state and "sol::this_state L, " or ""
  local lam = "  []( " .. state .. table.concat(decls, ", ") .. ")"

  out[#out + 1] = string.format("  /* %s (yaml block line %d) */", ir.op, ir.line)
  out[#out + 1] = "  mod.set_function(\"" .. fn .. "\","
  out[#out + 1] = lam

  if multi then
    -- C ABI out-array form: int64_t pd_op_x(pd_tensor* outs, int64_t
    -- nout, <params>). Tuple outputs have a compile-time count N
    -- (effective_returns) -> single call, fixed array, variadic_results.
    -- Single-vector outputs have a runtime count -> query-then-allocate
    -- (the query call computes and discards; documented ABI cost).
    out[#out + 1] = "  {"
    for _, l in ipairs(preludes) do out[#out + 1] = l end
    if is_vec_out then
      out[#out + 1] = "    /* runtime-length output: query the count, then fill (03-codegen.md sec.4 item 9) */"
      out[#out + 1] = "    int64_t n = " .. fn
        .. "(nullptr, 0" .. (callargs ~= "" and ", " or "") .. callargs .. ");"
      out[#out + 1] = "    pd::check();"
      out[#out + 1] = "    std::vector<pd_tensor> outs(static_cast<size_t>(n));"
      out[#out + 1] = "    int64_t got = " .. fn
        .. "(outs.data(), n" .. (callargs ~= "" and ", " or "") .. callargs .. ");"
      out[#out + 1] = "    pd::check();"
      out[#out + 1] = "    (void)got;"
      out[#out + 1] = "    std::vector<LuaTensor> r;"
      out[#out + 1] = "    r.reserve(outs.size());"
      out[#out + 1] = "    for (pd_tensor t : outs) r.emplace_back(t);"
      out[#out + 1] = "    return sol::as_table(std::move(r));"
    else
      local n = #eff
      out[#out + 1] = string.format(
        "    std::array<pd_tensor, %d> outs{};", n)
      out[#out + 1] = "    int64_t got = " .. fn
        .. "(outs.data(), " .. n
        .. (callargs ~= "" and ", " or "") .. callargs .. ");"
      out[#out + 1] = "    pd::check();"
      out[#out + 1] = "    (void)got;"
      -- variadic_results is a std::vector<object> (variadic_results.hpp:38);
      -- its variadic ctor (:59) forwards to the VECTOR ctor, which has no
      -- (object, object, ...) overload -- push_back one by one.
      out[#out + 1] = "    sol::variadic_results r;"
      for i = 1, n do
        out[#out + 1] = string.format(
          "    r.push_back(sol::make_object(L, LuaTensor(outs[%d])));", i - 1)
      end
      out[#out + 1] = "    return r;"
    end
    out[#out + 1] = "  });"
  else
    out[#out + 1] = "  {"
    for _, l in ipairs(preludes) do out[#out + 1] = l end
    out[#out + 1] = "    pd_tensor r = " .. fn
      .. "(" .. callargs .. ");"
    out[#out + 1] = "    pd::check();"
    out[#out + 1] = "    return LuaTensor(r);"
    out[#out + 1] = "  });"
  end
  out[#out + 1] = ""
end

-- Deterministic per-letter file + registry emission.
local function emit_all(ops, outdir, sha)
  local buckets = {}
  for _, ir in ipairs(ops) do
    -- guardrail mirrors emit_capi: a tuple containing std::vector<Tensor>
    -- needs the two-phase ABI (not implemented); refuse to emit.
    local eff = effective_returns(ir)
    if not all_tensor_outputs(eff) and not single_vec_output(ir) then
      error("emit_sol: op '" .. ir.op .. "' has a std::vector<Tensor> "
            .. "element in a multi-output tuple -- two-phase ABI not "
            .. "implemented (skip_rules.lua); refuse to emit")
    end
    local letter = string.sub(ir.op, 1, 1)
    buckets[letter] = buckets[letter] or {}
    buckets[letter][#buckets[letter] + 1] = ir
  end

  local letters = {}
  for letter in pairs(buckets) do letters[#letters + 1] = letter end
  table.sort(letters)

  -- per-letter files: each defines pd::register_ops_<letter>(sol::table&)
  for _, letter in ipairs(letters) do
    table.sort(buckets[letter], function(a, b) return a.op < b.op end)
    local cc = gen_header(sha)
    cc[#cc + 1] = ""
    cc[#cc + 1] = '#include "sol_common.hpp"'
    cc[#cc + 1] = ""
    cc[#cc + 1] = "#include <array>"
    cc[#cc + 1] = "#include <vector>"
    cc[#cc + 1] = ""
    -- pd_op_* C ABI declarations (generated by emit_capi.lua; csrc/capi_gen
    -- is on paddle_core's include path, CMakeLists.txt)
    cc[#cc + 1] = '#include "ops_gen.h"'
    cc[#cc + 1] = ""
    cc[#cc + 1] = "namespace pd {"
    cc[#cc + 1] = ""
    cc[#cc + 1] = string.format(
      "/* Register the pd_op_* functions whose names start with '%s' "
      .. "into the paddle_core module table. */", letter)
    cc[#cc + 1] = string.format(
      "void register_ops_%s(sol::table& mod) {", letter)
    for _, ir in ipairs(buckets[letter]) do
      emit_op(ir, cc)
    end
    cc[#cc + 1] = "}"
    cc[#cc + 1] = ""
    cc[#cc + 1] = "}  // namespace pd"
    local f = "gen_ops_" .. letter .. ".cc"
    local fh, err = io.open(outdir .. "/" .. f, "w")
    assert(fh, "emit_sol: cannot write " .. f .. ": " .. tostring(err))
    fh:write(table.concat(cc, "\n"))
    fh:close()
    io.stdout:write("  wrote " .. f .. " (" .. #buckets[letter] .. " ops)\n")
  end

  -- registry: pd::register_ops(lua_State*, int) calls every letter
  local all = gen_header(sha)
  all[#all + 1] = ""
  all[#all + 1] = '#include "sol_common.hpp"'
  all[#all + 1] = ""
  all[#all + 1] = "namespace pd {"
  all[#all + 1] = ""
  for _, letter in ipairs(letters) do
    all[#all + 1] = string.format("void register_ops_%s(sol::table&);", letter)
  end
  all[#all + 1] = ""
  all[#all + 1] = "/* Entry point called from luaopen_paddle_core (module.cc) "
    .. "after register_tensor / register_enums_place. */"
  all[#all + 1] = "void register_ops(lua_State* L, int module_index) {"
  all[#all + 1] = "  sol::state_view lua(L);"
  all[#all + 1] = "  sol::table mod(L, sol::absolute_index(L, module_index));"
  for _, letter in ipairs(letters) do
    all[#all + 1] = string.format("  register_ops_%s(mod);", letter)
  end
  all[#all + 1] = "}"
  all[#all + 1] = ""
  all[#all + 1] = "}  // namespace pd"
  local f = "gen_ops_all.cc"
  local fh, err = io.open(outdir .. "/" .. f, "w")
  assert(fh, "emit_sol: cannot write " .. f .. ": " .. tostring(err))
  fh:write(table.concat(all, "\n"))
  fh:close()
  io.stdout:write("  wrote " .. f .. " (" .. #letters .. " letters)\n")
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
    io.stderr:write("emit_sol: unknown arg '" .. tostring(a) .. "'\n")
    os.exit(2)
  end
  i = i + 1
end

if not outdir or #paths == 0 then
  io.stderr:write(
    "usage: lua tools/gen/emit_sol.lua <out-dir> <yaml> [<yaml> ...] "
    .. "[--ops a,b,c] [--sha <git-sha>]\n")
  os.exit(2)
end

local ok, ops = pcall(collect_ops, paths, whitelist)
if not ok then
  io.stderr:write("emit_sol FAIL: " .. tostring(ops) .. "\n")
  os.exit(1)
end
if #ops == 0 then
  io.stderr:write("emit_sol: no ops to emit\n")
  os.exit(1)
end
io.stdout:write("emit_sol: " .. #ops .. " ops, output -> " .. outdir .. "\n")
local ok2, err2 = pcall(emit_all, ops, outdir, sha)
if not ok2 then
  io.stderr:write("emit_sol FAIL: " .. tostring(err2) .. "\n")
  os.exit(1)
end
os.exit(0)
