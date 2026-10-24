-- tools/gen/emit_lua.lua
-- Generates the Lua wrapper layer (lua/paddle/_ops/<op>.lua) from the yaml
-- IR produced by yaml_subset.lua. Pure Lua 5.1 subset, zero Python
-- (C9 / R42).
--
-- Shape decisions (design sources):
--   - calling forms are exactly two (plan/api/README.md sec.2.1):
--     all-positional (trailing defaults may be omitted) OR one named/mixed
--     table. `f(x, {opt=1})` (positional + trailing options table) is a
--     bug, never generated (R36).
--   - parameter parsing is argrule's job (R27/D31, foundations sec.5.4.4)
--     but generated wrappers do NOT require it at runtime (api/README
--     sec.2.1: "构建期展开,运行时不调 argrule"). Each wrapper therefore
--     carries its OWN expanded argument mapping -- generated from the
--     argrule-style rule schema -- and calls the C ABI op directly.
--   - 1-based conversion (C4/D13): Tensor index semantics are converted
--     in the C ABI layer (emit_capi, pd_idx_minus1/plus1); scalar
--     axis/dim attrs and scalar takes_index attrs (index_select_strided
--     index) get their -1 conversion HERE, in the generated wrapper, per
--     tools/gen/index_semantics.lua + conventions.md sec.5.1. The wrapper
--     therefore never sees a bare Tensor it must subtract from -- it only
--     rewrites numbers, which Lua 5.1 can do.
--   - attrs with defaults ride as explicit arguments; the wrapper fills
--     defaults from the yaml (single source of truth, 03-codegen.md sec.4
--     item 3).
--   - bool attrs pass 1/0 (C ABI has no C99 bool, 01-c-abi.md sec.4).
--
-- Usage:
--   lua tools/gen/emit_lua.lua <out-dir> <yaml> [<yaml> ...] [--sha <sha>]
--     emits one file lua/paddle/_ops/<op>.lua per forward op.

-- Locate sibling modules: generators always run from the repo root
-- (layout.md workflow), so tools/gen/ is on the module path. MUST be
-- set before the requires below (require runs at first use).
package.path = "tools/gen/?.lua;" .. package.path

local yaml = require "yaml_subset"
local index_sem = require "index_semantics"
local skip_rules = require "skip_rules"

-- yaml `{1, 2}` literal -> Lua table literal "{ 1, 2 }" (array attrs:
-- IntArray / std::vector<T>; C ABI side is (ptr, n), Lua side is a table).
local function lua_array_literal(default_str)
  -- `{}` -> empty table; non-empty `{1, 2}` -> "{ 1, 2 }"
  local inner = string.gsub(default_str, "[{}]", " ")
  local items = {}
  for piece in string.gmatch(inner, "[^,]+") do
    local p = string.gsub(piece, "^%s*(.-)%s*$", "%1")  -- trim (gsub multi-return: parens)
    if p ~= "" then
      -- gsub returns (s, count); parens force single value or tonumber gets
      -- a bogus base arg (Lua 5.1 multi-return pitfall)
      local v = tonumber((string.gsub(p, "f$", "")))
      assert(v, "emit_lua: bad array default '" .. default_str .. "'")
      items[#items + 1] = string.format("%.17g", v)
    end
  end
  return "{ " .. table.concat(items, ", ") .. " }"
end

-- phi::DataType default -> paddle_core.PD_DTYPE_* constant (paddle_c.h
-- pd_dtype enum, values 1:1 with phi::DataType per paddle_c.h:58).
local DATA_TYPE_DEFAULTS = {
  ["phi::DataType::UNDEFINED"] = "paddle_core.PD_DTYPE_UNDEFINED",
  ["phi::DataType::BOOL"] = "paddle_core.PD_DTYPE_BOOL",
  ["phi::DataType::UINT8"] = "paddle_core.PD_DTYPE_UINT8",
  ["phi::DataType::INT8"] = "paddle_core.PD_DTYPE_INT8",
  ["phi::DataType::UINT16"] = "paddle_core.PD_DTYPE_UINT16",
  ["phi::DataType::INT16"] = "paddle_core.PD_DTYPE_INT16",
  ["phi::DataType::UINT32"] = "paddle_core.PD_DTYPE_UINT32",
  ["phi::DataType::INT32"] = "paddle_core.PD_DTYPE_INT32",
  ["phi::DataType::UINT64"] = "paddle_core.PD_DTYPE_UINT64",
  ["phi::DataType::INT64"] = "paddle_core.PD_DTYPE_INT64",
  ["phi::DataType::FLOAT32"] = "paddle_core.PD_DTYPE_FLOAT32",
  ["phi::DataType::FLOAT64"] = "paddle_core.PD_DTYPE_FLOAT64",
  ["phi::DataType::COMPLEX64"] = "paddle_core.PD_DTYPE_COMPLEX64",
  ["phi::DataType::COMPLEX128"] = "paddle_core.PD_DTYPE_COMPLEX128",
  ["phi::DataType::PSTRING"] = "paddle_core.PD_DTYPE_PSTRING",
  ["phi::DataType::FLOAT16"] = "paddle_core.PD_DTYPE_FLOAT16",
  ["phi::DataType::BFLOAT16"] = "paddle_core.PD_DTYPE_BFLOAT16",
  ["phi::DataType::FLOAT8_E4M3FN"] = "paddle_core.PD_DTYPE_FLOAT8_E4M3FN",
  ["phi::DataType::FLOAT8_E5M2"] = "paddle_core.PD_DTYPE_FLOAT8_E5M2",
}

-- Lua default literal for a yaml default-value string. Handles every type
-- present in the 4-yaml forward set (surveyed 2026-10-24, 130 distinct
-- combos); anything unknown errors loudly.
local function lua_default_literal(cpp_type, default_str)
  if default_str == nil then
    return nil
  end
  if cpp_type == "bool" then
    if default_str == "true" then return "true" end
    if default_str == "false" then return "false" end
  elseif cpp_type == "float" or cpp_type == "double" then
    -- yaml floats may carry an f suffix (0.9f) or a constant expression
    -- (1.0f/3, 1.6732632423543772848170429916717): strip the f, keep the
    -- expression (Lua evaluates division fine). Lua reads the exponent fine.
    local v = string.gsub(default_str, "f$", "")
    if tonumber(v) then return v end
    if string.find(v, "/") then return v end   -- 1.0/3, 1.0/8
  elseif cpp_type == "int" or cpp_type == "int64_t" then
    if tonumber(default_str) then return default_str end
    if string.find(default_str, "<<") then      -- (2 << 12) - 1 -> 2^12 - 1
      return string.gsub(default_str, "<<", "^")
    end
    local v = string.gsub(default_str, "L$", "")  -- 0L -> 0
    if tonumber(v) then return v end
  elseif cpp_type == "paddle::experimental::Scalar" then
    local v = string.gsub(default_str, "f$", "")
    if tonumber(v) then return v end
  elseif cpp_type == "std::string" then
    return string.format("%q", default_str)
  elseif cpp_type == "paddle::experimental::IntArray"
  or cpp_type == "std::vector<int64_t>"
  or cpp_type == "std::vector<int>"
  or cpp_type == "std::vector<float>"
  or cpp_type == "std::vector<double>" then
    return lua_array_literal(default_str)
  elseif cpp_type == "phi::DataType" then
    local c = DATA_TYPE_DEFAULTS[default_str]
    if c then return c end
  elseif cpp_type == "paddle::Place" then
    -- default place is CPU (paddle_c.h pd_place_new(PD_PLACE_CPU, 0));
    -- `{}` and `CPUPlace()` both mean that
    return "paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)"
  end
  error("emit_lua: cannot translate default '" .. tostring(default_str)
        .. "' for type '" .. cpp_type .. "' (add a case, don't guess)")
end

-- True if the C++ attr type is a plain scalar (axis/dim style 1-based).
-- Array types (IntArray / std::vector<T>) get per-element conversion.
local function is_scalar_attr(cpp_type)
  return cpp_type == "int" or cpp_type == "int64_t"
    or cpp_type == "float" or cpp_type == "double"
    or cpp_type == "paddle::experimental::Scalar"
end

-- yaml axis/dim defaults are C++-side (0-based, first dim = 0). The Lua
-- wrapper is 1-based, so a non-negative default must be shifted +1
-- (axis=0 -> Lua default 1, axis=1 -> 2); negative defaults (-1 = last
-- dim) stay as-is (conventions.md sec.5.1). Only the named axis/dim
-- attrs are shifted -- never a general attr.
local function shift_axis_default(cpp_type, default_str)
  if not default_str then return nil end
  if not is_scalar_attr(cpp_type) then return nil end   -- arrays: {} unchanged
  local v = tonumber(default_str)
  if not v then
    error("emit_lua: axis/dim default '" .. tostring(default_str)
          .. "' is not a plain number -- cannot shift to 1-based "
          .. "(add a case, don't guess)")
  end
  if v >= 0 then return tostring(v + 1) end
  return tostring(v)
end

-- Emit one wrapper module file body.
local function emit_wrapper(ir)
  local op = ir.op
  local params, callargs = {}, {}
  -- index params of this op (1-based names)
  local take_idx = {}
  for _, p in ipairs(index_sem.index_params(op)) do
    take_idx[p] = true
  end

  -- parameter handling: named/mixed table OR positional.
  -- Build one line per param, positional-or-keyword resolution.
  local setters = {}
  local order = {}   -- param names in yaml order
  for _, in_ in ipairs(ir.inputs or {}) do
    order[#order + 1] = in_.name
  end
  for _, a in ipairs(ir.attrs or {}) do
    order[#order + 1] = a.name
  end

  -- resolve: if a single table arg -> named/mixed; else positional.
  -- Named wins over array slot, BUT only when non-nil: `t.x or t[i]` would
  -- swallow an explicit `false` (bool attrs) -- must test nil explicitly.
  -- Positional branch index = 1-based position in `order`, independent of
  -- the table branch (a shared counter here made select() start at N+1).
  local lines = {}
  lines[#lines + 1] = "  local n = select(\"#\", ...)"
  lines[#lines + 1] = "  if n == 1 and type(select(1, ...)) == \"table\" then"
  lines[#lines + 1] = "    local t = select(1, ...)"
  for i, name in ipairs(order) do
    lines[#lines + 1] = "    local v = t." .. name
    lines[#lines + 1] = "    if v == nil then v = t[" .. i .. "] end"
    lines[#lines + 1] = "    " .. name .. " = v"
  end
  lines[#lines + 1] = "  else"
  for i, name in ipairs(order) do
    lines[#lines + 1] = string.format(
      "    %s = select(%d, ...)", name, i)
  end
  lines[#lines + 1] = "  end"

  -- apply yaml defaults for attrs (nil -> default). axis/dim defaults are
  -- shifted to 1-based Lua literals (shift_axis_default); every other attr
  -- keeps the yaml value verbatim (it IS the C++-side value).
  for _, a in ipairs(ir.attrs or {}) do
    local lit = lua_default_literal(a.type, a.default)
    if lit and (a.name == "axis" or a.name == "dim") then
      lit = shift_axis_default(a.type, a.default) or lit
    end
    if lit then
      lines[#lines + 1] = string.format(
        "  if %s == nil then %s = %s end", a.name, a.name, lit)
    end
  end

  -- 1-based conversion (conventions.md sec.5.1), emitted in the wrapper:
  --  * axis/dim attrs: positive value n -> n-1, negative stays (e.g. -1 =
  --    last dim). Scalar and array (IntArray / std::vector<T>) forms.
  --  * takes_index params that are scalar ATTRS (index_select_strided
  --    index:int64_t): value - 1. Tensor takes_index params are converted
  --    in the C ABI layer (emit_capi), never here.
  -- Runs AFTER defaults so the shifted default literals convert back to
  -- the yaml (0-based) value.
  for _, a in ipairs(ir.attrs or {}) do
    local name = a.name
    if take_idx[name] then
      lines[#lines + 1] = string.format(
        "  if %s ~= nil then %s = %s - 1 end", name, name, name)
    elseif name == "axis" or name == "dim" then
      if is_scalar_attr(a.type) then
        lines[#lines + 1] = string.format(
          "  if %s ~= nil and %s > 0 then %s = %s - 1 end",
          name, name, name, name)
      else
        lines[#lines + 1] = string.format("  if %s ~= nil then", name)
        lines[#lines + 1] = "    if type(" .. name .. ") == \"table\" then"
        lines[#lines + 1] = "      for i, v in ipairs(" .. name .. ") do"
        lines[#lines + 1] = "        if v > 0 then " .. name .. "[i] = v - 1 end"
        lines[#lines + 1] = "      end"
        lines[#lines + 1] = "    elseif " .. name .. " > 0 then"
        lines[#lines + 1] = "      " .. name .. " = " .. name .. " - 1"
        lines[#lines + 1] = "    end"
        lines[#lines + 1] = "  end"
      end
    end
  end

  -- build the call: inputs -> pd_tensor locals (already resolved), attrs
  -- with bool -> 1/0
  for _, in_ in ipairs(ir.inputs or {}) do
    callargs[#callargs + 1] = in_.name
  end
  for _, a in ipairs(ir.attrs or {}) do
    local v = a.name
    if a.type == "bool" then
      v = "(" .. a.name .. " and 1 or 0)"
    end
    callargs[#callargs + 1] = v
  end
  local call = "  return paddle_core.pd_op_" .. op .. "("
    .. table.concat(callargs, ", ") .. ")"

  -- multi-output ops (wall #3): the C ABI function returns int64_t count
  -- and fills a caller-allocated pd_tensor outs array (emit_capi
  -- emit_op_multi). The sol2 binding layer (2.4) wraps that into Lua
  -- multiple-return-values, so the wrapper's call line below needs no
  -- extra codegen -- it forwards all values: y1, y2 = paddle_core.pd_op_x(...).
  -- Note the count is the *effective* return count: `intermediate:` outputs
  -- (yaml scratch tensors, e.g. lstm batch_gate/batch_cell_pre_act) never
  -- appear in the ad_func tuple -- but the wrapper does not care, it just
  -- forwards the binding's values.

  local body = {}
  body[#body + 1] = "-- GENERATED BY tools/gen/emit_lua.lua"
  body[#body + 1] = "-- FROM  paddle/phi/ops/yaml/{ops,backward,...}.yaml @ sha"
  body[#body + 1] = "-- DO NOT EDIT"
  body[#body + 1] = ""
  body[#body + 1] = "local paddle_core = require \"paddle_core\""
  body[#body + 1] = ""
  body[#body + 1] = "local M = {}"
  body[#body + 1] = ""
  body[#body + 1] = "-- " .. op .. ": positional / named / mixed table call"
  body[#body + 1] = "-- positional: paddle." .. op .. "(" .. table.concat(order, ", ") .. ")"
  body[#body + 1] = "-- named/mixed: paddle." .. op .. "{ " .. table.concat(
    order, " = ..., ") .. " = ... }"
  body[#body + 1] = "function M." .. op .. "(...)"
  for _, l in ipairs(lines) do
    body[#body + 1] = l
  end
  body[#body + 1] = call
  body[#body + 1] = "end"
  body[#body + 1] = ""
  body[#body + 1] = "return M"
  return table.concat(body, "\n")
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
    io.stderr:write("emit_lua: unknown arg '" .. tostring(a) .. "'\n")
    os.exit(2)
  end
  i = i + 1
end

if not outdir or #paths == 0 then
  io.stderr:write(
    "usage: lua tools/gen/emit_lua.lua <out-dir> <yaml> [<yaml> ...] "
    .. "[--ops a,b,c] [--sha <git-sha>]\n")
  os.exit(2)
end

local ok, ops = pcall(function()
  local all = {}
  local want = {}
  for _, w in ipairs(whitelist) do
    want[w] = true
  end
  for _, p in ipairs(paths) do
    local blocks = yaml.parse_file(p)
    for _, b in ipairs(blocks) do
      if b.kind == "forward" then
        local why = skip_rules.skip_reason(b.name)
        if why then
          io.stderr:write("  skip " .. b.name .. " (" .. why .. ")\n")
        elseif #whitelist == 0 or want[b.name] then
          index_sem.require_annotated(b.name, "emit_lua (" .. p .. ")")
          local ok2, ir = pcall(yaml.block_ir, b)
          if not ok2 then
            error("emit_lua: block_ir failed for '" .. b.name .. "': "
                  .. tostring(ir))
          end
          all[#all + 1] = ir
        end
      end
    end
  end
  table.sort(all, function(a, b)
    if a.op ~= b.op then return a.op < b.op end
    return a.line < b.line
  end)
  return all
end)
if not ok then
  io.stderr:write("emit_lua FAIL: " .. tostring(ops) .. "\n")
  os.exit(1)
end

io.stdout:write("emit_lua: " .. #ops .. " ops, output -> " .. outdir .. "\n")
for _, ir in ipairs(ops) do
  local ok3, body = pcall(emit_wrapper, ir)
  if not ok3 then
    io.stderr:write("emit_lua FAIL op " .. ir.op .. ": " .. tostring(body) .. "\n")
    os.exit(1)
  end
  local f = outdir .. "/" .. ir.op .. ".lua"
  local fh, err = io.open(f, "w")
  assert(fh, "emit_lua: cannot write " .. f .. ": " .. tostring(err))
  fh:write(body .. "\n")
  fh:close()
  io.stdout:write("  wrote " .. f .. "\n")
end
os.exit(0)