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
--   - 1-based conversion (C4/D13) lives ONLY in the generated wrapper:
--     index-semantics params get `- 1`, index-semantics outputs get `+ 1`,
--     per tools/gen/index_semantics.lua. Operators with index semantics
--     are NOT supported by this emit yet -- they error loudly (never
--     silently emit a wrapper without the conversion, conventions sec.5.2).
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

-- Lua default literal for a yaml default-value string. Handles the types
-- present in the trial set; anything unknown errors loudly.
local function lua_default_literal(cpp_type, default_str)
  if default_str == nil then
    return nil
  end
  if cpp_type == "bool" then
    if default_str == "true" then return "true" end
    if default_str == "false" then return "false" end
  elseif cpp_type == "float" or cpp_type == "double" then
    -- yaml floats may carry an f suffix (0.9f) or exponent (1.0e-6f);
    -- strip the f, Lua reads the exponent fine.
    local v = string.gsub(default_str, "f$", "")
    if tonumber(v) then return v end
  elseif cpp_type == "int" or cpp_type == "int64_t" then
    if tonumber(default_str) then return default_str end
  elseif cpp_type == "std::string" then
    return string.format("%q", default_str)
  end
  error("emit_lua: cannot translate default '" .. tostring(default_str)
        .. "' for type '" .. cpp_type .. "' (add a case, don't guess)")
end

-- True if the op has index semantics (either direction).
local function has_index_semantics(op)
  return #index_sem.index_params(op) > 0 or #index_sem.index_outputs(op) > 0
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
  local ret_idx = {}
  for _, pos in ipairs(index_sem.index_outputs(op)) do
    ret_idx[pos] = true
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

  -- apply yaml defaults for attrs (nil -> default)
  for _, a in ipairs(ir.attrs or {}) do
    local lit = lua_default_literal(a.type, a.default)
    if lit then
      lines[#lines + 1] = string.format(
        "  if %s == nil then %s = %s end", a.name, a.name, lit)
    end
  end

  -- 1-based conversion for index params (skip -- none in this emit)
  for _, name in ipairs(order) do
    if take_idx[name] then
      error("emit_lua: op '" .. op .. "' has takes_index param '" .. name
            .. "' -- index conversion not implemented in this emit yet; "
            .. "refuse to emit silently wrong wrapper")
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

  -- output 1-based conversion: none in this emit; error if present
  local rpos = ir.returns or {}
  for pos = 1, #rpos do
    if ret_idx[pos] then
      error("emit_lua: op '" .. op .. "' returns index at position " .. pos
            .. " -- output conversion not implemented in this emit yet; "
            .. "refuse to emit silently wrong wrapper")
    end
  end

  -- single-output assumption for now; multi-output ops error
  if #rpos > 1 then
    error("emit_lua: op '" .. op .. "' has " .. #rpos
          .. " outputs -- multi-output wrappers not implemented yet; "
          .. "refuse to emit")
  end

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