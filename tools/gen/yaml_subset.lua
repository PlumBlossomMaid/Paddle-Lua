-- tools/gen/yaml_subset.lua
-- YAML subset parser for Paddle op definition files (ops.yaml / backward.yaml
-- and friends). Pure Lua 5.1 subset, zero Python (C9 / R42).
--
-- Behaviour reference: the Python generator we no longer run, kept as a
-- semantic dictionary (03-codegen.md sec.2):
--   paddle/fluid/eager/auto_code_generator/generator/codegen_utils.py
--     ReadFwdFile:116              -- load file as list of op dicts
--     ParseYamlArgs:321            -- args string -> inputs/attrs lists
--     ParseYamlReturns:365         -- returns string -> returns list
--     ParseYamlForwardFromBackward:425 -- "fwd (args) -> returns" string
--     ParseYamlInplaceInfo:477     -- "(x -> out), (y -> out2)" -> map
--     ParseYamlCompositeInfo:490   -- "f(args...)" -> {name, args}
--     RemoveSpecialSymbolsInName:292
--     yaml_types_mapping:78
--
-- Grammar subset (verified against the real files, 03-codegen.md sec.2/6):
--   - top-level list items: `- op : name` / `- backward_op : name`
--   - block fields: 2-space indent `key : value` (single-line scalars)
--   - nested blocks (`infer_meta :` etc.): 4+ space indent, skipped here
--   - multi-line scalars only occur inside nested kernel func values
--   - NO anchors/aliases, NO |/> block scalars, NO merge keys, NO inline
--     comments (all verified; unknown syntax -> error, never silently pass)
--
-- Usage:
--   lua tools/gen/yaml_subset.lua <file.yaml> [--dump]
--     prints one line per op block: "<kind> <name> (<#inputs> in, <#attrs> at, <#returns> out)"
--     with --dump prints the full IR as JSON (for diff against the Python
--     reference generator).

local M = {}

-- yaml_types_mapping from codegen_utils.py:78
local YAML_TYPES = {
  ["int"] = "int",
  ["int32_t"] = "int32_t",
  ["int64_t"] = "int64_t",
  ["size_t"] = "size_t",
  ["float"] = "float",
  ["double"] = "double",
  ["bool"] = "bool",
  ["str"] = "std::string",
  ["str[]"] = "std::vector<std::string>",
  ["float[]"] = "std::vector<float>",
  ["double[]"] = "std::vector<double>",
  ["bool[]"] = "std::vector<bool>",
  ["Place"] = "paddle::Place",
  ["DataLayout"] = "phi::DataLayout",
  ["DataType"] = "phi::DataType",
  ["int64_t[]"] = "std::vector<int64_t>",
  ["int[]"] = "std::vector<int>",
  ["Tensor"] = "Tensor",
  ["Tensor[]"] = "std::vector<Tensor>",
  ["Tensor[Tensor[]]"] = "std::vector<std::vector<Tensor>>",
  ["Scalar"] = "paddle::experimental::Scalar",
  ["Scalar(int)"] = "paddle::experimental::Scalar",
  ["Scalar(int64_t)"] = "paddle::experimental::Scalar",
  ["Scalar(float)"] = "paddle::experimental::Scalar",
  ["Scalar(double)"] = "paddle::experimental::Scalar",
  ["Scalar[]"] = "std::vector<phi::Scalar>",
  ["IntArray"] = "paddle::experimental::IntArray",
}

local function trim(s)
  return (string.gsub(s, "^%s*(.-)%s*$", "%1"))
end

-- RemoveSpecialSymbolsInName (codegen_utils.py:292): strip everything after '@'
local function remove_special_symbols(name)
  local at = string.find(name, "@", 1, true)
  if at then
    return string.sub(name, 1, at - 1)
  end
  return name
end

-- Split on ',' but NOT inside '{...}' (codegen_utils.py:321 supports
-- `int[] a={1,3}`): a comma is a separator iff the remainder does not start
-- with [^{]*\}.
local function split_args(s)
  local parts = {}
  local start = 1
  local i = 1
  local n = #s
  while i <= n do
    if string.sub(s, i, i) == "," then
      local rest = string.sub(s, i + 1)
      if not string.match(rest, "^[^{]*}") then
        parts[#parts + 1] = string.sub(s, start, i - 1)
        start = i + 1
      end
    end
    i = i + 1
  end
  parts[#parts + 1] = string.sub(s, start)
  return parts
end

-- Split on ',' everywhere (returns have no {..} forms)
local function split_plain(s)
  local parts = {}
  for piece in string.gmatch(s, "[^,]+") do
    parts[#parts + 1] = piece
  end
  return parts
end

-- ParseYamlArgs (codegen_utils.py:321)
-- "(Tensor x, Scalar(int64_t) axis, bool keepdims = false, ...)" ->
--   inputs = {{name, type, pos}, ...}   pos = original position, 0-based
--   attrs  = {{name, type, default, pos}, ...}
local function parse_args(s)
  local inputs, attrs = {}, {}
  local parts = split_args(s)
  for i, part in ipairs(parts) do
    local arg = trim(part)
    -- Python regex is ((const )?\S+) (.*)  (codegen_utils.py:321): the type is
    -- the first whitespace-delimited token, optionally prefixed with "const ".
    -- Lua 5.1 patterns have no (group)? optional captures, so emulate:
    --   "Tensor x"                -> type "Tensor",  rest "x"
    --   "Scalar(int64_t) axis = 0"-> type "Scalar(int64_t)", rest "axis = 0"
    --   "const Tensor& x"         -> type "const Tensor&" (asserted later;
    --                                 mapping has no "const" types, same as
    --                                 upstream which also fails the assert)
    local arg_type, rest
    local arg2 = arg
    if string.sub(arg2, 1, 6) == "const " then
      local token = string.match(string.sub(arg2, 7), "^%S+")
      arg_type = "const " .. (token or "")
      rest = trim(string.sub(arg2, 7 + #(token or "")))
    else
      arg_type = string.match(arg2, "^%S+")
      rest = trim(string.sub(arg2, #arg_type + 1))
    end
    if not arg_type then
      error("ParseYamlArgs: cannot split arg '" .. arg .. "'")
    end
    arg_type = trim(arg_type)
    local name, default = rest, nil
    local eq = string.find(rest, "=", 1, true)
    if eq then
      name = trim(string.sub(rest, 1, eq - 1))
      default = trim(string.sub(rest, eq + 1))
    end
    assert(YAML_TYPES[arg_type],
      "yaml_subset: unsupported arg type '" .. arg_type .. "' (yaml_types_mapping:78)")
    -- DataLayout/DataType default values get namespaced (codegen_utils.py:325)
    if arg_type == "DataLayout" and default ~= nil then
      default = "paddle::experimental::" .. default
    elseif arg_type == "DataType" and default ~= nil then
      default = "phi::" .. default
    end
    local cpp_type = YAML_TYPES[arg_type]
    name = remove_special_symbols(name)
    if string.find(cpp_type, "Tensor") then
      assert(default == nil,
        "yaml_subset: Tensor arg '" .. name .. "' must not have a default (codegen_utils.py:335)")
      inputs[#inputs + 1] = { name = name, type = cpp_type, pos = i - 1 }
    else
      -- Keep the Scalar subtype ("int" / "int64_t" / "float" / "double")
      -- when the yaml writes Scalar(...) (ops.yaml split:5313 declares
      -- `Scalar(int) axis`; infermeta branches on the Scalar's internal
      -- dtype, unary.cc:4929, so "Scalar(int)" must reach emit_capi as
      -- int64, not double). Plain "Scalar" has sub = nil.
      local sub = nil
      if string.sub(arg_type, 1, 7) == "Scalar("
         and string.sub(arg_type, -1) == ")" then
        sub = string.sub(arg_type, 8, -2)
      end
      attrs[#attrs + 1] = { name = name, type = cpp_type,
                            default = default, pos = i - 1, sub = sub }
    end
  end
  return inputs, attrs
end

-- ParseYamlReturns (codegen_utils.py:365)
-- "Tensor(out), Tensor" -> returns = {{name, type, pos}, ...}
local function parse_returns(s)
  local list = {}
  local parts = split_plain(s)
  for i, part in ipairs(parts) do
    local ret = trim(part)
    -- split("{")[0] -- defensive: strip any brace suffix
    local brace = string.find(ret, "{", 1, true)
    if brace then
      ret = trim(string.sub(ret, 1, brace - 1))
    end
    local ret_type, ret_name = ret, ""
    local lp = string.find(ret, "(", 1, true)
    if lp then
      -- "Tensor(out)" -> remove trailing ')' then split on '('
      local inner = string.sub(ret, 1, #ret - 1)
      local left = string.find(inner, "(", 1, true)
      ret_type = trim(string.sub(inner, 1, left - 1))
      ret_name = trim(string.sub(inner, left + 1))
    end
    assert(YAML_TYPES[ret_type],
      "yaml_subset: unsupported return type '" .. ret_type .. "' (yaml_types_mapping:78)")
    local cpp_type = YAML_TYPES[ret_type]
    assert(string.find(cpp_type, "Tensor"),
      "yaml_subset: non-Tensor return '" .. ret_type .. "' (codegen_utils.py:378)")
    ret_name = remove_special_symbols(ret_name)
    list[#list + 1] = { name = ret_name, type = cpp_type, pos = i - 1 }
  end
  return list
end

-- ParseYamlInplaceInfo (codegen_utils.py:477)
-- "(x -> out0), (y -> out2)" -> {x = "out0", y = "out2"}
local function parse_inplace(s)
  local map = {}
  local parts = split_plain(s)
  for _, part in ipairs(parts) do
    local p = trim(part)
    p = string.gsub(p, "^%(", "")
    p = string.gsub(p, "%)$", "")
    local k, v = string.match(p, "^(.-)%s*->%s*(.-)$")
    if k then
      map[k] = v
    end
  end
  return map
end

-- ParseYamlCompositeInfo (codegen_utils.py:490)
-- "abs_grad(x, out_grad, x_grad)" -> {name = ..., args = {...}}
local function parse_composite(s)
  local name, args_str = string.match(s, "^(.-)%s*%((.*)%)%s*$")
  assert(name, "yaml_subset: composite field not of form f(args): '" .. s .. "'")
  local args = {}
  for piece in string.gmatch(args_str, "[^,]+") do
    args[#args + 1] = trim(piece)
  end
  return { name = name, args = args }
end

-- ParseYamlForwardFromBackward (codegen_utils.py:425)
-- "abs (Tensor x, Tensor grad_out) -> Tensor(grad_x)" ->
--   {name, inputs, attrs, returns}
local function parse_forward(s)
  local name, args_str, rets_str =
    string.match(s, "^(.-)%s*%((.-)%)%s*->%s*(.*)$")
  assert(name, "yaml_subset: forward field not of form f (args) -> returns: '" .. s .. "'")
  local inputs, attrs = parse_args(args_str)
  local returns = parse_returns(rets_str)
  return { name = trim(name), inputs = inputs, attrs = attrs, returns = returns }
end

-- ---------------------------------------------------------------------------
-- File-level structural parser.
--
-- Block model: a block starts at a `- op :` / `- backward_op :` line. Its
-- fields are 2-space-indented `key : value` lines. A field with an EMPTY
-- value opens a nested block; every deeper-indented line is skipped until a
-- 2-space field (or a new top-level item) shows up again. This is enough for
-- every block in ops.yaml / backward.yaml / legacy/static_ops.yaml /
-- op_compat.yaml: the fields we need (args/output/inplace/composite/forward/
-- backward/optional) are single-line scalars, and all nested blocks are
-- irrelevant to codegen IR.
-- ---------------------------------------------------------------------------
local TOP_ITEM = "^%-(%s*)([%w_]+)%s*:%s*(.*)$"
local FIELD = "^  ([%w_]+)%s*:%s*(.*)$"

-- count leading spaces (tabs not used in the 4 yaml files, verified)
local function leading_spaces(line)
  local n = 0
  while string.sub(line, n + 1, n + 1) == " " do
    n = n + 1
  end
  return n
end

local function parse_file(path)
  local fh, err = io.open(path, "r")
  if not fh then
    error("yaml_subset: cannot open " .. path .. ": " .. tostring(err))
  end
  local blocks = {}
  local cur = nil
  local nesting = false -- inside a nested block (field with empty value),
                        -- skip 4+-space lines until a 2-space field appears
  local last_field = nil -- most recent single-line field value (may fold)
  local lineno = 0
  for line in fh:lines() do
    lineno = lineno + 1
    -- strip trailing CR (CRLF sources)
    if string.sub(line, -1) == "\r" then
      line = string.sub(line, 1, -2)
    end
    local content = trim(line)
    if content ~= "" and string.sub(content, 1, 1) ~= "#" then
      local kind, name = string.match(line, TOP_ITEM)
      if kind then
        nesting = false
        last_field = nil
        cur = {
          kind = name == "op" and "forward" or name, -- op / backward_op
          op = name,
          name = trim(string.match(line, ":%s*(.*)$")),
          line = lineno,
          fields = {},
          raw = {},
        }
        blocks[#blocks + 1] = cur
      else
        local indent = leading_spaces(line)
        if indent >= 2 then
          local key, value = string.match(line, FIELD)
          if key and cur then
            nesting = false
            last_field = nil
            local v = trim(value)
            cur.fields[key] = v
            cur.raw[#cur.raw + 1] = key
            if v == "" then
              nesting = true
            else
              last_field = key
            end
          elseif nesting then
            -- nested-block content (4+ spaces); skip
          elseif last_field and cur then
            -- multi-line plain scalar: a deeper-indented line that is NOT a
            -- `key :` mapping item folds into the current field value with a
            -- single space (YAML plain-scalar folding, e.g. ops.yaml:75).
            local fold = trim(line)
            if not string.match(fold, "^[%w_]+%s*:") then
              cur.fields[last_field] = cur.fields[last_field] .. " " .. fold
            else
              error("yaml_subset: unexpected line " .. lineno .. ": '" .. line .. "'")
            end
          else
            error("yaml_subset: unexpected line " .. lineno .. ": '" .. line .. "'")
          end
        else
          error("yaml_subset: unexpected line " .. lineno .. ": '" .. line .. "'")
        end
      end
    end
  end
  fh:close()
  return blocks
end

-- Build the codegen IR for one block. Unknown/unused fields are ignored here
-- (they are consumed by later emit stages from `fields` when needed).
local function block_ir(b)
  local ir = {
    kind = b.kind,
    op = b.name,
    line = b.line,
  }
  local f = b.fields
  if f.args then
    -- upstream ParseYamlForward (codegen_utils.py:447-453) strips the outer
    -- parentheses from "(Tensor x, ...)" before ParseYamlArgs
    local args = f.args
    if string.sub(args, 1, 1) == "(" and string.sub(args, -1) == ")" then
      args = trim(string.sub(args, 2, -2))
    end
    local inputs, attrs = parse_args(args)
    ir.inputs, ir.attrs = inputs, attrs
  end
  if f.output then
    ir.returns = parse_returns(f.output)
  end
  if f.inplace then
    ir.inplace = parse_inplace(f.inplace)
  end
  if f.composite then
    ir.composite = parse_composite(f.composite)
  end
  if f.forward then
    ir.forward = parse_forward(f.forward)
  end
  if f.backward then
    ir.backward = f.backward
  end
  if f.optional then
    ir.optional = f.optional
  end
  -- `intermediate:` lists output names that are scratch/state tensors the
  -- dygraph API does NOT return (they exist to avoid reallocation during
  -- training). The *_ad_func's tuple contains only the non-intermediate
  -- returns (verified: lstm yaml has 4 outputs, batch_gate/batch_cell_pre_act
  -- are intermediate, lstm_ad_func returns std::tuple<Tensor, Tensor> --
  -- dygraph_functions.h). Kept as the raw comma-separated string; consumers
  -- split it.
  if f.intermediate then
    ir.intermediate = f.intermediate
  end
  return ir
end

M.parse_file = parse_file
M.block_ir = block_ir
M.parse_args = parse_args
M.parse_returns = parse_returns
M.parse_inplace = parse_inplace
M.parse_composite = parse_composite
M.parse_forward = parse_forward
M.YAML_TYPES = YAML_TYPES

-- ---------------------------------------------------------------------------
-- CLI
-- ---------------------------------------------------------------------------
local function json_escape(s)
  s = string.gsub(s, "\\", "\\\\")
  s = string.gsub(s, '"', '\\"')
  s = string.gsub(s, "\n", "\\n")
  s = string.gsub(s, "\r", "\\r")
  s = string.gsub(s, "\t", "\\t")
  return s
end

local function is_array(t)
  local n = 0
  for _ in pairs(t) do
    n = n + 1
  end
  if n == 0 then
    return true -- empty table serializes as [] here; IR has no empty objects
  end
  for i = 1, n do
    if t[i] == nil then
      return false
    end
  end
  return true
end

local to_json
to_json = function(v, depth)
  local t = type(v)
  if t == "string" then
    return '"' .. json_escape(v) .. '"'
  elseif t == "number" then
    if v % 1 == 0 then
      return string.format("%d", v)
    end
    return string.format("%.17g", v)
  elseif t == "boolean" then
    return tostring(v)
  elseif t == "nil" then
    return "null"
  elseif t == "table" then
    local pad = string.rep("  ", depth or 0)
    local items = {}
    if is_array(v) then
      for i, item in ipairs(v) do
        items[#items + 1] = to_json(item, (depth or 0) + 1)
      end
      return "[" .. table.concat(items, ", ") .. "]"
    else
      for k, val in pairs(v) do
        items[#items + 1] = '"' .. json_escape(tostring(k)) .. '": ' .. to_json(val, (depth or 0) + 1)
      end
      return "{" .. table.concat(items, ", ") .. "}"
    end
  end
  return "null"
end

-- CLI entry: run only as the main script (lua tools/gen/yaml_subset.lua ...).
-- When require()d as a module, Lua 5.1 hands the module name to the chunk as
-- a vararg; without this guard the CLI would try to open "yaml_subset" as a
-- file and abort module loading (observed when the parity harness required
-- it). arg[0] is the main-script path and does not match when required.
if arg and arg[0] and string.find(arg[0], "yaml_subset%.lua$") then
  local dump = false
  local files = {}
for i = 1, select("#", ...) do
  local a = select(i, ...)
  if a == "--dump" then
    dump = true
  else
    files[#files + 1] = a
  end
end

if #files == 0 then
  io.stderr:write("usage: lua tools/gen/yaml_subset.lua <file.yaml> [--dump]\n")
  os.exit(2)
end

for _, path in ipairs(files) do
  local ok, blocks = pcall(parse_file, path)
  if not ok then
    io.stderr:write("yaml_subset FAIL " .. path .. "\n  " .. tostring(blocks) .. "\n")
    os.exit(1)
  end
  if dump then
    local irs = {}
    for _, b in ipairs(blocks) do
      local ok2, ir = pcall(block_ir, b)
      if not ok2 then
        io.stderr:write("yaml_subset FAIL block " .. b.name .. " (line " .. b.line .. ")\n  " .. tostring(ir) .. "\n")
        os.exit(1)
      end
      irs[#irs + 1] = ir
    end
    io.stdout:write(to_json(irs, 0) .. "\n")
  else
    io.stdout:write("yaml_subset " .. path .. ": " .. #blocks .. " blocks\n")
    for _, b in ipairs(blocks) do
      local ok2, ir = pcall(block_ir, b)
      if not ok2 then
        io.stderr:write("yaml_subset FAIL block " .. b.name .. " (line " .. b.line .. ")\n  " .. tostring(ir) .. "\n")
        os.exit(1)
      end
      io.stdout:write(string.format(
        "  %-10s %-30s %3d in, %3d attrs, %3d out\n",
        b.kind, b.name,
        ir.inputs and #ir.inputs or 0,
        ir.attrs and #ir.attrs or 0,
        ir.returns and #ir.returns or 0))
    end
  end
  os.exit(0)
  end
end

return M
