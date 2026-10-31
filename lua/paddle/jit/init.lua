-- lua/paddle/jit/init.lua -- paddle.jit.load (P8 prewire, M1 2.8)
--                             + paddle.jit.trace (P15, M3 4.1)
-- Stage: P8 / M1 (WORKPLAN 2.8) + P15 / M3 (WORKPLAN 4.1).
-- Thin Lua face over the LuaLayer / LuaTrace usertypes; the C ABI slices
-- are csrc/capi/jit.cc + csrc/capi/trace.cc, the sol slices
-- csrc/sol/jit_binding.cc + csrc/sol/trace_binding.cc.
--
--   paddle.jit.load{ path, skip_load_weights=false } -> TranslatedLayer
--   layer(x) / layer:forward(x)   single Tensor or multi-return
--   layer:to(place) / layer:function_names() / layer:clone()
--
--   paddle.jit.trace(fn, example_inputs) -> traced callable (P15-A,
--   inference-first per the 2026-10-31 scope decision): runs fn once on
--   the example inputs, the generated op wrappers record every op into a
--   pir::Program (eager-record), and the returned callable replays that
--   program through run_program_ad_func with fresh inputs. Trace limits
--   (15-to-static.md sec.4 pit 2/3): control flow and shapes freeze to
--   this run; Tensor:item() during trace ERRORS (never silently freezes
--   a branch). Ops outside the trace gate (optional/index-semantics
--   inputs, in-place ops) record nothing -- the traced program then
--   computes a wrong graph, so trace only ops inside the gate for now.
--
-- Lua 5.1 subset (C3).

local paddle_core = require "paddle_core"
local argrule = require "argrule"
local rule = require "argrule.rule"
local unpack = table.unpack or unpack

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

-- ---------------------------------------------------------------------
-- paddle.jit.trace (P15-A)
-- ---------------------------------------------------------------------

-- Shared LuaTensor metatable probe (same pattern as io.lua).
local probe = paddle_core.tensor{1}
local TENSOR_MT = getmetatable(probe)
probe = nil
local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end

-- fn: a plain function or a callable (nn.Layer instance -- pl.class sets
-- __call). type slot = callable contract (D32).
local Callable = argrule.alias { function(v)
  if type(v) == "function" then return true end
  local mt = getmetatable(v)
  return mt ~= nil and mt.__call ~= nil
end }

-- example_inputs: one Tensor or an array of Tensors.
local TensorOrList = argrule.alias { function(v)
  if is_tensor(v) then return true end
  if type(v) ~= "table" then return false end
  for _, e in ipairs(v) do
    if not is_tensor(e) then return false end
  end
  return #v > 0
end }

M.trace = rule {
  { name = "fn", type = Callable,
    doc = "function or nn.Layer to trace (its forward runs once)" },
  { name = "example_inputs", type = TensorOrList,
    doc = "Tensor or array of Tensors fed to fn for the recording run" },
}(function(fn, example_inputs)
  local inps = is_tensor(example_inputs) and { example_inputs }
               or example_inputs
  local t = paddle_core.trace_begin()
  for i, inp in ipairs(inps) do
    t:input(inp, "x" .. i)
  end
  -- The recording run: generated wrappers execute eagerly AND record.
  -- pcall: on failure the trace must be closed NOW -- the thread's
  -- active-trace slot would otherwise leak until GC and poison every
  -- later op (and the trace_is_active gate).
  local ok, out = pcall(fn, unpack(inps))
  if not ok then
    t:close()
    error(out, 0)
  end
  local outs, out_is_list
  if is_tensor(out) then
    outs, out_is_list = { out }, false
  elseif type(out) == "table" and #out > 0 and is_tensor(out[1]) then
    outs, out_is_list = out, true
  else
    t:close()
    error("paddle.jit.trace: fn must return a Tensor or an array of "
          .. "Tensors", 2)
  end
  for i, o in ipairs(outs) do
    t:output(o, "out" .. i)
  end
  t:finish()  -- recording ends; the program stays alive for run()
  local nin = #inps
  -- The replayed callable: fresh inputs in, traced program out.
  local traced = function(...)
    local args = { ... }
    if #args ~= nin then
      error(("paddle.jit.trace: traced callable expects %d input(s), "
             .. "got %d"):format(nin, #args), 2)
    end
    local r = t:run(args)
    if out_is_list then
      if #outs == 1 then return { r } end
      return r  -- run already returns an array for multi-output
    end
    return r
  end
  return traced
end)

-- P18 script lands here later (D20; api/jit.md sec.2.3).
M.to_static = nil

return M