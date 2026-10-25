-- tools/smoke/ops_smoke.lua
-- M1 2.3.4 closing item 3: sol2 op binding layer smoke. The generated
-- csrc/sol/gen_ops_*.cc register every pd_op_* C ABI function as a sol2
-- lambda on the paddle_core module table; this script proves the layer
-- works from Lua for every emitted form:
--   single output       pd_op_add -> LuaTensor
--   multi output        pd_op_accuracy -> N Lua return values
--   single vector out   pd_op_split -> 1-based Lua table
--   dtype/place attrs   pd_op_full with PD_DTYPE_* + pd_place_new
--   error path          bad arg -> Lua error (pd::check -> sol::error)
--
-- Build first:  cmake -S . -B build-test
--               cmake --build build-test --target paddle_core -j 6
-- Run (from the repo root), same recipe as require_smoke.lua:
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/ops_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
assert(core.init() == true, "init()")

-- --- single output: add / multiply / relu / sqrt ---------------------
local a = core.tensor{2, 3}              -- 0-filled F32 CPU {2,3}
local c = core.pd_op_add(a, a)
assert(c:ndim() == 2 and c:numel() == 6, "add ndim/numel")
assert(c:shape()[1] == 2 and c:shape()[2] == 3, "add shape {2,3}")
assert(c:dtype() == core.PD_DTYPE_FLOAT32, "add dtype F32")

local m = core.pd_op_multiply(a, a)
assert(m:shape()[1] == 2, "multiply ok")
local r = core.pd_op_relu(a)
assert(r:numel() == 6, "relu ok")
local s = core.pd_op_sqrt(core.pd_op_exp(a))
assert(s:numel() == 6, "exp->sqrt ok")

-- --- multi output: topk (2 values) + accuracy (3 values) --------------
-- topk(x, k, axis, largest, sorted): axis=1 along dim1 of {2,3} takes top1
-- per row -> value [2,1] F32 + indices [2,1] INT64 (variadic_results -> 2
-- Lua values). accuracy(x, indices, label) then consumes that pair:
-- AccuracyInferMeta (ternary.cc:55-58) requires label rank 2 and in
-- runtime its 2nd dim == 1; the kernel (accuracy_kernel.cc:22) reads
-- indices/label as int64. Outputs are 0-d scalars (make_ddim({})).
local tval, tidx = core.pd_op_topk(a, 1.0, 1, 1, 1)
assert(tval ~= nil and tidx ~= nil, "topk 2 outputs")
assert(tidx:dtype() == core.PD_DTYPE_INT64, "topk indices INT64")
local v1, v2, v3 = core.pd_op_accuracy(tval, tidx, tidx)
assert(v1 ~= nil and v2 ~= nil and v3 ~= nil, "accuracy 3 outputs")
assert(#v1:shape() == 0 and #v2:shape() == 0 and #v3:shape() == 0,
       "accuracy outputs are 0-d scalars")

-- --- single vector output: split -> 1-based table ---------------------
-- split(x, sections, axis): axis here is C-ABI level (0-based; the
-- 1-based shift lives in the emit_lua wrapper). x is {4,1}; sections
-- {1,3} along axis 0 -> two tensors {1,1} and {3,1}.
local x4 = core.tensor{4, 1}
local parts = core.pd_op_split(x4, {1, 3}, 0)
assert(type(parts) == "table", "split returns a table")
assert(parts[1] ~= nil and parts[2] ~= nil and parts[3] == nil,
       "split count = 2")
assert(parts[1]:shape()[1] == 1, "split part1 dim0 = 1")
assert(parts[2]:shape()[1] == 3, "split part2 dim0 = 3")

-- --- dtype/place attrs: full ------------------------------------------
-- full(shape, value, dtype, place): Lua passes PD_DTYPE_* int + a
-- LuaPlace from the pd_place_new factory.
local cpu = core.pd_place_new(core.PD_PLACE_CPU, 0)
assert(type(cpu) == "userdata", "pd_place_new returns LuaPlace userdata")
local f = core.pd_op_full({2, 2}, 3.5, core.PD_DTYPE_FLOAT32, cpu)
assert(f:shape()[1] == 2 and f:shape()[2] == 2, "full shape {2,2}")
assert(f:dtype() == core.PD_DTYPE_FLOAT32, "full dtype")
assert(core.PD_DTYPE_FLOAT64 == 11, "PD_DTYPE_FLOAT64 == 11")
assert(core.PD_PLACE_CUDA == 1, "PD_PLACE_CUDA == 1")

-- --- error path: bad arg type -> Lua error (not crash) ----------------
local okerr, err = pcall(function() core.pd_op_add(1, 2) end)
assert(not okerr and type(err) == "string", "add(1,2) errors")
local okerr2, err2 = pcall(function() core.pd_op_full({2}, 0, 999, cpu) end)
assert(not okerr2 and type(err2) == "string", "bad dtype errors")

print("OPS_SMOKE PASS")
