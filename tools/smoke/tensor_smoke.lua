-- tools/smoke/tensor_smoke.lua
-- M1 2.5: tensor.lua Lua-face smoke -- metatable augmentation (__index /
-- __tostring / arithmetic) + 1-based indexing (number / table / string).
--
-- Build first:  cmake -S . -B build-test
--               cmake --build build-test --target paddle_core -j 6
-- Run (from the repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/tensor_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, core = pcall(require, "paddle_core")
assert(ok, "require paddle_core failed: " .. tostring(core))
core.init()

local tensor_mod = require "paddle.tensor"

-- base tensors: 0-d scalar + 2x3 all-2.0
local cpu = core.pd_place_new(core.PD_PLACE_CPU, 0)
local z = core.pd_op_full({}, 2.5, core.PD_DTYPE_FLOAT32, cpu)
local x = core.pd_op_full({2, 3}, 2.0, core.PD_DTYPE_FLOAT32, cpu)

-- --- item(): 0-d scalar via C++ binding
assert(z:item() == 2.5, "item() 0-d")

-- --- number index: x[1] = first row (rank 1, size 3)
local r1 = x[1]
assert(#r1:shape() == 1 and r1:shape()[1] == 3, "x[1] first row")
assert(r1:item(2) == 2.0, "row element value")

-- x[-1] = last row (negative stays from-the-end)
local rn = x[-1]
assert(#rn:shape() == 1 and rn:shape()[1] == 3, "x[-1] last row")

-- --- table index: x[{2,1}] -> 0-d element (both axes decrease)
local el = x[{2, 1}]
assert(#el:shape() == 0, "x[{2,1}] is 0-d")
assert(el:item() == 2.0, "x[{2,1}] value")

-- --- string slice: x["1:3, :"] = whole rows (half-open: 1-based [1,3)
-- becomes 0-based [0,2)); x["1:2, :"] = first row
local all = x["1:3, :"]
assert(#all:shape() == 2 and all:shape()[1] == 2 and all:shape()[2] == 3,
       "x[\"1:3, :\"] = whole {2,3}")
local s1 = x["1:2, :"]
assert(#s1:shape() == 2 and s1:shape()[1] == 1 and s1:shape()[2] == 3,
       "x[\"1:2, :\"] first row")

-- x[":"] is the identity (whole dim dropped)
local ident = x[":"]
assert(ident:shape()[1] == 2 and ident:shape()[2] == 3, "x[\":\"]")

-- x["::-1"] reverses axis 0 (shape unchanged; all-2 so value-blind)
local rev = x["::-1"]
assert(rev:shape()[1] == 2 and rev:shape()[2] == 3, "x[\"::-1\"] shape")

-- step slicing: x["1:3:2, :"] = odd rows (1-based rows 1..2 step 2 -> row 1)
local odd = x["1:3:2, :"]
assert(#odd:shape() == 2 and odd:shape()[1] == 1 and odd:shape()[2] == 3,
       "x[\"1:3:2, :\"] one row")

-- open ends: x["2:, :"] = rows 2..end; x[":2, :"] = rows 1..1
local tail = x["2:, :"]
assert(#tail:shape() == 2 and tail:shape()[1] == 1 and tail:shape()[2] == 3,
       "x[\"2:, :\"] last row")
local head = x[":2, :"]
assert(#head:shape() == 2 and head:shape()[1] == 1 and head:shape()[2] == 3,
       "x[\":2, :\"] first row")

-- --- arithmetic operators
local sum = x + x
assert(sum:numel() == 6 and sum[1]:item(1) == 4.0, "x+x element 4.0")
local prod = x * x
assert(prod[1]:item(1) == 4.0, "x*x element 4.0")
local neg = -x
assert(neg[1]:item(1) == -2.0, "-x element -2.0")

-- --- sol2 methods still reachable (orig_index lookup wins)
assert(x:ndim() == 2, "ndim")
assert(x:numel() == 6, "numel")
assert(x:dtype() == core.PD_DTYPE_FLOAT32, "dtype")
assert(x:is_contiguous() == 1, "is_contiguous")

-- --- __tostring does not crash and mentions shape
local ts = tostring(x)
assert(type(ts) == "string" and ts:find("shape=", 1, true) ~= nil,
       "tostring mentions shape")

-- --- unknown-attribute guard (no ':' / ',' -> spelling error, not slice)
local ok2, err = pcall(function() return x.shpae end)
assert(not ok2 and tostring(err):find("shpae", 1, true) ~= nil,
       "spelling guard for shpae")

-- --- out-of-range guard reports the 1-based index the user wrote
local ok3, err3 = pcall(function() return x[5] end)
assert(not ok3 and tostring(err3):find("5", 1, true) ~= nil,
       "out-of-range guard reports 1-based 5")

print("TENSOR_SMOKE PASS")