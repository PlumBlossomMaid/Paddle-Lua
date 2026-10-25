-- tools/smoke/dtype_device_smoke.lua
-- M1 2.5 step 1: paddle.dtype / paddle.device surface smoke (P5,
-- layout.md P5 step 1 -- dtype.lua + device.lua).
--
-- Verifies, against upstream facts:
--   - dtype constants are Paddle's OWN names, canonical string IS the
--     value, aliases point to the primary (dtype.py:77-126);
--   - finfo/iinfo numbers match pybind.cc:487-591 + IEEE-754;
--   - default dtype global state (framework.py:31/127);
--   - place is a VALUE with fieldwise equality (dtype-device.md sec.6.2);
--   - device global state + place -> C ABI handle boundary.
--
-- Run (from the repo root), same recipe as require_smoke.lua:
--   PATH=<paddle runtime dll dirs>;$PATH luajit tools/smoke/dtype_device_smoke.lua
--
-- Lua 5.1 / LuaJIT compatible; matches the 5.1 subset (C3).

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path

local ok, P = pcall(require, "paddle")
assert(ok, "require paddle: " .. tostring(P))

-- ---- dtype constants (canonical string values + aliases) ------------
assert(P.float32 == "float32", "float32")
assert(P.float == "float32", "float alias -> float32 (dtype.py:89)")
assert(P.int64 == "int64", "int64")
assert(P.long == "int64", "long alias")
assert(P.bool == "bool", "bool")
assert(P.float16 == "float16", "float16")
assert(P.half == "float16", "half alias")
assert(P.raw == "pstring", "raw alias")

-- bare numbers must be rejected (api/README.md sec.2.1.1)
local okrej = pcall(P.to_pd_dtype, 5)
assert(not okrej, "dtype enum rejects bare numbers")

-- ---- finfo (pybind.cc:538-591, IEEE-754) ----------------------------
local f32 = P.finfo(P.float32)
assert(f32.max == 3.4028234663852886e+38, "f32 max")
assert(f32.eps == 1.1920928955078125e-07, "f32 eps")
assert(f32.bits == 32, "f32 bits")
local f64 = P.finfo(P.float64)
assert(f64.max == 1.7976931348623157e+308, "f64 max")
assert(f64.bits == 64, "f64 bits")
local f16 = P.finfo(P.float16)
assert(f16.max == 65504.0, "f16 max (float16.h:0x7bff)")
assert(f16.eps == 9.765625e-04, "f16 eps (2^-10)")

-- ---- iinfo (pybind.cc:487-539) --------------------------------------
local i8 = P.iinfo(P.int8)
assert(i8.min == -128 and i8.max == 127 and i8.bits == 8, "iinfo int8")
local u8 = P.iinfo(P.uint8)
assert(u8.min == 0 and u8.max == 255, "iinfo uint8")

-- ---- default dtype global state --------------------------------------
assert(P.get_default_dtype() == "float32", "default float32")
P.set_default_dtype(P.float64)
assert(P.get_default_dtype() == "float64", "set_default float64")
P.set_default_dtype("float32")

-- ---- place value semantics (dtype-device.md sec.6.2) ----------------
local c1 = P.CPUPlace()
local c2 = P.CPUPlace()
assert(c1 == c2, "CPUPlace fieldwise eq")
local g1 = P.CUDAPlace(0)
local g2 = P.CUDAPlace(0)
assert(g1 == g2, "CUDAPlace(0) eq")
assert(c1 ~= g1, "CPU != CUDA")
assert(g1 ~= P.CUDAPlace(1), "CUDAPlace(0) != (1)")
assert(tostring(c1) == "Place(cpu)", "tostring CPU")
assert(tostring(g1) == "Place(gpu:0)", "tostring GPU")

-- ---- device global state ---------------------------------------------
P.device.set_device("cpu")
assert(P.device.get_device() == "cpu", "get_device cpu")

-- ---- place -> core handle (op boundary) ------------------------------
local luaplace = P.device._to_core_place(c1)
assert(type(luaplace) == "userdata", "place -> LuaPlace userdata")

print("DTYPE_DEVICE_SMOKE PASS")