-- lua/paddle/dtype.lua -- paddle.dtype: dtype constants + finfo/iinfo
-- Stage: P5 / M1 (WORKPLAN 2.5, layout.md P5 step 1: dtype.lua +
-- device.lua -- no deps, easiest to align with Paddle naming).
--
-- Design: docs/migration/plan/api/dtype-device.md.
--  - the API surface is Paddle's OWN dtype names (foundations.md sec.3.2:
--    Insight7's constants are already Paddle-named), so no renaming layer;
--  - a dtype VALUE is the canonical string "float32" (dtype-device.md
--    sec.2.1: `paddle.cast{x, "float32"}` and `paddle.cast{x,
--    paddle.float32}` are the same thing; api/README.md sec.2.1.1: enum
--    params accept a string OR a typed constant, never a bare number --
--    the constant IS the string here, bare numbers still rejected);
--  - the bridge to the C extension is paddle_core.PD_DTYPE_* integers
--    (paddle_c.h:58-79 pd_dtype, 1:1 with phi::DataType), owned by
--    enums_place.cc (csrc/sol/enums_place.cc:39). The string table below
--    is the ONLY place the Lua name -> C enum mapping lives; both the
--    generated _ops wrappers and the public creation APIs consult it.

local paddle_core = require "paddle_core"

local M = {}

-- ----------------------------------------------------------------------
-- dtype values (api/dtype-device.md sec.1.1 / sec.2.1)
-- ----------------------------------------------------------------------
-- Canonical names -> paddle_core.PD_DTYPE_* (paddle_c.h:58-79; the enum
-- comment lists the upstream phi/DataTypes it mirrors 1:1).
local CANONICAL = {
  undefined     = paddle_core.PD_DTYPE_UNDEFINED,
  bool          = paddle_core.PD_DTYPE_BOOL,
  uint8         = paddle_core.PD_DTYPE_UINT8,
  int8          = paddle_core.PD_DTYPE_INT8,
  uint16        = paddle_core.PD_DTYPE_UINT16,
  int16         = paddle_core.PD_DTYPE_INT16,
  uint32        = paddle_core.PD_DTYPE_UINT32,
  int32         = paddle_core.PD_DTYPE_INT32,
  uint64        = paddle_core.PD_DTYPE_UINT64,
  int64         = paddle_core.PD_DTYPE_INT64,
  float32       = paddle_core.PD_DTYPE_FLOAT32,
  float64       = paddle_core.PD_DTYPE_FLOAT64,
  complex64     = paddle_core.PD_DTYPE_COMPLEX64,
  complex128    = paddle_core.PD_DTYPE_COMPLEX128,
  pstring       = paddle_core.PD_DTYPE_PSTRING,
  float16       = paddle_core.PD_DTYPE_FLOAT16,
  bfloat16      = paddle_core.PD_DTYPE_BFLOAT16,
  float8_e4m3fn = paddle_core.PD_DTYPE_FLOAT8_E4M3FN,
  float8_e5m2   = paddle_core.PD_DTYPE_FLOAT8_E5M2,
}

-- Aliases -> canonical name (dtype.py:80-98: short=int16, int=int32,
-- long=int64, float=float32, half=float16, double=float64,
-- cfloat=complex64, cdouble=complex128; raw -> pstring is the string
-- type's alias in the same block: `raw = VarDesc.VarType.RAW`).
local ALIAS = {
  short = "int16",
  int = "int32",
  long = "int64",
  float = "float32",
  half = "float16",
  double = "float64",
  cfloat = "complex64",
  cdouble = "complex128",
  raw = "pstring",
}

-- Exported constants: the canonical string IS the value. Paddle.float32
-- must be settable as a top-level paddle table field -- init.lua copies
-- these entries onto the top-level table after requiring this module.
for name in pairs(CANONICAL) do
  M[name] = name
end
-- Aliases POINT TO the primary value (dtype.py:80-98: `float = float32`
-- -- the same object in Python), so paddle.float == paddle.float32
-- here too ("float32").
for alias, canonical in pairs(ALIAS) do
  M[alias] = canonical
end

-- name (canonical or alias) -> paddle_core.PD_DTYPE_* integer.
-- Error on anything else (no bare numbers -- api/README.md sec.2.1.1).
function M.to_pd_dtype(name)
  local v = CANONICAL[name]
  if not v then
    local c = ALIAS[name]
    if c then v = CANONICAL[c] end
  end
  if not v then
    error(("paddle: unknown dtype '%s' (expected a paddle.dtype value "
      .. "like 'float32', or a string; bare numbers are rejected)")
      :format(tostring(name)), 2)
  end
  return v
end

-- paddle_core.PD_DTYPE_* integer -> canonical name (for tensor:dtype()
-- and friends). Reverse map built lazily.
local pd_to_name = nil
function M.to_name(pd_dtype)
  if pd_to_name == nil then
    pd_to_name = {}
    for name, id in pairs(CANONICAL) do
      pd_to_name[id] = name
    end
  end
  local n = pd_to_name[pd_dtype]
  if not n then
    error(("paddle: unknown pd_dtype %s"):format(tostring(pd_dtype)), 2)
  end
  return n
end

-- ----------------------------------------------------------------------
-- finfo / iinfo (api/dtype-device.md sec.1.3 / sec.2.4)
-- ----------------------------------------------------------------------
-- Numeric values from upstream paddle/fluid/pybind/pybind.cc finfo struct
-- (:538-591): eps=epsilon, min=lowest, max=max, tiny=smallest_normal,
-- smallest_normal=min, resolution=10^-digits10, bits=sizeof*8. The
-- float16/32/64 & bfloat16 constants are the IEEE-754 decimal values;
-- the raw bit patterns each numeric_limits returns are cited.
local FINFO = {
  float16 = { bits = 16, eps = 9.765625e-04,               -- 2^-10 (float16.h:1140 raw 0x1400)
              min = -65504.0, max = 65504.0,               -- 0xfbff / 0x7bff (float16.h:1132/1127)
              tiny = 6.103515625e-05,                      -- 2^-14 (min normal, float16.h:1122 raw 0x0400)
              smallest_normal = 6.103515625e-05,
              resolution = 1e-03,                          -- digits10=3 (float16.h:1121)
              dtype = "float16" },
  bfloat16 = { bits = 16, eps = 7.8125e-03,                -- 2^-7 (bfloat16.h:415 raw 0x3C00)
               min = -3.3895313892515355e+38,              -- 0xff7f (bfloat16.h:409)
               max = 3.3895313892515355e+38,               -- 0x7f7f (bfloat16.h:412)
               tiny = 1.1754943508222875e-38,              -- 2^-126 (bfloat16.h:406 raw 0x0080)
               smallest_normal = 1.1754943508222875e-38,
               resolution = 1e-02,                         -- digits10=2 (bfloat16.h:394)
               dtype = "bfloat16" },
  float32 = { bits = 32, eps = 1.1920928955078125e-07,     -- 2^-23 (numeric_limits<float>)
              min = -3.4028234663852886e+38,
              max = 3.4028234663852886e+38,
              tiny = 1.1754943508222875e-38,               -- 2^-126
              smallest_normal = 1.1754943508222875e-38,
              resolution = 1e-06,                          -- digits10=6
              dtype = "float32" },
  float64 = { bits = 64, eps = 2.2204460492503131e-16,     -- 2^-52 (numeric_limits<double>)
              min = -1.7976931348623157e+308,
              max = 1.7976931348623157e+308,
              tiny = 2.2250738585072014e-308,              -- 2^-1022
              smallest_normal = 2.2250738585072014e-308,
              resolution = 1e-15,                          -- digits10=15
              dtype = "float64" },
  complex64 = { bits = 32, eps = 1.1920928955078125e-07,   -- CASE_FINFO_BODY(float32, float) (pybind.cc:576-577)
                min = -3.4028234663852886e+38, max = 3.4028234663852886e+38,
                tiny = 1.1754943508222875e-38, smallest_normal = 1.1754943508222875e-38,
                resolution = 1e-06, dtype = "float32" },
  complex128 = { bits = 64, eps = 2.2204460492503131e-16,  -- CASE_FINFO_BODY(float64, double) (pybind.cc:579-580)
                 min = -1.7976931348623157e+308, max = 1.7976931348623157e+308,
                 tiny = 2.2250738585072014e-308, smallest_normal = 2.2250738585072014e-308,
                 resolution = 1e-15, dtype = "float64" },
}

-- float8 finfo deferred: upstream supports them (float8_e4m3fn.h:403 /
-- float8_e5m2.h custom numeric_limits) but the doc defers float8
-- compute support to P3 (dtype-device.md sec.7, "跟随 C++ 侧算子支持度"); a
-- clear error here beats a wrong number.

function M.finfo(dtype)
  local name = tostring(dtype)
  if not FINFO[name] then
    error(("paddle.finfo: unsupported dtype '%s' (expected float16/"
      .. "bfloat16/float32/float64/complex64/complex128; float8 deferred)")
      :format(name), 2)
  end
  return FINFO[name]
end

local IINFO = {
  uint8  = { bits = 8,  min = 0, max = 255 },
  uint16 = { bits = 16, min = 0, max = 65535 },
  uint32 = { bits = 32, min = 0, max = 4294967295 },
  uint64 = { bits = 64, min = 0, max = 18446744073709551615.0 }, -- double holds 2^64-1 rounded (see below)
  int8   = { bits = 8,  min = -128, max = 127 },
  int16  = { bits = 16, min = -32768, max = 32767 },
  int32  = { bits = 32, min = -2147483648, max = 2147483647 },
  int64  = { bits = 64, min = -9223372036854775808.0, max = 9223372036854775807.0 },
}

-- NOTE: Lua numbers are doubles (C3); int64/uint64 min/max lose
-- precision past 2^53. Upstream returns Python ints (exact) --
-- pybind.cc:1626-1630. Documented limitation: M1 targets float32 math,
-- and exact 2^63-range constants are not representable in the 5.1
-- value model. If someone needs exact big-int iinfo, that is a LuaJIT
-- cdata / Lua 5.3 integer task, not this layer.
function M.iinfo(dtype)
  local name = tostring(dtype)
  local info = IINFO[name]
  if not info then
    error(("paddle.iinfo: unsupported dtype '%s' (expected uint8/16/32/"
      .. "64 or int8/16/32/64)"):format(name), 2)
  end
  return { bits = info.bits, min = info.min, max = info.max,
           dtype = name }
end

-- ----------------------------------------------------------------------
-- default dtype global state (api/dtype-device.md sec.1.3 / sec.6.4)
-- ----------------------------------------------------------------------
-- Process-per-Lua-state global: creation APIs that take no explicit
-- dtype use this. Upstream framework.py:31/127 set/get. Each lane gets
-- its own copy in the vm (documented in dtype-device.md sec.6.4).
local default_dtype = "float32"

function M.get_default_dtype()
  return default_dtype
end

function M.set_default_dtype(dtype)
  local name = tostring(dtype)
  if not CANONICAL[name] and not ALIAS[name] then
    error(("paddle.set_default_dtype: unknown dtype '%s'"):format(name), 2)
  end
  default_dtype = name
end

return M