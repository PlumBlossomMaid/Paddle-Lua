-- lua/paddle/tensor.lua -- paddle.Tensor Lua face (P5 step 3)
-- Stage: P5 / M1 (WORKPLAN 2.5). After dtype.lua + device.lua.
--
-- Design: docs/migration/plan/modules/05-tensor.md + api/tensor.md +
-- 02-binding.md sec.3.5 (the __index triple duty: attribute/method lookup
-- MUST beat slicing, else x.shape gets parsed as a slice expression).
--
-- The C++ usertype (tensor_binding.cc) registers shape/ndim/dtype/numel/
-- is_contiguous/place_kind_index/item/to_host_bytes on the shared sol2
-- metatable. This file ENHANCES that same metatable from Lua (probe
-- verified: getmetatable(t) is a plain table, __index is a table lookup;
-- __tostring/__add are unset here). Indexing lands on the generated
-- pd_op_slice / pd_op_strided_slice C ABI functions.
--
-- 1-based indexing (D13/C4): positive Lua index n -> C++ n-1; negative
-- stays unchanged (-1 is the last element in both conventions) BUT the
-- strided_slice/slice kernels expect non-negative starts after clamping,
-- exactly like upstream ParseIndex does `start = start < 0 ? start + dim
-- : start` (slice_utils.h:407) -- so the clamp must happen HERE (Lua),
-- before the op call. End bounds: slice is [start, end); "1:3" (1-based
-- closed) becomes start 0, end 2 (see slice.cc lua_spec_to_cpp).

local paddle_core = require "paddle_core"

-- Ensure the eager runtime is up before touching any kernel. pd_init_once
-- is idempotent; a bare `require "paddle.tensor"` must leave a working
-- runtime behind (init.lua wires M.init but does not call it).
paddle_core.init()

local dtype_mod = require "paddle.dtype"
local device_mod = require "paddle.device"

local M = {}

-- ---------------------------------------------------------------------
-- shared usertype metatable + enhanced __index dispatch
-- ---------------------------------------------------------------------

-- A throwaway instance to reach the SHARED usertype metatable. sol2
-- registers one metatable per usertype; every later instance shares it,
-- so a single augmentation here covers all tensors. The instance itself
-- is dropped right after (GC frees the shell via __gc).
local probe = paddle_core.tensor{1}
local mt = getmetatable(probe)
local orig_index = mt.__index   -- sol2 method lookup table (table type)

-- Lua-side methods (operator methods arrive from the P3 generator later,
-- api/tensor.md sec.1.4 -- this table is where they land).
local METHODS = {}
-- Lua-side attribute values (getter functions of shape/ndim/... are
-- already C++ methods on orig_index; this table is for Lua-only attrs).
local ATTRS = {}

-- ---------------------------------------------------------------------
-- scalar decode (__tostring / item printing)
-- little-endian host layout, phi stores raw host bytes (float16.h:1107
-- etc. are byte-exact) -- verified via to_host_bytes round-trip.
-- ---------------------------------------------------------------------

local function u32_at(bytes, off)
  return (bytes:byte(off)) + (bytes:byte(off + 1)) * 256
      + (bytes:byte(off + 2)) * 65536 + (bytes:byte(off + 3)) * 16777216
end

local function f32_at(bytes, off)
  local u = u32_at(bytes, off)
  local sign = (u >= 2147483648) and -1 or 1
  local exp = math.floor((u % 2147483648) / 8388608)
  local frac = u % 8388608
  if exp == 0 then
    return sign * (frac / 8388608) * 2 ^ -126
  elseif exp == 255 then
    return sign * math.huge
  end
  return sign * (1 + frac / 8388608) * 2 ^ (exp - 127)
end

local function i32_at(bytes, off)
  local u = u32_at(bytes, off)
  if u >= 2147483648 then return u - 4294967296 end
  return u
end

local function i64_at(bytes, off)
  -- two 32-bit halves; double holds up to 2^53 exactly, warn via dtype
  local lo = u32_at(bytes, off)
  local hi = u32_at(bytes, off + 4)
  local v = hi * 4294967296 + lo
  if hi >= 2147483648 then v = v - 18446744073709551616.0 end
  return v
end

local function f64_at(bytes, off)
  local lo = u32_at(bytes, off)
  local hi = u32_at(bytes, off + 4)
  local sign = (hi >= 2147483648) and -1 or 1
  local exp = math.floor((hi % 2147483648) / 1048576)
  local frac = (hi % 1048576) * 4294967296 + lo
  if exp == 0 then
    return sign * (frac / 4503599627370496.0) * 2 ^ -1022
  elseif exp == 2047 then
    return sign * math.huge
  end
  return sign * (1 + frac / 4503599627370496.0) * 2 ^ (exp - 1023)
end

-- element size per dtype (paddle_c.h pd_dtype; esize also derivable from
-- #bytes / numel, but a table keeps decoding independent of numel=0).
local ESIZE = {
  [1] = 1,   -- BOOL
  [2] = 1,   -- UINT8
  [3] = 1,   -- INT8
  [4] = 2,   -- UINT16
  [5] = 2,   -- INT16
  [6] = 4,   -- UINT32
  [7] = 4,   -- INT32
  [8] = 8,   -- UINT64
  [9] = 8,   -- INT64
  [10] = 4,  -- FLOAT32
  [11] = 8,  -- FLOAT64
  [15] = 2,  -- FLOAT16 (decoding pending)
  [16] = 2,  -- BFLOAT16 (decoding pending)
}

local function decode_at(dtype_value, bytes, off)
  if dtype_value == 10 then return f32_at(bytes, off) end       -- FLOAT32
  if dtype_value == 11 then return f64_at(bytes, off) end       -- FLOAT64
  if dtype_value == 9 then return i64_at(bytes, off) end        -- INT64
  if dtype_value == 7 then return i32_at(bytes, off) end        -- INT32
  if dtype_value == 1 then return bytes:byte(off) end           -- BOOL
  if dtype_value == 8 then
    -- UINT64: decode unsigned (no sign flip in i64_at)
    return u32_at(bytes, off + 4) * 4294967296 + u32_at(bytes, off)
  end
  return nil  -- undecodable dtype for now
end

-- ---------------------------------------------------------------------
-- string slice: "1:3, :" -> strided_slice args (05-tensor.md sec.3.1)
-- core.slice_spec already rewrites 1-based positive fields to 0-based;
-- THIS side then clamps negatives per dimension length and fills missing
-- start/stop (Python slice.indices semantics), then drops whole-dim
-- slices (upstream ParseIndex skips start==0 && end==dim && step==1).
-- ---------------------------------------------------------------------

local function clamp_neg(v, dim)
  if v < 0 then v = v + dim end
  if v < 0 then v = 0 end
  if v > dim then v = dim end
  return v
end

-- field is one dimension of the already-0-based spec string
-- (e.g. "0:2", ":", "::-1", "2:", ":3", "::2"). Returns nil (whole dim)
-- or { start, stop, step } in 0-based Python slice.indices semantics.
-- NOTE: only the DEFAULTS are filled here (Python slice.indices); the
-- kernel's normalize_interval (phi/kernels/funcs/slice_utils.h:128) is
-- the authority on clamping/negative handling, and strided_slice_kernel
-- (stride/strided_slice_kernel.cc:60-61) turns the -dim-1 sentinel into
-- -1. So pass values through -- over-normalizing here breaks "::-1".
local function parse_dim_spec(field, dim)
  if field == "" then return nil end
  local a, b, c = field:match("^([^:]*):([^:]*):?(.*)$")
  if a == nil then
    -- bare number: integer index on this dim -> slice [n, n+1)
    local n = tonumber(field)
    if n == nil then
      error(("paddle: bad slice spec field '%s'"):format(field))
    end
    return { start = n, stop = n + 1, step = 1, bare = true }
  end
  local start, stop, step = a, b, c
  if step == "" then step = nil end
  local s = (start == "" or start == nil) and nil or tonumber(start)
  local e = (stop == "" or stop == nil) and nil or tonumber(stop)
  local st = step and tonumber(step) or 1
  if st == 0 then
    error("paddle: bad slice spec: step cannot be 0")
  end
  -- Python slice.indices defaults (both conventions from the end)
  if st > 0 then
    s = s or 0
    e = e or dim
  else
    s = s or (dim - 1)
    e = e or (-dim - 1)   -- sentinel: up to and including index 0
  end
  return { start = s, stop = e, step = st }
end

-- Build {axes, starts, ends, strides} for strided_slice from a spec
-- string against a shape (1-based dims already read as Lua table).
local function spec_to_strided(spec, shape)
  local axes, starts, ends, strides = {}, {}, {}, {}
  local nspec = 0
  for _ in spec:gmatch("[^,]+") do
    nspec = nspec + 1
  end
  if nspec > #shape then
    error(("paddle: slice spec has %d dims but tensor has %d"):format(
      nspec, #shape))
  end
  local d = 0
  for dim_field in spec:gmatch("[^,]+") do
    d = d + 1
    local dim = shape[d]
    local parsed = parse_dim_spec(dim_field, dim)
    if parsed == nil then
      -- whole-dim (":") is skipped entirely (upstream continue)
    else
      -- skip whole-dim slices (start==0 && end==dim && step==1)
      if not (parsed.start == 0 and parsed.stop == dim and parsed.step == 1) then
        axes[#axes + 1] = d - 1
        starts[#starts + 1] = parsed.start
        ends[#ends + 1] = parsed.stop
        strides[#strides + 1] = parsed.step
      end
    end
  end
  return axes, starts, ends, strides
end

-- ---------------------------------------------------------------------
-- indexing: number / table / string key -> tensor
-- ---------------------------------------------------------------------

local function shape_of(t)
  return t:shape()   -- Lua table, 1-based dims (tensor_binding.cc)
end

-- x[n] (number key): first-dim index, 1-based. Negative from the end.
-- integer indexing uses slice op with decrease_axis (upstream
-- getTensorWithBasicIndexing -> slice + decrease -> squeeze).
local function index_number(t, n)
  if type(n) ~= "number" then
    error("paddle: index must be a number, got " .. type(n))
  end
  local shape = shape_of(t)
  if #shape == 0 then
    error("paddle: cannot index a 0-d tensor with x[n], use x:item()")
  end
  local dim0 = shape[1]
  local i = math.floor(n)
  local start = i
  if start > 0 then start = start - 1 end           -- 1-based -> 0-based
  if start < 0 then start = start + dim0 end        -- negative from the end
  if start < 0 or start >= dim0 then
    error(("paddle: index %d out of range for axis 0 (size %d)"):format(
      i, dim0))
  end
  return paddle_core.pd_op_slice(t, {0}, {start}, {start + 1}, {1}, {0})
end

-- x[{i, j}] / x[{i, {a,b}}] (table key): multi-dim. Element-wise integer
-- indices (decrease per axis) or {lo, hi} closed sub-intervals; upstream
-- maps both through slice. v1: integers only (05-tensor.md sec.6 defers
-- masks).
local function index_table(t, idx)
  if type(idx) ~= "table" then
    error("paddle: multi-dim index must be a table")
  end
  local shape = shape_of(t)
  local n = #idx
  if n > #shape then
    error(("paddle: index has %d dims but tensor has %d"):format(n, #shape))
  end
  local axes, starts, ends, decrease = {}, {}, {}, {}
  for d = 1, n do
    local k = idx[d]
    if type(k) == "number" then
      local dim = shape[d]
      local i = math.floor(k)
      local start = i
      if start > 0 then start = start - 1 end       -- 1-based -> 0-based
      if start < 0 then start = start + dim end
      if start < 0 or start >= dim then
        error(("paddle: index %d out of range for axis %d (size %d)")
          :format(i, d, dim))
      end
      axes[#axes + 1] = d - 1
      starts[#starts + 1] = start
      ends[#ends + 1] = start + 1
      decrease[#decrease + 1] = d - 1
    elseif type(k) == "table" then
      -- {lo, hi} closed sub-interval on this dim
      local lo, hi = k[1], k[2]
      if type(lo) ~= "number" or type(hi) ~= "number" then
        error("paddle: sub-interval must be {lo, hi}")
      end
      local dim = shape[d]
      local s0 = lo > 0 and (lo - 1) or lo           -- 1-based -> 0-based
      local e0 = hi
      if s0 < 0 then s0 = s0 + dim end
      if e0 < 0 then e0 = e0 + dim + 1 end           -- closed -> open
      axes[#axes + 1] = d - 1
      starts[#starts + 1] = s0
      ends[#ends + 1] = e0
    else
      error("paddle: multi-dim index element must be number or {lo, hi}")
    end
  end
  if #axes == 0 then return t end
  -- infer_flags: all 1 (values always given; no infer needed)
  local infer = {}
  for i = 1, #axes do infer[i] = 1 end
  return paddle_core.pd_op_slice(t, axes, starts, ends, infer, decrease)
end

-- x["1:3, :"] (string key): D14 string indexing.
local function index_string(t, spec)
  if type(spec) ~= "string" then
    error("paddle: index must be number, table or string")
  end
  local zero = paddle_core.slice_spec(spec)  -- 1-based -> 0-based text
  local axes, starts, ends, strides =
    spec_to_strided(zero, shape_of(t))
  if #axes == 0 then return t end
  return paddle_core.pd_op_strided_slice(t, axes, starts, ends, strides)
end

-- ---------------------------------------------------------------------
-- __index dispatch (02-binding.md sec.3.5): methods/attrs BEAT slicing
-- ---------------------------------------------------------------------

local function tensor_index(t, k)
  if type(k) == "number" then
    return index_number(t, k)
  elseif type(k) == "table" then
    return index_table(t, k)
  end
  -- string key: Lua methods first (may override C++), then sol2 methods,
  -- then attributes, then slicing -- 02-binding.md sec.3.5.
  local m = METHODS[k]
  if m ~= nil then return m end
  m = rawget(orig_index, k)
  if m ~= nil then return m end
  local a = ATTRS[k]
  if a ~= nil then return a end
  -- not a method/attr: slicing, with a spelling-error guard
  if not string.find(k, ":") and not string.find(k, ",") then
    error(("paddle.Tensor: unknown attribute or method '%s' "
           .. "(did you mean a slice like \"1:3, :\"?)"):format(k), 2)
  end
  return index_string(t, k)
end

mt.__index = tensor_index

-- ---------------------------------------------------------------------
-- __tostring: truncated print (05-tensor.md sec.4 pitfall 3): shape /
-- dtype / place + first/last few elements, never the whole buffer.
-- ---------------------------------------------------------------------

local function tostring_tensor(t)
  local shape = t:shape()
  local dims = "{" .. table.concat(shape, ", ") .. "}"
  local dt = t:dtype()
  local dname = dtype_mod.to_name(dt) or "?"
  local kind, idx = t:place_kind_index()
  local pname = paddle_core.place_kind_name(kind)
  -- place_kind_index returns the raw device id (0-based); mirror the
  -- upstream Place repr ("Place(gpu:0)").
  local place = pname .. (idx > 0 and (":" .. idx) or "")
  local numel = t:numel()
  local head = {}
  if numel <= 12 then
    local bytes = t:to_host_bytes()
    local esize = ESIZE[dt]
    if esize and numel > 0 then
      for i = 0, numel - 1 do
        local v = decode_at(dt, bytes, i * esize + 1)
        head[#head + 1] = v == nil and "?" or tostring(v)
      end
    end
  else
    head[#head + 1] = ("... %d elements ..."):format(numel)
  end
  return ("Tensor(shape=%s, dtype=%s, place=%s, %s)"):format(
    dims, dname, place, table.concat(head, ", "))
end

mt.__tostring = tostring_tensor

-- ---------------------------------------------------------------------
-- arithmetic operators (05-tensor.md sec.3.4): elementwise ops. __eq is
-- deliberately NOT bound (pitfall 2: Lua __eq must return a boolean; use
-- paddle.equal{x,y}). Operands must be tensors for v1; scalar operands
-- are a later step (full_like broadcast).
-- ---------------------------------------------------------------------

local function arith(name, fn)
  mt["__" .. name] = function(a, b)
    if type(a) ~= "userdata" or type(b) ~= "userdata" then
      error("paddle: " .. name .. " expects two tensors for now")
    end
    return fn(a, b)
  end
end

arith("add", function(a, b) return paddle_core.pd_op_add(a, b) end)
arith("sub", function(a, b) return paddle_core.pd_op_subtract(a, b) end)
arith("mul", function(a, b) return paddle_core.pd_op_multiply(a, b) end)
arith("div", function(a, b) return paddle_core.pd_op_divide(a, b) end)

mt.__unm = function(a)
  return paddle_core.pd_op_scale(a, -1.0, 0.0, 1)
end

-- ---------------------------------------------------------------------
-- public Lua-side API additions (methods table)
-- ---------------------------------------------------------------------

-- x:item(i) -- 1-based flat index; the 0-d case goes straight to the C++
-- binding (tensor_binding.cc "item"), multi-element uses to_host_bytes
-- (api/tensor.md sec.2.3; coordinate form lands with P6).
M.item = function(t, ...)
  local n = select("#", ...)
  if n == 0 then
    return rawget(orig_index, "item")(t)
  end
  local i = select(1, ...)
  if n == 1 and type(i) == "number" then
    local numel = t:numel()
    if i < 1 or i > numel then
      error(("paddle: item(%d) out of range (numel=%d)"):format(i, numel))
    end
    local dt = t:dtype()
    local esize = ESIZE[dt]
    if not esize then
      error("paddle: item(): dtype not decodable yet")
    end
    local bytes = t:to_host_bytes()
    return decode_at(dt, bytes, (i - 1) * esize + 1)
  end
  error("paddle: item() coordinate form pending (P6)")
end

-- x:backward() / x:backward{grad_tensors=nil, retain_graph=false} /
-- x:backward(true) -- ONE bottom layer with paddle.autograd.backward
-- (api/autograd.md sec.6 ②): this wrapper delegates to core.backward so
-- retain-graph / accumulation semantics can never diverge.
--
-- NOTE: core.backward takes (tensors, grad_tensors, retain_graph). sol2
-- treats a nil MIDDLE argument as "argument list terminator" (measured
-- via _probe_optional, 2026-10-25): the optional param reads no value and
-- every LATER param is dropped too. So grad_tensors must NEVER be nil
-- here -- use {} (empty list == no seeds, same semantics).
METHODS.backward = function(self, ...)
  if select("#", ...) == 0 then
    return paddle_core.backward({ self }, {}, false)
  end
  local arg = select(1, ...)
  if type(arg) == "table" then
    return paddle_core.backward({ self }, arg.grad_tensors or {},
                                arg.retain_graph ~= nil and arg.retain_graph or false)
  end
  if select("#", ...) == 1 then
    return paddle_core.backward({ self }, {}, arg and true or false)
  end
  -- (grad_tensors, retain_graph) positional
  return paddle_core.backward({ self }, arg or {}, select(2, ...))
end

-- Mount the Lua-side methods so the __index dispatch finds them (the P3
-- generator appends its operator methods to this same table later).
METHODS.item = M.item

-- Expose the tables so the P3 generator can mount operator methods and
-- 05-tensor.md spelling-error checks can reuse them.
M.METHODS = METHODS
M.ATTRS = ATTRS
M._spec_to_strided = spec_to_strided
M._index_string = index_string
M._index_table = index_table
M._index_number = index_number

return M