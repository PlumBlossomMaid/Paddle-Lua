-- lua/paddle/device.lua -- paddle.device: place value type + constructors
-- Stage: P5 / M1 (WORKPLAN 2.5, layout.md P5 step 1: dtype.lua +
-- device.lua -- no deps, easiest to align with Paddle naming).
--
-- Design: docs/migration/plan/api/dtype-device.md sec.2.2/6.2.
--  - place is a VALUE, not a handle: fieldwise equality
--    (`paddle.CUDAPlace(0) == paddle.CUDAPlace(0)` must be true --
--    dtype-device.md sec.6.2). The C ABI LuaPlace shell (enums_place.cc)
--    is deliberately opaque/pointer-identity, so this module owns the
--    public value type and converts to LuaPlace at the op boundary
--    (`_to_core_place`), exactly where the handle is needed.
--  - constructors mirror upstream place.h types (CPUPlace/GPUPlace/
--    GPUPinnedPlace/XPUPlace/XPUPinnedPlace/IPUPlace/CustomPlace,
--    place.h:134-201), using Paddle's own names (no PyTorch "device").
--
-- Value layout: { kind = <paddle_core.PD_PLACE_*>, index = <int>,
--                 dev_type = <string|nil> } + __eq/__tostring.

local paddle_core = require "paddle_core"

local M = {}

-- ----------------------------------------------------------------------
-- kind int <-> canonical name (device.cc:155-170 pd_place_kind_name).
-- Declared BEFORE PlaceMT: __tostring closes over these (Lua 5.1: a
-- name referenced inside a function defined before the local declaration
-- would resolve to a global, which is nil at runtime).
local NAME_TO_KIND = {
  cpu = paddle_core.PD_PLACE_CPU,
  gpu = paddle_core.PD_PLACE_CUDA,
  gpu_pinned = paddle_core.PD_PLACE_CUDAPINNED,
  xpu = paddle_core.PD_PLACE_XPU,
  xpu_pinned = paddle_core.PD_PLACE_XPUPINNED,
  ipu = paddle_core.PD_PLACE_IPU,
  custom = paddle_core.PD_PLACE_CUSTOM,
}
local KIND_TO_NAME = {}
for n, k in pairs(NAME_TO_KIND) do
  KIND_TO_NAME[k] = n
end

-- ----------------------------------------------------------------------
-- Place value type
-- ----------------------------------------------------------------------
local PlaceMT = {}
PlaceMT.__index = PlaceMT

function PlaceMT.__eq(a, b)
  if type(a) ~= "table" or type(b) ~= "table" then
    return false
  end
  if a.kind ~= b.kind then return false end
  if a.index ~= b.index then return false end
  return a.dev_type == b.dev_type
end

function PlaceMT.__tostring(p)
  -- Same shape as upstream Place repr ("Place(cpu)" / "Place(gpu:0)",
  -- set_device doctest device/__init__.py:630-636).
  local name = KIND_TO_NAME[p.kind] or "?"
  if p.kind == paddle_core.PD_PLACE_CPU then
    return "Place(cpu)"
  end
  return ("Place(%s:%d)"):format(name, p.index)
end

local function new_place(kind, index, dev_type)
  return setmetatable({ kind = kind, index = index,
                        dev_type = dev_type }, PlaceMT)
end

-- ----------------------------------------------------------------------
-- place constructors (api/dtype-device.md sec.1.2 / sec.2.2)
-- ----------------------------------------------------------------------
function M.CPUPlace()
  return new_place(paddle_core.PD_PLACE_CPU, 0, nil)
end

function M.CUDAPlace(dev_id)
  -- dtype-device.md sec.6.3: CUDAPlace() default is -1 = auto-select
  -- (upstream GPUPlace(-1) picks the current card). Keep -1 IN THE
  -- VALUE; pd_place_new (tensor.cc:153) rejects index<0, so _to_core_place
  -- resolves -1 to the actual device (0 on this build) at the boundary.
  if dev_id == nil then dev_id = -1 end
  return new_place(paddle_core.PD_PLACE_CUDA, dev_id, nil)
end

function M.CUDAPinnedPlace()
  return new_place(paddle_core.PD_PLACE_CUDAPINNED, 0, nil)
end

function M.XPUPlace(dev_id)
  if dev_id == nil then dev_id = 0 end
  return new_place(paddle_core.PD_PLACE_XPU, dev_id, nil)
end

function M.XPUPinnedPlace()
  return new_place(paddle_core.PD_PLACE_XPUPINNED, 0, nil)
end

function M.IPUPlace(dev_id)
  if dev_id == nil then dev_id = 0 end
  return new_place(paddle_core.PD_PLACE_IPU, dev_id, nil)
end

function M.CustomPlace(dev_type, dev_id)
  -- upstream CustomPlace(dev_type, device_id) (place.h:197-198); the
  -- dev_type is a string identifier ("custom" device family).
  if dev_id == nil then dev_id = 0 end
  return new_place(paddle_core.PD_PLACE_CUSTOM, dev_id, dev_type)
end

-- Accept a canonical name too: paddle.device("cpu") == paddle.CPUPlace()
-- (upstream device constructor, device/__init__.py:2106-2146).
local DEVICE_RE = "^([%a_]+):?(%d*)$"
function M.device(spec)
  if type(spec) == "table" then
    return spec
  end
  local name, idx = string.match(tostring(spec), DEVICE_RE)
  local kind = NAME_TO_KIND[name]
  if not kind then
    error(("paddle.device: unknown device '%s'"):format(tostring(spec)), 2)
  end
  if idx == "" then idx = "0" end
  return new_place(kind, tonumber(idx), nil)
end

-- ----------------------------------------------------------------------
-- internal: value place -> C ABI LuaPlace handle (op boundary)
-- ----------------------------------------------------------------------
-- Op lambdas take `const LuaPlace&` (enums_place.cc), so the public value
-- must become a handle right before an op call. -1 (auto) resolves to the
-- default device id: CUDA picks 0 here -- a full auto-select query is
-- device-layer work, and this build is CPU-only (pd_device_count
-- CUDA=0, device.cc:59 comment).
function M._to_core_place(p)
  local index = p.index
  if index < 0 then
    index = 0
  end
  return paddle_core.pd_place_new(p.kind, index)
end

-- ----------------------------------------------------------------------
-- device global state (api/dtype-device.md sec.2.3)
-- ----------------------------------------------------------------------
-- Process-per-Lua-state global default device string ("cpu" / "gpu:0"),
-- mirroring upstream set_device/get_device (device/__init__.py:605-664).
-- NOTE: the C ABI has no eager device-set hook yet; this is Lua-side
-- state consulted by the creation APIs, wired in tensor.lua (P5 step 3).
local default_device = "cpu"

function M.get_device()
  return default_device
end

function M.set_device(spec)
  if type(spec) == "table" then
    default_device = tostring(spec)
    return spec
  end
  local name, idx = string.match(tostring(spec), DEVICE_RE)
  if not NAME_TO_KIND[name] then
    error(("paddle.device.set_device: unknown device '%s'")
      :format(tostring(spec)), 2)
  end
  default_device = idx == "" and name or (name .. ":" .. idx)
  return M.device(spec)
end

function M.is_compiled_with_cuda()
  -- pd_device_count(PD_PLACE_CUDA) > 0 on a CUDA build; this build is
  -- CPU-only, so false (paddle_c.h:199-201).
  return paddle_core.device_count(paddle_core.PD_PLACE_CUDA) > 0
end

return M