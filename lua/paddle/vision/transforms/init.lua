-- lua/paddle/vision/transforms/init.lua -- paddle.vision.transforms
-- Stage: M2 / 3.1.3 (WORKPLAN 3.1.3). Contract: api/vision.md sec 2.2 +
-- 12-dataset-vision.md sec 3.2.
--
-- Design rules (both docs):
--   * a transform is a callable object t(x); Compose applies a list in
--     order (vision.md sec 2.2);
--   * transforms operate on insight.Array (D26), NOT on paddle.Tensor
--     (12-dataset-vision.md sec 3.2 "transform 内部用 Insight7, 不用
--     Tensor"). ToTensor is the ONLY Insight->Tensor boundary, so it
--     lands last in a pipeline (vision.md sec 6.3);
--   * transforms are pure functions (12-dataset-vision.md sec 3.2 --
--     P13 multi-worker hard constraint).
--
-- Upstream numerics mirrored (read 2026-10-28):
--   Compose  transforms.py:117-137  (call each f in order)
--   ToTensor transforms.py:347-389 + functional_cv2.py:27-57
--     uint8 -> cast float32 /255.0; 2D -> [H,W,1]; CHW -> transpose(2,0,1)
--   Normalize transforms.py:883-923 + functional_cv2.py:682-714
--     scalar mean/std -> [m,m,m] (transforms.py:913-917); CHW mean
--     reshape(-1,1,1) / HWC (1,1,-1) broadcast; (x-mean)/std
--   Resize  transforms.py:415-468 + functional_cv2.py:59-120
--     int -> short-edge match (size*h/w); tuple -> (h,w); bilinear dflt
--
-- Lua 5.1 subset (C3).

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"

local M = {}

local function insight()
  local ok, ins = pcall(require, "insight")
  if not ok then
    error("paddle.vision.transforms: needs Insight7 (D26) to manipulate "
          .. "image arrays -- failed to load: " .. tostring(ins), 3)
  end
  return ins
end

-- ---- Compose: 依序应用 transform 列表 -------------------------------

local Compose = class()
Compose._name = "Compose"

Compose._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "transforms", type = argrule.table,
    doc = "list of callable transforms, applied in order" },
}(function(self, transforms)
  self._transforms = transforms
end)

Compose.__call = function(self, x)
  for _, f in ipairs(self._transforms) do
    x = f(x)
  end
  return x
end

-- ---- ToTensor: Array -> paddle.Tensor (Insight->Tensor boundary) ----

local ToTensor = class()
ToTensor._name = "ToTensor"

ToTensor._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "data_format", type = argrule.string, opt = true,
    default = "CHW", doc = "'CHW' (default) or 'HWC'" },
}(function(self, data_format)
  if data_format ~= "CHW" and data_format ~= "HWC" then
    error("ToTensor: data_format must be 'CHW' or 'HWC', got "
          .. tostring(data_format), 3)
  end
  self.data_format = data_format
end)

-- Array([H,W] | [H,W,C]) -> LuaTensor.
-- Mirror functional_cv2.py:27-57 (uint8 -> /255; 2D -> [H,W,1];
-- CHW -> transpose; output float32). The /255 and the layout work
-- happen INSIDE Insight (data plane), the tensor is minted once at the
-- end via core.from_host_bytes -- the single C ABI crossing.
ToTensor.__call = function(self, x)
  local ins = insight()
  local dt = x.dtype

  -- 2D grayscale -> [H,W,1] (functional_cv2.py:31 pic[:, :, None])
  if x.ndim == 2 then
    x = x:unsqueeze(3)
  end

  -- dtype policy mirrors functional_cv2.py:49-56: uint8 -> cast
  -- float32 / 255.0; every other dtype stays as-is (no scaling). The
  -- pd_dtype sent to from_host_bytes MUST match the final byte layout
  -- (bytes are copied verbatim from the contiguous Array payload).
  local pd_dt
  if dt == ins.uint8 then
    x = x:to(ins.float32) / 255.0
    x = x:to(ins.float32)      -- scalar div may widen; pin back to F32
    pd_dt = paddle_core.PD_DTYPE_FLOAT32
  elseif dt == ins.float32 then
    pd_dt = paddle_core.PD_DTYPE_FLOAT32
  elseif dt == ins.float64 then
    pd_dt = paddle_core.PD_DTYPE_FLOAT64
  elseif dt == ins.int32 then
    pd_dt = paddle_core.PD_DTYPE_INT32
  elseif dt == ins.int64 then
    pd_dt = paddle_core.PD_DTYPE_INT64
  else
    error("ToTensor: unsupported Array dtype " .. tostring(dt), 3)
  end

  -- HWC -> CHW (functional_cv2.py:44 transpose((2,0,1)))
  if self.data_format == "CHW" then
    x = x:transpose{ 3, 1, 2 }
  end
  -- transpose creates a strided VIEW; to_dlpack's .bytes copies the
  -- contiguous payload, so materialize first.
  x = x:contiguous()

  local dlm = ins.to_dlpack(x)
  local bytes = dlm:bytes()   -- method, NOT a property (dlm_type["bytes"]
                              -- = lambda, insight_lua.cpp:1108)
  local dims = { x.shape[1], x.shape[2], x.shape[3] }
  local place = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  return paddle_core.from_host_bytes(bytes, dims, pd_dt, place)
end

-- ---- Normalize: (x - mean) / std on an insight.Array ----------------

local Normalize = class()
Normalize._name = "Normalize"

-- mean/std accept a number (expanded to [v,v,v], transforms.py:913-917
-- mirrors the PIL convention) or a list (per-channel).
local function to_list(v)
  if type(v) == "table" then return v end
  return { v, v, v }
end

Normalize._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "mean", type = argrule.Union{ argrule.number, argrule.table },
    doc = "mean per channel (scalar = same for all, expanded to 3)" },
  { name = "std", type = argrule.Union{ argrule.number, argrule.table },
    doc = "std per channel (scalar = same for all, expanded to 3)" },
  { name = "data_format", type = argrule.string, opt = true,
    default = "CHW", doc = "'CHW' (default) or 'HWC'" },
}(function(self, mean, std, data_format)
  if data_format ~= "CHW" and data_format ~= "HWC" then
    error("Normalize: data_format must be 'CHW' or 'HWC', got "
          .. tostring(data_format), 3)
  end
  self.mean = to_list(mean)
  self.std = to_list(std)
  self.data_format = data_format
end)

Normalize.__call = function(self, x)
  local ins = insight()
  -- Broadcast shapes mirror functional_cv2.py:682-714:
  --   CHW: mean.reshape(-1,1,1); HWC: mean.reshape(1,1,-1)
  -- Build the mean/std arrays at those shapes, then let Insight's
  -- elementwise broadcasting do (x - mean) / std.
  local m = #self.mean
  local s = #self.std
  local mdim, sdim
  if self.data_format == "CHW" then
    mdim = { m, 1, 1 }
    sdim = { s, 1, 1 }
  else
    mdim = { 1, 1, m }
    sdim = { 1, 1, s }
  end
  local mean_arr = ins.Array(self.mean):reshape(mdim)
  local std_arr  = ins.Array(self.std):reshape(sdim)
  return (x - mean_arr) / std_arr
end

-- ---- Resize: 2D image resize on insight.Array (C ABI pd_image_resize)

local Resize = class()
Resize._name = "Resize"

-- insight DType -> pd_dtype (values differ: ins F32=9, pd FLOAT32=10).
local function pd_dtype_of(dt)
  local ins = insight()
  if dt == ins.uint8 then return paddle_core.PD_DTYPE_UINT8 end
  if dt == ins.float32 then return paddle_core.PD_DTYPE_FLOAT32 end
  error("Resize: unsupported Array dtype (uint8/float32 only)", 3)
end

Resize._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "size", type = argrule.Union{ argrule.number, argrule.table },
    doc = "int = short-edge match; {h,w} = exact output size" },
  { name = "interpolation", type = argrule.string, opt = true,
    default = "bilinear", doc = "'nearest' or 'bilinear' (default)" },
}(function(self, size, interpolation)
  if interpolation ~= "nearest" and interpolation ~= "bilinear" then
    error("Resize: interpolation must be 'nearest' or 'bilinear', got "
          .. tostring(interpolation), 3)
  end
  self.size = size
  self.interpolation = interpolation
end)

-- Resolve the target (oh, ow) per functional_cv2.py:81-118:
--   int  -> short edge = size, other edge = size * h / w (int trunc)
--   {h,w}-> exact
local function resolve(h, w, size)
  if type(size) == "table" then
    return size[1], size[2]
  end
  if w <= h and w == size then return h, w end   -- already matched
  if h <= w and h == size then return h, w end
  if w < h then
    return math.floor(size * h / w), size
  end
  return size, math.floor(size * w / h)
end

Resize.__call = function(self, x)
  local h, w = x.shape[1], x.shape[2]
  local channels = (x.ndim == 3) and x.shape[3] or 1
  local dtype = x.dtype
  local interp = (self.interpolation == "nearest") and 0 or 1

  local oh, ow = resolve(h, w, self.size)
  if oh == h and ow == w then return x end

  local x2 = x
  if x.is_contiguous == false then x2 = x:contiguous() end
  local dlm = insight().to_dlpack(x2)
  local src_bytes = dlm:bytes()   -- method (insight_lua.cpp:1108)
  local out = paddle_core.image_resize(
    src_bytes, h, w, channels, pd_dtype_of(dtype), oh, ow, interp)

  -- Rebuild an Array from the resized bytes (same dtype, grayscale
  -- stays 2D -- functional_cv2.py:120 keeps 2D/3D shape distinction).
  local shape = (x.ndim == 3) and { oh, ow, channels } or { oh, ow }
  local dtype_str = (dtype == insight().uint8) and "u1" or "f4"
  return insight().from_dlpack(shape, dtype_str, out)
end

M.Compose = Compose
M.ToTensor = ToTensor
M.Normalize = Normalize
M.Resize = Resize

return M