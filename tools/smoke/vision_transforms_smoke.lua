-- tools/smoke/vision_transforms_smoke.lua -- paddle.vision.transforms
-- Stage: M2 / 3.1.3. Mirrors upstream numerics (transforms.py +
-- functional_cv2.py):
--   ToTensor  uint8 -> float32 /255 (functional_cv2.py:27-57); 2D ->
--             [H,W,1]; CHW transpose(2,0,1)
--   Normalize (x-mean)/std, scalar mean/std expanded to [v,v,v]
--             (transforms.py:913-917), CHW mean reshape(-1,1,1) / HWC
--             (1,1,-1) (functional_cv2.py:682-714)
--   Resize    int short-edge match, tuple (h,w); nearest floor(dx*scale)
--             / bilinear (dx+0.5)*scale-0.5 4-neighbour (resize.cpp)
--   Compose   applies transforms in order (transforms.py:117-137)
--   FashionMNIST subclass overrides FILES/NAME only (mnist.py:333-340)
--
-- Run (repo root, x64 env): set PATH=... ; lua/jit tools/smoke/vision_transforms_smoke.lua

package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "E:/code/Insight7/bindings/lua/?.lua;E:/code/Insight7/bindings/lua/?/init.lua;" .. package.path
package.cpath = "E:/code/Insight7/build/bindings/lua/?.dll;" .. package.cpath
package.cpath = "E:/code/Insight7/build/backends/cpu/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()

local paddle = require "paddle"
local T = paddle.vision.transforms
local insight = require "insight"

local DATA = "build-test/mnist-data"
local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end
local function close_to(a, b, eps)
  return math.abs(a - b) <= (eps or 1e-5)
end

-- =====================================================================
-- ToTensor: uint8 [H,W] -> [1,H,W] float32, values /255
-- mirror functional_cv2.py:27-57 -- build the same pipeline by hand
-- =====================================================================
local raw = { { 0, 255, 17 }, { 128, 64, 200 } }   -- 2x3 uint8
local arr = insight.from_dlpack({ 2, 3 }, "u1",
  string.char(0, 255, 17, 128, 64, 200))
local t = T.ToTensor{}(arr)
check(t:ndim() == 3, "ToTensor 2D -> 3D")
local shp = t:shape()
check(shp[1] == 1 and shp[2] == 2 and shp[3] == 3,
  ("ToTensor shape {1,2,3} (got %d,%d,%d)"):format(shp[1], shp[2], shp[3]))
-- item(i) is 1-based flat index; element (r,c) of CHW [1,2,3] sits at
-- r*3+c in row-major. element [2,2] = 64/255 -- verify via to_host_bytes
local hb = t:to_host_bytes()
local v22 = string.byte(hb, (1 * 3 + 1) * 4 + 1)         -- float32 byte[6]
local v22f = string.unpack and string.unpack("<f", hb, 1 + 5 * 4)
-- LuaJIT has no string.unpack: decode manually (host little-endian)
local function f32_at(s, idx)
  local b1, b2, b3, b4 = s:byte(idx, idx + 3)
  local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local sign = bits >= 2147483648 and 1 or 0
  local exp = math.floor(bits / 8388608) % 256
  local mant = bits % 8388608
  if exp == 0 then return 0.0 end
  if exp == 255 then return math.huge end
  local v = (1 + mant / 8388608) * 2 ^ (exp - 127)
  return sign == 1 and -v or v
end
check(close_to(f32_at(hb, 1 + 1 * 4), 1.0, 1e-6), "ToTensor [1,1] == 1.0 (255/255)")
check(close_to(f32_at(hb, 1 + 4 * 4), 64 / 255, 1e-6),
  ("ToTensor [2,2] == 64/255 (got %s)"):format(f32_at(hb, 1 + 4 * 4)))

-- Explicit CHW vs HWC order
local t_hwc = T.ToTensor{ data_format = "HWC" }(arr)
local shp_hwc = t_hwc:shape()
check(shp_hwc[1] == 2 and shp_hwc[2] == 3 and shp_hwc[3] == 1,
  "ToTensor HWC keeps (2,3,1)")

-- =====================================================================
-- Normalize: (x-mean)/std with scalar expansion + channel broadcast
-- =====================================================================
local x3 = insight.Array{ 0.5, 0.25, 0.75 }:reshape{ 3, 1, 1 }
local n = T.Normalize{ mean = 0.5, std = 0.5 }(x3)
-- scalar reads: Array:get(idx) is the raw 0-based flat accessor
-- (insight_lua.cpp:685, no -1 like the 1-based [] metamethod)
check(close_to(n:get(0), 0.0, 1e-9), "Normalize ch1 (0.5-0.5)/0.5 == 0")
check(close_to(n:get(1), -0.5, 1e-9), "Normalize ch2 (0.25-0.5)/0.5 == -0.5")
check(close_to(n:get(2), 0.5, 1e-9), "Normalize ch3 (0.75-0.5)/0.5 == 0.5")

-- HWC path: mean broadcast over (1,1,3)
local xhwc = insight.Array{ 0.1, 0.2, 0.3 }:reshape{ 1, 1, 3 }
local nh = T.Normalize{ mean = { 0.0, 0.1, 0.2 }, std = 1.0,
                        data_format = "HWC" }(xhwc)
check(close_to(nh:get(0), 0.1, 1e-9), "Normalize HWC ch1 == 0.1")
check(close_to(nh:get(1), 0.1, 1e-9), "Normalize HWC ch2 == 0.1")
check(close_to(nh:get(2), 0.1, 1e-9), "Normalize HWC ch3 == 0.1")

-- =====================================================================
-- Resize: nearest + bilinear vs hand-computed reference
-- =====================================================================
-- 2x2 grayscale: [[0,255],[128,64]] -> 4x4 nearest (scale=2, floor(dx*2))
local src2 = string.char(0, 255, 128, 64)
local r_n = core.image_resize(src2, 2, 2, 1, core.PD_DTYPE_UINT8,
                              4, 4, 0)   -- nearest
check(#r_n == 16, "image_resize nearest output size 4x4")
check(r_n:byte(1) == 0 and r_n:byte(2) == 0, "resize nearest row1 col1-2")
check(r_n:byte(3) == 255 and r_n:byte(4) == 255, "resize nearest row1 col3-4")
check(r_n:byte(9) == 128 and r_n:byte(15) == 64, "resize nearest row3/row4")

-- 2x2 -> 3x3 bilinear: hand-compute center pixel.
-- scale = 2/3; for dst (1,1): fx=(1+0.5)*0.6667-0.5=0.5 -> x0=0,x1=1,wx=0.5
-- symmetric for y: v = mean of the four corners = (0+255+128+64)/4 = 111.75
-- -> round(111.75+0.5)=112
local r_b = core.image_resize(src2, 2, 2, 1, core.PD_DTYPE_UINT8,
                              3, 3, 1)   -- bilinear
check(#r_b == 9, "image_resize bilinear output size 3x3")
check(r_b:byte(5) == 112,
  ("resize bilinear center == 112 (got %d)"):format(r_b:byte(5)))

-- Float32 path: 1x2 -> 2x3 nearest keeps exact values
local srcf = string.pack and string.pack("<ff", 1.5, -2.0)
                or (function()
                     local function fl(b, i)
                       -- little-endian f32
                       local a1,a2,a3,a4 = b:byte(i,i+3)
                       return a1 + a2*256 + a3*65536 + a4*16777216
                     end
                     local b1 = fl(string.char(0,0,192,63),1)  -- 1.5
                     return string.char(0,0,192,63, 0,0,0,192)
                   end)()
local r_f = core.image_resize(srcf, 1, 2, 1, core.PD_DTYPE_FLOAT32,
                              2, 3, 0)
check(#r_f == 24, "image_resize float32 output size 2x3")
local function rf(row, col)  -- f32 at (row,col) row-major 3 cols
  return f32_at(r_f, 1 + (row * 3 + col) * 4)
end
check(close_to(rf(0, 0), 1.5, 1e-6), "resize f32 nearest v==1.5")
check(close_to(rf(1, 2), -2.0, 1e-6), "resize f32 nearest v==-2.0")

-- Transform-level Resize: Array -> Array (dtype/shape preserved)
local arr28 = insight.Array(raw):to(insight.uint8)  -- 2x3
local rs = T.Resize{ size = { 4, 6 }, interpolation = "nearest" }(arr28)
check(rs.shape[1] == 4 and rs.shape[2] == 6, "Resize transform output shape")
check(rs.dtype == insight.uint8, "Resize transform keeps dtype")

-- int size: short-edge match (2x3, h<w -> oh=4, ow=floor(4*3/2)=6)
local rs2 = T.Resize{ size = 4, interpolation = "nearest" }(arr28)
check(rs2.shape[1] == 4 and rs2.shape[2] == 6, "Resize int short-edge match")

-- =====================================================================
-- Compose: applies in order (transform -> array -> tensor)
-- =====================================================================
local pipe = T.Compose{ transforms = {
  T.Normalize{ mean = { 0.5 }, std = { 0.5 } },
  T.ToTensor{},
} }
-- a 1x1x1 (C,H,W) float [[1.0]] -> normalize (1-0.5)/0.5 = 1.0 ->
-- ToTensor (not uint8: no /255) -> single-element Tensor, item() == 1.0
local one = insight.Array{ 1.0 }:reshape{ 1, 1, 1 }
local out = pipe(one)
check(out:ndim() == 3, "Compose output is a tensor (boundary)")
check(out:numel() == 1, "Compose keeps single element")
check(close_to(out:item(), 1.0, 1e-6),
  ("Compose normalize+tensor [[1.0]] == 1.0 (got %s)"):format(out:item()))

-- =====================================================================
-- FashionMNIST: subclass overrides FILES/NAME; parse path identical to
-- MNIST (IDX), so feed the local MNIST .gz to prove inheritance.
-- =====================================================================
local F = paddle.vision.datasets.FashionMNIST
check(F.NAME == "fashion-mnist", "FashionMNIST.NAME == fashion-mnist")
check(F.URL_PREFIX == "https://dataset.bj.bcebos.com/fashion_mnist/",
  "FashionMNIST URL prefix matches mnist.py:333-340")
check(F.FILES.train.image.md5 == "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
  "FashionMNIST train image md5 matches mnist.py:336")
check(F.FILES.test.image.md5 == "bef4ecab320f06d8554ea6380940ec79",
  "FashionMNIST test image md5 matches mnist.py:333-334")

local fds = F{
  image_path = DATA .. "/train-images-idx3-ubyte.gz",
  label_path = DATA .. "/train-labels-idx1-ubyte.gz",
}
check(fds:len() == 60000, "FashionMNIST train len == 60000")
local fx, fy = fds:get(1)
check(type(fx) == "userdata", "FashionMNIST get(1) returns Array")
check(fx.shape[1] == 28 and fx.shape[2] == 28, "FashionMNIST image 28x28")
check(fy >= 1 and fy <= 10, "FashionMNIST label 1..10 (C4)")
check(fx.dtype == insight.uint8, "FashionMNIST image uint8 (pil mirror)")

-- inherited bad-mode guard still works
local ok, err = pcall(function() return F{ mode = "val" } end)
check(not ok, "FashionMNIST bad mode raises")
check(tostring(err):find("mode", 1, true) ~= nil, "error mentions mode")

print(("VISION_TRANSFORMS_SMOKE PASS (%d assertions)"):format(ntests))