-- tools/smoke/vision_cifar_smoke.lua -- paddle.vision.datasets.Cifar10
-- Stage: M2 / 3.4 (T-M2-05). Mirrors python test/dataset/cifar10_test.py
-- semantics (50000 train / 10000 test + basic instance access) with
-- byte-level spot checks against the raw archive (same rigor as
-- vision_mnist_smoke):
--   - reads the real cifar-10-python.tar.gz from the default dataset
--     cache (downloaded once by utils.download; UTILS_DOWNLOAD_SMOKE
--     covers transport); if absent, downloads it (170MB, bcebos)
--   - len(train)=50000 / len(test)=10000
--   - x is insight.Array [32,32,3] uint8 HWC (mirror upstream default
--     backend='pil', cifar.py:163-175 reshape+transpose)
--   - label 1-based (C4; upstream 0..9)
--   - spot pixels: the dataset permutes CHW -> HWC, so the HWC origin
--     pixel (h=0,w=0) = archive bytes raw[0], raw[1024], raw[2048]
--     (channel planes are 1024 apart in the CHW layout). Compare via
--     Array:get(idx) (0-based flat accessor, Insight7).
--
-- Run (repo root, x64 env): set PATH=... ; luajit tools/smoke/vision_cifar_smoke.lua

package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath
-- lua-pickle (CIFAR archive is py2 pickle)
package.path = "E:/code/lua-pickle/src/?.lua;E:/code/lua-pickle/src/?/init.lua;" .. package.path
-- Insight7 (image producer D26)
package.path = "E:/code/Insight7/bindings/lua/?.lua;E:/code/Insight7/bindings/lua/?/init.lua;" .. package.path
package.cpath = "E:/code/Insight7/build/bindings/lua/?.dll;" .. package.cpath
package.cpath = "E:/code/Insight7/build/backends/cpu/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()

local paddle = require "paddle"
local pickle = require "pickle"
local ins = require "insight"

-- same archive the dataset uses (default cache dir)
local home = os.getenv("USERPROFILE") or os.getenv("HOME")
local TAR = home .. "/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz"

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- 1) len per split
local test = paddle.vision.datasets.Cifar10{ data_file = TAR, mode = "test" }
check(test:len() == 10000, ("test len == 10000 (got %d)"):format(test:len()))

local train = paddle.vision.datasets.Cifar10{ data_file = TAR, mode = "train" }
check(train:len() == 50000, ("train len == 50000 (got %d)"):format(train:len()))

-- 2) get(i): x is insight.Array [32,32,3] uint8 HWC (upstream pil
--    backend: reshape [3,32,32] + transpose [1,2,0] -> [32,32,3]),
--    label 1-based
local x, y = train:get(1)
check(type(x) == "userdata", "get(1) x is userdata (insight.Array)")
check(x.shape[1] == 32 and x.shape[2] == 32 and x.shape[3] == 3,
      ("x shape == {32,32,3} (got {%d,%d,%d})")
        :format(x.shape[1], x.shape[2], x.shape[3]))
check(x.nbytes == 32 * 32 * 3, "x holds 3072 uint8 samples (HWC)")
check(y >= 1 and y <= 10, ("label 1-based (got %d)"):format(y))

-- 3) spot pixels vs the raw archive. Ground truth: pickle-decoded
--    ndarray row 0 (3072 bytes, CHW layout c*1024 + h*32 + w).
--    dataset CHW -> HWC permute means HWC origin pixel =
--    (raw[0], raw[1024], raw[2048]); read back via Array:get (0-based).
local function raw_row0()
  -- reuse the tar walker minimally: find test_batch (small, 1.5MB)
  local function parse_tar_members(blob, want)
    local out, pos, n = {}, 1, #blob
    while pos + 512 <= n do
      local hdr = blob:sub(pos, pos + 511)
      local name = hdr:match("^(.-)%z")
      if not name or name == "" then break end
      local size_str = hdr:sub(125, 136):match("^(.-)%z")
      local size = 0
      if size_str then
        for i = 1, #size_str do
          local c = size_str:byte(i)
          if c >= 48 and c <= 55 then size = size * 8 + (c - 48) end
        end
      end
      local tf = hdr:sub(157, 157)
      if (tf == "0" or tf == " " or tf == "\0") and name:find(want, 1, true) then
        out[#out + 1] = { name = name,
                          data = blob:sub(pos + 512, pos + 512 + size - 1) }
      end
      pos = pos + 512 + math.floor((size + 511) / 512) * 512
    end
    return out
  end
  local f = assert(io.open(TAR, "rb"))
  local gz = f:read("*a")
  f:close()
  local tar = core.gzip_inflate(gz)
  local batch1 = parse_tar_members(tar, "data_batch_1")[1]
  local obj = pickle.loads(batch1.data)
  local data_arr, labels = nil, nil
  for k, v in pairs(obj) do
    if type(k) == "table" and k.__pd_kind == "bytes" then
      if k.data == "data" then data_arr = v end
      if k.data == "labels" then labels = v end
    end
  end
  local raw_all = ins.to_dlpack(data_arr):bytes()   -- module-level API
  return raw_all, labels
end
local raw_all, labels = raw_row0()
check(#raw_all == 10000 * 3072, "data_batch_1 has 10000*3072 bytes")

local x1 = train:get(1)
check(x1:get(0) == raw_all:byte(1),     "HWC origin R == archive raw[0]")
check(x1:get(1) == raw_all:byte(1025),  "HWC origin G == archive raw[1024]")
check(x1:get(2) == raw_all:byte(2049),  "HWC origin B == archive raw[2048]")
-- label parity (archive labels are 0-based; ours 1-based)
check(y == labels[1] + 1, ("label[1] parity (got %d, archive %d)")
        :format(y, labels[1]))

-- 4) transform hook runs (ToTensor folds to a float32 CHW tensor)
local xd, yd = train:get(1)
local tx = paddle.vision.transforms.ToTensor{}
local t = tx(xd)
check(type(t) == "userdata" and t:ndim() == 3, "ToTensor on Cifar image -> 3-D tensor")
check(t:shape()[1] == 3 and t:shape()[2] == 32 and t:shape()[3] == 32,
      "tensor shape == {3,32,32}")
-- read the first HWC pixel back as float32: ToTensor did /255 on uint8,
-- so the tensor[1][1][1] (R at origin, CHW) must equal raw byte 1 / 255.
-- Lua 5.1 has no string.unpack; decode f32 LE by hand (pure 5.1 subset).
local function f32_from_le(b, off)
  local b1, b2, b3, b4 = b:byte(off), b:byte(off + 1),
                          b:byte(off + 2), b:byte(off + 3)
  local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local sign = (bits >= 0x80000000) and -1.0 or 1.0
  local exp = math.floor((bits % 0x80000000) / 0x800000)
  local mant = bits % 0x800000
  if exp == 0 then
    return sign * (mant / 0x800000) * 2 ^ -126
  elseif exp == 255 then
    return sign * (mant == 0 and math.huge or 0 / 0)
  end
  return sign * (1 + mant / 0x800000) * 2 ^ (exp - 127)
end
local dlm2 = ins.to_dlpack(xd)   -- module-level to_dlpack (insight_lua.cpp:981)
local expected_first = raw_all:byte(1) / 255.0
local tconv = t:to_host_bytes()
local first_f32 = f32_from_le(tconv, 1)
check(math.abs(first_f32 - expected_first) < 1e-6,
      ("ToTensor /255 R == %.4f (got %.4f)")
        :format(expected_first, first_f32))

-- 5) bad mode -> clear error
local okm, errm = pcall(function()
  return paddle.vision.datasets.Cifar10{ data_file = TAR, mode = "val" }
end)
check(not okm, "bad mode raises")
check(tostring(errm):find("mode", 1, true) ~= nil, "error mentions mode")

print(("VISION_CIFAR_SMOKE PASS (%d assertions)"):format(ntests))