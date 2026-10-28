-- tools/smoke/vision_mnist_smoke.lua -- paddle.vision.datasets.MNIST
-- Stage: M2 / 3.1.2. Mirrors python test/dataset/mnist_test.py (60000
-- train / 10000 test instances, max label) + core facts:
--   - downloads handled by paddle.utils.download (not re-tested here;
--     UTILS_DOWNLOAD_SMOKE covers the transport). We feed local .gz
--     files (build-test/mnist-data) so this runs offline.
--   - .gz -> core.gzip_inflate -> IDX big-endian parse, image [28,28]
--     as insight.Array (D26), label 1-based (C4; upstream 0..9).
--   - byte-identity: decompressed payload must equal the reference
--     .ubyte files byte for byte.
--
-- Run (repo root, x64 env): set PATH=... ; lua/jit tools/smoke/vision_mnist_smoke.lua

package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath
-- Insight7 (soft dep, image producer D26): Lua bindings + native
package.path = "E:/code/Insight7/bindings/lua/?.lua;E:/code/Insight7/bindings/lua/?/init.lua;" .. package.path
package.cpath = "E:/code/Insight7/build/bindings/lua/?.dll;" .. package.cpath
package.cpath = "E:/code/Insight7/build/backends/cpu/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()

local paddle = require "paddle"

local DATA = "build-test/mnist-data"
local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- read reference .ubyte-format files (already gunzipped) as raw bytes
local function read_bin(p)
  local f = assert(io.open(p, "rb"))
  local d = f:read("*a")
  f:close()
  return d
end

-- 1) train: len 60000, decompressed payload byte-identical to reference
local train = paddle.vision.datasets.MNIST{
  mode = "train",
  image_path = DATA .. "/train-images-idx3-ubyte.gz",
  label_path = DATA .. "/train-labels-idx1-ubyte.gz",
}
check(train:len() == 60000, "train len == 60000")

-- raw payload via gzip (same path MNIST uses internally)
local img_payload = core.gzip_inflate(read_bin(DATA .. "/train-images-idx3-ubyte.gz"))
local lab_payload = core.gzip_inflate(read_bin(DATA .. "/train-labels-idx1-ubyte.gz"))
local ref_img = read_bin(DATA .. "/train-images-idx3-ubyte")
local ref_lab = read_bin(DATA .. "/train-labels-idx1-ubyte")
-- the .ubyte reference IS the full IDX (header + payload); gzip inflate
-- reproduces it wholesale -- compare in full (probe_gzip proved labels
-- byte-identical; here images too)
check(img_payload == ref_img, "train image full IDX byte-identical")
check(lab_payload == ref_lab, "train label full IDX byte-identical")

-- mirror test/dataset/mnist_test.py: max label over the whole set
-- (upstream 0..9; ours 1..10 via C4). Scan payload bytes directly --
-- skip the 8-byte IDX header (magic+num, byte 7 of header holds
-- 0xEA from num=60000), cheap, no per-sample Array construction.
local max_label = 0
for i = 9, #lab_payload do
  local v = lab_payload:byte(i)
  if v > max_label then max_label = v end
end
check(max_label == 9, ("train max raw label == 9 (got %d)"):format(max_label))

-- 2) get(i): x is insight.Array [28,28] uint8 (mirror upstream default
-- 'pil' backend, mnist.py:262-266; ToTensor's /255 keys off uint8),
-- label 1-based == refbyte+1
local x, y = train:get(1)
check(type(x) == "userdata", "get(1) x is userdata (insight.Array)")
check(x.shape[1] == 28 and x.shape[2] == 28, "x shape {28,28}")
check(x.nbytes == 784 * 1, "x holds 784 uint8 samples")
check(y == ref_lab:byte(9) + 1, ("train label[1] 1-based (got %d)"):format(y))

-- spot check 3 more samples at boundaries
for _, idx in ipairs{ 100, 30000, 60000 } do
  local xi, yi = train:get(idx)
  check(xi.shape[1] == 28 and xi.shape[2] == 28, ("x[%d] shape"):format(idx))
  check(yi == ref_lab:byte(8 + idx) + 1, ("label[%d] 1-based"):format(idx))
end

-- 3) test split: len 10000, labels 1-based
local test = paddle.vision.datasets.MNIST{
  mode = "test",
  image_path = DATA .. "/t10k-images-idx3-ubyte.gz",
  label_path = DATA .. "/t10k-labels-idx1-ubyte.gz",
}
check(test:len() == 10000, "test len == 10000")
local ref_test_lab = read_bin(DATA .. "/t10k-labels-idx1-ubyte")
for _, idx in ipairs{ 1, 5000, 10000 } do
  local xj, yj = test:get(idx)
  check(yj == ref_test_lab:byte(8 + idx) + 1, ("test label[%d] 1-based"):format(idx))
end

-- 4) bad mode -> clear error
local ok, err = pcall(function()
  return paddle.vision.datasets.MNIST{ mode = "val" }
end)
check(not ok, "bad mode raises")
check(tostring(err):find("mode", 1, true) ~= nil, "error mentions mode")

print(("VISION_MNIST_SMOKE PASS (%d assertions)"):format(ntests))