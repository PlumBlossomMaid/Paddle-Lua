-- lua/paddle/vision/datasets/cifar10.lua -- paddle.vision.datasets.Cifar10
-- Stage: M2 / 3.4 (WORKPLAN 3.4, T-M2-05). Contract: api/vision.md sec 2.1
--   -- note: the Cifar10/100 rows are listed in vision.md:24/183 as P12
--   -- deliverables; this file is the P12 remaining piece for M2 acceptance
--   -- (ResNet18 on CIFAR-10, 10 epochs > 80%, roadmap.md:239).
-- Upstream: python/paddle/vision/datasets/cifar.py (read 2026-10-29,
-- full 308 lines). Semantics mirrored:
--   - URL/MD5 constants cifar.py:22-23 (bcebos, cifar-10-python.tar.gz)
--   - MODE_FLAG_MAP cifar.py:25-29: train10 -> 'data_batch' (5 files),
--     test10 -> 'test_batch' (1 file)
--   - _load_data cifar.py:141-158: tarfile open, keep members whose NAME
--     contains the flag, sorted, pickle.load(encoding='bytes') each
--   - __getitem__ cifar.py:163-175: sample (row of b'data', 3072 bytes =
--     CHW 3x32x32) -> reshape [3,32,32] -> transpose [1,2,0] -> HWC
--     [32,32,3] uint8 (mirrors upstream default backend='pil')
-- The tar container is NOT gzip'd at the member level: cifar-10-python
-- .tar.gz is a gzip stream of a POSIX tar file; Lua side decompresses
-- with core.gzip_inflate then walks the tar blocks itself (512-byte
-- headers; verified against the real 170MB archive 2026-10-29).
--
-- Pickle: data_batch_* / test_batch are PYTHON 2 pickles (protocol 2).
-- lua-pickle decodes them (dict keys are bytes-tagged; numpy uint8 array
-- -> insight.Array via Insight7 DLPack). A py2 bytes-dtype deficiency was
-- found and fixed upstream in lua-pickle (ndarray.lua as_str unwrap,
-- 2026-10-29, verified against the real archive + busted 14/14).
--
-- Lua mapping (api/vision.md sec 2.1, mirrors mnist.lua):
--   paddle.vision.datasets.Cifar10{ mode="train", data_file,
--                                    transform, download }
--   ds:len() -> 50000 / 10000 ; ds:get(i) -> (x, y), i 1-based
--   x = insight.Array [32,32,3] uint8 HWC (D26; ToTensor(0-based-compat
--       CHW) then folds to a float32 [3,32,32] Tensor, /255)
--   y = label 1-based (C4: upstream 0..9 -> +1)
--
-- Lua 5.1 subset (C3).

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"

local M = {}

-- ---- constants (python/paddle/vision/datasets/cifar.py:22-29) ------
-- Class fields (mnist.lua:115-123 pattern) so Cifar100 can override
-- wholesale: cifar.py:226 class Cifar100(Cifar10) only re-points
-- URL/MD5 + MODE_FLAG_MAP entries (cifar.py:24-25 + :277-283).
local URL_PREFIX = "https://dataset.bj.bcebos.com/cifar/"
local CIFAR10_URL = URL_PREFIX .. "cifar-10-python.tar.gz"
local CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

-- tar member names inside cifar-10-python.tar.gz:
--   cifar-10-batches-py/data_batch_{1..5}  (train, 50000 rows)
--   cifar-10-batches-py/test_batch         (test,  10000 rows)
-- The flag is a SUBSTRING of the member name (cifar.py:142-143 filters
-- with `self.flag in each_item.name`).
local MODE_FLAG = { train = "data_batch", test = "test_batch" }

-- Default cache root mirrors mnist.lua (common.py:51-55 + NAME subdir).
local function default_cache_dir(self)
  local home = os.getenv("USERPROFILE") or os.getenv("HOME") or "."
  local custom = os.getenv("PADDLE_LUA_DATASET_HOME")
  if custom and custom ~= "" then
    return custom .. "/" .. self.NAME
  end
  return home .. "/.cache/paddle/dataset/" .. self.NAME
end

local function read_binary(path)
  local f, err = io.open(path, "rb")
  if not f then
    error("paddle.vision.datasets.Cifar10: cannot open " .. tostring(path)
          .. ": " .. tostring(err), 3)
  end
  local data = f:read("*a")
  f:close()
  return data
end

-- POSIX ustar walker over the gunzipped tar byte stream. Returns a list
-- of { name = <member path>, data = <raw member bytes> } for members
-- whose name contains `want` (mirrors cifar.py:142-143 membership test).
-- 512-byte headers: name[0..100), size at offset 124 (octal ASCII), type
-- flag at 156. End-of-archive = a zero block. Verified against the real
-- cifar-10-python.tar.gz: 8 members (6 data sets + readme + meta).
local function parse_tar_members(blob, want)
  local out, pos, n = {}, 1, #blob
  while pos + 512 <= n do
    local hdr = blob:sub(pos, pos + 511)
    local name = hdr:match("^(.-)%z")
    if not name or name == "" then break end          -- zero block
    local size_str = hdr:sub(125, 136):match("^(.-)%z")
    local size = 0
    if size_str then
      for i = 1, #size_str do                          -- octal digits 0-7
        local c = size_str:byte(i)
        if c >= 48 and c <= 55 then size = size * 8 + (c - 48) end
      end
    end
    local typeflag = hdr:sub(157, 157)
    if (typeflag == "0" or typeflag == " " or typeflag == "\0")
      and name:find(want, 1, true) then
      out[#out + 1] = { name = name,
                        data = blob:sub(pos + 512, pos + 512 + size - 1) }
    end
    pos = pos + 512 + math.floor((size + 511) / 512) * 512
  end
  return out
end

-- pickle.loads -> dict; CIFAR keys are PY2 str = bytes tags
-- {__pd_kind="bytes", data=<key>} (lua-pickle types.lua). Find the
-- value whose key is the wanted bytes.
local function by_bytes_key(dict, want)
  for k, v in pairs(dict) do
    if type(k) == "table" and k.__pd_kind == "bytes" and k.data == want then
      return v
    end
  end
  return nil
end

local function make_downloader(download)
  if download ~= nil and type(download) == "table" then
    return function(url, path, md5)
      download{ url = url, path = path, md5 = md5 }
    end
  end
  return function(url, path, md5)
    local ok, utils = pcall(require, "paddle.utils.download")
    if not ok then
      error("paddle.vision.datasets.Cifar10: needs paddle.utils.download "
            .. "(luasocket+luasec rocks; failed to load: " .. tostring(utils)
            .. ")", 3)
    end
    utils.download{ url = url, path = path, md5 = md5 }
  end
end

-- ---- Cifar10: io.Dataset subclass --------------------------------------

local Cifar10 = class(require("paddle.io").Dataset)
Cifar10._name = "Cifar10"
-- Class fields (Cifar100 = class(Cifar10) overrides, cifar.py:226).
Cifar10.URL = CIFAR10_URL
Cifar10.MD5 = CIFAR10_MD5
Cifar10.NAME = "cifar"
Cifar10.FLAGS = MODE_FLAG

Cifar10._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "mode", type = argrule.string, opt = true, default = "train",
    doc = "'train' (50000) or 'test' (10000)" },
  { name = "data_file", type = argrule.string, opt = true,
    doc = "path to the tar.gz archive; nil = download to cache" },
  { name = "transform", type = argrule.table, opt = true,
    doc = "callable applied to x in get(i)" },
  { name = "download", type = argrule.table, opt = true,
    doc = "injectable download fn (testing); default = utils.download" },
}(function(self, mode, data_file, transform, download)
  if mode ~= "train" and mode ~= "test" then
    error("Cifar10: mode must be 'train' or 'test', got "
          .. tostring(mode), 3)
  end
  local cls = getmetatable(self)
  local flag = cls.FLAGS[mode]
  local cache = default_cache_dir(self)

  local dl = make_downloader(download)
  if data_file == nil then
    local name = cls.URL:match("([^/]+)$")
    data_file = cache .. "/" .. name
    dl(cls.URL, data_file, cls.MD5)
  end

  -- gunzip + walk tar, then pickle.loads each relevant member
  local tar = paddle_core.gzip_inflate(read_binary(data_file))
  local members = parse_tar_members(tar, flag)
  if #members == 0 then
    error(("Cifar10: no tar member matches flag '%s'"):format(flag), 3)
  end

  -- Determine a stable member ORDER (cifar.py:144-146 sorts by name so
  -- data_batch_10 sorts before data_batch_2 numerically like Python's
  -- sorted() on strings -- for Cifar10 flags this is just 1..5 lexicographic).
  table.sort(members, function(a, b) return a.name < b.name end)

  local ok_pickle, pickle = pcall(require, "pickle")
  if not ok_pickle then
    error("Cifar10: needs the 'pickle' rock (lua-pickle) to decode the "
          .. "CIFAR archive: " .. tostring(pickle), 3)
  end

  -- Concatenate all rows (uint8, 3072 bytes each) into one byte string +
  -- one labels table. Memory: 50000x3072 = 153.6MB resident for 'train'
  -- (same tradeoff as upstream np uint8 arrays; 24GB boxes fine).
  local n_img, img_bytes, lab = 0, {}, {}
  for _, m in ipairs(members) do
    local batch = pickle.loads(m.data)
    local data_arr = by_bytes_key(batch, "data")
    local labels = by_bytes_key(batch, "labels")
      or by_bytes_key(batch, "fine_labels")
    if data_arr == nil or labels == nil then
      error("Cifar10: " .. m.name .. " missing b'data'/labels keys", 3)
    end
    -- numpy uint8 [10000,3072] -> insight.Array (native mapping) or raw
    -- bytes. Grab the raw row bytes once via DLPack export.
    local raw
    if type(data_arr) == "userdata" then
      local ok_ins, ins = pcall(require, "insight")
      if not ok_ins then
        error("Cifar10: needs Insight7 (D26) to materialize numpy arrays: "
              .. tostring(ins), 3)
      end
      raw = ins.to_dlpack(data_arr):bytes()
    elseif type(data_arr) == "table" and data_arr.data then
      raw = data_arr.data           -- degraded: raw bytes kept by lua-pickle
    else
      error("Cifar10: unhandled b'data' form " .. type(data_arr), 3)
    end
    local rows = #raw / 3072
    if rows ~= #labels then
      error(("Cifar10: %s rows %d != labels %d"):format(m.name, rows,
            #labels), 3)
    end
    n_img = n_img + rows
    img_bytes[#img_bytes + 1] = raw
    for j = 1, #labels do lab[#lab + 1] = labels[j] end
  end

  self._num = n_img
  self._img = table.concat(img_bytes)
  self._lab = lab
  self._flags = flag
  self._transform = transform
end)

function Cifar10:len()
  return self._num
end

-- x = insight.Array [32,32,3] uint8 HWC, mirroring upstream pil backend
-- (cifar.py:163-175 reshape CHW + transpose -> HWC, uint8; the default
-- get_image_backend() is 'pil'). Row i raw bytes are CHW 3x32x32; build
-- the HWC view with one DLPack import (CHW) + one permute (1-based axis
-- {2,3,1} = H,W,C order) -- pure Insight7, no per-byte Lua loop.
function Cifar10:get(i)
  assert(i >= 1 and i <= self._num,
    ("Cifar10: index %d out of range [1,%d]"):format(i, self._num))
  local ok_ins, ins = pcall(require, "insight")
  if not ok_ins then
    error("Cifar10: needs Insight7 (D26) to produce image arrays: "
          .. tostring(ins), 3)
  end
  local off = (i - 1) * 3072
  local row = self._img:sub(off + 1, off + 3072)
  local chw = ins.from_dlpack({ 3, 32, 32 }, "u1", row)
  local hwc = ins.permute(chw, { 2, 3, 1 })
  local x = hwc
  if self._transform ~= nil then
    x = self._transform(x)
  end
  return x, self._lab[i] + 1    -- label 1-based (C4; upstream 0..9)
end

M.Cifar10 = Cifar10
return M