-- lua/paddle/vision/datasets/mnist.lua -- paddle.vision.datasets.MNIST
-- Stage: M2 / 3.1.2 (WORKPLAN 3.1.2). Contract: api/vision.md sec 2.1.
-- Upstream: python/paddle/vision/datasets/mnist.py (MNIST class, 2026-10-27
-- read full 340 lines). Semantics mirrored:
--   - URL/MD5 constants mnist.py:112-120 (bcebos, 4 files)
--   - download via _check_exists_and_download (mnist.py:141-158) -> keeps
--     the .gz on disk (common.py download() only fetches, no decompress)
--   - _parse_dataset (mnist.py:160-218): gzip + IDX big-endian parse
--   - __getitem__ (mnist.py:262-280): image reshape [28,28], label int64
-- Lua mapping (api/vision.md sec 2.1 / api/dataset.md sec 2.2):
--   paddle.vision.datasets.MNIST{ mode="train", image_path, label_path,
--                                  transform, backend }
--   ds:len() -> 60000/10000 ; ds:get(i) -> (x, y), i 1-based
--   x = insight.Array [28,28] (D26; Insight7 replaces numpy's position)
--   y = label 1-based (C4: upstream 0..9 -> +1)
-- Decompress: core.gzip_inflate (paddle_c.h Group 8, gzip.cc) -- Paddle
-- bundles zlib; no bindable inflate exists upstream (grepped, zero hits).
--
-- Lua 5.1 subset (C3).

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"

local M = {}

-- ---- constants (python/paddle/vision/datasets/mnist.py:112-120) ------
-- Class fields (not upvalues) so subclasses override them wholesale --
-- FashionMNIST (mnist.py:333-340) replaces URL_PREFIX + the four
-- URL/MD5 pairs on the subclass, exactly like the Python class attrs.
local URL_PREFIX = "https://dataset.bj.bcebos.com/mnist/"
local FILES = {
  train = {
    image = { url = URL_PREFIX .. "train-images-idx3-ubyte.gz",
              md5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873" },
    label = { url = URL_PREFIX .. "train-labels-idx1-ubyte.gz",
              md5 = "d53e105ee54ea40749a09fcbcd1e9432" },
  },
  test = {
    image = { url = URL_PREFIX .. "t10k-images-idx3-ubyte.gz",
              md5 = "9fb629c4189551a2d022fa330f9573f3" },
    label = { url = URL_PREFIX .. "t10k-labels-idx1-ubyte.gz",
              md5 = "ec29112dd5afa0611ce80d1b7f02629c" },
  },
}

-- Default cache root mirrors python/paddle/dataset/common.py:51-55
-- (DATA_HOME = ~/.cache/paddle/dataset, plus the module subdir).
-- NAME is a class field (mnist.py:141-158 _check_exists_and_download
-- keys on self.NAME); FashionMNIST overrides it to "fashion-mnist".
local function default_cache_dir(self)
  local home = os.getenv("USERPROFILE") or os.getenv("HOME") or "."
  local custom = os.getenv("PADDLE_LUA_DATASET_HOME")
  if custom and custom ~= "" then
    return custom .. "/" .. self.NAME
  end
  return home .. "/.cache/paddle/dataset/" .. self.NAME
end

-- read a whole file as a binary string
local function read_binary(path)
  local f, err = io.open(path, "rb")
  if not f then
    error("paddle.vision.datasets.MNIST: cannot open " .. tostring(path)
          .. ": " .. tostring(err), 3)
  end
  local data = f:read("*a")
  f:close()
  return data
end

-- big-endian u32 (IDX headers are network byte order, mnist.py:174-177)
local function be_u32(s, off)
  return s:byte(off) * 16777216 + s:byte(off + 1) * 65536
       + s:byte(off + 2) * 256 + s:byte(off + 3)
end

-- Parse an IDX byte stream (already gunzipped). Returns
-- (num, rows, cols, payload) where payload is the raw sample bytes.
-- images: magic 2051, then num/rows/cols (mnist.py:174-177)
-- labels: magic 2049, then num         (mnist.py:182-184)
local function parse_idx(all, want_magic)
  local magic = be_u32(all, 1)
  if magic ~= want_magic then
    error(("IDX bad magic 0x%08x, want 0x%08x"):format(magic, want_magic), 3)
  end
  local num = be_u32(all, 5)
  if want_magic == 2051 then
    local rows = be_u32(all, 9)
    local cols = be_u32(all, 13)
    return num, rows, cols, string.sub(all, 17)
  end
  return num, nil, nil, string.sub(all, 9)
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
      error("paddle.vision.datasets.MNIST: needs paddle.utils.download "
            .. "(luasocket+luasec rocks; failed to load: " .. tostring(utils)
            .. ")", 3)
    end
    utils.download{ url = url, path = path, md5 = md5 }
  end
end

-- ---- MNIST: io.Dataset subclass --------------------------------------

local MNIST = class(require("paddle.io").Dataset)
MNIST._name = "MNIST"
-- Class fields (FashionMNIST overrides wholesale, mnist.py:333-340):
-- the Python class attrs (URL_PREFIX + four URL/MD5 pairs) live on the
-- subclass; pl.class __index chain resolves getmetatable(self).FILES.
MNIST.URL_PREFIX = URL_PREFIX
MNIST.FILES = FILES
MNIST.NAME = "mnist"

MNIST._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "mode", type = argrule.string, opt = true, default = "train",
    doc = "'train' or 'test'" },
  { name = "image_path", type = argrule.string, opt = true,
    doc = "path to the (gz) image file; nil = download to cache" },
  { name = "label_path", type = argrule.string, opt = true,
    doc = "path to the (gz) label file; nil = download to cache" },
  { name = "transform", type = argrule.table, opt = true,
    doc = "callable applied to x in get(i) (default collate entry)" },
  { name = "download", type = argrule.table, opt = true,
    doc = "injectable download fn (testing); default = utils.download" },
}(function(self, mode, image_path, label_path, transform, download)
  if mode ~= "train" and mode ~= "test" then
    error("MNIST: mode must be 'train' or 'test', got " .. tostring(mode), 3)
  end
  -- Class fields (layer.lua:180 pattern: getmetatable(self)._name):
  -- subclasses (FashionMNIST) override FILES / NAME on their own class
  -- table and the lookup lands there first (pl.class __index chain).
  local cls = getmetatable(self)
  local files = cls.FILES[mode]
  local cache = default_cache_dir(self)

  local dl = make_downloader(download)
  local function ensure(p, spec)
    if p then return p end
    -- mirror common.py:55 filename = url basename; check then download
    local name = spec.url:match("([^/]+)$")
    local path = cache .. "/" .. name
    dl(spec.url, path, spec.md5)
    return path
  end
  local img_path = ensure(image_path, files.image)
  local lab_path = ensure(label_path, files.label)

  -- gunzip + parse (mnist.py:160-218 uses gzip.GzipFile + struct)
  local img_raw = paddle_core.gzip_inflate(read_binary(img_path))
  local lab_raw = paddle_core.gzip_inflate(read_binary(lab_path))
  local nimg, rows, cols, img_payload = parse_idx(img_raw, 2051)
  local nlab, _, _, lab_payload = parse_idx(lab_raw, 2049)
  if nimg ~= nlab then
    error(("MNIST: image count %d != label count %d"):format(nimg, nlab), 3)
  end

  self._num = nimg
  self._rows = rows
  self._cols = cols
  self._img = img_payload
  self._lab = lab_payload
  self._transform = transform
  self._has_insight = false
end)

function MNIST:len()
  return self._num
end

-- x = insight.Array [28,28] uint8 (mirror upstream DEFAULT backend:
-- get_image_backend() default is 'pil', mnist.py:262-266 astype
-- 'uint8'). D26: Insight7 replaces numpy's position. Build straight
-- from the raw bytes via DLPack import -- no per-byte Lua loop, and the
-- dtype is genuinely uint8 so ToTensor's /255 (only uint8) fires,
-- exactly like upstream MNIST(pil) + ToTensor.
local function image_to_array(self, bytes)
  local ok, insight = pcall(require, "insight")
  if not ok then
    error("MNIST: got image bytes but no insight.Array producer -- "
          .. "install Insight7 (D26) to use this dataset", 3)
  end
  return insight.from_dlpack({ self._rows, self._cols }, "u1", bytes)
end

function MNIST:get(i)
  assert(i >= 1 and i <= self._num,
    ("MNIST: index %d out of range [1,%d]"):format(i, self._num))
  local im_off = (i - 1) * self._rows * self._cols
  local img_bytes = self._img:sub(im_off + 1,
                                  im_off + self._rows * self._cols)
  local x = image_to_array(self, img_bytes)
  if self._transform ~= nil then
    x = self._transform(x)
  end
  local y = self._lab:byte(i)
  return x, y + 1    -- label 1-based (C4; upstream 0..9)
end

M.MNIST = MNIST
return M