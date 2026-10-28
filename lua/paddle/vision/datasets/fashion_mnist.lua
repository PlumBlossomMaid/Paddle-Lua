-- lua/paddle/vision/datasets/fashion_mnist.lua -- paddle.vision.datasets.FashionMNIST
-- Stage: M2 / 3.1.3 (WORKPLAN 3.1.3). Contract: api/vision.md sec 1.1/2.1.
-- Upstream: python/paddle/vision/datasets/mnist.py:280-340 (class
-- FashionMNIST(MNIST): overrides NAME/URL_PREFIX + four URL/MD5 pairs,
-- mnist.py:333-340). Everything else (parse / get / transform hook) is
-- inherited from MNIST -- same IDX format, 10 classes, 28x28.
--
-- Lua mapping mirrors mnist.lua: class fields FILES/NAME override on
-- the subclass table; MNIST._init resolves getmetatable(self).FILES
-- through the pl.class __index chain (mnist.lua:115-123).
-- Lua 5.1 subset (C3).

local class = require "pl.class"
local MNIST = require("paddle.vision.datasets.mnist").MNIST

local M = {}

local FashionMNIST = class(MNIST)
FashionMNIST._name = "FashionMNIST"
-- mnist.py:333-340 (read 2026-10-28): URL_PREFIX + 4 URL/MD5 pairs
FashionMNIST.URL_PREFIX = "https://dataset.bj.bcebos.com/fashion_mnist/"
FashionMNIST.FILES = {
  train = {
    image = { url = FashionMNIST.URL_PREFIX .. "train-images-idx3-ubyte.gz",
              md5 = "8d4fb7e6c68d591d4c3dfef9ec88bf0d" },
    label = { url = FashionMNIST.URL_PREFIX .. "train-labels-idx1-ubyte.gz",
              md5 = "25c81989df183df01b3e8a0aad5dffbe" },
  },
  test = {
    image = { url = FashionMNIST.URL_PREFIX .. "t10k-images-idx3-ubyte.gz",
              md5 = "bef4ecab320f06d8554ea6380940ec79" },
    label = { url = FashionMNIST.URL_PREFIX .. "t10k-labels-idx1-ubyte.gz",
              md5 = "bb300cfdad3c16e7a12a480ee83cd310" },
  },
}
FashionMNIST.NAME = "fashion-mnist"

M.FashionMNIST = FashionMNIST
return M