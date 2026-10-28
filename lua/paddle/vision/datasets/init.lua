-- lua/paddle/vision/datasets/init.lua -- paddle.vision.datasets
-- Stage: M2 / 3.1.2. Contract: api/vision.md sec 1.1 / sec 2.1.
-- Lua 5.1 subset (C3).

local M = {}

M.MNIST = require("paddle.vision.datasets.mnist").MNIST
M.FashionMNIST = require("paddle.vision.datasets.fashion_mnist").FashionMNIST

return M