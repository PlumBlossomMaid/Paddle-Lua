-- lua/paddle/vision/init.lua -- paddle.vision
-- Stage: M2 / 3.1.2. Contract: api/vision.md. datasets land first
-- (MNIST, M1 training); transforms/models/ops follow in 3.1.x.
-- Lua 5.1 subset (C3).

local M = {}

M.datasets = require "paddle.vision.datasets"
M.transforms = require "paddle.vision.transforms"

return M