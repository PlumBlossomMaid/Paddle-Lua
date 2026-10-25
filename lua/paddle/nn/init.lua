-- lua/paddle/nn/init.lua -- paddle.nn module entry
-- Stage: P9 / M1 (WORKPLAN 2.9.1+). First submodule of nn: Layer base.
-- Layers/containers/functional/initializer land in later 2.9.x leaves
-- (09-nn.md sec.3.3 list); this file is where they mount.
--
-- Lazy-loaded from the top-level paddle table (api/README.md sec.2.5):
-- require "paddle" does not pull nn; first access to paddle.nn does.

local layer = require "paddle.nn.layer"
local linear = require "paddle.nn.layers.linear"
local conv = require "paddle.nn.layers.conv"
local activation = require "paddle.nn.layers.activation"
local pooling = require "paddle.nn.layers.pooling"
local norm = require "paddle.nn.layers.norm"
local loss = require "paddle.nn.layers.loss"
local common = require "paddle.nn.layers.common"
local functional = require "paddle.nn.functional"

local M = {}

-- Layer base class (2.9.1). Subclasses: class(paddle.nn.Layer) then
-- assign fields -- auto-registration takes over (09-nn.md sec.3.2).
M.Layer = layer.Layer

-- First real layer (2.9.2): proves auto-registration + create_parameter
-- + forward end to end (api/nn.md sec.2.3).
M.Linear = linear

-- 2.9.4 层类(MNIST 验收路径):Conv2D/ReLU/BatchNorm2D/
-- CrossEntropyLoss/Flatten/Dropout/Embedding。构造函数全部 argrule
-- 包装(self 规则,用户 2026-10-25 纪律)。
M.Conv2D = conv
M.ReLU = activation
M.BatchNorm2D = norm
M.CrossEntropyLoss = loss
M.Flatten = common.Flatten
M.Dropout = common.Dropout
M.Embedding = common.Embedding

-- 2.9.4 池化层类全 12 个(pooling.lua,forward 转发 functional 2.9.3.3)
M.AvgPool1D = pooling.AvgPool1D
M.AvgPool2D = pooling.AvgPool2D
M.AvgPool3D = pooling.AvgPool3D
M.MaxPool1D = pooling.MaxPool1D
M.MaxPool2D = pooling.MaxPool2D
M.MaxPool3D = pooling.MaxPool3D
M.AdaptiveAvgPool1D = pooling.AdaptiveAvgPool1D
M.AdaptiveAvgPool2D = pooling.AdaptiveAvgPool2D
M.AdaptiveAvgPool3D = pooling.AdaptiveAvgPool3D
M.AdaptiveMaxPool1D = pooling.AdaptiveMaxPool1D
M.AdaptiveMaxPool2D = pooling.AdaptiveMaxPool2D
M.AdaptiveMaxPool3D = pooling.AdaptiveMaxPool3D

-- Functionals (2.9.3): activation + more, all argrule-wrapped (D31;
-- api/nn-functional.md). First batch = 16 activations.
M.functional = functional

-- 2.9.1 internals exposed for tests/CI greps (rawset discipline).
M._FIELDS = layer.FIELDS

return M
