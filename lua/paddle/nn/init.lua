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
local containers = require "paddle.nn.containers"
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
-- 2.9.4.2b:conv.lua 泛化为 1D/2D/3D + Transpose 六个变体(表导出)
M.Conv1D = conv.Conv1D
M.Conv2D = conv.Conv2D
M.Conv3D = conv.Conv3D
M.Conv1DTranspose = conv.Conv1DTranspose
M.Conv2DTranspose = conv.Conv2DTranspose
M.Conv3DTranspose = conv.Conv3DTranspose
-- 2.9.4.2b:activation.lua 批量激活层类(表导出)。ReLU 原单类→表中。
M.CELU = activation.CELU
M.ELU = activation.ELU
M.GELU = activation.GELU
M.Hardshrink = activation.Hardshrink
M.Hardswish = activation.Hardswish
M.Tanh = activation.Tanh
M.Hardtanh = activation.Hardtanh
M.PReLU = activation.PReLU
M.RReLU = activation.RReLU
M.ReLU = activation.ReLU
M.ReLU6 = activation.ReLU6
M.SELU = activation.SELU
M.LeakyReLU = activation.LeakyReLU
M.Sigmoid = activation.Sigmoid
M.Hardsigmoid = activation.Hardsigmoid
M.Softplus = activation.Softplus
M.Softshrink = activation.Softshrink
M.Softsign = activation.Softsign
M.Swish = activation.Swish
M.Mish = activation.Mish
M.Tanhshrink = activation.Tanhshrink
M.ThresholdedReLU = activation.ThresholdedReLU
M.Silu = activation.Silu
M.LogSigmoid = activation.LogSigmoid
M.Softmax = activation.Softmax
M.LogSoftmax = activation.LogSoftmax
M.Maxout = activation.Maxout
M.Softmax2D = activation.Softmax2D
-- 2.9.4.2b:norm.lua 泛化为 BatchNorm1D/2D/3D + LayerNorm + GroupNorm
M.BatchNorm1D = norm.BatchNorm1D
M.BatchNorm2D = norm.BatchNorm2D
M.BatchNorm3D = norm.BatchNorm3D
M.LayerNorm = norm.LayerNorm
M.GroupNorm = norm.GroupNorm
-- 2.9.4.2b:loss.lua 表导出(CrossEntropyLoss + MSELoss/L1Loss/
-- SmoothL1Loss/BCELoss/BCEWithLogitsLoss,forward 转发 functional)
M.CrossEntropyLoss = loss.CrossEntropyLoss
M.MSELoss = loss.MSELoss
M.L1Loss = loss.L1Loss
M.SmoothL1Loss = loss.SmoothL1Loss
M.BCELoss = loss.BCELoss
M.BCEWithLogitsLoss = loss.BCEWithLogitsLoss
M.Flatten = common.Flatten
M.Dropout = common.Dropout
M.Dropout2D = common.Dropout2D
M.Dropout3D = common.Dropout3D
M.Embedding = common.Embedding

-- 2.9.5 容器(containers.lua,D27 全继承 Layer)
M.Sequential = containers.Sequential
M.LayerList = containers.LayerList
M.LayerDict = containers.LayerDict

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
