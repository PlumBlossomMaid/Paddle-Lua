-- lua/paddle/nn/init.lua -- paddle.nn module entry
-- Stage: P9 / M1 (WORKPLAN 2.9.1+). First submodule of nn: Layer base.
-- Layers/containers/functional/initializer land in later 2.9.x leaves
-- (09-nn.md sec.3.3 list); this file is where they mount.
--
-- Lazy-loaded from the top-level paddle table (api/README.md sec.2.5):
-- require "paddle" does not pull nn; first access to paddle.nn does.

local layer = require "paddle.nn.layer"
local linear = require "paddle.nn.layers.linear"
local functional = require "paddle.nn.functional"

local M = {}

-- Layer base class (2.9.1). Subclasses: class(paddle.nn.Layer) then
-- assign fields -- auto-registration takes over (09-nn.md sec.3.2).
M.Layer = layer.Layer

-- First real layer (2.9.2): proves auto-registration + create_parameter
-- + forward end to end (api/nn.md sec.2.3).
M.Linear = linear

-- Functionals (2.9.3): activation + more, all argrule-wrapped (D31;
-- api/nn-functional.md). First batch = 16 activations.
M.functional = functional

-- 2.9.1 internals exposed for tests/CI greps (rawset discipline).
M._FIELDS = layer.FIELDS

return M
