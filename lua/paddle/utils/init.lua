-- lua/paddle/utils/init.lua -- paddle.utils entry point
-- Stage: 3.1 download base. Contract: api/utils.md sec 1/2.1.
-- Submodules: download (P4 core, dataset/vision dependency).
-- Lua 5.1 subset (C3).

local M = {}

-- download.lua is small and network-oriented; expose it eagerly so
-- paddle.utils.download.* works without extra require ceremony. Missing
-- luasocket/luasec rocks surface an actionable error at call time
-- (download.lua transport()).
M.download = require "paddle.utils.download"

return M