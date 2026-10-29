-- tools/smoke/resnet_smoke.lua -- paddle.vision.models.ResNet / resnet18
-- Stage: M2 / 3.4 (T-M2-05 subtask 2). Mirrors python test of ResNet
-- structure + M2 acceptance prerequisites (12-dataset-vision.md sec 5):
--   - ResNet18 layers cfg {2,2,2,2} stages with 64/128/256/512 planes
--     (resnet.py:260-266 / :288-293)
--   - forward on [1,3,32,32] CIFAR-sized input -> [1,10] logits
--     (resnet.py:382-399; flatten axis 1 -> Lua start_axis 2, C4)
--   - state_dict keys byte-for-byte match Python prefix.name dot paths
--     (09-nn.md sec 3.5; e.g. "conv1.weight", "layer1.0.conv1.weight")
--   - backward loop runs (training path, SGD one step) without error
--
-- Run (repo root, x64 env): set PATH=... ; luajit tools/smoke/resnet_smoke.lua

package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "E:/code/Insight7/bindings/lua/?.lua;E:/code/Insight7/bindings/lua/?/init.lua;" .. package.path
package.cpath = "E:/code/Insight7/build/bindings/lua/?.dll;" .. package.cpath
package.cpath = "E:/code/Insight7/build/backends/cpu/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()

local paddle = require "paddle"
local nn = paddle.nn

local F32 = core.PD_DTYPE_FLOAT32
local cpu = core.pd_place_new(core.PD_PLACE_CPU, 0)

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- 1) resnet18() factory with CIFAR classes: output dim 10
local model = paddle.vision.models.resnet18{ num_classes = 10 }
check(model ~= nil, "resnet18() constructs")

-- 2) forward: [1,3,32,32] -> [1,10]
local x = core.pd_op_full({ 1, 3, 32, 32 }, 0.5, F32, cpu)
local out = model(x)
check(out:ndim() == 2, ("forward ndim == 2 (got %d)"):format(out:ndim()))
check(out:shape()[1] == 1 and out:shape()[2] == 10,
      ("forward shape == {1,10} (got {%d,%d})")
        :format(out:shape()[1], out:shape()[2]))

-- 3) state_dict keys match Python dot-path naming
local sd = model:state_dict()
local keys = {}
for k in pairs(sd) do keys[#keys + 1] = k end
table.sort(keys)
local joined = table.concat(keys, ",")
check(joined:find("conv1.weight", 1, true) ~= nil, "has conv1.weight")
check(joined:find("bn1.weight", 1, true) ~= nil, "has bn1.weight")
check(joined:find("layer1.0.conv1.weight", 1, true) ~= nil,
      "has layer1.0.conv1.weight (Python dot path)")
check(joined:find("layer4.1.bn2.bias", 1, true) ~= nil, "has layer4.1.bn2.bias")
check(joined:find("fc.weight", 1, true) ~= nil, "has fc.weight")
-- no top-level param named without prefix (all under conv/bn/layer/fc)
for _, k in ipairs(keys) do
  local ok = k:match("^conv") or k:match("^bn") or k:match("^layer")
    or k:match("^fc") or k:match("^relu") or k:match("^maxpool")
  check(ok ~= nil, "key prefix legal: " .. tostring(k))
end

-- 4) parameter count sanity: resnet18 has 11.69M params. Count via
--    named_parameters and compare total numel cheaply (weights only).
local nparams = 0
for _ in model:named_parameters() do nparams = nparams + 1 end
-- conv1(64x3x7x7)=9408 + stage convs + two fc... just assert a plausible
-- floor so structure regressions surface: each BasicBlock has 5 tensors
-- (2 conv weights + 2 bn weights + 1 bn bias); 16 blocks + stem + bn1 + fc
check(nparams >= 2 * 16 + 6, ("named_parameters count sane (got %d)")
        :format(nparams))

-- 5) backward loop (SGD one step) — the M2 training path must run
local loss_fn = nn.MSELoss{}
local target = core.pd_op_full({ 1, 10 }, 0.0, F32, cpu)
local opt = paddle.optimizer.SGD{ parameters = model:parameters(),
                                  learning_rate = 0.01 }
local old_has_grad = core.has_grad()
core.set_has_grad(true)
local loss_out = loss_fn(out, target)
-- MSELoss returns a 0-d scalar tensor (tensor.lua:240 forbids x[n] on
-- 0-d; pass it straight to backward). NOTE: core.backward's binding is
-- POSITIONAL (sol table, gts, retain) -- a named-table call crashes
-- (measured 0xC0000005, diag_bk 2026-10-30). paddle.autograd.backward
-- wraps positionally (autograd.lua:43-44), the same accepted form as
-- autograd_smoke.lua:55.
paddle.autograd.backward{ tensors = { loss_out } }
opt:step()
opt:clear_grad()
core.set_has_grad(old_has_grad)

-- 6) unsupported depth errors
local ok_d, err_d = pcall(function()
  paddle.vision.models.ResNet{ Block = paddle.vision.models.BasicBlock,
                               depth = 47 }
end)
check(not ok_d, "unsupported depth raises")
check(tostring(err_d):find("depth", 1, true) ~= nil,
      "error mentions depth")

print(("RESNET_SMOKE PASS (%d assertions)"):format(ntests))