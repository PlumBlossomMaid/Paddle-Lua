-- lua/paddle/vision/models/resnet.lua -- paddle.vision.models.ResNet
-- Stage: M2 / 3.4 (T-M2-05 subtask 2). Contract: api/vision.md sec 2.3
--   local net = paddle.vision.models.ResNet{ depth = 18, num_classes = 10 }
-- Upstream: python/paddle/vision/models/resnet.py (read 2026-10-29,
-- full 899 lines). Structure mirrored verbatim:
--   BasicBlock    resnet.py:121-197  expansion=1; conv1 3x3/s / conv2
--                 3x3/s1, both bias_attr=False; downsample = 1x1 conv
--                 (stride) + BN when stride!=1 or channels change
--   ResNet        resnet.py:229-399  stem conv1 7x7/s2/p3 (no bias) +
--                 bn1 + relu + maxpool 3x3/s2/p1; layers cfg
--                 {18:{2,2,2,2}, 34:{3,4,6,3}, 50:{3,4,6,3}, ...};
--                 avgpool AdaptiveAvgPool2D(1,1); fc Linear(512*exp)
--   forward       resnet.py:382-399  conv1..layer4, [avgpool],
--                 flatten(start_axis=2), fc
-- BottleneckBlock (resnet.py:200-281, expansion=4) is included so
-- ResNet{depth=50} etc. construct; 1x1->3x3(stride)->1x1 with width
-- scaling (base_width/64 * groups).
-- Pretrained weights (resnet.py:403-421 _resnet + model_urls :67-96)
-- load via paddle.load(.pdparams) -- P9 chain, wired but only exercised
-- when pretrained=true is passed (M2 acceptance trains from scratch).
--
-- Residual add: out += identity is paddle.add (out and identity have
-- identical shape; conv.lua:122 uses the same pd_op_add).
-- Flatten: upstream calls paddle.flatten(x, 1) -- 0-based axis 1 =
-- batch; Lua 1-based equivalent start_axis=2 (C4; nn/common.lua:7-9
-- documents the same default).
--
-- Lua 5.1 subset (C3).

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local nn = require "paddle.nn"
local paddle_core = require "paddle_core"
-- 5.1 global `unpack` vs 5.2+ `table.unpack` (C3/C5 compat)
local unpack = table.unpack or unpack

local M = {}

-- model_urls (resnet.py:67-96) -- arch -> {url, md5}. Kept for the
-- pretrained chain; the loader mirrors _resnet (resnet.py:403-421).
local MODEL_URLS = {
  resnet18 = { "https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams",
               "cf548f46534aa3560945be4b95cd11c4" },
  resnet34 = { "https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams",
               "8d2275cf8706028345f78ac0e1d31969" },
  resnet50 = { "https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams",
               "ca6f485ee1ab0492d38f323885b0ad80" },
  resnet101 = { "https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams",
                "02f35f034ca3858e1e54d4036443c92d" },
  resnet152 = { "https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams",
                "7ad16a3f1e7333859ff986138630fd7a" },
}

local LAYER_CFG = {          -- resnet.py:260-266 layer_cfg
  [18]  = { 2, 2, 2, 2 },
  [34]  = { 3, 4, 6, 3 },
  [50]  = { 3, 4, 6, 3 },
  [101] = { 3, 4, 23, 3 },
  [152] = { 3, 8, 36, 3 },
}

local function conv3x3(inplanes, planes, stride)
  return nn.Conv2D{ in_channels = inplanes, out_channels = planes,
                    kernel_size = 3, stride = stride or 1, padding = 1,
                    bias_attr = false }
end

local function conv1x1(inplanes, planes, stride)
  return nn.Conv2D{ in_channels = inplanes, out_channels = planes,
                    kernel_size = 1, stride = stride or 1,
                    bias_attr = false }
end

-- ---- BasicBlock (resnet.py:121-197) ---------------------------------

local BasicBlock = class(nn.Layer)
BasicBlock._name = "BasicBlock"
BasicBlock.expansion = 1

BasicBlock._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "inplanes", type = argrule.number },
  { name = "planes", type = argrule.number },
  { name = "stride", type = argrule.number, opt = true, default = 1 },
  { name = "downsample", type = argrule.any, opt = true,
    doc = "nn.Layer or nil (resnet.py:128)" },
  { name = "groups", type = argrule.number, opt = true, default = 1 },
  { name = "base_width", type = argrule.number, opt = true, default = 64 },
  { name = "dilation", type = argrule.number, opt = true, default = 1 },
}(function(self, inplanes, planes, stride, downsample, groups, base_width,
            dilation)
  -- resnet.py:137-140: dilation > 1 not supported in BasicBlock
  if dilation and dilation > 1 then
    error("BasicBlock: Dilation > 1 not supported in BasicBlock", 3)
  end
  self:super()                                          -- layer.lua FIELDS 注册
  self.conv1 = conv3x3(inplanes, planes, stride)      -- resnet.py:141-145
  self.bn1 = nn.BatchNorm2D{ num_features = planes }  -- :146
  self.relu = nn.ReLU{}                               -- :147
  self.conv2 = conv3x3(planes, planes, 1)             -- :148-150
  self.bn2 = nn.BatchNorm2D{ num_features = planes }  -- :151
  self.downsample = downsample                        -- :152
  self.stride = stride                                -- :153
end)

function BasicBlock:forward(x)                          -- resnet.py:155-178
  local identity = x
  local out = self.conv1(x)
  out = self.bn1(out)
  out = self.relu(out)
  out = self.conv2(out)
  out = self.bn2(out)
  if self.downsample ~= nil then
    identity = self.downsample(x)
  end
  out = paddle_core.pd_op_add(out, identity)            -- out += identity
  out = self.relu(out)
  return out
end

-- ---- BottleneckBlock (resnet.py:200-281) ----------------------------

local BottleneckBlock = class(nn.Layer)
BottleneckBlock._name = "BottleneckBlock"
BottleneckBlock.expansion = 4

BottleneckBlock._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "inplanes", type = argrule.number },
  { name = "planes", type = argrule.number },
  { name = "stride", type = argrule.number, opt = true, default = 1 },
  { name = "downsample", type = argrule.any, opt = true },
  { name = "groups", type = argrule.number, opt = true, default = 1 },
  { name = "base_width", type = argrule.number, opt = true, default = 64 },
  { name = "dilation", type = argrule.number, opt = true, default = 1 },
}(function(self, inplanes, planes, stride, downsample, groups, base_width,
            dilation)
  self:super()
  -- resnet.py:217 width = int(planes * (base_width / 64.0)) * groups
  local width = math.floor(planes * (base_width / 64.0)) * (groups or 1)
  self.conv1 = conv1x1(inplanes, width, 1)              -- resnet.py:222
  self.bn1 = nn.BatchNorm2D{ num_features = width }     -- :223
  self.conv2 = nn.Conv2D{ in_channels = width, out_channels = width,
                          kernel_size = 3, stride = stride or 1,
                          padding = dilation or 1,
                          groups = groups or 1,
                          dilation = dilation or 1,
                          bias_attr = false }           -- :225-233
  self.bn2 = nn.BatchNorm2D{ num_features = width }     -- :234
  self.conv3 = conv1x1(width, planes * self.expansion, 1)  -- :236-238
  self.bn3 = nn.BatchNorm2D{ num_features = planes * self.expansion }  -- :239
  self.relu = nn.ReLU{}                                 -- :240
  self.downsample = downsample                          -- :241
  self.stride = stride                                  -- :242
end)

function BottleneckBlock:forward(x)                     -- resnet.py:244-278
  local identity = x
  local out = self.conv1(x)
  out = self.bn1(out)
  out = self.relu(out)
  out = self.conv2(out)
  out = self.bn2(out)
  out = self.relu(out)
  out = self.conv3(out)
  out = self.bn3(out)
  if self.downsample ~= nil then
    identity = self.downsample(x)
  end
  out = paddle_core.pd_op_add(out, identity)
  out = self.relu(out)
  return out
end

-- ---- ResNet (resnet.py:229-399) --------------------------------------

local ResNet = class(nn.Layer)
ResNet._name = "ResNet"
M.BasicBlock = BasicBlock
M.BottleneckBlock = BottleneckBlock

ResNet._init = rule {
  { name = "self", type = argrule.table, doc = "pl.class instance (auto)" },
  { name = "Block", type = argrule.table,
    doc = "BasicBlock or BottleneckBlock class" },
  { name = "depth", type = argrule.number, opt = true, default = 50 },
  { name = "width", type = argrule.number, opt = true, default = 64 },
  { name = "num_classes", type = argrule.number, opt = true, default = 1000 },
  { name = "with_pool", type = argrule.boolean, opt = true, default = true },
  { name = "groups", type = argrule.number, opt = true, default = 1 },
}(function(self, Block, depth, width, num_classes, with_pool, groups)
  local layers = LAYER_CFG[depth]
  if layers == nil then
    error("ResNet: unsupported depth " .. tostring(depth)
          .. " (resnet.py layer_cfg has 18/34/50/101/152)", 3)
  end
  self:super()                                          -- layer.lua FIELDS 注册
  self.groups = groups                                  -- resnet.py:270
  self.base_width = width                               -- :271
  self.num_classes = num_classes                        -- :272
  self.with_pool = with_pool                            -- :273
  self.inplanes = 64                                    -- :275
  self.dilation = 1                                     -- :276

  self.conv1 = nn.Conv2D{ in_channels = 3, out_channels = 64,
                          kernel_size = 7, stride = 2, padding = 3,
                          bias_attr = false }           -- :278-284
  self.bn1 = nn.BatchNorm2D{ num_features = 64 }        -- :285
  self.relu = nn.ReLU{}                                 -- :286
  self.maxpool = nn.MaxPool2D{ kernel_size = 3, stride = 2, padding = 1 } -- :287
  self.layer1 = self:_make_layer(Block, 64, layers[1])  -- :288
  self.layer2 = self:_make_layer(Block, 128, layers[2], 2)
  self.layer3 = self:_make_layer(Block, 256, layers[3], 2)
  self.layer4 = self:_make_layer(Block, 512, layers[4], 2)
  if with_pool then
    self.avgpool = nn.AdaptiveAvgPool2D{ output_size = 1 }  -- :290-292
  end
  if num_classes > 0 then
    self.fc = nn.Linear{ in_features = 512 * Block.expansion,
                         out_features = num_classes }   -- :294-296
  end
end)

-- _make_layer (resnet.py:299-380): first block carries stride/downsample,
-- inplanes advances to planes*expansion, remaining blocks repeat.
function ResNet:_make_layer(Block, planes, blocks, stride)
  stride = stride or 1
  local downsample = nil
  -- :303-306 stride != 1 or inplanes != planes*expansion
  if stride ~= 1 or self.inplanes ~= planes * Block.expansion then
    downsample = nn.Sequential{
      conv1x1(self.inplanes, planes * Block.expansion, stride),
      nn.BatchNorm2D{ num_features = planes * Block.expansion },
    }
  end
  local items = {}
  items[1] = Block{ inplanes = self.inplanes, planes = planes,
                    stride = stride, downsample = downsample,
                    groups = self.groups, base_width = self.base_width,
                    dilation = self.dilation }
  self.inplanes = planes * Block.expansion              -- :362
  for _ = 2, blocks do
    items[#items + 1] = Block{ inplanes = self.inplanes, planes = planes,
                               stride = 1, downsample = nil,
                               groups = self.groups,
                               base_width = self.base_width, dilation = 1 }
  end
  -- Sequential takes sibling layer args (containers.lua:182-215 walks
  -- whole[2..]); unpack items into the constructor table.
  return nn.Sequential{ unpack(items) }
end

function ResNet:forward(x)                              -- resnet.py:382-399
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)
  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)
  if self.with_pool then
    x = self.avgpool(x)
  end
  if self.num_classes > 0 then
    -- paddle.flatten(x, 1): 0-based start_axis=1. pd_op_flatten takes
    -- 0-based axes (Flatten:forward converts in common.lua:51-55, then
    -- calls pd_op_flatten); so pass (1, -1) -- NOT Lua 1-based (2, -1).
    x = paddle_core.pd_op_flatten(x, 1, -1)
    x = self.fc(x)
  end
  return x
end

-- ---- resnet18/34/50 factories (resnet.py:424+, _resnet :403-421) ----

local function make_factory(arch, Block, depth)
  return rule {
    { name = "pretrained", type = argrule.boolean, opt = true,
      default = false },
    -- kwargs 透传到 ResNet(镜像 resnet.py:424-427 resnet18(pretrained,
    -- **kwargs) -> _resnet(...) -> ResNet(block, depth, **kwargs))
    { name = "num_classes", type = argrule.number, opt = true,
      default = 1000 },
    { name = "width", type = argrule.number, opt = true, default = 64 },
    { name = "with_pool", type = argrule.boolean, opt = true,
      default = true },
    { name = "groups", type = argrule.number, opt = true, default = 1 },
  }(function(pretrained, num_classes, width, with_pool, groups)
    local model = ResNet{ Block = Block, depth = depth,
                          num_classes = num_classes, width = width,
                          with_pool = with_pool, groups = groups }
    if pretrained then
      local spec = MODEL_URLS[arch]
      assert(spec, arch .. " model do not have a pretrained model now")
      -- P9/P12 chain: paddle.load reads .pdparams via lua-pickle
      local paddle = require "paddle"
      local utils = require "paddle.utils.download"
      local home = os.getenv("USERPROFILE") or os.getenv("HOME") or "."
      local path = home .. "/.cache/paddle/hapi/weights/"
                    .. arch .. ".pdparams"
      utils.download{ url = spec[1], path = path, md5 = spec[2] }
      model:set_state_dict(paddle.load{ path = path })
    end
    return model
  end)
end

M.resnet18 = make_factory("resnet18", BasicBlock, 18)
M.resnet34 = make_factory("resnet34", BasicBlock, 34)
M.resnet50 = make_factory("resnet50", BottleneckBlock, 50)
M.ResNet = ResNet

return M