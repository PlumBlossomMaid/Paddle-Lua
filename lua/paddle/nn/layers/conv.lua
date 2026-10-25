-- lua/paddle/nn/layers/conv.lua -- paddle.nn.Conv2D
-- Stage: P9 / M1 (WORKPLAN 2.9.4). MNIST 路径卷积层。
--
-- Upstream blueprint: python/paddle/nn/layer/conv.py:712 (Conv2D) +
-- _ConvNd.__init__ conv.py:150-230 + _update_padding_nd
-- (nn/functional/conv.py:83-136) + _conv_nd (nn/functional/conv.py:209-262).
--
-- Parameter shapes (conv.py:180-200): filter_shape = [out_channels,
-- in_channels // groups, kh, kw]; default init Normal(0.0, std) with
-- std = (2.0 / filter_elem_num)^0.5, filter_elem_num = prod(kernel)*in
-- channels (conv.py:186-189); bias = create_parameter([out_channels]).
--
-- Padding 三态解析:_update_padding_nd(padding, channel_last, 2) → 返回
-- (padding_list, padding_algorithm)(conv.py:83-136)。字符串 'SAME'/
-- 'VALID' → 算法;2*ndim 对称 list 折叠为 ndim;pair-list 剔除 batch/
-- channel 维度。padding 是空间偏移不是索引(C4 例外,nn.md sec.3)。
--
-- Forward (conv.py:864-916): padding_mode='zeros' 时直接
-- _C_ops.conv2d(x, w, stride, padding, padding_algorithm, dilation,
-- groups, data_format) (conv.py:249);bias 时 new_shape=[1]*ndim、
-- new_shape[channel_dim]=-1, reshape 后 add (conv.py:259-262)。
-- channel_dim: NCHW→1, NHWC→3 (0-based)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local Layer = require("paddle.nn.layer").Layer

local Conv2D = class(Layer)
Conv2D._name = "Conv2D"

-- IntOrList: kernel/stride/dilation 三态(int/{a,b}/[a,b])
local IntOrList = argrule.Union { argrule.number, argrule.table }

-- _update_padding_nd Lua 版(conv.py:83-136)。channel_last =
-- data_format == 'NHWC';ndim=2。返回 (padding_list, algorithm)。
local function update_padding_nd(padding, channel_last, ndim)
  if type(padding) == "string" then
    local u = string.upper(padding)
    if u ~= "SAME" and u ~= "VALID" then
      error(("paddle.nn.Conv2D: unknown padding %q, only 'SAME'/'VALID'")
        :format(padding), 3)
    end
    local zeros = {}
    for _ = 1, ndim do zeros[#zeros + 1] = 0 end
    return zeros, u
  end
  if type(padding) == "table" then
    -- pair-list: len == 2 + ndim,首元素是表 → 剔除 batch/channel
    if #padding == 2 + ndim and type(padding[1]) == "table" then
      -- _zero_padding_in_batch_and_channel(conv.py:70):NCHW 时前两对
      -- 是 batch/channel,NHWC 时首尾两对
      local flat = {}
      local keep
      if channel_last then
        keep = { 2, 3 }  -- 剔除索引 1、4(NHWC:batch 末、channel 首?)
      else
        keep = { 3, 4 }  -- 剔除索引 1、2(NCHW:batch、channel)
      end
      -- 校验剔除维为 0
      local skip = channel_last and { 1, 4 } or { 1, 2 }
      for _, i in ipairs(skip) do
        if padding[i][1] ~= 0 or padding[i][2] ~= 0 then
          error("paddle.nn.Conv2D: non-zero padding in batch/channel "
            .. "dimension is not supported", 3)
        end
      end
      for _, i in ipairs(keep) do
        flat[#flat + 1] = padding[i][1]
        flat[#flat + 1] = padding[i][2]
      end
      padding = flat
    end
    -- 对称折叠:成对相等 → 取每对第 1 个
    if #padding == 2 * ndim and type(padding[1]) == "number" then
      local sym = true
      for i = 1, ndim do
        if padding[i * 2 - 1] ~= padding[i * 2] then sym = false end
      end
      if sym then
        local half = {}
        for i = 1, ndim do half[#half + 1] = padding[i * 2 - 1] end
        padding = half
      end
    end
    if #padding ~= ndim and #padding ~= 2 * ndim then
      error(("paddle.nn.Conv2D: invalid padding table (len %d, want %d "
        .. "or %d)"):format(#padding, ndim, 2 * ndim), 3)
    end
  else
    -- 整数 → 每维同值
    local lst = {}
    for _ = 1, ndim do lst[#lst + 1] = padding end
    padding = lst
  end
  for _, p in ipairs(padding) do
    if p < 0 then
      error("paddle.nn.Conv2D: padding must be >= 0", 3)
    end
  end
  return padding, "EXPLICIT"
end

-- 标量 → {v,v}(ndim 个)
local function to_list(v, ndim)
  if type(v) == "table" then return v end
  local lst = {}
  for _ = 1, ndim do lst[#lst + 1] = v end
  return lst
end

Conv2D._init = rule {
  { name = "self", type = Layer },
  { name = "in_channels", type = argrule.number, doc = "input channels" },
  { name = "out_channels", type = argrule.number, doc = "output channels" },
  { name = "kernel_size", type = IntOrList, doc = "kernel (int or {kh,kw})" },
  { name = "stride", type = IntOrList, opt = true, default = 1 },
  { name = "padding", type = argrule.any, opt = true, default = 0,
    doc = "int / list / 'SAME' / 'VALID'(空间偏移,非索引)" },
  { name = "dilation", type = IntOrList, opt = true, default = 1 },
  { name = "groups", type = argrule.number, opt = true, default = 1 },
  { name = "padding_mode", type = argrule.string, opt = true, default = "zeros",
    doc = "'zeros' only in v1 (reflect etc. F.pad 链后续批)" },
  { name = "weight_attr", type = argrule.any, opt = true,
    doc = "initializer (v2, 同 Linear)" },
  { name = "bias_attr", type = argrule.any, opt = true,
    doc = "false = no bias (conv.py:846 bias=False)" },
  { name = "data_format", type = argrule.string, opt = true, default = "NCHW",
    doc = "'NCHW' or 'NHWC'" },
}(function(self, in_channels, out_channels, kernel_size, stride, padding,
          dilation, groups, padding_mode, weight_attr, bias_attr,
          data_format)
  if padding_mode ~= "zeros" then
    error("paddle.nn.Conv2D: padding_mode '" .. tostring(padding_mode)
      .. "' lands with the F.pad chain in a later batch "
      .. "-- tracked in tasks.md", 2)
  end
  if data_format ~= "NCHW" and data_format ~= "NHWC" then
    error("paddle.nn.Conv2D: data_format must be 'NCHW' or 'NHWC'", 2)
  end
  self:super()
  local ks = to_list(kernel_size, 2)
  local st = to_list(stride, 2)
  local dl = to_list(dilation, 2)
  local ch_last = (data_format == "NHWC")
  local pad, alg = update_padding_nd(padding, ch_last, 2)
  if in_channels % groups ~= 0 then
    error("paddle.nn.Conv2D: in_channels must be divisible by groups", 2)
  end
  -- conv.py:180:filter_shape = [out, in//groups, kh, kw]
  local filter_shape = {
    out_channels, math.floor(in_channels / groups), ks[1], ks[2],
  }
  -- conv.py:186-189:Normal(0, (2/filter_elem_num)^0.5)
  local filter_elem_num = ks[1] * ks[2] * in_channels
  local std = (2.0 / filter_elem_num) ^ 0.5
  self._dtype = paddle_core.PD_DTYPE_FLOAT32
  self.weight = self:create_parameter{
    shape = filter_shape,
    dtype = self._dtype,
    default_initializer = function(v)
      paddle_core.pd_op_gaussian_inplace(v, 0.0, std, 0)  -- gen_ops_g.cc:71
    end,
  }
  if bias_attr ~= false then
    self.bias = self:create_parameter{
      shape = { out_channels },
      dtype = self._dtype,
    }
  end
  self._stride = st
  self._paddings = pad
  self._padding_algorithm = alg
  self._dilation = dl
  self._groups = groups
  self._data_format = data_format
  self._channel_dim = (data_format == "NCHW") and 1 or 3  -- 0-based
  self.name = nil
end)

function Conv2D:forward(x)
  local out = paddle_core.pd_op_conv2d(
    x, self.weight, self._stride, self._paddings,
    self._padding_algorithm, self._dilation, self._groups,
    self._data_format)  -- ops_gen.h:100
  if self.bias ~= nil then
    -- conv.py:259-262:bias reshape 成 [1,-1,1,1](NCHW)后 add
    local new_shape = { 1, 1, 1, 1 }
    new_shape[self._channel_dim + 1] = -1
    local b = paddle_core.pd_op_reshape(self.bias, new_shape)
    out = paddle_core.pd_op_add(out, b)
  end
  return out
end

return Conv2D
