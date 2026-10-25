-- lua/paddle/nn/layers/conv.lua -- paddle.nn Conv 层类(1D/2D/3D + Transpose)
-- Stage: P9 / M1 (WORKPLAN 2.9.4.2b)。Conv2D 本批次前已有(2.9.4.1),
-- 本批泛化为 _ConvNd 工厂,产出 6 个类。
--
-- Upstream blueprint: python/paddle/nn/layer/conv.py
--   _ConvNd.__init__:conv.py:82-254
--   Conv1D:253 / Conv1DTranspose:459 / Conv2D:712 / Conv2DTranspose:917
--   Conv3D:1161 / Conv3DTranspose:1337
--   functional: conv.py:438(conv1d)/958(conv1d_transpose)/
--   1227(conv2d_transpose)/1731(conv3d_transpose)
--   padding 三态解析:conv.py:83-136(_update_padding_nd)
--
-- 关键差异处理:
--   1D 卷积无独立算子 —— 上游 functional conv1d 用 unsqueeze→conv2d→
--   squeeze 链(conv.py:653-708):weight unsqueeze(-2)、x 按
--   squeeze_axis 插入 1 维、stride/dilation 前插 1、padding 前插 0,
--   再以 NCHW/NHWC 调 conv2d。conv1d_transpose 同理
--   (conv.py:1175-1226,weight unsqueeze(-1)、output_padding 后插 0)。
--   Transpose 卷积 data_format 归一化成 NCHW/NHWC(conv.py:1913),
--   正向 3D 直接传 NCDHW/NDHWC(conv.py:289 分支)。
--
-- 生成物(直调 paddle_core):
--   pd_op_conv2d(gen_ops_c.cc:275)/conv2d_transpose(284)/
--   conv3d(302)/conv3d_transpose(311);pd_op_unsqueeze(gen_ops_u.cc:119)
--   /pd_op_squeeze(gen_ops_s.cc:431)。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle_core = require "paddle_core"
local Layer = require("paddle.nn.layer").Layer

-- IntOrList: kernel/stride/dilation 三态(int/{a,b}/[a,b])
local IntOrList = argrule.Union { argrule.number, argrule.table }

-- _update_padding_nd Lua 版(conv.py:83-136),ndim 泛化。
-- 返回 (padding_list, algorithm)。
local function update_padding_nd(padding, channel_last, ndim)
  if type(padding) == "string" then
    local u = string.upper(padding)
    if u ~= "SAME" and u ~= "VALID" then
      error(("paddle.nn: unknown padding %q, only 'SAME'/'VALID'")
        :format(padding), 3)
    end
    local zeros = {}
    for _ = 1, ndim do zeros[#zeros + 1] = 0 end
    return zeros, u
  end
  if type(padding) == "table" then
    -- pair-list: len == 2 + ndim,首元素是表 → 剔除 batch/channel
    if #padding == 2 + ndim and type(padding[1]) == "table" then
      -- _zero_padding_in_batch_and_channel:非 channel_last 时前两对是
      -- batch/channel(保留 3..2+ndim);channel_last 时首尾两对(保
      -- 留 2..1+ndim)
      local flat = {}
      local keep = {}
      local skip = {}
      if channel_last then
        for i = 2, 1 + ndim do keep[#keep + 1] = i end
        skip = { 1, 2 + ndim }
      else
        for i = 3, 2 + ndim do keep[#keep + 1] = i end
        skip = { 1, 2 }
      end
      for _, i in ipairs(skip) do
        if padding[i][1] ~= 0 or padding[i][2] ~= 0 then
          error("paddle.nn: non-zero padding in batch/channel "
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
      error(("paddle.nn: invalid padding table (len %d, want %d "
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
      error("paddle.nn: padding must be >= 0", 3)
    end
  end
  return padding, "EXPLICIT"
end

-- 标量 → ndim 个同值的列表
local function to_list(v, ndim)
  if type(v) == "table" then return v end
  local lst = {}
  for _ = 1, ndim do lst[#lst + 1] = v end
  return lst
end

-- reshape bias 成 [1,-1,1,1](NCHW 系)后 add(conv.py:259-262)。
-- channel_dim 是 0-based 通道轴(由 data_format 决定)。
local function add_bias(out, bias, ndim, channel_dim)
  local new_shape = {}
  for _ = 1, ndim + 2 do new_shape[#new_shape + 1] = 1 end
  new_shape[channel_dim + 1] = -1
  local b = paddle_core.pd_op_reshape(bias, new_shape)
  return paddle_core.pd_op_add(out, b)
end

-- 1D 前插/后插 helper(给 stride/dilation/padding 前插 1 或 0)
local function prepend1(v)
  local r = { 1 }
  for _, e in ipairs(v) do r[#r + 1] = e end
  return r
end
local function append1(v)
  local r = {}
  for _, e in ipairs(v) do r[#r + 1] = e end
  r[#r + 1] = 1
  return r
end

-- _ConvNd 工厂:按 dims(1/2/3)与 transposed 生成类。
-- 构造参数与上游一一对应(conv.py:82-254)。
local function build_conv_class(name, dims, transposed)
  local Cls = class(Layer)
  Cls._name = name

  -- 该维度的合法 data_format
  local valid_fmts = { "NCL", "NLC" }
  if dims == 2 then valid_fmts = { "NCHW", "NHWC" } end
  if dims == 3 then valid_fmts = { "NCDHW", "NDHWC" } end

  Cls._init = rule {
    { name = "self", type = Layer },
    { name = "in_channels", type = argrule.number, doc = "input channels" },
    { name = "out_channels", type = argrule.number, doc = "output channels" },
    { name = "kernel_size", type = IntOrList, doc = "kernel (int or list)" },
    { name = "stride", type = IntOrList, opt = true, default = 1 },
    { name = "padding", type = argrule.any, opt = true, default = 0,
      doc = "int / list / 'SAME' / 'VALID'(空间偏移,非索引)" },
    { name = "dilation", type = IntOrList, opt = true, default = 1 },
    { name = "groups", type = argrule.number, opt = true, default = 1 },
    { name = "padding_mode", type = argrule.string, opt = true,
      default = "zeros", doc = "'zeros' only (reflect 等 F.pad 链后续批)" },
    { name = "weight_attr", type = argrule.any, opt = true,
      doc = "initializer (v2, 同 Linear)" },
    { name = "bias_attr", type = argrule.any, opt = true,
      doc = "false = no bias" },
    { name = "data_format", type = argrule.string, opt = true,
      default = valid_fmts[1],
      doc = table.concat(valid_fmts, "/") },
    { name = "output_padding", type = IntOrList, opt = true, default = 0,
      doc = "transpose only; ignored for conv" },
    { name = "bias", type = argrule.boolean, opt = true, default = true,
      doc = "bias=False → bias_attr=false (conv.py:846)" },
    { name = "name", type = argrule.Optional { argrule.string }, opt = true },
  }(function(self, in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, padding_mode, weight_attr, bias_attr,
            data_format, output_padding, bias, name)
    -- conv.py:146-147
    if padding_mode ~= "zeros" then
      error("paddle.nn." .. name .. ": padding_mode '" .. tostring(
        padding_mode) .. "' lands with the F.pad chain in a later batch", 2)
    end
    local ok_fmt = false
    for _, f in ipairs(valid_fmts) do
      if data_format == f then ok_fmt = true end
    end
    if not ok_fmt then
      error("paddle.nn." .. name .. ": data_format must be one of "
        .. table.concat(valid_fmts, "/"), 2)
    end
    if bias == false then bias_attr = false end
    self:super()
    local ks = to_list(kernel_size, dims)
    local st = to_list(stride, dims)
    local dl = to_list(dilation, dims)
    -- 1D 的 padding 解析在 forward(functional conv1d 里做);其余在构造
    local pad, alg = {}, "EXPLICIT"
    if dims ~= 1 then
      pad, alg = update_padding_nd(padding,
        (data_format == valid_fmts[2]), dims)
    end
    self._raw_padding = padding
    if in_channels % groups ~= 0 then
      error("paddle.nn." .. name .. ": in_channels must be divisible by "
        .. "groups", 2)
    end
    -- transposed: [in, out//groups, *ks];否则 [out, in//groups, *ks]
    local filter_shape
    if transposed then
      filter_shape = { in_channels, math.floor(out_channels / groups) }
    else
      filter_shape = { out_channels, math.floor(in_channels / groups) }
    end
    for _, k in ipairs(ks) do filter_shape[#filter_shape + 1] = k end
    local filter_elem_num = 1
    for _, k in ipairs(ks) do filter_elem_num = filter_elem_num * k end
    filter_elem_num = filter_elem_num * in_channels
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
    self._channel_dim = (data_format == valid_fmts[2]) and (dims + 1) or 1
    -- 0-based;NLC/NHWC/NDHWC 通道在末位
    self._output_padding = output_padding
    self.name = name
  end)

  Cls.forward = function(self, x, output_size)
    -- 每个变体的 forward 单独在下方构造时替换
    return x
  end

  return Cls
end

-- ---------------------------------------------------------------------
-- Conv1D / Conv1DTranspose:unsqueeze→conv2d→squeeze 链
-- ---------------------------------------------------------------------

local Conv1D = build_conv_class("Conv1D", 1, false)
function Conv1D:forward(x)
  -- functional conv1d(conv.py:438):NCL/NLC,经 conv2d 模拟
  local ch_last = (self._data_format == "NLC")
  local pad, alg = update_padding_nd(self._raw_padding, ch_last, 1)
  if #pad == 2 then pad = { 0, 0, pad[1], pad[2] }
  elseif #pad == 1 then pad = { 0, pad[1] } end
  local stride = prepend1(self._stride)
  local dilation = prepend1(self._dilation)
  -- weight [M,C,K] → [M,C,1,K];x [N,C,L] → [N,C,1,L](NCL)
  local w = paddle_core.pd_op_unsqueeze(self.weight, { -2 })
  local squeeze_axis = ch_last and -3 or -2
  local x2 = paddle_core.pd_op_unsqueeze(x, { squeeze_axis })
  local df = ch_last and "NHWC" or "NCHW"
  local out = paddle_core.pd_op_conv2d(
    x2, w, stride, pad, alg, dilation, self._groups, df)
  if self.bias ~= nil then out = add_bias(out, self.bias, 1,
    ch_last and 3 or 1) end
  return paddle_core.pd_op_squeeze(out, { squeeze_axis })
end

local Conv1DTranspose = build_conv_class("Conv1DTranspose", 1, true)
function Conv1DTranspose:forward(x, output_size)
  -- functional conv1d_transpose(conv.py:958)
  local ch_last = (self._data_format == "NLC")
  local pad, alg = update_padding_nd(self._raw_padding, ch_last, 1)
  if #pad == 2 then pad = { pad[1], pad[2], 0, 0 }
  elseif #pad == 1 then pad = { pad[1], 0 } end
  local stride = append1(self._stride)
  local dilation = append1(self._dilation)
  local out_pad = 0
  if self._output_padding ~= 0 then
    local op = to_list(self._output_padding, 1)
    out_pad = { op[1], 0 }
  end
  local out_size = {}
  if output_size ~= nil then
    local os = to_list(output_size, 1)
    out_size = { os[1], 1 }
  end
  -- weight [Cin,Cout,K] → [Cin,Cout,K,1];x [N,C,L] → [N,C,L,1](NCL)
  local w = paddle_core.pd_op_unsqueeze(self.weight, { -1 })
  local squeeze_axis = ch_last and -2 or -1
  local x2 = paddle_core.pd_op_unsqueeze(x, { squeeze_axis })
  local df = ch_last and "NHWC" or "NCHW"
  local out = paddle_core.pd_op_conv2d_transpose(
    x2, w, stride, pad, out_pad, out_size, alg, self._groups, dilation, df)
  if self.bias ~= nil then out = add_bias(out, self.bias, 1,
    ch_last and 3 or 1) end
  return paddle_core.pd_op_squeeze(out, { squeeze_axis })
end

-- ---------------------------------------------------------------------
-- Conv2D(2.9.4.1 已有,本文件迁移为类表导出)
-- ---------------------------------------------------------------------

local Conv2D = build_conv_class("Conv2D", 2, false)
function Conv2D:forward(x)
  local out = paddle_core.pd_op_conv2d(
    x, self.weight, self._stride, self._paddings,
    self._padding_algorithm, self._dilation, self._groups,
    self._data_format)  -- ops_gen.h:100
  if self.bias ~= nil then
    out = add_bias(out, self.bias, 2, self._channel_dim)
  end
  return out
end

local Conv2DTranspose = build_conv_class("Conv2DTranspose", 2, true)
function Conv2DTranspose:forward(x, output_size)
  -- functional conv2d_transpose(conv.py:1227)
  local ch_last = (self._data_format == "NHWC")
  local pad, alg = update_padding_nd(self._raw_padding, ch_last, 2)
  local stride = to_list(self._stride, 2)
  local dilation = to_list(self._dilation, 2)
  local out_pad = {}
  if self._output_padding ~= 0 then
    out_pad = to_list(self._output_padding, 2)
  end
  local out_size = {}
  if output_size ~= nil then out_size = to_list(output_size, 2) end
  local out = paddle_core.pd_op_conv2d_transpose(
    x, self.weight, stride, pad, out_pad, out_size, alg, self._groups,
    dilation, self._data_format)
  if self.bias ~= nil then
    out = add_bias(out, self.bias, 2, self._channel_dim)
  end
  return out
end

-- ---------------------------------------------------------------------
-- Conv3D / Conv3DTranspose:直调 pd_op_conv3d(_transpose)
-- ---------------------------------------------------------------------

local Conv3D = build_conv_class("Conv3D", 3, false)
function Conv3D:forward(x)
  local out = paddle_core.pd_op_conv3d(
    x, self.weight, self._stride, self._paddings,
    self._padding_algorithm, self._groups, self._dilation,
    self._data_format)  -- gen_ops_c.cc:302(NCDHW/NDHWC 原样)
  if self.bias ~= nil then
    out = add_bias(out, self.bias, 3, self._channel_dim)
  end
  return out
end

local Conv3DTranspose = build_conv_class("Conv3DTranspose", 3, true)
function Conv3DTranspose:forward(x, output_size)
  -- functional conv3d_transpose(conv.py:1731);data_format 归一化成
  -- NCHW/NHWC 传给算子(conv.py:1913)
  local ch_last = (self._data_format == "NDHWC")
  local pad, alg = update_padding_nd(self._raw_padding, ch_last, 3)
  local stride = to_list(self._stride, 3)
  local dilation = to_list(self._dilation, 3)
  local out_pad = {}
  if self._output_padding ~= 0 then
    out_pad = to_list(self._output_padding, 3)
  end
  local out_size = {}
  if output_size ~= nil then out_size = to_list(output_size, 3) end
  local df = ch_last and "NHWC" or "NCHW"
  local out = paddle_core.pd_op_conv3d_transpose(
    x, self.weight, stride, pad, out_pad, out_size, alg, self._groups,
    dilation, df)
  if self.bias ~= nil then
    out = add_bias(out, self.bias, 3, ch_last and 4 or 1)
  end
  return out
end

return {
  Conv1D = Conv1D,
  Conv1DTranspose = Conv1DTranspose,
  Conv2D = Conv2D,
  Conv2DTranspose = Conv2DTranspose,
  Conv3D = Conv3D,
  Conv3DTranspose = Conv3DTranspose,
}