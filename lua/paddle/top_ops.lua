-- lua/paddle/top_ops.lua -- paddle 顶层算子层(api/top.md 契约兑现,首批)
-- Stage: M3 / 4.4 前置。metrics-lua 只调 paddle-lua 的公开 API
-- (overview.md §1:"metrics-lua 只能调用 paddle-lua 的公开 API"),而 top.md
-- §2 契约的 paddle.zeros/cast/sum/... 在 init.lua 里从未接线 —— 实测
-- 顶层全 nil,只有 paddle_core.pd_op_* 内部 C ABI 可用。本文件把生成物
-- (_ops/<op>.lua,529 个薄 wrapper)包装成公开契约,负责生成物漏掉的两件:
--   * dtype:字符串/常量 → 枚举(paddle.dtype.to_pd_dtype)。pd_op_* 只收
--     枚举,实测 paddle_core.pd_op_cast(t, "int32") 崩、PD_DTYPE_INT32 才
--     行;top.md §2 dtype 参数契约是字符串(paddle.float32)。
--   * IntArray 轴:标量 → {v}。pd_op_sum 的 sol 绑定收
--     std::vector<int64_t>,实测传 number 崩(gen_ops_s.cc:509)。
-- 生成物负责 1-based→0-based 轴转换与默认值(emit_lua 已做);本层只做
-- 契约。调用约定:一张参数表 paddle.f{x, opt=...}(api/README.md §2.1.5),
-- 位置形态也支持(argrule)。

local argrule = require "argrule"
local rule = require "argrule.rule"
local dtype = require "paddle.dtype"

-- Shared LuaTensor usertype metatable (tensor.lua augments the same one).
local probe = require("paddle_core").tensor{1}
local TENSOR_MT = getmetatable(probe)
probe = nil
local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end
local Tensor = argrule.alias { function(v) return is_tensor(v) end }

local M = {}

-- dtype name/constant -> pd_dtype enum (nil stays nil: the generated
-- wrapper then applies its own default).
local function pd_dtype(v)
  if v == nil then return nil end
  return dtype.to_pd_dtype(v)
end

-- IntArray axis: scalar -> {v}; a table passes through. nil stays nil.
local function intarray(v)
  if v == nil or type(v) == "table" then return v end
  return { v }
end

-- Generated wrapper lookup: require "paddle._ops.<op>" -> M.<op>.
local function gen(opname)
  local mod = require("paddle._ops." .. opname)
  return mod[opname]
end

-- ---------------------------------------------------------------------
-- 创建(top.md §2.1):zeros / ones / full
-- ---------------------------------------------------------------------

-- zeros{shape, dtype} -- creation.py:1807, size_args 不移植(D35);
-- 默认 float32(生成物 default)。
M.zeros = rule {
  { name = "shape", type = argrule.List { argrule.number },
    doc = "output shape (1-based, e.g. {2,3})" },
  { name = "dtype", type = argrule.string, opt = true, default = nil,
    doc = "dtype name, default float32" },
}(function(shape, d)
  return gen("zeros")(shape, pd_dtype(d))
end)

-- ones{shape, dtype} -- creation.py:2118
M.ones = rule {
  { name = "shape", type = argrule.List { argrule.number },
    doc = "output shape" },
  { name = "dtype", type = argrule.string, opt = true, default = nil,
    doc = "dtype name, default float32" },
}(function(shape, d)
  return gen("ones")(shape, pd_dtype(d))
end)

-- full{shape, fill_value, dtype} -- creation.py:2160; top.md §2.1.5 判据
-- full{{2,3}, 1.0} 是合法表内位置形式。
M.full = rule {
  { name = "shape", type = argrule.List { argrule.number },
    doc = "output shape" },
  { name = "fill_value", type = argrule.number,
    doc = "constant fill value" },
  { name = "dtype", type = argrule.string, opt = true, default = nil,
    doc = "dtype name, default float32" },
}(function(shape, fill, d)
  return gen("full")(shape, fill, pd_dtype(d))
end)

-- ---------------------------------------------------------------------
-- 逻辑与比较(top.md §2.5):cast / equal / greater_than(首批)
-- ---------------------------------------------------------------------

-- cast{x, dtype} -- manipulation.py:251-259;dtype 必填字符串/常量。
M.cast = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "dtype", type = argrule.string, doc = "target dtype name" },
}(function(x, d)
  pd_dtype(d)  -- 报错前置:未知 dtype 在调算子前失败
  return gen("cast")(x, pd_dtype(d))
end)

-- equal{x, y} -- logic.py:221-227;输出 bool 且 stop_gradient(top.md §2.5)
local equal_rule = {
  { name = "x", type = Tensor, doc = "first tensor" },
  { name = "y", type = Tensor, doc = "second tensor" },
}
M.equal = rule(equal_rule)(function(x, y) return gen("equal")(x, y) end)
M.greater_than = rule(equal_rule)(function(x, y)
  return gen("greater_than")(x, y)
end)

-- ---------------------------------------------------------------------
-- 归约与统计(top.md §2.3):sum / mean(axis 标量或表,1-based,生成物转换)
-- ---------------------------------------------------------------------

-- sum{x, axis, dtype, keepdim} -- stat.py;axis 全量省略或 1-based
-- 标量/表轴;dtype 字符串。
M.sum = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "axis", type = argrule.any, opt = true, default = nil,
    doc = "1-based axis or axis list; omit for full reduce" },
  { name = "dtype", type = argrule.string, opt = true, default = nil,
    doc = "dtype name (optional downcast)" },
  { name = "keepdim", type = argrule.boolean, opt = true, default = false,
    doc = "keep reduced dims as size-1" },
}(function(x, axis, d, keepdim)
  return gen("sum")(x, intarray(axis), pd_dtype(d), keepdim)
end)

-- mean{x, axis, keepdim} -- stat.py:65-112(无 dtype 参数)
M.mean = rule {
  { name = "x", type = Tensor, doc = "input tensor" },
  { name = "axis", type = argrule.any, opt = true, default = nil,
    doc = "1-based axis or axis list; omit for full reduce" },
  { name = "keepdim", type = argrule.boolean, opt = true, default = false,
    doc = "keep reduced dims as size-1" },
}(function(x, axis, keepdim)
  return gen("mean")(x, intarray(axis), keepdim)
end)

-- ---------------------------------------------------------------------
-- 基础算子(top.md §2.2,一元/二元,纯 re-export)
-- ---------------------------------------------------------------------

local function una(opname, a, b)
  M[opname] = rule {
    { name = "x", type = Tensor, doc = "input tensor" },
  }(function(x)
    return gen(opname)(x)
  end)
end
local function bin(opname)
  M[opname] = rule(equal_rule)(function(x, y)
    return gen(opname)(x, y)
  end)
end

una("abs")
una("sqrt")
una("square")
bin("subtract")
bin("divide")
bin("maximum")
bin("minimum")

return M