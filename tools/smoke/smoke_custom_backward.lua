-- tools/smoke/smoke_custom_backward.lua -- P16 EAGER autograd smoke
-- Validates 16-cpp-extension.md acceptance #2: "同一个算子的反向正确,
-- 能参与 backward()". Uses the official custom_linear op (PD_BUILD_OP +
-- PD_BUILD_GRAD_OP, source ZERO-modified):
--
--   out = custom_linear(X, W, B) = matmul(X, W) + B
--   gradients (hand-computable, matmul rules):
--     dX = grad @ W^T
--     dW = X^T @ grad
--     dB = sum(grad, axis=0)
--
-- Path: load{} -> ops.custom_linear(x, w, b) -- our Lua face routes custom
-- ops through the EAGER C ABI (pd_custom_op_eager_run) so inputs with
-- stop_gradient=false build a RunCustomOpNode; paddle.grad{} then runs the
-- user's PD_BUILD_GRAD_OP kernel via egr::Backward.
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()
local paddle = require "paddle"
local cpp_extension = paddle.utils.cpp_extension
local CPU = core.pd_place_new(core.PD_PLACE_CPU, 0)

local function setgrad(t)
  t:set_stop_gradient(false)
  return t
end

-- f32 LE decode, pure Lua 5.1 (clone of vision_transforms_smoke).
local function f32_at(s, idx)
  local b1, b2, b3, b4 = s:byte(idx, idx + 3)
  local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local sign = bits >= 2147483648 and 1 or 0
  local exp = math.floor(bits / 8388608) % 256
  local mant = bits % 8388608
  if exp == 0 then return 0.0 end
  if exp == 255 then return math.huge end
  local v = (1 + mant / 8388608) * 2 ^ (exp - 127)
  return sign == 1 and -v or v
end

-- Build an f32 tensor with arbitrary values: bytes -> host buffer copy.
local function from_lua(shape, vals)
  local bytes = {}
  for _, v in ipairs(vals) do
    -- IEEE-754 encode f32 LE by hand (no string.pack / bit lib in 5.1).
    local sign = v < 0 and 1 or 0
    if v < 0 then v = -v end
    local exp, mant
    if v == 0 then
      exp, mant = 0, 0
    else
      exp = math.floor(math.log(v) / math.log(2)) + 127
      -- normalize: v = m * 2^(exp-127), m in [1,2)
      local m = v / 2 ^ (exp - 127)
      mant = math.floor((m - 1) * 8388608 + 0.5)
      if mant >= 8388608 then
        mant = mant - 8388608
        exp = exp + 1
      end
    end
    if exp > 254 then exp = 254 end -- clamp (test values are tame)
    local bits = sign * 2147483648 + exp * 8388608 + mant
    bytes[#bytes + 1] = string.char(bits % 256,
                                    math.floor(bits / 256) % 256,
                                    math.floor(bits / 65536) % 256,
                                    math.floor(bits / 16777216) % 256)
  end
  return core.from_host_bytes(table.concat(bytes), shape,
                              core.PD_DTYPE_FLOAT32, CPU)
end

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- Official custom_linear (PD_BUILD_GRAD_OP registered, 3 inputs -> 1 out).
local SRC = "E:\\code\\Paddle\\test\\custom_op\\custom_linear_op.cc"
local f = io.open(SRC, "rb")
if not f then
  print("SMOKE SKIP: custom_linear_op.cc not found")
  os.exit(77)
end
f:close()

local ops = cpp_extension.load{
  name = "cpp_ext_linear",
  sources = { SRC },
}
local fn = ops.custom_linear
check(type(fn) == "function", "custom_linear callable registered")

-- X [2,3], W [3,2], B [2]
local X = { 1, 2, 3, 4, 5, 6 }
local W = { 1, 0, 0, 1, 0.5, 0.5 }
local B = { 0.1, -0.1 }
local x = setgrad(from_lua({ 2, 3 }, X))
local w = setgrad(from_lua({ 3, 2 }, W))
local b = setgrad(from_lua({ 2 }, B))

local y = fn(x, w, b)
check(type(y) == "userdata", "eager forward returns a Tensor")

-- y = X@W + B; X [2,3], W [3,2], B [2]
--   X@W row0 = [1*1+2*0+3*0.5, 1*0+2*1+3*0.5] = [2.5, 3.5]  (+B => [2.6, 3.4])
--   X@W row1 = [4+0+3, 0+5+3]                  = [7.0, 8.0]  (+B => [7.1, 7.9])
local yb = y:to_host_bytes()
check(math.abs(f32_at(yb, 1) - 2.6) < 1e-5, "y[0,0] ~ 2.6")
check(math.abs(f32_at(yb, 5) - 3.4) < 1e-5, "y[0,1] ~ 3.4")

-- ---- backward: seed grad = ones [2,2] ----
local seed = core.pd_op_full({ 2, 2 }, 1.0, core.PD_DTYPE_FLOAT32, CPU)
local grads = paddle.autograd.grad{ outputs = y, inputs = { x, w, b },
                                    grad_outputs = seed }
check(type(grads) == "table" and #grads == 3, "grad returns 3 grads")
local gx, gw, gb = grads[1], grads[2], grads[3]
-- dX = grad @ W^T:  grad [2,2] ones; W [3,2]
--   dX[0,*] = sum_k grad[0,k] * W[*,k] = [1*1+1*0, 1*0+1*1, 1*0.5+1*0.5]
--           = [1, 1, 1]; dX[1,*] same = [1, 1, 1]
local gxb = gx:to_host_bytes()
check(math.abs(f32_at(gxb, 1) - 1) < 1e-5, "gX[0,0] == 1")
check(math.abs(f32_at(gxb, 9) - 1) < 1e-5, "gX[0,2] == 1")
-- dW = X^T @ grad: X [2,3], grad ones [2,2], dW[*,*] = sum over batch
--   dW[0,0] = x00*g00 + x10*g10 = 1+4 = 5
--   dW[2,0] = x02*g00 + x12*g10 = 3+6 = 9
local gwb = gw:to_host_bytes()
check(math.abs(f32_at(gwb, 1) - 5) < 1e-5, "gW[0,0] == 5")
check(math.abs(f32_at(gwb, 17) - 9) < 1e-5, "gW[2,0] == 9") -- 5th f32: 1-based (5-1)*4+1 = 17
-- dB = sum over rows of grad = [2, 2]
local gbb = gb:to_host_bytes()
check(math.abs(f32_at(gbb, 1) - 2) < 1e-4, "gB[0] == 2")
check(math.abs(f32_at(gbb, 5) - 2) < 1e-4, "gB[1] == 2")

print(("SMOKE PASS: custom op eager backward OK (%d assertions)"
       ):format(ntests))
os.exit(0)