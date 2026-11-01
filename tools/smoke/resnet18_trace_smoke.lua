-- tools/smoke/resnet18_trace_smoke.lua -- P15 验收(15-to-static.md §5)
-- trace 一个真实 ResNet18(CIFAR 结构),验证:
--   1. traced(x) == eager(x) 逐位一致(eval 模式,推理语义)
--   2. trace 耗时 < 前向 10x
--   3. save() 产物 + Python paddle.jit.load 读回一致
--      (py_resnet_readback.py 复用 py_jit_readback 思路)
--
-- Invariant (P15-A 推理先行):net 必须 eval 模式下 trace —— bn 的
-- is_test attr 在录制时被固化,eval 录 = 推理语义,与 run 的
-- is_test=true 一致;train 模式录制会让 run 时的 bn 走训练语义。
--
-- Run (repo root, x64 env):
--   set PATH=E:\code\Paddle\build-nopy\paddle\common;E:\code\Paddle\build-nopy\paddle\phi;%PATH%
--   luajit tools/smoke/resnet18_trace_smoke.lua

package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()
local paddle = require "paddle"

local CPU = core.pd_place_new(core.PD_PLACE_CPU, 0)
local function full(shape, v)
  return core.pd_op_full(shape, v, core.PD_DTYPE_FLOAT32, CPU)
end

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- ---- model: CIFAR ResNet18 + 显式 fc 初始化(与 cifar10.lua 同款) ----
local nn = paddle.nn
local net = paddle.vision.models.resnet18{ num_classes = 10 }
local old_g = core.has_grad()
core.set_has_grad(false)
for _, p in ipairs(net:parameters()) do
  local s = p:shape()
  if #s == 2 then
    local fan_in = s[1]
    core.pd_op_gaussian_inplace(p, 0.0, (2 / fan_in) ^ 0.5, 0)
  end
end
core.set_has_grad(old_g)
net:eval()  -- bn 推理语义(bn 的 is_test 在录制时固化)

-- stub 输入:B=2, 常量 0.5(确定性,不依赖 CIFAR 数据)
local x = full({ 2, 3, 32, 32 }, 0.5)

-- eager 参考在 TRACE 之前全部算好(net 内部状态必须零污染)
local t0 = os.clock()
local e_ref
for i = 1, 10 do
  e_ref = net(x)
end
local fwd_avg = (os.clock() - t0) / 10
local x2 = full({ 2, 3, 32, 32 }, 0.25)
local e2 = net(x2)  -- 与 x2 同批次的预 trace 参考
local x2b = x2:to_host_bytes()
local b0, b1, b2, b3 = string.byte(x2b, 1, 4)
print(("DIAG: x2 first float bytes = %02x %02x %02x %02x (0.25=0x3e800000)"
   ):format(b0, b1, b2, b3))

-- trace(eval 下录制)
local t1 = os.clock()
local traced = paddle.jit.trace(net, x)
local trace_time = os.clock() - t1
check(trace_time < 10 * fwd_avg,
      ("trace %.3fs >= 10x forward %.3fs"):format(trace_time, fwd_avg))

-- 逐位一致
local r = traced(x)
local function head4(t)
  local s = t:to_host_bytes()
  local out = {}
  for i = 1, 4 do
    local b0, b1, b2, b3 = string.byte(s, (i - 1) * 4 + 1, (i - 1) * 4 + 4)
    local v = (b0 or 0) + (b1 or 0) * 256 + (b2 or 0) * 65536
               + (b3 or 0) * 16777216
    if v >= 2147483648 then v = v - 4294967296 end
    out[i] = string.format("%.6f", v / 4294967296)
  end
  return table.concat(out, " ")
end
print(("DIAG: eager(x)  = %s"):format(head4(e_ref)))
print(("DIAG: traced(x) = %s"):format(head4(r)))
local bytes_e = e_ref:to_host_bytes()
local bytes_t = r:to_host_bytes()
check(bytes_t == bytes_e, "ResNet18 traced(x) != eager(x) bitwise")
local v1 = r:item(1)
check(type(v1) == "number", "traced output not a number")

-- 再跑一次新输入:输入替换仍一致(x2/e2 已在 trace 前就绪)
print(("DIAG: eager(x)  = %s"):format(head4(e_ref)))
print(("DIAG: eager(x2) = %s"):format(head4(e2)))
local r2 = traced(x2)
print(("DIAG: traced(x2) = %s"):format(head4(r2)))
-- 诊断:替换是否根本没生效(r2 == routed == 用旧输入的 traced(x))
local rout = traced(x)
print(("DIAG: traced(x2)==traced(x)? %s"):format(
  r2:to_host_bytes() == rout:to_host_bytes() and "YES (substitution broke)"
  or "no (substitution applied)"))
if r2:to_host_bytes() ~= e2:to_host_bytes() then
  local be, bt = e2:to_host_bytes(), r2:to_host_bytes()
  local n = math.min(#be, #bt)
  local diff, first = 0, nil
  for i = 1, n, 4 do
    if be:sub(i, i + 3) ~= bt:sub(i, i + 3) then
      diff = diff + 1
      if first == nil then first = i end
    end
  end
  print(("DIAG: x2 bytes len e=%d t=%d, float diffs=%d first at byte %d"
     ):format(#be, #bt, diff, first or -1))
  if first then
    local function de(bytes, off)
      local s = bytes:sub(off, off + 3)
      local _, _, b0, b1, b2, b3 =
        string.byte(s, 1, 4)
      local v = (b0 or 0) + (b1 or 0) * 256 + (b2 or 0) * 65536
                 + (b3 or 0) * 16777216
      if v >= 2147483648 then v = v - 4294967296 end
      return v / 4294967296
    end
    print(("DIAG: eager@%d=%.6f traced@%d=%.6f"):format(
      first, de(be, first), first, de(bt, first)))
  end
end
check(r2:to_host_bytes() == e2:to_host_bytes(),
      "ResNet18 traced(x2) != eager(x2) input substitution broken")

-- save + 落盘供 Python 读回
traced.save("build-test/resnet18_trace")

local fjson = io.open("build-test/resnet18_trace.json", "rb")
local fparams = io.open("build-test/resnet18_trace.pdiparams", "rb")
check(fjson ~= nil and fparams ~= nil, "save() did not write both files")
if fjson then fjson:close() end
if fparams then fparams:close() end
local ref = io.open("build-test/resnet18_trace.out.bin", "wb")
ref:write(r2:to_host_bytes())
ref:close()

print(("RESNET18_TRACE_SMOKE PASS (%d assertions): trace %.3fs (10x fwd)"
   ):format(ntests, trace_time))