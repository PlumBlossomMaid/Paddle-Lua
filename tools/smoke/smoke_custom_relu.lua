-- tools/smoke/smoke_custom_relu.lua -- P16 custom-op 核心探针
-- 目标:验证自定义算子 DLL 能被(pd_load_op_meta_info)加载、注册到
-- custom_op. 前缀的 PIR 方言,并能经 Lua 侧调用执行 forward。
--
-- 本探针覆盖 16-cpp-extension.md 验收第一条的一半:CPU 自定义算子
-- load() 后能调用。先由外部编译 custom_relu_op.cc 成 custom_relu.dll
-- (见 build_test.cmd),再：
--   1. core.load_custom_op(dso)   -> 返回 {op_name -> callable}
--   2. 调用 callable(x),对拍 relu
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()

local CPU = core.pd_place_new(core.PD_PLACE_CPU, 0)
local function full(shape, v)
  return core.pd_op_full(shape, v, core.PD_DTYPE_FLOAT32, CPU)
end

local DSO = "build-test/custom_relu.dll"
local f = io.open(DSO, "rb")
if not f then
  print(("SMOKE SKIP: %s missing -- run the build step first"):format(DSO))
  os.exit(77)
end
f:close()

-- 1. 加载
local ops = core.load_custom_op(DSO)
print("loaded ops:", table.concat(ops, ", "))

-- 2. relu 数值对拍
local x = full({ 2, 3 }, -1.0)
local x2 = full({ 2, 3 }, 0.5)
-- 经 paddle_core 的算子分发：custom op 以函数形态出现在 ops 表里
local fn = ops.custom_relu
if type(fn) ~= "function" then
  print("SMOKE FAIL: ops.custom_relu is not a function")
  os.exit(1)
end

local y = fn(x)
print("custom_relu(x) bytes:", y:to_host_bytes():sub(1, 4))
local y2 = fn(x2)
print("custom_relu(x2) bytes:", y2:to_host_bytes():sub(1, 4))

-- 期望:relu(-1) = 0,relu(0.5) = 0.5 (0x3f000000)
local fb = y:to_host_bytes():sub(1, 4)
local fb2 = y2:to_host_bytes():sub(1, 4)
local zero = string.char(0, 0, 0, 0)
local half = string.char(0, 0, 0x80, 0x3f) -- 0.5 = 0x3f800000, 小端 00 00 80 3f
local ok1 = fb == zero
local ok2 = fb2 == half
print(("relu(-1)=0: %s  relu(0.5)=0.5: %s"):format(
  ok1 and "YES" or ("no got %02x%02x%02x%02x"):format(fb:byte(1), fb:byte(2), fb:byte(3), fb:byte(4)),
  ok2 and "YES" or ("no got %02x%02x%02x%02x"):format(fb2:byte(1), fb2:byte(2), fb2:byte(3), fb2:byte(4))))

if ok1 and ok2 then
  print("SMOKE PASS: custom_relu forward OK (2 assertions)")
  os.exit(0)
else
  print("SMOKE FAIL")
  os.exit(1)
end