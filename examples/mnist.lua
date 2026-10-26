-- examples/mnist.lua -- M1 验收(roadmap.md 2.12)
-- 判据:lua examples/mnist.lua,2 epoch,测试集准确率 > 95%
-- 上游对照:python/paddle/dataset/mnist.py(URL/MD5/归一化 mnist.py:31-76)。
--
-- 数据:build-test/mnist-data/ 下 4 个 IDX 文件(下载/解压/MD5 校验见
-- v1 一次性命令已完成,12-dataset-vision.md §3.1 的下载方案交给 P12)。
--   images:magic 2051, num, rows=28, cols=28, num*784 uint8(大端)
--   labels:magic 2049, num, num 个 uint8(大端)
-- 归一化:images/255*2-1 → [-1,1](mnist.py:73-75);label 0-9 → 1-10(C4)。
--
-- 已知缺口(2026-10-26 实测):nn.Linear 默认权重 Constant(0)
--   (layer.lua:573 注释自认,2.9.3 nn.initializer 未落地)——全零权重
--   + ReLU 死区 = 梯度全 0、训练必然失败。Conv 层自带 gaussian 初始化
--   (conv.lua),本脚本对 Linear 权重显式 gaussian(用户侧初始化,
--   上游 Linear 默认是 ParamAttr 的默认 initializer)。initializer 批次
--   落地后这行可以删。
--
-- 1-based 纪律(C4):CrossEntropyLoss 的 label、argmax 的 index 全是
-- 1-based;layer/argmax 内部已把 1-based 转 0-based 喂算子。

-- ---- 运行环境(smoke 同款:仓库根执行,Insight 非必需) ----------
-- package.path/cpath 指向本仓库的 Lua 层与 build-test 下的 paddle_core
-- (io_smoke.lua 同款;libpaddle 运行 DLL 由调用方放进 PATH)
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local paddle_core = require "paddle_core"
paddle_core.init()

require "paddle.tensor"
local paddle = require "paddle"

local DATA_DIR = os.getenv("MNIST_DATA_DIR") or "build-test/mnist-data"

local function be_u32(s, off)
  return s:byte(off) * 16777216 + s:byte(off + 1) * 65536
       + s:byte(off + 2) * 256 + s:byte(off + 3)
end

-- 读 IDX,返回 (num, rows, cols, payload) —— 数字全大端
local function read_idx(path, want_magic)
  local f = assert(io.open(path, "rb"), "cannot open " .. path)
  local all = f:read("*a")
  f:close()
  local magic = be_u32(all, 1)
  assert(magic == want_magic,
    ("%s: bad magic 0x%08x, want 0x%08x"):format(path, magic, want_magic))
  local num = be_u32(all, 5)
  if want_magic == 2051 then
    local rows = be_u32(all, 9)
    local cols = be_u32(all, 13)
    return num, rows, cols, string.sub(all, 17)
  end
  return num, nil, nil, string.sub(all, 9)
end

-- CPU place(全程 CPU;paddle_memory 追踪由 scope 负责,P5)
local CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)

-- 造 [N,1,28,28] FLOAT32 归一化 images;原始字节整块进入(60000*784
-- 逐算子不可行,from_host_bytes 是 2.12 数据管线基座):
--   from_host_bytes(data, dims, dtype, place) → UINT8 tensor
--   cast → FLOAT32
--   scale(2/255, -1, 1):out = (2/255)*x - 1 = x/255*2 - 1(mnist.py)
local function make_images(rows, cols, payload)
  local u8 = paddle_core.from_host_bytes(
    payload, { math.floor(#payload / (rows * cols)), 1, rows, cols },
    paddle_core.PD_DTYPE_UINT8, CPU_PLACE)
  local f32 = paddle_core.pd_op_cast(u8, paddle_core.PD_DTYPE_FLOAT32)
  return paddle_core.pd_op_scale(f32, 2 / 255, -1, 1)
end

-- 造 [N] INT64 1-based labels(0-9 → 1-10,C4)
local function make_labels(payload)
  local u8 = paddle_core.from_host_bytes(
    payload, { #payload }, paddle_core.PD_DTYPE_UINT8, CPU_PLACE)
  local i64 = paddle_core.pd_op_cast(u8, paddle_core.PD_DTYPE_INT64)
  local ones = paddle_core.pd_op_full(
    { #payload }, 1, paddle_core.PD_DTYPE_INT64, CPU_PLACE)
  return paddle_core.pd_op_add(i64, ones)
end

-- ---- 数据加载 ------------------------------------------------
local tr_n, tr_rows, tr_cols, tr_img_payload =
  read_idx(DATA_DIR .. "/train-images-idx3-ubyte", 2051)
local _, _, _, tr_label_payload =
  read_idx(DATA_DIR .. "/train-labels-idx1-ubyte", 2049)
local te_n, te_rows, te_cols, te_img_payload =
  read_idx(DATA_DIR .. "/t10k-images-idx3-ubyte", 2051)
local _, _, _, te_label_payload =
  read_idx(DATA_DIR .. "/t10k-labels-idx1-ubyte", 2049)
assert(tr_rows == 28 and tr_cols == 28, "MNIST rows/cols must be 28")
assert(tr_n == 60000 and te_n == 10000, "MNIST sizes must be 60k/10k")

local tr_x = make_images(tr_rows, tr_cols, tr_img_payload)
local tr_y = make_labels(tr_label_payload)
local te_x = make_images(te_rows, te_cols, te_img_payload)
local te_y = make_labels(te_label_payload)

-- ---- 模型(LeNet 风格):28->14->7 ---------------------------------
-- Conv2D 权重自带 gaussian(conv.lua);Linear 权重显式初始化
local nn = paddle.nn
local net = nn.Sequential{
  nn.Conv2D{ in_channels = 1, out_channels = 32, kernel_size = 3,
             padding = 1 },
  nn.ReLU(),
  nn.MaxPool2D{ kernel_size = 2 },
  nn.Conv2D{ in_channels = 32, out_channels = 64, kernel_size = 3,
             padding = 1 },
  nn.ReLU(),
  nn.MaxPool2D{ kernel_size = 2 },
  nn.Flatten(),
  nn.Linear{ in_features = 64 * 7 * 7, out_features = 128 },
  nn.ReLU(),
  nn.Linear{ in_features = 128, out_features = 10 },
}

-- Linear 默认 Constant(0) → 显式 gaussian 初始化(He 风格,同 conv)。
-- 见文件头"已知缺口"。shape=[in,out],fan_in=shape[1]。
-- 必须 no_grad 下做:gaussian_inplace 带自己的反向节点
-- (gaussian_inplace_grad,ops.yaml:2564),has_grad 下初始化会把节点挂进
-- autograd 图,之后训练 backward 到同一参数崩(FatalError:
-- "TensorWrappers of GaussianInplaceGradNode do not exist",
-- backward.cc:215 —— 2026-10-27 实证)。
local old_g = paddle_core.has_grad()
paddle_core.set_has_grad(false)
for _, p in ipairs(net:parameters()) do
  local s = p:shape()
  if #s == 2 then
    local fan_in = s[1]
    paddle_core.pd_op_gaussian_inplace(p, 0.0, (2 / fan_in) ^ 0.5, 0)
  end
end
paddle_core.set_has_grad(old_g)

-- ---- 训练循环(2 epoch,batch=64,slicing 沿 axis 0) -------------
local BATCH = 64
local EPOCHS = 2
local criterion = nn.CrossEntropyLoss{ reduction = "mean" }
local opt = paddle.optimizer.Adam{
  parameters = net:parameters(), learning_rate = 0.001,
}

-- slice 的 starts/ends 是坐标(非 index 语义),Lua 传入即 0-based 区间
local function batch_of(t, start0, end0)
  return paddle_core.pd_op_slice(t, { 0 }, { start0 }, { end0 },
    { 1 }, {})
end

for epoch = 1, EPOCHS do
  local total_loss = 0
  local n_batches = 0
  local i0 = 0
  while i0 < tr_n do
    local i1 = math.min(i0 + BATCH, tr_n)
    local x = batch_of(tr_x, i0, i1)
    local y = batch_of(tr_y, i0, i1)
    local loss = criterion(net(x), y)
    total_loss = total_loss + loss:item()
    n_batches = n_batches + 1
    loss:backward()
    opt:step()
    opt:clear_grad()
    -- 显式 GC(2026-10-27 实证):LuaJIT 默认 GC 阈值不回收逐 batch
    -- 累积的 LuaTensor shell → C 侧(autograd 图/phi 张量)堆积,
    -- ~300 batch 后 0xC0000005。每 100 batch 收一次(同 Python
    -- gc.collect 的心智;探针 12:显式 GC 后 938 batch 全跑完)。
    if n_batches % 100 == 0 then collectgarbage("collect") end
    i0 = i1
  end
  print(("epoch %d  loss %.4f"):format(epoch, total_loss / n_batches))
end

-- ---- 测试集准确率 ------------------------------------------------
-- argmax(logits, axis=-1) → 1-based index(returns_index 已 +1),
-- 与 1-based label 直接比较。
paddle_core.set_has_grad(false)
local correct = 0
local i0 = 0
while i0 < te_n do
  local i1 = math.min(i0 + BATCH, te_n)
  local x = batch_of(te_x, i0, i1)
  local y = batch_of(te_y, i0, i1)
  local logits = net(x)
  local pred = paddle_core.pd_op_argmax(
    logits, -1, 0, 0, paddle_core.PD_DTYPE_INT64)
  local eq = paddle_core.pd_op_equal(pred, y)
  local eqi = paddle_core.pd_op_cast(eq, paddle_core.PD_DTYPE_INT32)
  correct = correct + paddle_core.pd_op_sum(
    eqi, {}, paddle_core.PD_DTYPE_UNDEFINED, 0):item()
  i0 = i1
end
paddle_core.set_has_grad(true)

local acc = correct / te_n
print(("test accuracy %.4f (%d/%d)"):format(acc, correct, te_n))
assert(acc > 0.95,
  ("M1 验收失败:test accuracy %.4f <= 0.95"):format(acc))
print("MNIST M1 ACCEPT: test accuracy > 95%")