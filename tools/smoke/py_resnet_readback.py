# tools/smoke/py_resnet_readback.py -- ResNet18 trace 读回验证(M3 闸门)
# Lua 侧 resnet18_trace_smoke.lua 产物(build-test/resnet18_trace.{json,
# pdiparams})必须能被 Python paddle.jit.load 读回,且对同一输入 x2 的
# 推理输出与 Lua traced 输出逐位一致(对照 .out.bin)。
#
# 运行:Lua smoke 先跑,然后
#   python tools/smoke/py_resnet_readback.py
import os
import sys

import numpy as np
import paddle

PREFIX = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build-test"))
MODEL = os.path.join(PREFIX, "resnet18_trace")

if not (os.path.exists(MODEL + ".json")
        and os.path.exists(MODEL + ".pdiparams")):
    print("PY_READBACK FAIL: resnet18_trace artifacts missing -- run "
          "resnet18_trace_smoke.lua first")
    sys.exit(1)

tl = paddle.jit.load(MODEL)
x = paddle.full([2, 3, 32, 32], 0.25, "float32")  # 与 smoke 的 x2 一致
y = tl(x).numpy().ravel()
print("PY_LOADED:", y[:8].tolist(), "...")

ref = np.frombuffer(
    open(os.path.join(PREFIX, "resnet18_trace.out.bin"), "rb").read(),
    dtype=np.float32)

# 判据采用"数值一致"而非逐位一致:Python 端是 pip 安装的 GPU 构建
# paddle,而 Lua traced 产物由我们自己的 no-Python CPU 构建(paddle_core)
# 跑出——两个独立构建的卷积/BN kernel 浮点 accumulate 顺序不同,逐位
# 级差异不可避免(实测 max ~9.5e-7)。Python 自身的 dyn vs static 在同
# 构建内是逐位一致的(selfcheck 实测 0.0);1e-5 atol 覆盖跨构建 roundoff
# 且远小于任何结构错误(结构错误会是 O(1) 量级)。
ATOL = 1e-5
max_abs = float(np.max(np.abs(y - ref))) if y.shape == ref.shape else np.inf
ok = (y.shape == ref.shape and max_abs <= ATOL)
print("PY_READBACK " + ("PASS" if ok else "FAIL")
      + (f" (max|diff|={max_abs:.2e} <= {ATOL:.0e}, same weights, "
         f"cross-build roundoff)" if ok
         else f" (n={ref.size}: shape {y.shape} vs {ref.shape}, "
              f"max_diff={max_abs:.3e}, lua_first={ref.ravel()[:4].tolist()} "
              f"py_first={y[:4].tolist()})"))
sys.exit(0 if ok else 1)