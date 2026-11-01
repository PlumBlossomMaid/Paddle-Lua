# tools/smoke/py_jit_readback.py -- P15-A 步骤 4 读回验证(M3 闸门)
# Lua 侧 paddle.jit.trace 产物(build-test/lua_mlp.json + .pdiparams)
# 必须能被 Python paddle.jit.load 读回,且 x=full([2,3],2.0) 的推理输出
# 与 Lua traced 输出逐位一致(对照 build-test/lua_mlp.out.bin,应全 5.0)。
#
# 运行:Lua smoke 先跑(jit_trace_smoke.lua 产出 3 个文件),然后
#   python tools/smoke/py_jit_readback.py
import os
import sys

import numpy as np
import paddle

PREFIX = os.path.join(os.path.dirname(__file__), "..", "..", "build-test")
PREFIX = os.path.normpath(PREFIX)

model = os.path.join(PREFIX, "lua_mlp")

if not (os.path.exists(model + ".json")
        and os.path.exists(model + ".pdiparams")):
    print("PY_READBACK FAIL: lua_mlp artifacts missing -- run "
          "jit_trace_smoke.lua first")
    sys.exit(1)

tl = paddle.jit.load(model)
x = paddle.full([2, 3], 2.0, "float32")
y = tl(x).numpy().ravel()
print("PY_LOADED:", y.tolist())

ref = np.frombuffer(
    open(os.path.join(PREFIX, "lua_mlp.out.bin"), "rb").read(),
    dtype=np.float32)
ok = (y.shape == ref.shape
      and np.array_equal(y, ref)
      and np.allclose(y, 5.0, atol=1e-6))
print("PY_READBACK " + ("PASS" if ok else "FAIL")
      + (" (bitwise == Lua traced, all ~= 5.0)" if ok
         else f" (lua_ref={ref.tolist()} py={y.tolist()})"))
sys.exit(0 if ok else 1)