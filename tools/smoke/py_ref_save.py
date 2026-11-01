# py_ref_save.py -- Python 侧参考基准(P15-A 步骤 4)
# 用 paddle 3.5.0 PIR 模式保存一个 matmul->add->relu MLP,看 jit.save
# 产物的 .json 结构(pd_op.data / pd_op.fetch / persistable),并验证
# jit.load 读回推理一致。这是 Lua trace 产物要做到的对照形态。
import paddle
import numpy as np

paddle.enable_static() if False else None


class MLP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = paddle.create_parameter(
            [3, 4], 'float32',
            default_initializer=paddle.nn.initializer.Constant(1.0))
        self.b = paddle.create_parameter(
            [4], 'float32',
            default_initializer=paddle.nn.initializer.Constant(-1.0))

    def forward(self, x):
        return paddle.nn.functional.relu(paddle.matmul(x, self.w) + self.b)


m = MLP()
x = paddle.full([2, 3], 2.0, 'float32')
out = m(x)
print("EAGER:", out.numpy().ravel().tolist())

# jit.save 需要 to_static;PIR 模式下直接存
sm = paddle.jit.to_static(m, input_spec=[paddle.static.InputSpec([2, 3], 'float32', 'x')])
paddle.jit.save(sm, 'py_mlp')
print("SAVED")

# load 读回
tl = paddle.jit.load('py_mlp')
y = tl(x)
print("LOADED:", y.numpy().ravel().tolist())
print("MATCH:", np.allclose(out.numpy(), y.numpy()))
