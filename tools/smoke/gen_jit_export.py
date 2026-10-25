# one-off: generate a NON-PIR jit.save export for the 2.8 smoke (v2).
# RUN + DELETE. matmul hit an old-IR dy2static bug (eager_utils.cc:2350)
# -- use only old-IR-safe ops: add, bias-broadcast, relu.
import paddle


class TinyNet(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.b = paddle.create_parameter(
            shape=[dim], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        y = paddle.add(x, x)          # elementwise_add
        y = paddle.add(y, self.b)     # broadcast bias
        return paddle.nn.functional.relu(y)


net = TinyNet(4)
net.eval()
x_spec = [paddle.static.InputSpec(shape=[2, 4], dtype="float32")]
with paddle.pir_utils.OldIrGuard():
    paddle.jit.save(net, ".qwen/tmp/jit_export/tiny", input_spec=x_spec)
print("JIT_EXPORT_DONE_NONPIR")