// custom_relu_cuda_stub.cc -- link-time CUDA stubs for the official
// custom_relu_op.cc on a CPU-only Windows build.
//
// The official example (paddle/test/custom_op/custom_relu_op.cc) forward-
// declares relu_cuda_* (defined in custom_relu_op.cu, only built when a GPU
// toolchain exists). Linux/macOS shared libraries tolerate undefined
// symbols at link time (resolved from the global scope at load), so the
// CPU-only Python build links it as-is. Windows DLL links do NOT: every
// referenced symbol must resolve. These stubs exist only to satisfy the
// linker -- ReluForward/Backward dispatch on x.is_cpu()/is_gpu() and this
// CPU-only host never reaches a CUDA branch; a reached stub throws loudly
// instead of silently misbehaving.
//
// The user-facing op source itself is UNCHANGED (P16 acceptance: Python
// example works zero-modified); the stub is a build-host concern only,
// equivalent to what a no-CUDA toolchain does implicitly on ELF.
//
// Signatures MUST match the forward declarations in custom_relu_op.cc
// exactly (read from paddle/test/custom_op/custom_relu_op.cc):
//   relu_cuda_forward(const paddle::Tensor& x)                       -> {out}
//   relu_cuda_backward(const paddle::Tensor& x,
//                      const paddle::Tensor& out,
//                      const paddle::Tensor& grad_out)               -> {grad_x}
//   relu_cuda_double_backward(const paddle::Tensor& out,
//                             const paddle::Tensor& ddx)             -> {ddout}
//   relu_cuda_backward_without_x(const paddle::Tensor& out,
//                                const paddle::Tensor& grad_out)     -> {grad_x}
//   relu_cuda_forward_out(const paddle::Tensor& x, paddle::Tensor* out)
//   relu_cuda_backward_out(const paddle::Tensor& x,
//                          const paddle::Tensor& out,
//                          const paddle::Tensor& grad_out,
//                          paddle::Tensor* grad_x)

#include <vector>
#include "paddle/extension.h"

namespace {
[[noreturn]] void cuda_unreachable(const char* fn) {
  PD_THROW(std::string("custom_relu: ") + fn +
           " reached on a CPU-only build (no CUDA toolchain)");
}
}  // namespace

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x) {
  cuda_unreachable("relu_cuda_forward");
}

std::vector<paddle::Tensor> relu_cuda_backward(
    const paddle::Tensor& x, const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  cuda_unreachable("relu_cuda_backward");
}

std::vector<paddle::Tensor> relu_cuda_double_backward(
    const paddle::Tensor& out, const paddle::Tensor& ddx) {
  cuda_unreachable("relu_cuda_double_backward");
}

std::vector<paddle::Tensor> relu_cuda_backward_without_x(
    const paddle::Tensor& out, const paddle::Tensor& grad_out) {
  cuda_unreachable("relu_cuda_backward_without_x");
}

void relu_cuda_forward_out(const paddle::Tensor& x, paddle::Tensor* out) {
  cuda_unreachable("relu_cuda_forward_out");
}

void relu_cuda_backward_out(const paddle::Tensor& x,
                            const paddle::Tensor& out,
                            const paddle::Tensor& grad_out,
                            paddle::Tensor* grad_x) {
  cuda_unreachable("relu_cuda_backward_out");
}