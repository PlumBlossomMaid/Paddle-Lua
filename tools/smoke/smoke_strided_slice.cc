// tools/smoke/smoke_strided_slice.cc
// T-M0-14: verify that x[::-1] via strided_slice is a VIEW (shares buffer,
// offset + negative stride) and that autograd gradients flow through it in a
// WITH_PYTHON=OFF + ON_INFER=OFF build.
//
// Question under test (plan/modules/13-lanes.md / variable_index.py):
//   paddle.Tensor.__getitem__ with step != 1 selects strided_slice
//   (variable_index.py:409-415). Is the eager op a view (shares buffer)
//   or a copy (fresh buffer)? And does egr::Backward propagate through it?
//
// MEASURED ANSWER (2026-10-23, this smoke):
//   VIEW. FLAGS_use_stride_kernel defaults to true (kernel_factory.cc:31), so
//   eager picks the STRIDED kernel which does out->ResetHolder(input.Holder())
//   and rewrites meta offset/strides (stride/strided_slice_kernel.cc:95-99).
//   Physically py == px + 3 floats (offset 12 bytes); logical y[i] == x[3-i].
//   The Alloc+Eigen path in funcs/strided_slice.h is the FALLBACK used only
//   when FLAGS_use_stride_kernel=false.
//
// References:
//   paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h:1038
//                                               -> strided_slice_ad_func (signature)
//   paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_grad_functions.h:656
//                                               -> strided_slice_grad_ad_func
//   paddle/phi/kernels/stride/strided_slice_kernel.cc:95-99
//                                               -> STRIDED kernel: out->ResetHolder(
//                                                  input.Holder()); meta.offset/strides
//                                                  rewritten -> shared-buffer VIEW
//   paddle/phi/core/kernel_factory.cc:31        -> FLAGS_use_stride_kernel default true
//   paddle/phi/kernels/funcs/strided_slice.h:266-281
//                                               -> StridedSliceCompute: Alloc + Eigen
//                                                  stridedSlice (FALLBACK copy path)
//   python/paddle/base/variable_index.py:330-343 -> x[::-1]: step<0, start=None ->
//                                                  MAX_INTEGER, end=None -> MIN_INTEGER
//   paddle/fluid/eager/backward.h:26-31          -> egr::Backward(tensors, grad_tensors, ...)
//   paddle/fluid/platform/init.h                 -> InitMemoryMethod / InitDevices
//   paddle/fluid/eager/api/utils/global_utils.h  -> egr::Controller::Instance()
//   paddle/fluid/eager/utils.h:123              -> EagerUtils::autograd_meta
//   paddle/fluid/eager/autograd_meta.h:122      -> SetStopGradient
//   paddle/fluid/eager/autograd_meta.h:71       -> AutogradMeta::Grad
//   paddle/phi/api/include/tensor.h             -> paddle::Tensor
//   paddle/phi/api/include/api.h                -> paddle::experimental::full
//   paddle/phi/core/dense_tensor.h              -> phi::DenseTensor
//   test/cpp/eager/task_tests/backward_test.cc:31-32
//                                               -> PD_DECLARE_KERNEL pattern
//
// Build: same 483-lib baseline as tools/smoke/smoke.cc (see process/m0-report.md
//   section 2), /FORCE:MULTIPLE, /bigobj. Run: smoke_strided_slice.exe
//
// Runtime DLLs must be on PATH (smoke output showed 0xC0000005 without them):
//   E:\code\Paddle\build-nopy\paddle\phi
//   E:\code\Paddle\build-nopy\paddle\common
//   E:\code\Paddle\build-nopy\third_party\install\onednn\bin
//   E:\code\Paddle\build-nopy\third_party\install\mklml\lib

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_grad_functions.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"

// The eager framework needs the kernels linked in explicitly (backward_test.cc
// convention). strided_slice runs on CPU in this build.
PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(strided_slice, CPU, ALL_LAYOUT);

int main() {
  // MemoryMethod must be initialized before any phi allocation.
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  // x = [1, 2, 3, 4] (1-D float32)
  paddle::Tensor x = paddle::experimental::full(
      {4},
      paddle::experimental::Scalar(0.0),
      phi::DataType::FLOAT32,
      phi::CPUPlace());
  {
    auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
    float* data = dense->mutable_data<float>(phi::CPUPlace());
    for (int i = 0; i < 4; ++i) data[i] = static_cast<float>(i + 1);
  }
  egr::EagerUtils::autograd_meta(&x)->SetStopGradient(false);

  // y = x[::-1]  (Python: start=None/step<0 -> MAX_INTEGER, end=None -> MIN_INTEGER,
  // strided_slice_ad_func(x, axes={0}, starts, ends, strides))
  const int64_t kMax = std::numeric_limits<int64_t>::max();
  const int64_t kMin = std::numeric_limits<int64_t>::min();
  paddle::Tensor y = strided_slice_ad_func(
      x, std::vector<int>{0}, paddle::experimental::IntArray({kMax}),
      paddle::experimental::IntArray({kMin}),
      paddle::experimental::IntArray({-1}));

  // 1) View semantics: y shares x's holder with meta.offset=12 bytes.
  //    STRIDED kernel (stride/strided_slice_kernel.cc:95-99) does
  //    out->ResetHolder(input.Holder()) and rewrites meta.offset/strides.
  auto dense_x = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<phi::DenseTensor>(y.impl());
  const float* px = dense_x->data<float>();
  const float* py = dense_y->data<float>();
  bool shares_holder = (dense_x->Holder().get() == dense_y->Holder().get());
  bool offset_is_12 = (dense_y->meta().offset == 12);
  // Write-through proof: mutating the view's buffer mutates x.
  // py points 12 bytes (3 floats) into x; y[i] physically == x[3-i].
  float* py_mut = const_cast<float*>(py);
  py_mut[0] = 99.0f;
  bool write_through = (std::abs(px[3] - 99.0f) < 1e-6f);
  py_mut[0] = 4.0f;  // restore
  // Logical values through the strided view: y[i] == x[3-i].
  const float expect[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  bool values_ok = true;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(px[3 - i] - expect[i]) > 1e-6f) values_ok = false;
  }

  // 2) Gradient flows: seed grad = [1, 2, 3, 4]. dy[i] = dx[3-i], so
  //    d/dx[3-i] picks up grad[i] -> x.grad should be [4, 3, 2, 1].
  paddle::Tensor seed = paddle::experimental::full(
      {4},
      paddle::experimental::Scalar(0.0),
      phi::DataType::FLOAT32,
      phi::CPUPlace());
  {
    auto dense_s = std::dynamic_pointer_cast<phi::DenseTensor>(seed.impl());
    float* data = dense_s->mutable_data<float>(phi::CPUPlace());
    for (int i = 0; i < 4; ++i) data[i] = static_cast<float>(i + 1);
  }
  egr::Backward({y}, {seed}, false, false);

  auto* meta_x = egr::EagerUtils::unsafe_autograd_meta(x);
  const paddle::Tensor& grad_x = meta_x->Grad();
  auto dense_gx = std::dynamic_pointer_cast<phi::DenseTensor>(grad_x.impl());
  const float* gx = dense_gx->data<float>();
  bool grad_ok = true;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(gx[i] - expect[i]) > 1e-6f) grad_ok = false;
  }

  if (shares_holder && offset_is_12 && write_through && values_ok &&
      grad_ok) {
    std::cout << "SMOKE PASS: strided_slice is VIEW (shared holder, "
              << "meta.offset=12, write-through x[3] changed), "
              << "y=[4,3,2,1] logical, x.grad=[4,3,2,1] "
              << "(gradient flowed through autograd)" << std::endl;
    return 0;
  }
  std::cout << "SMOKE FAIL: shares_holder=" << shares_holder
            << " offset_is_12=" << offset_is_12
            << " write_through=" << write_through
            << " values_ok=" << values_ok << " grad_ok=" << grad_ok
            << " (px=" << px << ", py=" << py << ")" << std::endl;
  return 1;
}
