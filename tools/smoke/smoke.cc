// tools/smoke/smoke.cc
// M0 acceptance smoke test: prove that a WITH_PYTHON=OFF + ON_INFER=OFF
// build of Paddle can actually compute gradients through egr::Backward.
//
// References:
//   paddle/fluid/eager/backward.h:26            -> egr::Backward
//   paddle/phi/api/include/api.h:506            -> paddle::experimental::full
//   paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h
//                                               -> *_ad_func (generated)
//   paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h:37
//                                               -> multiply_ad_func (manual)
//   paddle/fluid/eager/utils.h:123              -> EagerUtils::autograd_meta
//   paddle/fluid/eager/autograd_meta.h:122      -> SetStopGradient
//   paddle/fluid/eager/autograd_meta.h:71       -> AutogradMeta::Grad

#include <cmath>
#include <iostream>
#include <vector>

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"

int main() {
  // MemoryMethod must be initialized before any phi allocation.
  // See test/cpp/eager task runner (paddle_gtest_main.cc:73) and
  // paddle/fluid/platform/init.cc:443.
  paddle::framework::InitMemoryMethod();

  // Init device contexts and tracer (mirrors eager_test::InitEnv).
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  // a = full({2,2}, 3.0), b = full({2,2}, 4.0)
  paddle::Tensor a = paddle::experimental::full(
      {2, 2},
      paddle::experimental::Scalar(3.0),
      phi::DataType::FLOAT32,
      phi::CPUPlace());
  paddle::Tensor b = paddle::experimental::full(
      {2, 2},
      paddle::experimental::Scalar(4.0),
      phi::DataType::FLOAT32,
      phi::CPUPlace());

  // Leaf tensors must allow gradient accumulation.
  egr::EagerUtils::autograd_meta(&a)->SetStopGradient(false);
  egr::EagerUtils::autograd_meta(&b)->SetStopGradient(false);

  // c = a * b (elementwise), d = sum(c).  d = 48.0
  paddle::Tensor c = multiply_ad_func(a, b);
  paddle::Tensor d = sum_ad_func(c, {}, phi::DataType::FLOAT32, false);

  // Backward from d:  d/da == b == 4.0, d/db == a == 3.0
  egr::Backward({d}, {});

  // Read gradients back from the autograd meta of a and b.
  auto* meta_a = egr::EagerUtils::unsafe_autograd_meta(a);
  const paddle::Tensor& grad_a = meta_a->Grad();
  auto dense_a = std::dynamic_pointer_cast<phi::DenseTensor>(grad_a.impl());
  const float* data_a = dense_a->data<float>();

  auto* meta_b = egr::EagerUtils::unsafe_autograd_meta(b);
  const paddle::Tensor& grad_b = meta_b->Grad();
  auto dense_b = std::dynamic_pointer_cast<phi::DenseTensor>(grad_b.impl());
  const float* data_b = dense_b->data<float>();

  float max_err = 0.0f;
  for (int i = 0; i < 4; ++i) {
    max_err = std::max(max_err, std::abs(data_a[i] - 4.0f));
    max_err = std::max(max_err, std::abs(data_b[i] - 3.0f));
  }

  if (max_err < 1e-6f) {
    std::cout << "SMOKE PASS: a.grad == 4.0, b.grad == 3.0, max_err="
              << max_err << std::endl;
    return 0;
  }
  std::cout << "SMOKE FAIL: max_err=" << max_err << std::endl;
  return 1;
}
