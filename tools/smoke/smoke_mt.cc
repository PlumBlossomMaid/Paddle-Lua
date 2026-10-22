// tools/smoke/smoke_mt.cc
// M0 #3 acceptance: is it safe for multiple threads to share phi CPU kernels /
// the framework DeviceContext? Answers the open question in
// research/dataloader.md §3.3 and plan/modules/13-lanes.md 坑①.
//
// Design: N threads run the same graph as smoke.cc concurrently, many rounds.
//   a=full({2,2},3), b=full({2,2},4), c=multiply_ad_func(a,b), d=sum_ad_func(c),
//   egr::Backward({d},{});  expect a.grad==4.0, b.grad==3.0 every round.
// All threads pull the DeviceContext from the framework pool (shared), exactly
// like the binding layer + DataLoader workers will.
//
// References: see smoke.cc header (same sources). Threads use std::thread only;
// no Lanes involvement here - this isolates the phi/DeviceContext question.

#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"

namespace {

// One full forward+backward round; returns max_err over a/b grads.
float RunRound() {
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

  egr::EagerUtils::autograd_meta(&a)->SetStopGradient(false);
  egr::EagerUtils::autograd_meta(&b)->SetStopGradient(false);

  // c = a * b (elementwise), d = sum(c).  d = 48.0
  paddle::Tensor c = multiply_ad_func(a, b);
  paddle::Tensor d = sum_ad_func(c, {}, phi::DataType::FLOAT32, false);

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
  return max_err;
}

}  // namespace

int main(int argc, char** argv) {
  // Init is single-threaded, before any workers spawn. See smoke.cc for the
  // two mandatory calls (InitMemoryMethod / InitDevices).
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());

  const int kThreads = (argc > 1) ? std::atoi(argv[1]) : 4;
  const int kRounds = (argc > 2) ? std::atoi(argv[2]) : 50;
  if (kThreads < 1 || kRounds < 1) {
    std::cerr << "usage: smoke_mt [threads] [rounds-per-thread]" << std::endl;
    return 2;
  }

  std::atomic<int> failures{0};
  std::atomic<int> rounds_done{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      for (int r = 0; r < kRounds; ++r) {
        float err = RunRound();
        if (err >= 1e-6f) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
        rounds_done.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }

  if (failures.load() == 0) {
    std::cout << "SMOKE MT PASS: " << kThreads << " threads x " << kRounds
              << " rounds (" << rounds_done.load()
              << " total), all grads == 4.0/3.0, max_err=0" << std::endl;
    return 0;
  }
  std::cout << "SMOKE MT FAIL: " << failures.load() << " bad rounds of "
            << rounds_done.load() << std::endl;
  return 1;
}
