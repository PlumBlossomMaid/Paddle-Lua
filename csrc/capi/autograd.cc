// autograd.cc -- autograd entry points for the C ABI layer.
// Stage: P1 / M1 (WORKPLAN 2.1.4). Follows tensor.cc (layout.md sec.3).
//
// Design: docs/migration/plan/modules/01-c-abi.md sec.3.3 group 4.
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, v3.5.0):
//   paddle/fluid/eager/backward.h:26-31          egr::Backward signature
//   paddle/fluid/eager/utils.h:123               EagerUtils::autograd_meta
//   paddle/fluid/eager/utils.h:149               EagerUtils::nullable_autograd_meta
//   paddle/fluid/eager/autograd_meta.h:71        AutogradMeta::Grad
//   paddle/fluid/eager/autograd_meta.h:118       AutogradMeta::StopGradient
//   paddle/fluid/eager/autograd_meta.h:122       AutogradMeta::SetStopGradient
// Reference usage (proven by M0 smoke, T-M0-14): tools/smoke/smoke_strided_slice.cc

#include <vector>

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/utils.h"

#include "capi_impl.h"

extern "C" {

pd_status pd_backward(const pd_tensor* tensors,
                      int ntensors,
                      const pd_tensor* inputs,
                      int ninputs,
                      int retain_graph) {
  PD_GUARD_BEGIN;
  (void)inputs;   // reserved: egr::Backward accumulates grad on every
  (void)ninputs;  // autograd-enabled leaf; per-input selection is Grad().
  if (tensors == nullptr || ntensors <= 0) {
    pd_set_error(PD_INVALID_ARG, "tensors empty");
    return PD_INVALID_ARG;
  }
  std::vector<paddle::Tensor> vec;
  vec.reserve(static_cast<size_t>(ntensors));
  for (int i = 0; i < ntensors; ++i) {
    if (tensors[i] == nullptr) {
      pd_set_error(PD_INVALID_ARG, "tensors[i] is null");
      return PD_INVALID_ARG;
    }
    vec.push_back(reinterpret_cast<const pd_tensor_t*>(tensors[i])->value);
  }
  // backward.h:26-31 -- Backward(tensors, grad_tensors, retain_graph,
  // create_graph=false). Empty grad_tensors => Paddle seeds ones.
  egr::Backward(vec, {}, retain_graph != 0, false);
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

int pd_tensor_get_stop_gradient(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return 1;
  egr::AutogradMeta* meta = egr::EagerUtils::nullable_autograd_meta(
      reinterpret_cast<const pd_tensor_t*>(t)->value);  // utils.h:149
  if (meta == nullptr) return 1;  // no meta => stop by default
  return meta->StopGradient() ? 1 : 0;  // autograd_meta.h:118
  PD_GUARD_END(1);
}

pd_status pd_tensor_set_stop_gradient(pd_tensor t, int stop) {
  PD_GUARD_BEGIN;
  if (t == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor is null");
    return PD_INVALID_ARG;
  }
  // utils.h:123 -- EagerUtils::autograd_meta(Tensor*) creates meta if absent
  // (only leaves need explicit false; smoke_strided_slice.cc does exactly
  // this before Backward). autograd_meta.h:122 -- SetStopGradient(bool).
  egr::EagerUtils::autograd_meta(
      &reinterpret_cast<pd_tensor_t*>(t)->value)
      ->SetStopGradient(stop != 0);
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

pd_tensor pd_tensor_grad(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return nullptr;
  egr::AutogradMeta* meta = egr::EagerUtils::nullable_autograd_meta(
      reinterpret_cast<const pd_tensor_t*>(t)->value);  // utils.h:149
  if (meta == nullptr) return nullptr;
  const paddle::Tensor& g = meta->Grad();  // autograd_meta.h:71
  if (!g.initialized()) return nullptr;    // no grad accumulated yet
  // New shell sharing the grad tensor's storage; caller owns & frees.
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{g});
  PD_GUARD_END(nullptr);
}

}  // extern "C"