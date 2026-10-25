// serialize.cc -- single-tensor save/load for the C ABI layer.
// Stage: P1 / M1 (WORKPLAN 2.7). Follows autograd.cc (layout.md sec.3).
//
// Design: docs/migration/plan/modules/07-serialization.md sec.3.2/3.3.
//   - paddle.save/load of a SINGLE tensor uses Paddle's own wire format
//     via framework::SaveTensor / LoadTensor (D19); pickle stays a
//     read-only bonus (lua-pickle, M2).
//   - every symbol below was read from $PADDLE_ROOT sources:
//     paddle/fluid/framework/io/save_load_tensor.h:20-24   SaveTensor/LoadTensor
//     paddle/fluid/framework/io/save_load_tensor.cc:24-54  impl: ofstream + phi::
//                                                           SerializeToStream /
//                                                           DeserializeFromStream
//     paddle/phi/api/include/tensor.h:115/418/425          Tensor(shared_ptr) /
//                                                           impl() / set_impl()

#include <memory>
#include <string>

#include "paddle/fluid/framework/io/save_load_tensor.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"

#include "capi_impl.h"

extern "C" {

// Save a tensor to `path` in Paddle's single-tensor wire format
// (save_load_tensor.h:20). overwrite is accepted for signature parity with
// upstream but not consulted: the impl opens with ofstream (truncates).
pd_status pd_tensor_save(const pd_tensor t,
                         const char* path,
                         int overwrite) {
  PD_GUARD_BEGIN;
  if (t == nullptr || path == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor/path is null");
    return PD_INVALID_ARG;
  }
  const paddle::Tensor& src = reinterpret_cast<const pd_tensor_t*>(t)->value;
  // tensor.h:418 -- impl() -> shared_ptr<TensorBase>; a validated dense
  // tensor holds a phi::DenseTensor impl.
  auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(src.impl());
  if (dense == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor has no DenseTensor impl");
    return PD_INVALID_ARG;
  }
  // save_load_tensor.h:20 -- SaveTensor(const DenseTensor&, path, overwrite)
  paddle::framework::SaveTensor(*dense, std::string(path), overwrite != 0);
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

// Load the single-tensor wire format at `path` into a NEW pd_tensor the
// caller owns (free it). The dtype/place are whatever the file holds.
pd_status pd_tensor_load(const char* path, pd_tensor* out) {
  PD_GUARD_BEGIN;
  if (path == nullptr || out == nullptr) {
    pd_set_error(PD_INVALID_ARG, "path/out is null");
    return PD_INVALID_ARG;
  }
  phi::DenseTensor dense;
  // save_load_tensor.h:24 -- LoadTensor(path, DenseTensor*)
  paddle::framework::LoadTensor(std::string(path), &dense);
  // tensor.h:115 -- Tensor(shared_ptr<TensorBase>) takes ownership.
  auto impl = std::make_shared<phi::DenseTensor>(std::move(dense));
  paddle::Tensor value(impl);
  *out = reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(value)});
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

}  // extern "C"