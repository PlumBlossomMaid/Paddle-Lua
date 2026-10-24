// tensor.cc -- Tensor lifetime / metadata / data-movement for the C ABI
// layer. Stage: P1 / M1 (WORKPLAN 2.1.3). Second .cc after status.cc
// (layout.md sec.3: exception boundary first, then the real API).
//
// Handle layout (01-c-abi.md sec.3.1 / sec.4 note 1):
//   pd_tensor/pd_place are OPAQUE pointers across the C boundary; the real
//   structs are defined HERE (this .cc), so C++ type safety applies where
//   it counts and C callers only ever see an opaque handle. The shell owns
//   a paddle::Tensor whose storage lives via its internal shared_ptr --
//   freeing the shell never frees the buffer prematurely.
//
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, v3.5.0).

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

#include "capi_impl.h"

namespace {

// Compile-time alignment of the C-A-B-I dtype enum with phi::DataType
// (01-c-abi.md sec.3.4): an upstream enum change breaks the build instead
// of silently producing wrong dtypes at runtime.
#define PD_STATIC_ASSERT_DTYPE(pd, phi)                        \
  static_assert(static_cast<int>(pd) == static_cast<int>(phi), \
                "pd_dtype drift vs phi::DataType at " #pd)

PD_STATIC_ASSERT_DTYPE(PD_DTYPE_UNDEFINED, phi::DataType::UNDEFINED);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_BOOL, phi::DataType::BOOL);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_UINT8, phi::DataType::UINT8);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_INT8, phi::DataType::INT8);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_UINT16, phi::DataType::UINT16);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_INT16, phi::DataType::INT16);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_UINT32, phi::DataType::UINT32);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_INT32, phi::DataType::INT32);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_UINT64, phi::DataType::UINT64);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_INT64, phi::DataType::INT64);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_FLOAT32, phi::DataType::FLOAT32);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_FLOAT64, phi::DataType::FLOAT64);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_COMPLEX64, phi::DataType::COMPLEX64);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_COMPLEX128, phi::DataType::COMPLEX128);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_PSTRING, phi::DataType::PSTRING);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_FLOAT16, phi::DataType::FLOAT16);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_BFLOAT16, phi::DataType::BFLOAT16);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_FLOAT8_E4M3FN, phi::DataType::FLOAT8_E4M3FN);
PD_STATIC_ASSERT_DTYPE(PD_DTYPE_FLOAT8_E5M2, phi::DataType::FLOAT8_E5M2);
#undef PD_STATIC_ASSERT_DTYPE

// pd_place_kind -> phi::AllocationType (place.h:32-43).
phi::AllocationType to_allocation_type(pd_place_kind kind) {
  switch (kind) {
    case PD_PLACE_CPU:
      return phi::AllocationType::CPU;
    case PD_PLACE_CUDA:
      return phi::AllocationType::GPU;
    case PD_PLACE_CUDAPINNED:
      return phi::AllocationType::GPUPINNED;
    case PD_PLACE_XPU:
      return phi::AllocationType::XPU;
    case PD_PLACE_XPUPINNED:
      return phi::AllocationType::XPUPINNED;
    case PD_PLACE_IPU:
      return phi::AllocationType::IPU;
    case PD_PLACE_CUSTOM:
      return phi::AllocationType::CUSTOM;
  }
  return phi::AllocationType::UNDEFINED;  // unreachable; kind is validated
}

// reverse map for pd_tensor_place (place.h:32-43).
pd_place_kind to_pd_place_kind(phi::AllocationType type) {
  switch (type) {
    case phi::AllocationType::CPU:
      return PD_PLACE_CPU;
    case phi::AllocationType::GPU:
      /* CUDA is an alias of GPU (place.h:36) -- one case covers both. */
      return PD_PLACE_CUDA;
    case phi::AllocationType::GPUPINNED:
      return PD_PLACE_CUDAPINNED;
    case phi::AllocationType::XPU:
      return PD_PLACE_XPU;
    case phi::AllocationType::XPUPINNED:
      return PD_PLACE_XPUPINNED;
    case phi::AllocationType::IPU:
      return PD_PLACE_IPU;
    case phi::AllocationType::CUSTOM:
      return PD_PLACE_CUSTOM;
    default:
      return PD_PLACE_CPU;  // UNDEFINED encodes as CPU-kind; not expected
  }
}

// Dispatch a host-buffer copy by dtype. `dst` must already be shaped;
// t.mutable_data<T>() allocates on the stored place (tensor.h:346) -- the
// templated overload is the {dtype-dependent} way to ask Paddle to allocate.
template <typename T>
void* copy_into(paddle::Tensor* t, const void* src) {
  T* dst = t->mutable_data<T>();  // tensor.h:346
  if (dst != nullptr && src != nullptr) {
    std::memcpy(dst, src, t->numel() * sizeof(T));  // tensor.h:186 numel
  }
  return dst;
}

void* copy_into_by_dtype(paddle::Tensor* t, pd_dtype d, const void* src) {
  switch (d) {
    case PD_DTYPE_BOOL:
      return copy_into<bool>(t, src);
    case PD_DTYPE_UINT8:
      return copy_into<uint8_t>(t, src);
    case PD_DTYPE_INT8:
      return copy_into<int8_t>(t, src);
    case PD_DTYPE_UINT16:
      return copy_into<uint16_t>(t, src);
    case PD_DTYPE_INT16:
      return copy_into<int16_t>(t, src);
    case PD_DTYPE_UINT32:
      return copy_into<uint32_t>(t, src);
    case PD_DTYPE_INT32:
      return copy_into<int32_t>(t, src);
    case PD_DTYPE_UINT64:
      return copy_into<uint64_t>(t, src);
    case PD_DTYPE_INT64:
      return copy_into<int64_t>(t, src);
    case PD_DTYPE_FLOAT32:
      return copy_into<float>(t, src);
    case PD_DTYPE_FLOAT64:
      return copy_into<double>(t, src);
    case PD_DTYPE_FLOAT16:
      return copy_into<phi::dtype::float16>(t, src);
    case PD_DTYPE_BFLOAT16:
      return copy_into<phi::dtype::bfloat16>(t, src);
    default:
      pd_set_error(PD_INVALID_ARG, "dtype not host-copyable");
      return nullptr;
  }
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------ */
/* place                                                              */
/* ------------------------------------------------------------------ */

pd_place pd_place_new(pd_place_kind kind, int index) {
  PD_GUARD_BEGIN;
  if (index < 0) {
    pd_set_error(PD_INVALID_ARG, "device index must be >= 0");
    return nullptr;
  }
  phi::AllocationType at = to_allocation_type(kind);
  if (at == phi::AllocationType::UNDEFINED) {
    pd_set_error(PD_INVALID_ARG, "bad pd_place_kind");
    return nullptr;
  }
  // place.h:68 -- Place(AllocationType, int8_t id, dev_type="")
  return reinterpret_cast<pd_place>(new pd_place_t{
      phi::Place(at, static_cast<int8_t>(index))});
  PD_GUARD_END(nullptr);
}

void pd_place_free(pd_place p) {
  PD_GUARD_BEGIN;
  delete reinterpret_cast<pd_place_t*>(p);
  PD_GUARD_END_VOID;
}

/* ------------------------------------------------------------------ */
/* Group 1: lifetime                                                  */
/* ------------------------------------------------------------------ */

pd_tensor pd_tensor_new_empty(pd_place place) {
  PD_GUARD_BEGIN;
  if (place == nullptr) {
    pd_set_error(PD_INVALID_ARG, "place is null");
    return nullptr;
  }
  // tensor.h:121 declares Tensor(const Place&) but upstream NEVER implements
  // it (paddle/phi/api/lib/tensor.cc has only 4 ctors; phi.dll exports
  // none) -- calling it is a link error. Default ctor (tensor.h:98) gives
  // the empty shell this API promises; place binds at first allocation.
  paddle::Tensor t;
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(t)});
  PD_GUARD_END(nullptr);
}

pd_tensor pd_tensor_from_host_buffer(pd_place place,
                                     const int64_t* dims,
                                     int ndim,
                                     int dtype,
                                     const void* data,
                                     size_t nbytes) {
  PD_GUARD_BEGIN;
  (void)nbytes;  // byte count is implied by shape+dtype; caller may pass 0
  if (place == nullptr) {
    pd_set_error(PD_INVALID_ARG, "place is null");
    return nullptr;
  }
  if (ndim <= 0 || dims == nullptr) {
    pd_set_error(PD_INVALID_ARG, "ndim must be > 0 with dims != null");
    return nullptr;
  }
  std::vector<int64_t> shape(dims, dims + ndim);
  // tensor.h:135 declares Tensor(const Place&, shape) but upstream NEVER
  // implements it (same ghost-ctor issue as tensor.h:121) -- use the
  // exported factory instead (api.h:384).
  paddle::Tensor t = paddle::experimental::empty(
      shape, static_cast<phi::DataType>(dtype),
      reinterpret_cast<pd_place_t*>(place)->value);
  void* dst = copy_into_by_dtype(&t, static_cast<pd_dtype>(dtype), data);
  if (dst == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(t)});
  PD_GUARD_END(nullptr);
}

void pd_tensor_free(pd_tensor t) {
  PD_GUARD_BEGIN;
  delete reinterpret_cast<pd_tensor_t*>(t);  // shell only; storage inside
  PD_GUARD_END_VOID;
}

pd_tensor pd_tensor_clone(const pd_tensor t, pd_place place, int blocking) {
  PD_GUARD_BEGIN;
  if (t == nullptr || place == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor/place is null");
    return nullptr;
  }
  // tensor.h:492 / 501 -- copy_to(place) / copy_to(place, blocking)
  paddle::Tensor copied =
      reinterpret_cast<const pd_tensor_t*>(t)->value.copy_to(
          reinterpret_cast<pd_place_t*>(place)->value, blocking != 0);
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(copied)});
  PD_GUARD_END(nullptr);
}

pd_tensor pd_tensor_retain(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor is null");
    return nullptr;
  }
  // New shell sharing the same storage: copy ctor is value-semantics
  // (sec.4 note 1). NOT a deep copy -- use clone() for that.
  paddle::Tensor copy = reinterpret_cast<const pd_tensor_t*>(t)->value;
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(copy)});
  PD_GUARD_END(nullptr);
}

/* ------------------------------------------------------------------ */
/* Group 2: metadata                                                  */
/* ------------------------------------------------------------------ */

int pd_tensor_ndim(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return 0;
  return static_cast<int>(
      reinterpret_cast<const pd_tensor_t*>(t)->value.shape().size());
  // tensor.h:199 shape()
  PD_GUARD_END(0);
}

int64_t pd_tensor_numel(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return 0;
  return reinterpret_cast<const pd_tensor_t*>(t)->value.numel();  // tensor.h:186
  PD_GUARD_END(0);
}

int pd_tensor_dtype(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return static_cast<int>(PD_DTYPE_UNDEFINED);
  return static_cast<int>(
      reinterpret_cast<const pd_tensor_t*>(t)->value.dtype());  // tensor.h:224
  PD_GUARD_END(static_cast<int>(PD_DTYPE_UNDEFINED));
}

int pd_tensor_shape(const pd_tensor t, int64_t* dims, int ndim_out) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return 0;
  std::vector<int64_t> shape =
      reinterpret_cast<const pd_tensor_t*>(t)->value.shape();  // tensor.h:199
  int n = static_cast<int>(shape.size());
  if (dims != nullptr && ndim_out > 0) {
    int write = n < ndim_out ? n : ndim_out;
    for (int i = 0; i < write; ++i) dims[i] = shape[i];
  }
  return n;
  PD_GUARD_END(0);
}

int pd_tensor_is_contiguous(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return 0;
  return reinterpret_cast<const pd_tensor_t*>(t)->value.is_contiguous() ? 1
                                                                        : 0;
  // tensor.h:675
  PD_GUARD_END(0);
}

void pd_tensor_place(const pd_tensor t, int* kind, int* index) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return;
  const phi::Place& p =
      reinterpret_cast<const pd_tensor_t*>(t)->value.place();  // tensor.h:292
  if (kind != nullptr) *kind = static_cast<int>(to_pd_place_kind(p.GetType()));
  if (index != nullptr) *index = static_cast<int>(p.GetDeviceId());  // place.h:94
  PD_GUARD_END_VOID;
}

/* ------------------------------------------------------------------ */
/* Group 3: data movement                                             */
/* ------------------------------------------------------------------ */

pd_tensor pd_tensor_copy_to(const pd_tensor t, pd_place place, int blocking) {
  PD_GUARD_BEGIN;
  if (t == nullptr || place == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor/place is null");
    return nullptr;
  }
  paddle::Tensor copied =
      reinterpret_cast<const pd_tensor_t*>(t)->value.copy_to(
          reinterpret_cast<pd_place_t*>(place)->value, blocking != 0);
  // tensor.h:501
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(copied)});
  PD_GUARD_END(nullptr);
}

int64_t pd_tensor_to_host_buffer(const pd_tensor t, void* data, size_t nbytes) {
  PD_GUARD_BEGIN;
  if (t == nullptr || data == nullptr) {
    pd_set_error(PD_INVALID_ARG, "tensor/data is null");
    return -1;
  }
  const paddle::Tensor& src = reinterpret_cast<const pd_tensor_t*>(t)->value;
  const void* p = src.data();  // tensor.h:397 -- non-const data() also OK
  if (p == nullptr) {
    pd_set_error(PD_ERROR, "tensor has no data (uninitialized?)");
    return -1;
  }
  size_t total = static_cast<size_t>(src.numel()) *          // tensor.h:186
                 pd_dtype_sizeof(static_cast<int>(src.dtype()));  // :224
  size_t copy = total < nbytes ? total : nbytes;
  std::memcpy(data, p, copy);
  return static_cast<int64_t>(copy);
  PD_GUARD_END(-1);
}

size_t pd_dtype_sizeof(int dtype) {
  PD_GUARD_BEGIN;
  // Byte sizes mirror phi::SizeOf (data_type.h): bool/int8/uint8=1,
  // fp16/bf16=2, fp32/int32/uint32=4, fp64/int64/uint64=8, c64=8, c128=16.
  switch (static_cast<pd_dtype>(dtype)) {
    case PD_DTYPE_BOOL:
    case PD_DTYPE_UINT8:
    case PD_DTYPE_INT8:
    case PD_DTYPE_FLOAT8_E4M3FN:
    case PD_DTYPE_FLOAT8_E5M2:
      return 1;
    case PD_DTYPE_UINT16:
    case PD_DTYPE_INT16:
    case PD_DTYPE_FLOAT16:
    case PD_DTYPE_BFLOAT16:
      return 2;
    case PD_DTYPE_UINT32:
    case PD_DTYPE_INT32:
    case PD_DTYPE_FLOAT32:
    case PD_DTYPE_COMPLEX64:
      return 4;
    case PD_DTYPE_UINT64:
    case PD_DTYPE_INT64:
    case PD_DTYPE_FLOAT64:
    case PD_DTYPE_COMPLEX128:
      return 8;
    case PD_DTYPE_PSTRING:
    case PD_DTYPE_UNDEFINED:
    case PD_DTYPE__NUM:
      break;
  }
  pd_set_error(PD_INVALID_ARG, "unknown pd_dtype");
  return 0;
  PD_GUARD_END(0);
}

}  // extern "C"