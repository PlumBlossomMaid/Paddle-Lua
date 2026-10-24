/*
 * paddle_c.h -- paddle-lua C ABI layer (the only public header of csrc/capi)
 *
 * Design: docs/migration/plan/modules/01-c-abi.md
 * Stage: P1 / M1 (WORKPLAN 2.1.1)
 *
 * Hard constraints (01-c-abi.md sec.1/3/4):
 *   1. This file is plain C (C99). It must stay free of C++ keywords:
 *      acceptance is a keyword-scan of this file returning zero hits.
 *   2. No C99 bool (LuaJIT FFI is picky). Use int, 0/1 (01-c-abi.md sec.4).
 *   3. No struct in the ABI. Pass shape as (ptr, n) pair (sec.4).
 *   4. pd_dtype integer values MUST match phi::DataType one-to-one; a
 *      static_assert table in paddle_c.cc catches drift at compile time.
 *
 * Ownership rule (01-c-abi.md sec.3.1):
 *   Every function returning a new pd_tensor / pd_place transfers ownership
 *   to the caller, who must call the matching _free. No borrowed returns.
 */

#ifndef PADDLE_C_H
#define PADDLE_C_H

/* Standard C headers only (int64_t, size_t, NULL). This file is consumed
 * by C99 compilers and by sol2 C++ TUs alike; these are always available. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Error handling (01-c-abi.md sec.3.2)                               */
/* ------------------------------------------------------------------ */

typedef enum {
  PD_OK = 0,
  PD_ERROR = 1,
  PD_OOM = 2,
  PD_INVALID_ARG = 3
} pd_status;

/* Thread-local (NOT a global buffer -- Lanes will call these in parallel,
 * P13). Valid until the next pd_* call. */
pd_status    pd_last_status(void);
const char*  pd_last_error(void);

/* ------------------------------------------------------------------ */
/* dtype (01-c-abi.md sec.3.4)                                        */
/* ------------------------------------------------------------------ */

/* Mirrors the enum in paddle/phi/common/data_type.h, values 1:1
 * (data_type.h:43-81, read 2026-10-24):
 *   UNDEFINED=0, BOOL, UINT8, INT8, UINT16, INT16, UINT32, INT32,
 *   UINT64, INT64, FLOAT32, FLOAT64, COMPLEX64, COMPLEX128, PSTRING,
 *   FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, NUM_DATA_TYPES.
 * NUM_DATA_TYPES (upstream data_type.h:79) is a sentinel, not exported. */
typedef enum {
  PD_DTYPE_UNDEFINED = 0,
  PD_DTYPE_BOOL,
  PD_DTYPE_UINT8,
  PD_DTYPE_INT8,
  PD_DTYPE_UINT16,
  PD_DTYPE_INT16,
  PD_DTYPE_UINT32,
  PD_DTYPE_INT32,
  PD_DTYPE_UINT64,
  PD_DTYPE_INT64,
  PD_DTYPE_FLOAT32,
  PD_DTYPE_FLOAT64,
  PD_DTYPE_COMPLEX64,
  PD_DTYPE_COMPLEX128,
  PD_DTYPE_PSTRING,
  PD_DTYPE_FLOAT16,
  PD_DTYPE_BFLOAT16,
  PD_DTYPE_FLOAT8_E4M3FN,
  PD_DTYPE_FLOAT8_E5M2,
  PD_DTYPE__NUM   /* count sentinel, not aligned with upstream */
} pd_dtype;

/* ------------------------------------------------------------------ */
/* Handles (01-c-abi.md sec.3.1)                                      */
/* ------------------------------------------------------------------ */

/* pd_tensor wraps a heap paddle::Tensor* (see tensor.cc).
 * pd_place wraps a heap phi::Place* (see device.cc, 2.1.6).
 *
 * Typedefs come BEFORE the place/lifetime declarations below: the C ABI
 * functions reference these handle types, and C needs them declared first. */

typedef struct pd_tensor_s*  pd_tensor;
typedef struct pd_place_s*   pd_place;

/* ------------------------------------------------------------------ */
/* place / device (01-c-abi.md sec.3.4)                              */
/* ------------------------------------------------------------------ */

/* No pd_place struct in the ABI (sec.4). A place is the pair
 * (device kind, device index); CPU always uses index 0.
 *
 * Kinds mirror paddle/phi/common/place.h AllocationType (place.h:32-43):
 *   UNDEFINED=0, CPU=1, GPU=2(CUDA), GPUPINNED=3, XPU=4, XPUPINNED=5,
 *   IPU=7, CUSTOM=9. MLU/NPU are NOT in upstream's enum, so they are not
 * represented here either. */
typedef enum {
  PD_PLACE_CPU = 0,
  PD_PLACE_CUDA,
  PD_PLACE_CUDAPINNED,
  PD_PLACE_XPU,
  PD_PLACE_XPUPINNED,
  PD_PLACE_IPU,
  PD_PLACE_CUSTOM
} pd_place_kind;

/* Creates a place handle. Caller owns it; must call pd_place_free. */
pd_place   pd_place_new(pd_place_kind kind, int index);
void       pd_place_free(pd_place p);

/* ------------------------------------------------------------------ */
/* Group 1: lifetime (6 fns, 01-c-abi.md sec.3.3)                     */
/* ------------------------------------------------------------------ */

/* Empty tensor (no data allocated). */
pd_tensor  pd_tensor_new_empty(pd_place place);

/* Create by COPYING host data (a copy, never a borrow -- sec.4: data()
 * pointers must not be held long-term). */
pd_tensor  pd_tensor_from_host_buffer(pd_place place,
                                      const int64_t* dims, int ndim,
                                      int dtype,
                                      const void* data, size_t nbytes);

/* Release. Underlying memory lifetime is decided by refcounting
 * (sec.4: pd_tensor owns only the outer shell). */
void       pd_tensor_free(pd_tensor t);

/* Deep copy (dst device = place; blocking=0 => caller syncs device). */
pd_tensor  pd_tensor_clone(const pd_tensor t, pd_place place, int blocking);

/* New shell handle SHARING the same underlying storage (paddle::Tensor is
 * value-semantics: its copy ctor shares the TensorBase, sec.4 note 1).
 * Use it to hand the same tensor to Lua more than once; the caller owns the
 * returned handle and must free it. This is NOT a deep copy. */
pd_tensor  pd_tensor_retain(const pd_tensor t);

/* ------------------------------------------------------------------ */
/* Group 2: metadata (10 fns)                                         */
/* ------------------------------------------------------------------ */

/* All getters tolerate t==NULL: return 0/NULL instead of crashing
 * (plain C has no exceptions to lean on). */

int        pd_tensor_ndim(const pd_tensor t);
int64_t    pd_tensor_numel(const pd_tensor t);       /* element count */
int        pd_tensor_dtype(const pd_tensor t);       /* pd_dtype; 0=UNDEFINED */
/* Writes shape into caller buffer (capacity ndim_out); returns actual rank.
 * ndim_out==0 or dims==NULL => query only (two-phase pattern). */
int        pd_tensor_shape(const pd_tensor t, int64_t* dims, int ndim_out);
int        pd_tensor_is_contiguous(const pd_tensor t);   /* 0/1 */
/* Unpacks place: *kind gets pd_place_kind, *index gets device id. */
void       pd_tensor_place(const pd_tensor t, int* kind, int* index);

/* ------------------------------------------------------------------ */
/* Group 3: data movement (8 fns)                                     */
/* ------------------------------------------------------------------ */

/* Cross-device copy (semantics = paddle::Tensor::copy_to, tensor.h:492). */
pd_tensor  pd_tensor_copy_to(const pd_tensor t, pd_place place, int blocking);

/* Copy back to host buffer: caller allocates (capacity >=
 * numel * dtype_size). Returns bytes written; <0 => error (status set). */
int64_t    pd_tensor_to_host_buffer(const pd_tensor t, void* data, size_t nbytes);

/* dtype size in bytes (table lookup, no tensor needed). 0 on failure. */
size_t     pd_dtype_sizeof(int dtype);

/* ------------------------------------------------------------------ */
/* Group 4: autograd (8 fns; main impl 2.1.4)                         */
/* ------------------------------------------------------------------ */

/* Backward pass (tensors = leaf list to compute grads for;
 * retain_graph keeps the graph alive). Impl: evaluate Backward()
 * (01-c-abi.md sec.2, eager/backward.h:25). */
pd_status  pd_backward(const pd_tensor* tensors, int ntensors,
                       const pd_tensor* inputs, int ninputs,
                       int retain_graph);

/* Read/write stop_gradient (autograd meta via eager/utils.h:123/149). */
int        pd_tensor_get_stop_gradient(const pd_tensor t);
pd_status  pd_tensor_set_stop_gradient(pd_tensor t, int stop);

/* Get grad (NULL if none). Caller does NOT own it -- retain() to keep. */
pd_tensor  pd_tensor_grad(const pd_tensor t);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PADDLE_C_H */
