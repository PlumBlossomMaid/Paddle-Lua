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

/* Deep-copy src's data INTO dst's EXISTING storage: dst keeps its handle
 * identity, autograd meta and place -- exactly Python's in-place
 * parameter update (_C_ops.sgd_/adam_ write ParamOut back into Param,
 * 10-optimizer.md sec.4). Source: tensor.h:510 Tensor::copy_ /
 * tensor_method.cc:98. Requires dtype / tensor-type / place to match
 * (tensor_method.cc:106-118); failure sets the status slot. */
pd_status  pd_tensor_copy_from(pd_tensor dst, const pd_tensor src);

/* Copy back to host buffer: caller allocates (capacity >=
 * numel * dtype_size). Returns bytes written; <0 => error (status set). */
int64_t    pd_tensor_to_host_buffer(const pd_tensor t, void* data, size_t nbytes);

/* dtype size in bytes (table lookup, no tensor needed). 0 on failure. */
size_t     pd_dtype_sizeof(int dtype);

/* ------------------------------------------------------------------ */
/* Group 4: autograd (8 fns; main impl 2.1.4)                         */
/* ------------------------------------------------------------------ */

/* Backward pass (tensors = leaf list to compute grads for;
 * grad_tensors seeds the initial gradients (empty = all ones);
 * retain_graph keeps the graph alive. Impl: evaluate Backward()
 * (01-c-abi.md sec.2, eager/backward.h:25). */
pd_status  pd_backward(const pd_tensor* tensors, int ntensors,
                       const pd_tensor* grad_tensors, int ngrad_tensors,
                       int retain_graph);

/* Grad() -- compute grads of `inputs` w.r.t. `tensors` WITHOUT
 * accumulating onto .grad (backward.h:31-40). Caller allocates out_grads
 * (capacity nout >= ninputs); each slot gets a NEW pd_tensor the caller
 * owns (free it), or NULL when that input has no grad (allow_unused).
 * only_inputs=1 restricts the path to the inputs. */
pd_status  pd_grad(const pd_tensor* tensors, int ntensors,
                   const pd_tensor* inputs, int ninputs,
                   const pd_tensor* grad_tensors, int ngrad_tensors,
                   int retain_graph, int create_graph,
                   int only_inputs, int allow_unused,
                   pd_tensor* out_grads, int nout);

/* Global grad-enabled flag: egr::Controller::Instance().HasGrad() /
 * SetHasGrad (global_utils.h:75/77). Python no_grad/is_grad_enabled go
 * through core._has_grad/_set_has_grad (imperative.cc:641-642). */
int        pd_has_grad(void);
void       pd_set_has_grad(int has_grad);

/* Read/write stop_gradient (autograd meta via eager/utils.h:123/149). */
int        pd_tensor_get_stop_gradient(const pd_tensor t);
pd_status  pd_tensor_set_stop_gradient(pd_tensor t, int stop);

/* Get grad (NULL if none). Caller does NOT own it -- retain() to keep. */
pd_tensor  pd_tensor_grad(const pd_tensor t);

/* ------------------------------------------------------------------ */
/* Group 5: device / stream (6 fns, 01-c-abi.md sec.3.3; impl 2.1.6)  */
/* ------------------------------------------------------------------ */

/* Number of devices of the given kind (pd_place_kind). Error => -1.
 * Source: paddle/phi/backends/device_manager.h:221 GetDeviceCount. */
int        pd_device_count(int kind);

/* Current device index of the given kind, or -1 on error.
 * Source: paddle/phi/backends/gpu/gpu_info.h:58 GetCurrentDeviceId. */
int        pd_device_current(int kind);

/* Make `index` of the given kind the current device. Returns PD_OK.
 * Source: paddle/phi/backends/gpu/gpu_info.h:72 SetDeviceId. */
pd_status  pd_device_set_current(int kind, int index);

/* Block until all work on the given place (kind,index) has finished.
 * Source: paddle/phi/core/device_context.h:184 DeviceContext::Wait(). */
pd_status  pd_device_sync(int kind, int index);

/* Allocated bytes of the kind's memory stats; -1 => unavailable. */
int64_t    pd_device_allocated_bytes(int kind);

/* Pointer to a NUL-terminated device type name for a pd_place_kind
 * (e.g. "CPU", "GPU", "XPU"; "?" if unknown). Never NULL. */
const char* pd_place_kind_name(int kind);

/* One-time eager runtime init (idempotent). MUST be called before any pd_*
 * call that reaches the phi kernel factory (pd_tensor_from_host_buffer,
 * ops). Without it the kernel factory throws. Only the C ABI layer links
 * Paddle (02-binding.md sec.2), so every process embedding paddle_core
 * goes through this -- never static-link Paddle in the embedder itself
 * (duplicate gflags registration, flags_native.cc:309). */
pd_status  pd_init_once(void);

/* ------------------------------------------------------------------ */
/* Group 6: jit layers (6 fns, 08-jit-infer.md sec.3.2; impl 2.1.6)   */
/* ------------------------------------------------------------------ */

/* Opaque handle to a loaded jit::Layer (layer.h:44). */
typedef struct pd_layer_s*  pd_layer;

/* Load a jit-saved program (prefix/dir path). Returns handle or NULL on
 * error (status set). Source: paddle/fluid/jit/serializer.h:78. */
pd_layer   pd_jit_layer_load(const char* path, pd_place place);

/* Release a loaded layer. NULL-safe. */
void       pd_jit_layer_free(pd_layer layer);

/* Forward inputs (tensors array, ntensors count) through the layer.
 * Outputs are written to out array (capacity nout); returns output count.
 * Source: paddle/fluid/jit/layer.h:57. Inputs/outputs borrow; caller owns. */
int        pd_jit_layer_forward(pd_layer layer,
                                const pd_tensor* inputs, int ntensors,
                                pd_tensor* out, int nout);

/* Move the layer to a new place (layer.h:63 to()). Returns PD_OK. */
pd_status  pd_jit_layer_to(pd_layer layer, pd_place place);

/* Function names of the layer (layer.h:73 FunctionNames()).
 * Writes names into buf (capacity nb); returns count. */
int        pd_jit_layer_function_names(pd_layer layer,
                                       char* buf, size_t nbuf, size_t* len);

/* Clone the layer (layer.h:78 Clone() with null stream). */
pd_layer   pd_jit_layer_clone(pd_layer layer);

/* ------------------------------------------------------------------ */
/* Group 7: single-tensor serialization (2 fns, 07-serialization.md)  */
/* ------------------------------------------------------------------ */

/* Save a tensor in Paddle's single-tensor wire format (D19, T-M0-19
 * verified: versioned, lossless, byte-identical re-save).
 * Source: paddle/fluid/framework/io/save_load_tensor.h:20. */
pd_status  pd_tensor_save(const pd_tensor t, const char* path, int overwrite);

/* Load the single-tensor wire format at `path` into a NEW pd_tensor the
 * caller owns (free it). dtype/place come from the file.
 * Source: paddle/fluid/framework/io/save_load_tensor.h:24. */
pd_status  pd_tensor_load(const char* path, pd_tensor* out);

/* ------------------------------------------------------------------ */
/* Group 8: gzip decompression (1 fn, 12-dataset-vision.md sec.4.2)   */
/* ------------------------------------------------------------------ */

/* Decompress a gzip stream in memory. Two-pass protocol (mirrors
 * pd_jit_layer_function_names):
 *   pass 1: out=NULL, cap=0 -> *out_len = required byte count
 *   pass 2: out=buffer, cap>=*out_len -> decompressed bytes written
 * Paddle ships no bindable inflate (grepped paddle/ for gzopen/inflate
 * Init/uncompress: zero hits), so this links zlib (zlib.h:1113-1116
 * inflateInit2 windowBits 15+16 auto-detect gzip). Used by
 * paddle.vision.datasets.MNIST to read the .gz IDX files. */
pd_status  pd_gzip_inflate(const unsigned char* src, size_t src_len,
                           unsigned char* out, size_t cap, size_t* out_len);

/* ------------------------------------------------------------------ */
/* Group 9: image ops (2 fn, 12-dataset-vision.md sec.3.2/4.2)        */
/* ------------------------------------------------------------------ */

/* Resize an image buffer (H,W,C) row-major, mirroring the OpenCV
 * coordinate mapping functional_cv2.py:59-120 delegates to (resize.cpp:
 * nearest floor(dx*scale); bilinear (dx+0.5)*scale-0.5 + 4-neighbour).
 * Insight7 has no 2D resize and Paddle's interp ops are Tensor-level,
 * so transforms (which must stay on insight.Array, 12-dataset-vision.md
 * sec.3.2) call this directly.
 *   interp: 0=nearest, 1=bilinear
 *   dtype:  PD_DTYPE_UINT8 or PD_DTYPE_FLOAT32 only
 * Output size is computable, so one call with dst_cap >=
 * dst_h*dst_w*channels*itemsize suffices; *dst_len gets the count. */
pd_status  pd_image_resize(const unsigned char* src, size_t src_len,
                           int src_h, int src_w, int channels, int dtype,
                           int dst_h, int dst_w, int interpolation,
                           unsigned char* out, size_t dst_cap,
                           size_t* dst_len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PADDLE_C_H */
