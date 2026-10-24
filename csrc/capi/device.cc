// device.cc -- device / stream queries for the C ABI layer.
// Stage: P1 / M1 (WORKPLAN 2.1.6). Follows autograd.cc (layout.md sec.3).
//
// Design: docs/migration/plan/modules/01-c-abi.md sec.3.3 group 5.
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, v3.5.0):
//   paddle/phi/backends/device_manager.h:221   DeviceManager::GetDeviceCount
//   paddle/phi/backends/gpu/gpu_info.h:58      GetCurrentDeviceId
//   paddle/phi/backends/gpu/gpu_info.h:72      SetDeviceId
//   paddle/phi/core/device_context.h:184       DeviceContext::Wait()
//   paddle/phi/core/memory/stats.h:143         DeviceMemoryStatCurrentValue
//   paddle/phi/core/memory/stats.h:150         HostMemoryStatCurrentValue
//   paddle/fluid/platform/init.h:443           InitMemoryMethod / InitDevices
//   paddle/fluid/eager/api/utils/global_utils.h  egr::Controller
//
// CPU-only build (WITH_GPU=OFF): GPU/XPU branches are #ifdef'd out below
// so this file links everywhere. Revisit when a GPU build lands.

#include <cstring>
#include <mutex>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/memory/stats.h"

#include "capi_impl.h"

extern "C" {

pd_status pd_init_once(void) {
  PD_GUARD_BEGIN;
  // Idempotent eager init. MUST run before any pd_* call that reaches the
  // phi kernel factory (e.g. pd_tensor_from_host_buffer -> empty), otherwise
  // the kernel factory throws (m0-report sec.4.2). Only the C ABI layer is
  // allowed to link Paddle (02-binding.md sec.2); the sol layer / smoke EXEs
  // call this through the Lua module instead of statically linking Paddle
  // themselves -- this also avoids duplicate gflags registration
  // (flags_native.cc:309) when paddle_core.dll is loaded into a process
  // that would otherwise link the same static libs.
  static std::once_flag init_flag;
  std::call_once(init_flag, [] {
    paddle::framework::InitMemoryMethod();  // init.h:443
    paddle::framework::InitDevices();
    egr::Controller::Instance().SetExpectedPlace(phi::CPUPlace());
  });
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

int pd_device_count(int kind) {
  PD_GUARD_BEGIN;
  switch (static_cast<pd_place_kind>(kind)) {
    case PD_PLACE_CPU:
      // CPU is not a DeviceManager-registered backend (device_manager.cc
      // only knows plugin devices like GPU/XPU); it always exists exactly
      // once. Do NOT call GetDeviceCount("CPU") -- it throws and we would
      // turn every query into an error.
      return 1;
    default:
      // -1 means "error" per 01-c-abi.md group 5; set the status so the
      // sol2 layer's pd::check() turns it into a Lua error.
      pd_set_error(PD_INVALID_ARG, "pd_device_count: unsupported kind");
      return -1;
  }
  PD_GUARD_END(-1);
}

int pd_device_current(int kind) {
  PD_GUARD_BEGIN;
  switch (static_cast<pd_place_kind>(kind)) {
    case PD_PLACE_CPU:
      return 0;  // CPU has a single implicit device id 0
#ifdef PADDLE_WITH_CUDA
    case PD_PLACE_CUDA:
      return phi::backends::gpu::GetCurrentDeviceId();  // gpu_info.h:58
    case PD_PLACE_CUDAPINNED:
      return phi::backends::gpu::GetCurrentDeviceId();
#endif
    default:
      pd_set_error(PD_INVALID_ARG, "pd_device_current: unsupported kind");
      return -1;
  }
  PD_GUARD_END(-1);
}

pd_status pd_device_set_current(int kind, int index) {
  PD_GUARD_BEGIN;
  switch (static_cast<pd_place_kind>(kind)) {
    case PD_PLACE_CPU:
      if (index != 0) {
        pd_set_error(PD_INVALID_ARG, "CPU device index must be 0");
        return PD_INVALID_ARG;
      }
      return PD_OK;  // nothing to switch; CPU is always current
#ifdef PADDLE_WITH_CUDA
    case PD_PLACE_CUDA:
      phi::backends::gpu::SetDeviceId(index);  // gpu_info.h:72
      return PD_OK;
#endif
    default:
      pd_set_error(PD_INVALID_ARG, "pd_device_set_current: unsupported kind");
      return PD_INVALID_ARG;
  }
  PD_GUARD_END(PD_ERROR);
}

pd_status pd_device_sync(int kind, int index) {
  PD_GUARD_BEGIN;
  // local kind->AllocationType map (tensor.cc keeps its own copy in an
  // anonymous namespace; do not share -- one function, kept local).
  phi::AllocationType at;
  switch (static_cast<pd_place_kind>(kind)) {
    case PD_PLACE_CPU:      at = phi::AllocationType::CPU;      break;
    case PD_PLACE_CUDA:     at = phi::AllocationType::GPU;      break;
    case PD_PLACE_CUDAPINNED: at = phi::AllocationType::GPUPINNED; break;
    case PD_PLACE_XPU:      at = phi::AllocationType::XPU;      break;
    case PD_PLACE_XPUPINNED: at = phi::AllocationType::XPUPINNED; break;
    case PD_PLACE_IPU:      at = phi::AllocationType::IPU;      break;
    case PD_PLACE_CUSTOM:   at = phi::AllocationType::CUSTOM;   break;
    default:
      pd_set_error(PD_INVALID_ARG, "pd_device_sync: bad kind");
      return PD_INVALID_ARG;
  }
  // CPU DeviceContext::Wait() is a no-op (device_context.h:184), but going
  // through the pool keeps the same code path for every backend.
  phi::Place place(at, static_cast<int8_t>(index));  // place.h:68
  phi::DeviceContextPool::Instance().Get(place)->Wait();  // context_pool.h:78
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

int64_t pd_device_allocated_bytes(int kind) {
  PD_GUARD_BEGIN;
  switch (static_cast<pd_place_kind>(kind)) {
    case PD_PLACE_CPU:
      // stats.h:150 -- HostMemoryStatCurrentValue("Allocated", 0)
      return paddle::memory::HostMemoryStatCurrentValue("Allocated", 0);
#ifdef PADDLE_WITH_CUDA
    case PD_PLACE_CUDA: {
      // stats.h:143 -- per-device stat; index 0 unless switched
      int dev = phi::backends::gpu::GetCurrentDeviceId();
      return paddle::memory::DeviceMemoryStatCurrentValue("Allocated", dev);
    }
#endif
    default:
      pd_set_error(PD_INVALID_ARG, "pd_device_allocated_bytes: unsupported kind");
      return -1;
  }
  PD_GUARD_END(-1);
}

const char* pd_place_kind_name(int kind) {
  switch (static_cast<pd_place_kind>(kind)) {
    case PD_PLACE_CPU:
      return "CPU";
    case PD_PLACE_CUDA:
      return "GPU";
    case PD_PLACE_CUDAPINNED:
      return "GPUPINNED";
    case PD_PLACE_XPU:
      return "XPU";
    case PD_PLACE_XPUPINNED:
      return "XPUPINNED";
    case PD_PLACE_IPU:
      return "IPU";
    case PD_PLACE_CUSTOM:
      return "CUSTOM";
    default:
      return "?";
  }
}

}  // extern "C"