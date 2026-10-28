// registry.cc -- cross-lane tensor registry for the C ABI layer.
// Stage: M2 / 3.2 (P13, 13-lanes.md sec.4.1 handle-table fallback).
//
// Why this exists: Lanes cannot deep-copy a sol2 usertype. When it copies
// a LuaTensor it FIRST copies the userdata's metatable (tools.c:1581-1601
// inter_copy_one(mt)), and sol2 metatables hold C closures registered
// under sol2-internal registry names (e.g. `_R/7/place_kind_index`); Lanes
// requires every C function crossing a lane to be in its transfer
// database (populate_func_lookup_table), which sol2 does not do. Verified
// 2026-10-28: lanes_tensor_probe FAILED with
//   function '_R/7/place_kind_index' not found in main origin transfer
//   database.
// Q-02 is therefore CLOSED with "usertype metatable not transferable";
// D24's __lanesclone slot stays (harmless), but the DataLoader worker
// protocol sends only integer ids through the Linda and exchanges tensors
// through this registry -- zero data copy (shared_ptr refcount +1), which
// is exactly the "只传一个整数" fallback of 13-lanes.md sec.4.1.
//
// Refcount design (paddle_c.h contract: pd_tensor owns an outer shell; the
// underlying buffer lives via paddle::Tensor's internal shared_ptr):
//   put:  store paddle::Tensor value (copy ctor shares TensorBase)  -> +1
//   take: copy the stored value out (shared_ptr +1), erase, wrap in a NEW
//         pd_tensor shell the caller owns and must free (pd_tensor_free).
// The registry's own reference is dropped by erase; main/worker shells
// then hold the only remaining references. Last shell out frees the
// buffer -- identical to ordinary in-process tensor passing.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "paddle/phi/api/include/tensor.h"

#include "capi_impl.h"

namespace {

std::mutex g_registry_mutex;
std::unordered_map<int64_t, paddle::Tensor> g_registry;
std::atomic<int64_t> g_next_id{1};  // ids are never reused; 0 means absent

}  // namespace

extern "C" {

int64_t pd_tensor_registry_put(const pd_tensor t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_tensor_registry_put: tensor is null");
    return 0;
  }
  int64_t id = g_next_id.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    // Copy: paddle::Tensor is value-semantics (shared TensorBase), so this
    // bumps the shared_ptr refcount ("+1" per the header contract).
    g_registry.emplace(id, reinterpret_cast<pd_tensor_t*>(t)->value);
  }
  return id;
  PD_GUARD_END(0);
}

pd_tensor pd_tensor_registry_take(int64_t id) {
  PD_GUARD_BEGIN;
  paddle::Tensor value;
  {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_registry.find(id);
    if (it == g_registry.end()) {
      pd_set_error(PD_INVALID_ARG, "pd_tensor_registry_take: unknown id");
      return nullptr;
    }
    value = std::move(it->second);  // move out; erase drops the registry ref
    g_registry.erase(it);
  }
  // New shell over the same storage; caller owns and frees it.
  return reinterpret_cast<pd_tensor>(new pd_tensor_t{std::move(value)});
  PD_GUARD_END(nullptr);
}

void pd_tensor_registry_clear(void) {
  PD_GUARD_BEGIN;
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  g_registry.clear();
  PD_GUARD_END_VOID;
}

}  // extern "C"