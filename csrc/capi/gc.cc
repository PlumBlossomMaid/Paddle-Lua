// gc.cc -- GC heap tracking (layers 1-3 of the 9-layer scheme).
// Stage: M2 / 3.3 (P14, 14-gc.md sec.3.1).
//
// Faithful port of Torch7's THGeneral.c heap tracking (verified line by
// line against docs/migration/research/_ref/THGeneral.c:134-238):
//   :138  heapMaxDelta = 1e6          +/-1MB batch before touching global
//   :140  heapSoftmax  = 3e8          300MB soft cap, grows x1.4
//   :142  growthThresh = 0.8          grow softmax if post-GC heap > 80%
//   :143  growthFactor = 1.4
//   :134  __thread handler+data       each thread registers its own bridge
//
// Layering (14-gc.md table):
//   layer 1  pd_gc_account(bytes) per LuaTensor create/destroy
//   layer 2  1MB batch merge into the global atomic heapSize
//   layer 3  x1.4 softmax growth so GC does not thrash at a fixed cap
//   layer 6-8 (OOM callback) land in a later slice (register_oom).
//
// CRITICAL safety rule (THGeneral.c:219 `if (size > 0)`): the handler
// fires ONLY on positive flushes. LuaTensor destructors run inside Lua
// GC finalizers (sol2 usertype __gc); calling lua_gc from a finalizer
// would re-enter the collector. Allocation happens in normal Lua call
// context, so the bridge (which calls lua_gc) is only ever invoked there.
//
// Thread model: heap_delta + softmax + handler + handler_data are
// thread_local; heap_size is a global atomic. A lane's flush reads the
// GLOBAL heap size against its OWN softmax, then runs lua_gc on its OWN
// lua_State (never another lane's -- 14-gc.md sec.4 坑②). Default ON
// per D6; paddle.set_heap_tracking(false) turns it off (14-gc.md 坑④).
//
// The fast path (within the +/-1MB batch) is two adds + one compare --
// no atomics, no locking; ~3 extra trivial pd_* calls per tensor on the
// enabled path (numel+dtype are plain field reads).

#include <atomic>
#include <cstdint>

#include "paddle_c.h"

namespace {

/* THGeneral.c:136-140 */
constexpr int64_t kHeapMaxDelta = 1000000;    // +1MB flush threshold
constexpr int64_t kHeapMinDelta = -1000000;   // -1MB flush threshold
constexpr int64_t kHeapSoftmaxInit = 300000000;  // 300MB (THGeneral.c:140)
constexpr int64_t kGrowthThreshNumer = 8;     // softmax * 0.8 (integer)
constexpr int64_t kGrowthFactorNumer = 14;    // softmax * 1.4 (integer)
constexpr int64_t kGrowthFactorDenom = 10;

std::atomic<int64_t> g_heap_size{0};          // global: merged live bytes
std::atomic<int>     g_enabled{1};            // D6: ON by default

/* THGeneral.c:134-135 -- thread-local per-lane state. Re-initialized per
 * thread; each lane registers its own lua_State-bound bridge. */
thread_local int64_t g_heap_delta = 0;
thread_local int64_t g_heap_softmax = kHeapSoftmaxInit;
thread_local void (*g_handler)(void* data) = nullptr;
thread_local void* g_handler_data = nullptr;

/* Flush the thread-local delta into the global size; return the new
 * global value (THGeneral.c:169-180 applyHeapDelta). */
int64_t apply_heap_delta() {
  int64_t old_size =
      g_heap_size.fetch_add(g_heap_delta, std::memory_order_relaxed);
  int64_t new_size = old_size + g_heap_delta;
  g_heap_delta = 0;
  return new_size;
}

/* THGeneral.c:182-193 maybeTriggerGC. Only ever called from a POSITIVE
 * flush (allocation path -- normal Lua context, safe to run lua_gc). */
void maybe_trigger_gc(int64_t cur_heap_size) {
  if (g_handler != nullptr && cur_heap_size > g_heap_softmax) {
    g_handler(g_handler_data);

    // Re-check post-GC: if the heap is still > 80% of the softmax, the
    // GC could not reclaim enough -- raise the soft cap x1.4 so we do
    // not thrash (THGeneral.c:142-143, integer math: after > 0.8*softmax
    //  <=> after*10 > softmax*8).
    int64_t after = apply_heap_delta();
    if (after * 10 > g_heap_softmax * kGrowthThreshNumer) {
      g_heap_softmax =
          g_heap_softmax * kGrowthFactorNumer / kGrowthFactorDenom;
    }
  }
}

}  // namespace

extern "C" {

void pd_gc_account(int64_t bytes) {
  if (g_enabled.load(std::memory_order_relaxed) == 0) {
    return;
  }
  g_heap_delta += bytes;

  // layer 2: batch. Stay within the +/-1MB band without touching the
  // global atomic (THGeneral.c:214-218).
  if (g_heap_delta <= kHeapMaxDelta && g_heap_delta >= kHeapMinDelta) {
    return;
  }
  int64_t new_size = apply_heap_delta();

  // layer 3 (and the handler): only allocation triggers GC (see the
  // safety rule in the file header; THGeneral.c:219).
  if (bytes > 0) {
    maybe_trigger_gc(new_size);
  }
}

int pd_gc_set_enabled(int on) {
  int prev = g_enabled.exchange(on ? 1 : 0, std::memory_order_relaxed);
  // Disabling is a RESET, not a pause: drop the global ledger and any
  // pending delta. Heap tracking is a GC-pressure heuristic, not a
  // memory auditor -- a freeze would leave stale +bytes from objects
  // created before the toggle to unbalance the next run.
  if (!on) {
    g_heap_size.store(0, std::memory_order_relaxed);
    g_heap_delta = 0;
  }
  return prev;
}

int pd_gc_enabled(void) {
  return g_enabled.load(std::memory_order_relaxed);
}

void pd_gc_set_handler(void (*handler)(void* data), void* data) {
  g_handler = handler;
  g_handler_data = data;
}

void pd_gc_set_softmax(int64_t bytes) {
  g_heap_softmax = bytes > 0 ? bytes : kHeapSoftmaxInit;
}

int64_t pd_gc_tracked_bytes(void) {
  return g_heap_size.load(std::memory_order_relaxed);
}

}  // extern "C"