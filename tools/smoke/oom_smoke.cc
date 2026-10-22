// tools/smoke/oom_smoke.cc
// M0 #7 acceptance: prove RegisterOOMCallback + a releasing callback can
// rescue a failed allocation (the "lua_gc" idea from research/gc.md §3.1).
//
// We cannot wait for a real OOM on this machine, so we simulate the failure
// side: a MockAllocator that throws BadAlloc for the first N allocations,
// wrapped in a RetryAllocator (paddle/phi/core/memory/allocation/
// retry_allocator.h). We then register an OOM callback and check the retry
// loop calls it and retries when it reports freed bytes > 0.
//
// References:
//   retry_allocator.h:32     -> RegisterOOMCallback signature
//   retry_allocator.cc:60-82 -> retry loop: callback returns freed bytes,
//                                >0 means "keep retrying"
//   allocator.h:55           -> BadAlloc
//   allocator.h:191          -> Allocator base (pure virtual AllocateImpl)

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/common/place.h"

COMMON_DECLARE_int64(offload_retry_times);

namespace paddle::memory::allocation {

// Allocator that throws BadAlloc for the first `fail_times_` calls, then
// succeeds using plain malloc. Simulates an OOM that can be recovered.
class MockOomAllocator : public Allocator {
 public:
  explicit MockOomAllocator(int fail_times) : fail_times_(fail_times) {}

  phi::Allocation* AllocateImpl(size_t size) override {
    if (fail_times_.load() > 0) {
      fail_times_.fetch_sub(1);
      throw BadAlloc(std::string("mock OOM for size ") + std::to_string(size),
                     __FILE__,
                     __LINE__);
    }
    void* ptr = std::malloc(size);
    if (ptr == nullptr) {
      throw BadAlloc("malloc returned nullptr", __FILE__, __LINE__);
    }
    return new Allocation(ptr, size, phi::CPUPlace());
  }

  void FreeImpl(phi::Allocation* allocation) override {
    std::free(allocation->ptr());
    delete allocation;
  }

  bool IsAllocThreadSafe() const override { return true; }
  uint64_t ReleaseImpl(const phi::Place&) override { return 1; }

 private:
  std::atomic<int> fail_times_;
};

}  // namespace paddle::memory::allocation

int main() {
  using namespace paddle::memory::allocation;
  const phi::CPUPlace cpu;

  // OOM-recovery retry loop only runs when this flag is > 0
  // (retry_allocator.cc:67, default -1 in common/flags.cc:2363).
  FLAGS_offload_retry_times = 100;

  // --- Case 1: callback reports >0 freed bytes -> retry until success ---
  {
    int callback_calls = 0;
    RegisterOOMCallback([&](phi::Place, size_t) -> size_t {
      callback_calls++;
      return 1024;  // pretend lua_gc freed 1024 bytes -> keep retrying
    });
    auto mock = std::make_shared<MockOomAllocator>(/*fail_times=*/2);
    RetryAllocator retry(mock, cpu, /*retry_ms=*/0);
    bool threw = false;
    void* ptr = nullptr;
    try {
      auto alloc = retry.Allocate(4096);
      ptr = alloc->ptr();
    } catch (const BadAlloc&) {
      threw = true;
    }
    if (threw || ptr == nullptr) {
      std::fprintf(stderr,
                   "FAIL case1: callback rescued nothing (callback_calls=%d)\n",
                   callback_calls);
      return 1;
    }
    if (callback_calls != 2) {
      std::fprintf(stderr,
                   "FAIL case1: expected 2 callback calls, got %d\n",
                   callback_calls);
      return 1;
    }
    std::printf("case1 OK: callback called %d times, allocation rescued\n",
                callback_calls);
  }

  // --- Case 2: callback reports 0 freed bytes -> give up, still throws ---
  {
    int callback_calls = 0;
    RegisterOOMCallback([&](phi::Place, size_t) -> size_t {
      callback_calls++;
      return 0;  // lua_gc could not free anything -> stop retrying
    });
    auto mock = std::make_shared<MockOomAllocator>(/*fail_times=*/100);
    RetryAllocator retry(mock, cpu, /*retry_ms=*/0);
    bool threw = false;
    try {
      auto alloc = retry.Allocate(4096);
      (void)alloc;
    } catch (const BadAlloc&) {
      threw = true;
    }
    if (!threw) {
      std::fprintf(stderr, "FAIL case2: expected BadAlloc to escape\n");
      return 1;
    }
    // retry loop runs once (callback returns 0 -> has_offloaded=false), then
    // the final alloc_func() throws again; callback called exactly once.
    if (callback_calls != 1) {
      std::fprintf(stderr,
                   "FAIL case2: expected 1 callback call, got %d\n",
                   callback_calls);
      return 1;
    }
    std::printf("case2 OK: callback called once, BadAlloc propagated\n");
  }

  // --- Case 3: no callback registered -> no OOM-callback retry, but the
  // RetryAllocator's outer waiter (retry_allocator.cc:88-115) still retries
  // once after retry_time_ ms; with retry_ms=0 it retries immediately and
  // the last attempt throws -> BadAlloc escapes. fail_times must be large
  // enough that even the final attempt fails.
  {
    RegisterOOMCallback(nullptr);
    auto mock = std::make_shared<MockOomAllocator>(/*fail_times=*/100);
    RetryAllocator retry(mock, cpu, /*retry_ms=*/0);
    bool threw = false;
    try {
      auto alloc = retry.Allocate(4096);
      (void)alloc;
    } catch (const BadAlloc&) {
      threw = true;
    }
    if (!threw) {
      std::fprintf(stderr, "FAIL case3: expected BadAlloc without callback\n");
      return 1;
    }
    std::printf("case3 OK: no callback -> no OOM-callback retry, BadAlloc propagates\n");
  }

  std::printf("OOM SMOKE PASS: RegisterOOMCallback retry semantics verified\n");
  return 0;
}