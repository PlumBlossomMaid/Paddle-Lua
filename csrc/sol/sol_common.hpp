// sol_common.hpp -- shared internal header for csrc/sol (NOT part of the
// public ABI). Each .cc in this directory includes it; version differences
// between Lua 5.1/5.2/5.3/5.4/LuaJIT are handled by sol2 (02-binding.md
// sec.3.1) -- our code must never contain #if LUA_VERSION.
//
// Design: docs/migration/plan/modules/02-binding.md sec.3.3.
// The C ABI returns pd_status; Lua must see error() instead. pd::check()
// throws a sol::error carrying the C ABI's thread-local message, which
// sol2 converts into a lua_error at the call boundary.

#ifndef PADDLE_LUA_SOL_COMMON_H
#define PADDLE_LUA_SOL_COMMON_H

extern "C" {
#include "paddle_c.h"
}

#include <sol/sol.hpp>

#include <stdexcept>
#include <string>

namespace pd {

/* Throw sol::error with the C ABI's last error message. [[noreturn]]. */
inline void throw_status() {
  const char* msg = pd_last_error();
  throw sol::error(std::string("paddle: ") + (msg ? msg : "unknown error"));
}

/* Call after any pd_* call that can fail (02-binding.md sec.3.3). */
inline void check() {
  if (pd_last_status() != PD_OK) {
    throw_status();
  }
}

/* 3.3 GC layer 1: bytes a LuaTensor shell claims against the heap-tracking
 * ledger (14-gc.md sec.3.1). Zero when tracking is disabled -- one atomic
 * read, keeping the disabled path inside the <5% on/off budget (14-gc.md
 * sec.5) -- or when the handle is null. Claim = numel x dtype_sizeof. */
inline int64_t track_claim(pd_tensor h) {
  if (pd_gc_enabled() == 0 || h == nullptr) {
    return 0;
  }
  int64_t n = pd_tensor_numel(h);
  size_t esize = pd_dtype_sizeof(pd_tensor_dtype(h));
  return n * static_cast<int64_t>(esize);
}

/* One Lua-visible Tensor (02-binding.md sec.3.2). Ownership is
 * transfer-on-create, release on GC. The struct lives HERE, not in
 * tensor_binding.cc, because the generated gen_ops_*.cc must construct
 * and unwrap it (AGENTS sec.8.3) -- both TUs include sol_common.hpp.
 *
 * Copy shares storage via pd_tensor_retain (new shell, same data);
 * required by the __lanesclone slot and by sol2 value returns.
 *
 * 3.3 GC hooks (14-gc.md sec.3.1 layer 1): every shell that lives on the
 * Lua side accounts numel*dtype_sizeof bytes against the C-side ledger
 * (pd_gc_account, gc.cc). +bytes at birth, -bytes at death, so the
 * ledger equals "sum of live Lua-visible tensor bytes" -- the signal the
 * softmax path (=collectgarbage bridge) reacts to. A retain-created
 * shell re-accounts the same storage (2x while both live): that is the
 * conservative direction (GC pressure earlier, never OOM), the lifecycle
 * sum is still exactly zero per underlying tensor, and the disabled
 * path costs one pd_gc_enabled() atomic read only (<5% on/off
 * acceptance, 14-gc.md sec.5).
 *
 * Finalizer safety: ~LuaTensor runs inside Lua GC. pd_gc_account(-bytes)
 * only merges into the atomic ledger, it can never invoke the handler
 * (gc.cc only fires it when bytes>0, Torch7 THGeneral.c:219 `if (size>0)`
 * -- calling collectgarbage from a finalizer would re-enter the
 * collector). Zero accounting for h==nullptr (moved-from shells). */
struct LuaTensor {
  pd_tensor h;      // opaque C ABI handle
  int64_t tracked;  // bytes this shell claims (0 when tracking off)

  explicit LuaTensor(pd_tensor handle)
      : h(handle), tracked(track_claim(handle)) {
    if (tracked != 0) {
      pd_gc_account(tracked);
    }
  }
  ~LuaTensor() {
    // free() is null/idempotent: moved-from shells (h==nullptr) and
    // shells already released by the user's explicit t:free() reach here
    // as no-ops (pd_tensor_free(NULL) = delete nullptr, tensor.cc:229).
    free();
  }

  /* 3.3 GC layer 5 (14-gc.md sec.3): explicit release. Drops this
   * shell's handle NOW -- ledger is charged -tracked immediately and
   * pd_tensor_free(h) releases this shell's reference to the underlying
   * storage (paddle::Tensor shared_ptr; other retain-created shells
   * stay alive). After free() the shell is DEAD: h==nullptr, so the
   * destructor is a no-op (no double free, no double charge). Idempotent
   * -- a second free() on the same shell does nothing. Mirrors Torch7's
   * tensor:free() (user says "I am done with this tensor early", the
   * memory returns without waiting for Lua GC). */
  void free() {
    if (h == nullptr) {
      return;
    }
    if (tracked != 0) {
      pd_gc_account(-tracked);
      tracked = 0;
    }
    pd_tensor_free(h);
    h = nullptr;
  }

  LuaTensor(const LuaTensor& other)
      : h(pd_tensor_retain(other.h)), tracked(track_claim(other.h)) {
    if (tracked != 0) {
      pd_gc_account(tracked);
    }
  }
  LuaTensor(LuaTensor&& other) noexcept
      : h(other.h), tracked(other.tracked) {
    other.h = nullptr;
    other.tracked = 0;
  }
  LuaTensor& operator=(const LuaTensor&) = delete;
};

/* One Lua-visible Place (dtype-device.md sec.2.2: place is a value, but
 * at the paddle_core layer it is a RAII shell over the opaque C ABI
 * handle -- pd_place_new transfers ownership, ~LuaPlace frees it.
 * Not copyable: pd_place has no retain (paddle_c.h ownership rule), so
 * an op lambda borrows `const LuaPlace&` and the factory returns by
 * move. pd_place_free(NULL) is `delete nullptr` (tensor.cc:173) -- a
 * moved-from shell frees safely. */
struct LuaPlace {
  pd_place h;

  explicit LuaPlace(pd_place handle) : h(handle) {}
  ~LuaPlace() { pd_place_free(h); }

  LuaPlace(const LuaPlace&) = delete;
  LuaPlace& operator=(const LuaPlace&) = delete;
  LuaPlace(LuaPlace&& other) noexcept : h(other.h) { other.h = nullptr; }
};

/* Register the LuaTensor usertype + tensor factory into the module table
 * that sits at stack index `module_index` (defined in tensor_binding.cc,
 * 2.2.2). Called at the end of luaopen_paddle_core. */
void register_tensor(lua_State* L, int module_index);

/* Register core.slice_spec (lua spec -> 0-based string, D14) into a sol
 * table (slice.cc, 2.2.4). Pure string transform for P5 consumption. */
void register_slice(sol::table& mod);

/* Register the PD_DTYPE_* / PD_PLACE_* integer constants, the LuaPlace
 * usertype and the pd_place_new factory (enums_place.cc, 2.2.5). The
 * generated _ops wrappers reference these constants for attr defaults
 * (emit_lua DATA_TYPE_DEFAULTS), so this MUST run before any op call. */
void register_enums_place(lua_State* L, int module_index);

/* Register every pd_op_* C ABI function as a sol2 lambda on the module
 * table (generated: gen_ops_*.cc / gen_ops_all.cc, 2.3.4 closing item
 * 3). Runs last in luaopen_paddle_core, after the usertypes it needs. */
void register_ops(lua_State* L, int module_index);

/* Register the LuaLayer usertype (jit::Layer face) + core.jit_load
 * (jit_binding.cc, 2.8). Runs after register_ops. */
void register_jit(lua_State* L, int module_index);

/* Register the cross-lane tensor registry (core.tensor_registry_put /
 * take / clear, tensor_binding.cc, 3.2). Handle-table fallback of
 * 13-lanes.md sec.4.1: Lanes cannot deep-copy a sol2 usertype (its
 * metatable holds C functions under sol2-internal registry names), so
 * DataLoader workers exchange only integer ids and tensors travel
 * through this process-wide registry (zero data copy). */
void register_tensor_registry(sol::table& mod);

}  // namespace pd

#endif  // PADDLE_LUA_SOL_COMMON_H