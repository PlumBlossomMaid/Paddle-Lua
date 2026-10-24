// tensor_binding.cc -- LuaTensor usertype (shape/dtype surface for 2.2.2)
// Stage: P2 / M1 (WORKPLAN 2.2.2). After module.cc / error.cc
// (layout.md P2: step 2 "tensor_binding.cc <- usertype, bind shape/dtype").
//
// Design: docs/migration/plan/modules/02-binding.md.
//   - LuaTensor is a plain sol2 usertype, NOT a Lanes deep userdata;
//     the shell owns a pd_tensor handle, storage lives in paddle::Tensor's
//     shared_ptr (sec.3.2, tensor.h:142 ctor). GC frees the shell only.
//   - __lanesclone slot reserved now for P13 (sec.3.2): copy ctor via
//     pd_tensor_retain (new shell sharing storage) -- never memcpy.
//   - 5.1 has no __len on userdata (sec.4 pitfall 2), so size is numel()/
//     ndim() methods, never #x.
//
// This TU touches paddle_c.h ONLY (sec.2: the sol layer never includes
// Paddle headers), so version-porting = recompiling the same source.

#include "sol_common.hpp"

#include <cstdint>
#include <vector>

namespace pd {

/* One Lua-visible Tensor. Ownership is transfer-on-create, release on GC. */
struct LuaTensor {
  pd_tensor h;                        // opaque C ABI handle

  explicit LuaTensor(pd_tensor handle) : h(handle) {}
  ~LuaTensor() { pd_tensor_free(h); }

  /* Copy shares storage via retain (new shell, same data). Required by the
   * __lanesclone slot (02-binding.md sec.3.2) and by sol2 value returns. */
  LuaTensor(const LuaTensor& other) : h(pd_tensor_retain(other.h)) {}
  LuaTensor(LuaTensor&& other) noexcept : h(other.h) { other.h = nullptr; }
  LuaTensor& operator=(const LuaTensor&) = delete;
};

/* Factory: paddle_core.tensor{dims} -> 0-filled FLOAT32 CPU tensor.
 * 1-based dims come in as a Lua table; C ABI wants int64_t array.
 * NOTE: pd::check() must sit right after the fallible call -- every
 * PD_GUARD entry clears the error slot, so ANY intermediate pd_* call
 * (here: pd_place_free) would wipe the failure before we read it. */
void push_tensor_factory(sol::table& mod) {
  mod.set_function(
      "tensor",
      [](sol::table dims) -> LuaTensor {
        std::vector<int64_t> shape;
        for (auto& kv : dims) {
          shape.push_back(kv.second.as<int64_t>());
        }
        pd_place cpu = pd_place_new(PD_PLACE_CPU, 0);
        pd_tensor t = pd_tensor_from_host_buffer(
            cpu, shape.data(), static_cast<int>(shape.size()),
            static_cast<int>(PD_DTYPE_FLOAT32), nullptr, 0);
        pd::check();   // read the status NOW, before pd_place_free clears it
        pd_place_free(cpu);
        return LuaTensor(t);
      });
}

/* Register the LuaTensor usertype + factory into the module table (which
 * sits at stack index `module_index`, as created by luaopen_paddle_core).
 * Extra methods arrive in slices (2.2.4) and the Lua API layer (P5). */
void register_tensor(lua_State* L, int module_index) {
  sol::state_view lua(L);
  sol::table mod(L, sol::absolute_index(L, module_index));

  lua.new_usertype<LuaTensor>(
      "paddle_core.Tensor",
      sol::constructors<LuaTensor(pd_tensor)>(),
      /* methods on the C ABI metadata group (paddle_c.h Group 2) */
      "ndim",
      [](const LuaTensor& t) { return pd_tensor_ndim(t.h); },
      "numel",
      [](const LuaTensor& t) { return pd_tensor_numel(t.h); },
      "dtype",
      [](const LuaTensor& t) { return pd_tensor_dtype(t.h); },
      "shape",
      [](const LuaTensor& t) {
        std::vector<int64_t> dims(pd_tensor_ndim(t.h), 0);
        pd_tensor_shape(t.h, dims.data(), static_cast<int>(dims.size()));
        return sol::as_table(dims);   // Lua table, not a sol container usertype
      },
      "is_contiguous",
      [](const LuaTensor& t) { return pd_tensor_is_contiguous(t.h); },
      /* reserved now for P13 (02-binding.md sec.3.2) */
      "__lanesclone",
      [](void* dst, const LuaTensor& src) { new (dst) LuaTensor(src); });

  push_tensor_factory(mod);
  register_slice(mod);   // slice.cc: core.slice_spec (D14 string transform)
}

}  // namespace pd