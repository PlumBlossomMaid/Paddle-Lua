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

/* NOTE: struct LuaTensor now lives in sol_common.hpp (the generated
 * gen_ops_*.cc needs to construct/unwrap it, AGENTS sec.8.3); this TU
 * only registers the usertype and the tensor factory. */

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
      /* Group-2 unpack: place(kind,index) -> two Lua ints (device.lua
       * reconstructs the value-semantic Place). */
      "place_kind_index",
      [](const LuaTensor& t) {
        int kind = 0, index = 0;
        pd_tensor_place(t.h, &kind, &index);
        return std::make_pair(kind, index);
      },
      /* 2.5 tensor.lua: item() needs a scalar read. Only 0-d / single
       * element; multi-element is a Lua-side error (api/tensor.md 2.3).
       * pd_tensor_to_host_buffer sets the error slot on failure, so
       * pd::check() must sit right after it (PD_GUARD entry clears the
       * previous error). Layout is host endianness = phi's memory
       * layout (float16.h etc. are byte-exact), so memcpy + cast. */
      "item",
      [](const LuaTensor& t) -> double {
        int64_t numel = pd_tensor_numel(t.h);
        if (numel != 1) {
          throw sol::error("paddle: item() on a non-scalar tensor, use "
                           "item(i) (1-based coordinate) -- P5 pending");
        }
        size_t esize = pd_dtype_sizeof(pd_tensor_dtype(t.h));
        if (esize == 0) {
          throw sol::error("paddle: item(): unsupported dtype");
        }
        std::vector<char> buf(esize);
        pd_tensor_to_host_buffer(t.h, buf.data(), buf.size());
        pd::check();
        switch (pd_tensor_dtype(t.h)) {
          case PD_DTYPE_FLOAT32: {
            float f;
            memcpy(&f, buf.data(), sizeof(f));
            return static_cast<double>(f);
          }
          case PD_DTYPE_FLOAT64: {
            double d;
            memcpy(&d, buf.data(), sizeof(d));
            return d;
          }
          case PD_DTYPE_INT64: {
            int64_t i;
            memcpy(&i, buf.data(), sizeof(i));
            return static_cast<double>(i);
          }
          case PD_DTYPE_INT32: {
            int32_t i;
            memcpy(&i, buf.data(), sizeof(i));
            return static_cast<double>(i);
          }
          case PD_DTYPE_BOOL:
          case PD_DTYPE_UINT8:
          case PD_DTYPE_INT8: {
            uint8_t u;
            memcpy(&u, buf.data(), sizeof(u));
            return static_cast<double>(u);
          }
          default:
            throw sol::error("paddle: item(): dtype not decodable yet");
        }
      },
      /* 2.5 tensor.lua: raw host bytes for __tostring truncation and
       * item(i) later. Returns a Lua string (byte-exact copy); Lua side
       * decodes per dtype (float16.h:1107 etc. are byte-exact). */
      "to_host_bytes",
      [](const LuaTensor& t) {
        int64_t numel = pd_tensor_numel(t.h);
        size_t esize = pd_dtype_sizeof(pd_tensor_dtype(t.h));
        if (numel <= 0 || esize == 0) {
          return std::string();
        }
        std::vector<char> buf(static_cast<size_t>(numel) * esize);
        pd_tensor_to_host_buffer(t.h, buf.data(), buf.size());
        pd::check();
        return std::string(buf.data(), buf.size());
      },
      /* reserved now for P13 (02-binding.md sec.3.2) */
      "__lanesclone",
      [](void* dst, const LuaTensor& src) { new (dst) LuaTensor(src); });

  push_tensor_factory(mod);
  register_slice(mod);   // slice.cc: core.slice_spec (D14 string transform)
}

}  // namespace pd