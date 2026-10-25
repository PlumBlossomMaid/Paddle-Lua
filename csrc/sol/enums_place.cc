// enums_place.cc -- dtype/place constant table + LuaPlace usertype
// Stage: P2 / M1 (WORKPLAN 2.2.5, closing gap found while wiring the sol2
// op layer 2026-10-25).
//
// Design: docs/migration/plan/modules/02-binding.md sec.3.4 ("dtype /
// place 枚举表" is a hand-written slice of the sol layer) + api/
// dtype-device.md.
//
// Why this file exists (2026-10-25): the generated lua/paddle/_ops/*.lua
// wrappers compute attr defaults by referencing paddle_core.PD_DTYPE_*
// and paddle_core.pd_place_new(...) on the module table (emit_lua.lua
// DATA_TYPE_DEFAULTS), but nothing registered them -- every op call with
// a defaulted dtype/place attr would pass nil into the sol2 op lambda.
// This TU registers:
//   - PD_DTYPE_* integer constants (values from the paddle_c.h pd_dtype
//     enum, which is 1:1 with phi::DataType -- paddle_c.h:58);
//   - PD_PLACE_* integer constants (paddle_c.h pd_place_kind);
//   - pd_place_new(kind, index) -> LuaPlace (wraps the C ABI owning
//     handle; GC frees it via ~LuaPlace -> pd_place_free);
//   - the LuaPlace usertype so sol2 op lambdas can take `const
//     LuaPlace&` (their pd_place attr param rides as .h).
//
// LuaPlace is deliberately NOT the P5 value-semantic place (dtype-device.md
// sec.2.2/6.2 -- fieldwise equality). That equality layer belongs to the
// Lua API (device.lua); here place is an opaque handle shell, exactly like
// Tensor at this layer.

#include "sol_common.hpp"

namespace pd {

void register_enums_place(lua_State* L, int module_index) {
  sol::state_view lua(L);
  sol::table mod(L, sol::absolute_index(L, module_index));

  // dtype constants (paddle_c.h pd_dtype -- the enum IS the single source;
  // referencing the enum constant here, never a magic number).
  mod["PD_DTYPE_UNDEFINED"] = PD_DTYPE_UNDEFINED;
  mod["PD_DTYPE_BOOL"] = PD_DTYPE_BOOL;
  mod["PD_DTYPE_UINT8"] = PD_DTYPE_UINT8;
  mod["PD_DTYPE_INT8"] = PD_DTYPE_INT8;
  mod["PD_DTYPE_UINT16"] = PD_DTYPE_UINT16;
  mod["PD_DTYPE_INT16"] = PD_DTYPE_INT16;
  mod["PD_DTYPE_UINT32"] = PD_DTYPE_UINT32;
  mod["PD_DTYPE_INT32"] = PD_DTYPE_INT32;
  mod["PD_DTYPE_UINT64"] = PD_DTYPE_UINT64;
  mod["PD_DTYPE_INT64"] = PD_DTYPE_INT64;
  mod["PD_DTYPE_FLOAT32"] = PD_DTYPE_FLOAT32;
  mod["PD_DTYPE_FLOAT64"] = PD_DTYPE_FLOAT64;
  mod["PD_DTYPE_COMPLEX64"] = PD_DTYPE_COMPLEX64;
  mod["PD_DTYPE_COMPLEX128"] = PD_DTYPE_COMPLEX128;
  mod["PD_DTYPE_PSTRING"] = PD_DTYPE_PSTRING;
  mod["PD_DTYPE_FLOAT16"] = PD_DTYPE_FLOAT16;
  mod["PD_DTYPE_BFLOAT16"] = PD_DTYPE_BFLOAT16;
  mod["PD_DTYPE_FLOAT8_E4M3FN"] = PD_DTYPE_FLOAT8_E4M3FN;
  mod["PD_DTYPE_FLOAT8_E5M2"] = PD_DTYPE_FLOAT8_E5M2;

  // place-kind constants (paddle_c.h pd_place_kind).
  mod["PD_PLACE_CPU"] = PD_PLACE_CPU;
  mod["PD_PLACE_CUDA"] = PD_PLACE_CUDA;
  mod["PD_PLACE_CUDAPINNED"] = PD_PLACE_CUDAPINNED;
  mod["PD_PLACE_XPU"] = PD_PLACE_XPU;
  mod["PD_PLACE_XPUPINNED"] = PD_PLACE_XPUPINNED;
  mod["PD_PLACE_IPU"] = PD_PLACE_IPU;
  mod["PD_PLACE_CUSTOM"] = PD_PLACE_CUSTOM;

  // LuaPlace usertype: no constructor from Lua (places are created via
  // pd_place_new below; GC runs ~LuaPlace -> pd_place_free).
  lua.new_usertype<LuaPlace>("paddle_core.Place");

  // Factory used by generated _ops defaults:
  //   paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)
  // pd_place_new (tensor.cc:156) sets PD_INVALID_ARG and returns NULL for
  // a bad kind / negative index -- pd::check() must sit right after the
  // call (the failure slot is cleared by any later pd_* call).
  mod.set_function(
      "pd_place_new",
      [](int kind, int index) -> LuaPlace {
        pd_place p =
            pd_place_new(static_cast<pd_place_kind>(kind), index);
        pd::check();
        return LuaPlace(p);
      });
}

}  // namespace pd