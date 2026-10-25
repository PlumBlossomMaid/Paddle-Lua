// jit_binding.cc -- LuaLayer usertype: jit::Layer face for Lua (2.8)
// Stage: P2 / M1 (WORKPLAN 2.8, layout.md P8 load.lua).
//
// Design: docs/migration/plan/modules/08-jit-infer.md sec.3.2 + api/jit.md
// sec.2.2. Mirrors the LuaTensor pattern (tensor_binding.cc): a RAII shell
// over the opaque C ABI pd_layer handle; GC frees via ~LuaLayer.
//
//   core.jit_load{path, skip_load_weights=false} -> LuaLayer
//   layer:forward(inputs)          inputs: Tensor or array of Tensor;
//                                  single output -> one Tensor, multiple ->
//                                  multiple returns (08-jit-infer sec.4 ②)
//   layer:to(place)                layer.h:63 to()
//   layer:function_names()         layer.h:73 FunctionNames()
//   layer:clone()                  layer.h:78 Clone(nullptr) -- P13 prewire
//   __lanesclone                    P13 prewire, same convention as LuaTensor
//
// This TU touches paddle_c.h ONLY (02-binding.md sec.2: the sol layer never
// includes Paddle headers), so version-porting = recompiling the same source.

#include "sol_common.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace pd {

/* RAII shell over the C ABI jit layer handle (paddle_c.h Group 6). */
struct LuaLayer {
  pd_layer h;

  explicit LuaLayer(pd_layer handle) : h(handle) {}
  ~LuaLayer() { pd_jit_layer_free(h); }

  LuaLayer(const LuaLayer&) = delete;
  LuaLayer& operator=(const LuaLayer&) = delete;
  LuaLayer(LuaLayer&& other) noexcept : h(other.h) { other.h = nullptr; }
};

/* Register core.jit_load + the LuaLayer usertype (08-jit-infer.md sec.3.2
 * -- 100-line sol slice; the 200-line C ABI slice is csrc/capi/jit.cc). */
void register_jit(lua_State* L, int module_index) {
  sol::state_view lua(L);
  sol::table mod(L, sol::absolute_index(L, module_index));

  /* forward: x:forward(t) or x:forward({t1, t2}) -> one Tensor, or
   * multiple return values for multiple outputs (08-jit-infer.md sec.4 ②).
   * Shared with __call so layer(x) works (api/jit.md sec.2.2).
   * NOTE: pass the lambda BY VALUE (sol2's lua_bind_traits rejects a
   * pointer-to-lambda: it needs the closure type, not a pointer to it). */
  auto layer_forward =
      [](sol::this_state L, const LuaLayer& layer,
         sol::variadic_args va) -> sol::object {
    std::vector<pd_tensor> inputs;
    for (auto v : va) {
      if (v.is<LuaTensor>()) {  // single Tensor passed directly
        LuaTensor& t = v.as<LuaTensor&>();
        inputs.push_back(t.h);
      } else if (v.is<sol::table>()) {  // array of Tensors
        sol::table tbl = v.as<sol::table>();
        for (auto& kv : tbl) {
          LuaTensor& t = kv.second.as<LuaTensor&>();
          inputs.push_back(t.h);
        }
      } else {
        throw sol::error("paddle: forward inputs must be Tensor(s)");
      }
    }
    if (inputs.empty()) {
      throw sol::error("paddle: forward needs at least one input Tensor");
    }
    // Two-phase output query (jit.cc pattern): first call with nout=0
    // returns the count, then allocate exactly that many.
    int n = pd_jit_layer_forward(layer.h, inputs.data(),
                                 static_cast<int>(inputs.size()), nullptr, 0);
    pd::check();
    std::vector<pd_tensor> outs(static_cast<size_t>(n > 0 ? n : 0), nullptr);
    if (n > 0) {
      int got = pd_jit_layer_forward(layer.h, inputs.data(),
                                     static_cast<int>(inputs.size()),
                                     outs.data(), n);
      pd::check();
      if (got != n) {
        throw sol::error("paddle: forward output count changed");
      }
    }
    if (n == 0) return sol::make_object(L, sol::nil);
    if (n == 1) return sol::make_object(L, LuaTensor(outs[0]));
    sol::table results(L, sol::create);
    for (int i = 0; i < n; ++i) {
      results[i + 1] = LuaTensor(outs[static_cast<size_t>(i)]);
    }
    return sol::make_object(L, results);
  };

  /* core.jit_load{path, place?, skip_load_weights=false} -> LuaLayer
   * place defaults to CPU (device.cc pd_place_new; this build is CPU-only,
   * api/jit.md sec.2.2 has no place param -- load targets CPU, :to() moves).
   * skip_load_weights is accepted for signature parity; the C ABI Load
   * always reads weights (serializer.h:78 Load has no skip flag), so a
   * true value is rejected to avoid silently wrong behavior. */
  mod.set_function(
      "jit_load",
      [](sol::table args) -> LuaLayer {
        std::string path = args["path"];
        if (path.empty()) {
          throw sol::error("paddle: jit_load{path} requires a path");
        }
        if (args["skip_load_weights"] == true) {
          throw sol::error(
              "paddle: skip_load_weights=true not supported (Load always "
              "reads weights, serializer.h:78)");
        }
        pd_place place = pd_place_new(PD_PLACE_CPU, 0);
        pd_layer layer = pd_jit_layer_load(path.c_str(), place);
        pd::check();  // read the status NOW, before pd_place_free clears it
        pd_place_free(place);
        return LuaLayer(layer);
      });

  lua.new_usertype<LuaLayer>(
      "paddle_core.Layer",
      /* x:forward(t) or x:forward({t1, t2}) -> one Tensor, or multiple
       * return values for multiple outputs (08-jit-infer.md sec.4 ②).
       * Shared with __call so layer(x) works (api/jit.md sec.2.2).
       * layer_forward is passed BY VALUE: sol2's lua_bind_traits needs the
       * closure type; a pointer-to-lambda fails (call.hpp function_pointer_type). */
      "forward", layer_forward,
      "__call", layer_forward,
      /* x:to(place) -- layer.h:63 to() */
      "to",
      [](const LuaLayer& layer, const LuaPlace& place) {
        pd_status st = pd_jit_layer_to(layer.h, place.h);
        pd::check();
        return 0;
      },
      /* x:function_names() -> array of strings (layer.h:73) */
      "function_names",
      [](sol::this_state L, const LuaLayer& layer) -> sol::object {
        size_t len = 0;
        int n = pd_jit_layer_function_names(layer.h, nullptr, 0, &len);
        pd::check();
        if (n <= 0) return sol::make_object(L, sol::nil);
        std::vector<char> buf(len);
        int got = pd_jit_layer_function_names(layer.h, buf.data(), buf.size(),
                                              &len);
        pd::check();
        sol::table names(L, sol::create);
        size_t off = 0;
        int idx = 0;
        while (idx < got && off < buf.size()) {
          names[idx + 1] = std::string(buf.data() + off);
          off += std::strlen(buf.data() + off) + 1;
          ++idx;
        }
        return sol::make_object(L, names);
      },
      /* x:clone() -> new LuaLayer (08-jit-infer.md sec.3.3 P13 prewire) */
      "clone",
      [](const LuaLayer& layer) -> LuaLayer {
        pd_layer c = pd_jit_layer_clone(layer.h);
        pd::check();
        return LuaLayer(c);
      },
      /* reserved now for P13 (08-jit-infer.md sec.3.3: Clone is the
       * per-worker share mechanism). pd_layer has NO retain, so the new
       * shell must own its own clone -- pd_jit_layer_clone, never a
       * shared raw handle (double free). */
      "__lanesclone",
      [](void* dst, const LuaLayer& src) {
        pd_layer c = pd_jit_layer_clone(src.h);
        pd::check();
        new (dst) LuaLayer(c);
      });
}

}  // namespace pd