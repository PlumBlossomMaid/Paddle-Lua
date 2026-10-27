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

  /* 2.12 MNIST 数据管线:一次性把 Lua 字节串(host 端原始布局)造进
   * Tensor。60000x784 的数据逐算子填充不可行,必须整块导入。
   * 上游对照:paddle/phi/api/include/api.h:384 paddle::experimental::
   * empty(shape, dtype, place) + memcpy(字节由 shape*dtype 隐含,
   * tensor.cc:198 处 nbytes 被忽略)。C ABI 入口 pd_tensor_from_host_buffer
   * (paddle_c.h:128),这里只是 sol 薄绑定。
   * 参数纪律:place 用 const LuaPlace&(与全部 pd_op_* lambda 一致,
   * sol2 不接受按值 move-only —— enums_place.cc 删了拷贝构造);
   * data 用 std::string(Lua 字符串可含 \0,不能用 const char* 裸收)。 */
  mod.set_function(
      "from_host_bytes",
      [](const std::string& data, sol::table dims, int dtype,
         const LuaPlace& place) -> LuaTensor {
        std::vector<int64_t> shape;
        for (auto& kv : dims) {
          shape.push_back(kv.second.as<int64_t>());
        }
        pd_tensor t = pd_tensor_from_host_buffer(
            place.h, shape.data(), static_cast<int>(shape.size()),
            dtype, data.data(), data.size());
        pd::check();
        return LuaTensor(t);
      });
}

/* ---- 2.6 module-level autograd entries (paddle.autograd.* backend) --
 * Accept Lua arrays of LuaTensor; validate and forward to the C ABI.
 * Keep LuaTensor unwrap + pd::check() in exactly this order (the error
 * slot is cleared by ANY intermediate pd_* call). */
void register_autograd_fns(sol::table& mod) {
  auto as_tensor_vec = [](sol::table tbl) -> std::vector<pd_tensor> {
    std::vector<pd_tensor> vec;
    for (auto& kv : tbl) {
      LuaTensor& t = kv.second.as<LuaTensor&>();
      vec.push_back(t.h);
    }
    return vec;
  };

  /* core.backward{tensors, grad_tensors=nil, retain_graph=false} */
  mod.set_function(
      "backward",
      [as_tensor_vec](sol::table tensors, sol::optional<sol::table> gts,
                      sol::optional<bool> retain_graph) {
        std::vector<pd_tensor> tvec = as_tensor_vec(tensors);
        std::vector<pd_tensor> gvec;
        if (gts) gvec = as_tensor_vec(*gts);
        pd_status st = pd_backward(tvec.data(), static_cast<int>(tvec.size()),
                                   gvec.empty() ? nullptr : gvec.data(),
                                   static_cast<int>(gvec.size()),
                                   retain_graph.value_or(false) ? 1 : 0);
        pd::check();
        return 0;
      });

  /* core.grad{outputs, inputs, grad_outputs=nil, retain_graph=false,
   * create_graph=false, only_inputs=true, allow_unused=false} -> table
   * of gradient tensors (nil slot where a input has no grad). */
  mod.set_function(
      "grad",
      [as_tensor_vec](sol::this_state L, sol::table outputs,
                      sol::table inputs, sol::optional<sol::table> gts,
                      sol::optional<bool> retain_graph,
                      sol::optional<bool> create_graph,
                      sol::optional<bool> only_inputs,
                      sol::optional<bool> allow_unused) -> sol::object {
        std::vector<pd_tensor> tvec = as_tensor_vec(outputs);
        std::vector<pd_tensor> ivec = as_tensor_vec(inputs);
        std::vector<pd_tensor> gvec;
        if (gts) gvec = as_tensor_vec(*gts);
        std::vector<pd_tensor> out_grads(ivec.size(), nullptr);
        pd_status st = pd_grad(
            tvec.data(), static_cast<int>(tvec.size()),
            ivec.data(), static_cast<int>(ivec.size()),
            gvec.empty() ? nullptr : gvec.data(), static_cast<int>(gvec.size()),
            retain_graph.value_or(false) ? 1 : 0,
            create_graph.value_or(false) ? 1 : 0,
            only_inputs.value_or(true) ? 1 : 0,
            allow_unused.value_or(false) ? 1 : 0,
            out_grads.data(), static_cast<int>(out_grads.size()));
        pd::check();
        sol::table out(L, sol::create);
        for (size_t i = 0; i < out_grads.size(); ++i) {
          if (out_grads[i] != nullptr) {
            out[i + 1] = LuaTensor(out_grads[i]);   // ownership transfer
          } else {
            out[i + 1] = sol::nil;
          }
        }
        return out;
      });

  /* core.has_grad() / core.set_has_grad(bool) — no_grad/is_grad_enabled
   * backend (imperative.cc:641-642 binds these same Controller calls). */
  mod.set_function("has_grad", []() { return pd_has_grad() != 0; });
  mod.set_function("set_has_grad", [](bool v) { pd_set_has_grad(v ? 1 : 0); });
}

/* ---- 2.7 module-level serialization entries (paddle_c.h Group 7) ----
 * Single-tensor save/load on Paddle's own wire format (D19). The Lua API
 * layer (paddle.save/load) dispatches state_dict objects to lua-pickle
 * and plain tensors to these two (07-serialization.md sec.3.2). */
void register_serialize_fns(sol::table& mod) {
  /* core.save_tensor(tensor, path, overwrite=false) */
  mod.set_function(
      "save_tensor",
      [](const LuaTensor& t, const char* path, sol::optional<bool> overwrite) {
        pd_status st = pd_tensor_save(
            t.h, path, overwrite.has_value() ? (*overwrite ? 1 : 0) : 1);
        pd::check();
        return 0;
      });

  /* core.load_tensor(path) -> LuaTensor (ownership transfer; caller gets
   * a NEW shell over the loaded storage, GC frees it on collect). */
  mod.set_function(
      "load_tensor",
      [](const char* path) -> LuaTensor {
        pd_tensor t = nullptr;
        pd_status st = pd_tensor_load(path, &t);
        pd::check();  // read status NOW; t may be null on error
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
      /* ---- 2.6 autograd group (paddle_c.h Group 4) -----------------
       * Same underlying egr::Backward / egr::Grad as paddle.autograd.*
       * (api/autograd.md sec.2 note: Tensor:backward and
       * paddle.autograd.backward MUST share one bottom layer, else
       * retain-graph / accumulation semantics diverge). */
      "backward",
      [](const LuaTensor& t, sol::optional<bool> retain_graph) {
        pd_tensor arr[1] = {t.h};
        pd_status st = pd_backward(arr, 1, nullptr, 0,
                                   retain_graph.value_or(false) ? 1 : 0);
        pd::check();
        return 0;
      },
      /* x:grad() -> LuaTensor | nil (pd_tensor_grad returns a NEW shell
       * the caller owns; sol2 takes ownership into the usertype). */
      "grad",
      [](sol::this_state L, const LuaTensor& t) -> sol::object {
        pd_tensor g = pd_tensor_grad(t.h);
        if (g == nullptr) {
          return sol::make_object(L, sol::nil);
        }
        return sol::make_object(L, LuaTensor(g));
      },
      "stop_gradient",
      [](const LuaTensor& t) { return pd_tensor_get_stop_gradient(t.h) != 0; },
      "set_stop_gradient",
      [](const LuaTensor& t, bool v) {
        pd_status st = pd_tensor_set_stop_gradient(t.h, v ? 1 : 0);
        pd::check();
        return 0;
      },
      /* 2.10 optimizer: t:copy_from(src) -- deep-copy src's data INTO
       * t's existing storage; t keeps its handle identity / autograd
       * meta / place. This is the Lua face of the in-place parameter
       * update that Python gets from _C_ops.sgd_/adam_ (ParamOut back
       * into Param, 10-optimizer.md sec.4). Requires dtype/place match
       * (pd_tensor_copy_from -> Tensor::copy_, tensor.h:510). */
      "copy_from",
      [](const LuaTensor& t, const LuaTensor& src) {
        pd_status st = pd_tensor_copy_from(t.h, src.h);
        pd::check();
        return 0;
      },
      /* reserved now for P13 (02-binding.md sec.3.2) */
      "__lanesclone",
      [](void* dst, const LuaTensor& src) { new (dst) LuaTensor(src); });

  push_tensor_factory(mod);
  register_slice(mod);          // slice.cc: core.slice_spec (D14 string transform)
  register_autograd_fns(mod);   // 2.6: core.backward/grad/has_grad/set_has_grad
  register_serialize_fns(mod);  // 2.7: core.save_tensor/load_tensor (Group 7)

  /* 3.1.2: core.gzip_inflate(str) -> str -- in-memory gzip decompress
   * (paddle_c.h Group 8). Two-pass: query size, then decompress into a
   * fresh buffer. Feeds paddle.vision.datasets.MNIST's .gz IDX files. */
  mod.set_function(
      "gzip_inflate",
      [](sol::string_view src) -> std::string {
        size_t need = 0;
        pd_status st = pd_gzip_inflate(
            reinterpret_cast<const unsigned char*>(src.data()), src.size(),
            nullptr, 0, &need);
        pd::check();
        std::string out(need, '\0');
        if (need > 0) {
          st = pd_gzip_inflate(
              reinterpret_cast<const unsigned char*>(src.data()), src.size(),
              reinterpret_cast<unsigned char*>(&out[0]), need, &need);
          pd::check();
        }
        return out;
      });
}

}  // namespace pd