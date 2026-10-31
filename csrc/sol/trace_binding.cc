// trace_binding.cc -- LuaTrace usertype: the P15 static-graph trace face
// (paddle.jit.trace, 15-to-static.md sec.3 eager-record).
//
// Stage: P15 / M3 (WORKPLAN 4.1). Mirrors the LuaLayer pattern
// (jit_binding.cc): a RAII shell over the opaque C ABI pd_trace handle
// (paddle_c.h Group 12); GC frees via ~LuaTrace. This TU touches
// paddle_c.h ONLY (02-binding.md sec.2).
//
//   core.trace_begin()                    -> LuaTrace
//   core.trace_is_active()                -> bool (tensor.lua item gate)
//   trace:input(tensor, name)             declare a feed (before forward)
//   trace:output(tensor, name)            mark a fetch (after forward)
//   trace:run() / trace:run({t1, ...})    replay; xs replace the declared
//                                         inputs (pd_trace_run2)

#include "sol_common.hpp"

#include <vector>

namespace pd {

/* RAII shell over the C ABI trace handle (paddle_c.h Group 12). */
struct LuaTrace {
  pd_trace h;

  explicit LuaTrace(pd_trace handle) : h(handle) {}
  ~LuaTrace() { pd_trace_free(h); }

  LuaTrace(const LuaTrace&) = delete;
  LuaTrace& operator=(const LuaTrace&) = delete;
  LuaTrace(LuaTrace&& other) noexcept : h(other.h) { other.h = nullptr; }
};

void register_trace(lua_State* L, int module_index) {
  sol::state_view lua(L);
  sol::table mod(L, sol::absolute_index(L, module_index));

  mod.set_function("trace_begin", []() -> LuaTrace {
    pd_trace t = pd_trace_begin();
    pd::check();
    if (t == nullptr) {
      throw sol::error("paddle: trace_begin failed");
    }
    return LuaTrace(t);
  });
  mod.set_function("trace_is_active",
                   []() { return pd_trace_is_active() != 0; });

  lua.new_usertype<LuaTrace>(
      "paddle_core.Trace",
      /* trace:input(tensor, name) -- paddle_c.h Group 12 */
      "input",
      [](const LuaTrace& tr, const LuaTensor& t, const std::string& name) {
        pd_status st = pd_trace_input(tr.h, t.h, name.c_str());
        (void)st;
        pd::check();
        return 0;
      },
      /* trace:output(tensor, name) */
      "output",
      [](const LuaTrace& tr, const LuaTensor& t, const std::string& name) {
        pd_status st = pd_trace_output(tr.h, t.h, name.c_str());
        (void)st;
        pd::check();
        return 0;
      },
      /* trace:run() replays the captured inputs; trace:run({t1, ...})
       * substitutes the declared inputs (pd_trace_run2). Two-phase output
       * query like jit forward: nout=0 yields the count. Single output ->
       * one Tensor; multiple -> an array (matches layer:forward). */
      "run",
      [](sol::this_state L, const LuaTrace& tr,
         sol::optional<sol::table> inputs) -> sol::object {
        std::vector<pd_tensor> xs;
        const pd_tensor* xsp = nullptr;
        int nx = 0;
        if (inputs) {
          for (auto& kv : *inputs) {
            LuaTensor& t = kv.second.as<LuaTensor&>();
            xs.push_back(t.h);
          }
          xsp = xs.data();
          nx = static_cast<int>(xs.size());
        }
        int n = pd_trace_run2(tr.h, xsp, nx, nullptr, 0);
        pd::check();
        if (n < 0) pd::throw_status();
        if (n == 0) return sol::make_object(L, sol::nil);
        std::vector<pd_tensor> outs(static_cast<size_t>(n), nullptr);
        int got = pd_trace_run2(tr.h, xsp, nx, outs.data(), n);
        pd::check();
        if (got != n) {
          throw sol::error("paddle: trace run output count changed");
        }
        if (n == 1) return sol::make_object(L, LuaTensor(outs[0]));
        sol::table results(L, sol::create);
        for (int i = 0; i < n; ++i) {
          results[i + 1] = LuaTensor(outs[static_cast<size_t>(i)]);
        }
        return sol::make_object(L, results);
      },
      /* trace:finish() -- end the recording phase (pd_trace_end): later
       * eager ops stop recording into this program; the handle stays
       * alive for run(). */
      "finish",
      [](const LuaTrace& tr) {
        pd_trace_end(tr.h);
        return 0;
      },
      /* trace:close() -- explicit release (idempotent; __gc then no-ops).
       * Used by paddle.jit.trace when the recording run fails: the
       * thread's active-trace slot must be cleared NOW, not at GC. */
      "close",
      [](LuaTrace& tr) {
        pd_trace_free(tr.h);
        tr.h = nullptr;
        return 0;
      });
}

}  // namespace pd
