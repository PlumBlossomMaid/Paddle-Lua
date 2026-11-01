// custom_op_binding.cc -- core.load_custom_op(dso) (P16, 16-cpp-extension.md).
// This TU touches paddle_c.h ONLY (02-binding.md sec.2).
//
//   core.load_custom_op(absolute_dso_path) -> { <op_name> = true, ... }
//
// Loading registers every op in the DLL into the PIR dialect (custom_op.
// prefix) AND the legacy fluid OpInfo table (custom_operator.cc:1279-1315).
// The returned table lists the newly registered op names so the Lua
// cpp_extension face can build callables for each (execution lands in the
// C ABI's pd_custom_op_run, added in the P16 executor step).

#include <cstring>
#include <string>
#include <vector>

#include "sol_common.hpp"

namespace pd {

void register_custom_op(lua_State* L, int module_index) {
  sol::state_view lua(L);
  sol::table mod(L, sol::absolute_index(L, module_index));

  mod.set_function(
      "load_custom_op",
      [](sol::this_state L, const std::string& dso_path) -> sol::object {
        // Query size first (names = NULL, cap = 0), then fill. Each name
        // is NUL-terminated; a fresh buffer per op-name avoids growth.
        int n = pd_load_op_meta_info(dso_path.c_str(), nullptr, 0);
        pd::check();
        if (n < 0) pd::throw_status();
        std::vector<std::string> names;
        names.reserve(static_cast<size_t>(n));
        std::vector<char> buf(static_cast<size_t>(n) * 512);
        int got = pd_load_op_meta_info(dso_path.c_str(), buf.data(),
                                       static_cast<int>(buf.size()));
        pd::check();
        if (got != n) {
          throw sol::error("paddle: load_custom_op op count changed "
                           "between probe and fill");
        }
        size_t off = 0;
        for (int i = 0; i < got; ++i) {
          names.emplace_back(buf.data() + off);
          off += names.back().size() + 1;
        }
        sol::state_view lv(L);
        sol::table ops = lv.create_table(0, static_cast<int>(names.size()));
        for (size_t i = 0; i < names.size(); ++i) {
          ops[names[i]] = true;
        }
        return sol::make_object(L, ops);
      });
}

}  // namespace pd