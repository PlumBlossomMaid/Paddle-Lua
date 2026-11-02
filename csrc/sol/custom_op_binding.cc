// custom_op_binding.cc -- custom-op C ABI face (P16, 16-cpp-extension.md).
// This TU touches paddle_c.h ONLY (02-binding.md sec.2).
//
//   core.load_custom_op(absolute_dso_path) -> { <op_name> = true, ... }
//     Loading registers every op in the DLL into the PIR dialect (custom_op.
//     prefix) AND the legacy fluid OpInfo table (custom_operator.cc:1279-1315).
//     The returned table lists the newly registered op names so the Lua
//     cpp_extension face can build callables for each.
//
//   core.custom_op_run(op_type, {inputs...}, {attrs...}) -> {outputs...}
//     Execute a loaded custom op in PIR inference mode (pd_custom_op_run,
//     paddle_c.h Group 14). attrs is an optional array of
//     { type = pd_attr_type, value = scalar-or-array, len = n } in the op's
//     attrs_ declaration order (len only used for vector attrs; scalars and
//     strings use 1).

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

  /* core.custom_op_meta(op_type) -> { inputs = {..}, outputs = {..},
   * attrs = { {name=.., type=..}, .. } } -- the op schema in Lua types
   * (paddle_c.h Group 14). The cpp_extension face builds arg-aware
   * callables from this: how many positional tensors and which named
   * attrs. */
  mod.set_function(
      "custom_op_meta",
      [](sol::this_state L, const std::string& op_type) -> sol::object {
        int n = pd_custom_op_meta(op_type.c_str(), nullptr, 0);
        pd::check();
        if (n < 0) pd::throw_status();
        std::vector<char> buf(static_cast<size_t>(n) + 1, '\0');
        int got = pd_custom_op_meta(op_type.c_str(), buf.data(),
                                    static_cast<int>(buf.size()));
        pd::check();
        if (got < 0) pd::throw_status();
        std::string schema(buf.data());
        // split on '\n' into 3 lines, then ',' into names
        auto split_lines = [](const std::string& s) {
          std::vector<std::string> lines;
          size_t start = 0;
          while (true) {
            size_t p = s.find('\n', start);
            if (p == std::string::npos) break;
            lines.emplace_back(s.substr(start, p - start));
            start = p + 1;
          }
          return lines;
        };
        auto split_comma = [](const std::string& line) {
          std::vector<std::string> parts;
          size_t start = 0;
          while (true) {
            size_t p = line.find(',', start);
            if (p == std::string::npos) {
              if (start < line.size()) parts.emplace_back(line.substr(start));
              break;
            }
            parts.emplace_back(line.substr(start, p - start));
            start = p + 1;
          }
          return parts;
        };
        auto lines = split_lines(schema);
        sol::state_view lv(L);
        auto make_list = [&](const std::vector<std::string>& v) {
          sol::table t = lv.create_table(0, static_cast<int>(v.size()));
          for (size_t i = 0; i < v.size(); ++i) {
            t[static_cast<int>(i) + 1] = v[i];  // 1-based (C4)
          }
          return t;
        };
        sol::table result = lv.create_table();
        result["inputs"] = make_list(lines.size() > 0
                                         ? split_comma(lines[0])
                                         : std::vector<std::string>{});
        result["outputs"] = make_list(lines.size() > 1
                                          ? split_comma(lines[1])
                                          : std::vector<std::string>{});
        // attrs: "name:type" strings -> { {name, type}, .. }
        auto attr_parts = lines.size() > 2 ? split_comma(lines[2])
                                           : std::vector<std::string>{};
        sol::table attrs = lv.create_table(0, static_cast<int>(attr_parts.size()));
        for (size_t i = 0; i < attr_parts.size(); ++i) {
          const std::string& at = attr_parts[i];
          size_t colon = at.find(':');
          sol::table a = lv.create_table();
          if (colon != std::string::npos) {
            a["name"] = at.substr(0, colon);
            a["type"] = at.substr(colon + 1);
          } else {
            a["name"] = at;
            a["type"] = "";
          }
          attrs[static_cast<int>(i) + 1] = a;
        }
        result["attrs"] = attrs;
        return sol::make_object(L, result);
      });

  /* core.custom_op_run(op, inputs, attrs) -> table of output LuaTensors.
   * attr entries: { type = <pd_attr_type int>, value = <number | string |
   * array of number/string>, len = <int, vector only> }. Types map to the
   * paddle_c.h PD_ATTR_* enum. Attr VALUES are marshalled into C-owned
   * storage for the duration of the call. */
  mod.set_function(
      "custom_op_run",
      [](sol::this_state L, const std::string& op_type, sol::table inputs,
         sol::optional<sol::table> attrs_tbl) -> sol::object {
        std::vector<pd_tensor> xs;
        for (auto& kv : inputs) {
          LuaTensor& t = kv.second.as<LuaTensor&>();
          xs.push_back(t.h);
        }
        std::vector<int> attr_types;
        std::vector<int> attr_lens;
        std::vector<std::vector<int>> ints_buf;
        std::vector<std::vector<float>> floats_buf;
        std::vector<std::vector<double>> doubles_buf;
        std::vector<std::vector<int64_t>> i64_buf;
        std::vector<std::vector<std::string>> strs_buf;
        std::vector<std::vector<const char*>> cstrs_buf;
        std::vector<const void*> attr_values;
        if (attrs_tbl) {
          for (auto& kv : *attrs_tbl) {
            sol::table a = kv.second.as<sol::table>();
            int type = a["type"].get<int>();
            attr_types.push_back(type);
            int len = 1;
            if (a["len"].valid()) len = a["len"].get<int>();
            attr_lens.push_back(len);
            switch (type) {
              case PD_ATTR_BOOL:
              case PD_ATTR_INT: {
                ints_buf.emplace_back();
                auto& v = ints_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<int>());
                } else {
                  v.push_back(a["value"].get<int>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_FLOAT: {
                floats_buf.emplace_back();
                auto& v = floats_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<float>());
                } else {
                  v.push_back(a["value"].get<float>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_DOUBLE: {
                doubles_buf.emplace_back();
                auto& v = doubles_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<double>());
                } else {
                  v.push_back(a["value"].get<double>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_INT64: {
                i64_buf.emplace_back();
                auto& v = i64_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<int64_t>());
                } else {
                  v.push_back(a["value"].get<int64_t>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_STRING: {
                strs_buf.emplace_back();
                auto& v = strs_buf.back();
                v.push_back(a["value"].get<std::string>());
                attr_values.push_back(
                    static_cast<const void*>(v.front().c_str()));
                break;
              }
              case PD_ATTR_VEC_STRING: {
                strs_buf.emplace_back();
                auto& v = strs_buf.back();
                sol::table arr = a["value"].get<sol::table>();
                for (int i = 1; i <= len; ++i)
                  v.push_back(arr[i].get<std::string>());
                cstrs_buf.emplace_back();
                auto& cv = cstrs_buf.back();
                for (size_t i = 0; i < v.size(); ++i)
                  cv.push_back(v[i].c_str());
                attr_values.push_back(cv.data());
                break;
              }
              default:
                throw sol::error("paddle: custom_op_run unsupported attr "
                                 "type code");
            }
          }
        }
        int n = pd_custom_op_run(
            op_type.c_str(), xs.empty() ? nullptr : xs.data(),
            static_cast<int>(xs.size()), attr_types.empty() ? nullptr
                                                            : attr_types.data(),
            attr_values.empty() ? nullptr : attr_values.data(),
            attr_lens.empty() ? nullptr : attr_lens.data(),
            static_cast<int>(attr_types.size()), nullptr, 0);
        pd::check();
        if (n < 0) pd::throw_status();
        std::vector<pd_tensor> outs(static_cast<size_t>(n), nullptr);
        int got = pd_custom_op_run(
            op_type.c_str(), xs.empty() ? nullptr : xs.data(),
            static_cast<int>(xs.size()), attr_types.empty() ? nullptr
                                                            : attr_types.data(),
            attr_values.empty() ? nullptr : attr_values.data(),
            attr_lens.empty() ? nullptr : attr_lens.data(),
            static_cast<int>(attr_types.size()), outs.data(),
            static_cast<int>(outs.size()));
        pd::check();
        if (got != n) {
          throw sol::error("paddle: custom_op_run output count changed");
        }
        sol::state_view lv(L);
        sol::table results = lv.create_table(0, n);
        for (int i = 0; i < n; ++i) {
          results[i + 1] = LuaTensor(outs[static_cast<size_t>(i)]);
        }
        return sol::make_object(L, results);
      });

  /* core.custom_op_eager_run(op, inputs, attrs) -> outputs. SAME attrs
   * marshalling as custom_op_run, but the execution is EAGER (autograd-
   * tracked): when an input requires grad, the forward builds a
   * RunCustomOpNode so a later core.grad/core.backward can run the op's
   * PD_BUILD_GRAD_OP kernel. Attr handling duplicated here (a helper
   * would need to thread 6 buffers through two call shapes; this is the
   * project's explicit 3-lines-over-abstraction budget). */
  mod.set_function(
      "custom_op_eager_run",
      [](sol::this_state L, const std::string& op_type, sol::table inputs,
         sol::optional<sol::table> attrs_tbl) -> sol::object {
        std::vector<pd_tensor> xs;
        for (auto& kv : inputs) {
          LuaTensor& t = kv.second.as<LuaTensor&>();
          xs.push_back(t.h);
        }
        std::vector<int> attr_types;
        std::vector<int> attr_lens;
        std::vector<std::vector<int>> ints_buf;
        std::vector<std::vector<float>> floats_buf;
        std::vector<std::vector<double>> doubles_buf;
        std::vector<std::vector<int64_t>> i64_buf;
        std::vector<std::vector<std::string>> strs_buf;
        std::vector<std::vector<const char*>> cstrs_buf;
        std::vector<const void*> attr_values;
        if (attrs_tbl) {
          for (auto& kv : *attrs_tbl) {
            sol::table a = kv.second.as<sol::table>();
            int type = a["type"].get<int>();
            attr_types.push_back(type);
            int len = 1;
            if (a["len"].valid()) len = a["len"].get<int>();
            attr_lens.push_back(len);
            switch (type) {
              case PD_ATTR_BOOL:
              case PD_ATTR_INT: {
                ints_buf.emplace_back();
                auto& v = ints_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<int>());
                } else {
                  v.push_back(a["value"].get<int>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_FLOAT: {
                floats_buf.emplace_back();
                auto& v = floats_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<float>());
                } else {
                  v.push_back(a["value"].get<float>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_DOUBLE: {
                doubles_buf.emplace_back();
                auto& v = doubles_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<double>());
                } else {
                  v.push_back(a["value"].get<double>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_INT64: {
                i64_buf.emplace_back();
                auto& v = i64_buf.back();
                if (len > 1) {
                  sol::table arr = a["value"].get<sol::table>();
                  for (int i = 1; i <= len; ++i)
                    v.push_back(arr[i].get<int64_t>());
                } else {
                  v.push_back(a["value"].get<int64_t>());
                }
                attr_values.push_back(v.data());
                break;
              }
              case PD_ATTR_STRING: {
                strs_buf.emplace_back();
                auto& v = strs_buf.back();
                v.push_back(a["value"].get<std::string>());
                attr_values.push_back(
                    static_cast<const void*>(v.front().c_str()));
                break;
              }
              case PD_ATTR_VEC_STRING: {
                strs_buf.emplace_back();
                auto& v = strs_buf.back();
                sol::table arr = a["value"].get<sol::table>();
                for (int i = 1; i <= len; ++i)
                  v.push_back(arr[i].get<std::string>());
                cstrs_buf.emplace_back();
                auto& cv = cstrs_buf.back();
                for (size_t i = 0; i < v.size(); ++i)
                  cv.push_back(v[i].c_str());
                attr_values.push_back(cv.data());
                break;
              }
              default:
                throw sol::error("paddle: custom_op_eager_run unsupported "
                                 "attr type code");
            }
          }
        }
        int n = pd_custom_op_eager_run(
            op_type.c_str(), xs.empty() ? nullptr : xs.data(),
            static_cast<int>(xs.size()), attr_types.empty() ? nullptr
                                                            : attr_types.data(),
            attr_values.empty() ? nullptr : attr_values.data(),
            attr_lens.empty() ? nullptr : attr_lens.data(),
            static_cast<int>(attr_types.size()), nullptr, 0);
        pd::check();
        if (n < 0) pd::throw_status();
        std::vector<pd_tensor> outs(static_cast<size_t>(n), nullptr);
        int got = pd_custom_op_eager_run(
            op_type.c_str(), xs.empty() ? nullptr : xs.data(),
            static_cast<int>(xs.size()), attr_types.empty() ? nullptr
                                                            : attr_types.data(),
            attr_values.empty() ? nullptr : attr_values.data(),
            attr_lens.empty() ? nullptr : attr_lens.data(),
            static_cast<int>(attr_types.size()), outs.data(),
            static_cast<int>(outs.size()));
        pd::check();
        if (got != n) {
          throw sol::error("paddle: custom_op_eager_run output count changed");
        }
        sol::state_view lv(L);
        sol::table results = lv.create_table(0, n);
        for (int i = 0; i < n; ++i) {
          results[i + 1] = LuaTensor(outs[static_cast<size_t>(i)]);
        }
        return sol::make_object(L, results);
      });
}

}  // namespace pd