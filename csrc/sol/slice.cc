// slice.cc -- Lua string-slice spec -> C++ 0-based spec
// Stage: P2 / M1 (WORKPLAN 2.2.4). Last file of csrc/sol (layout.md P2).
//
// Design: docs/migration/plan/overview.md sec.6.1.2 (D14).
// Lua has no colon-slicing syntax and __index receives a single key, so
// slicing is expressed as a STRING:  x["1:3, :"]  (1-based, per C4).
// This TU port the proven converter from Insight7:
//   E:\code\Insight7\bindings\lua\insight_lua.cpp:357  lua_spec_to_cpp
// Rules (verified independently in overview.md sec.6.1.1):
//   - fields 0 (start) and 1 (stop) are 1-based: positive value -> -1;
//     negative values already count from the end in both conventions and
//     pass through unchanged;
//   - field 2 (step) passes through UNTOUCHED (an old Insight7 revision
//     decremented positive steps too -- that bug is fixed in the current
//     source and is NOT ported).
// Examples: "1:3" -> "0:2", "1:5:2" -> "0:4:2", "2,1:-1" -> "1,0:-1",
//           "::-1" -> "::-1", "::2" -> "::2".
//
// The produced string is a pure textual transform; consuming it into a
// strided_slice op happens at P5 (lua/paddle/tensor.lua + P3 ops).

#include "sol_common.hpp"

#include <cstdint>
#include <string>

namespace pd {

std::string lua_spec_to_cpp(const std::string& spec) {
  auto trim = [](const std::string& s) -> std::string {
    size_t begin = s.find_first_not_of(" \t");
    if (begin == std::string::npos) return std::string();
    return s.substr(begin, s.find_last_not_of(" \t") - begin + 1);
  };
  auto to_zero_based = [](const std::string& field) -> std::string {
    if (field.empty()) return field;
    try {
      size_t pos = 0;
      long long val = std::stoll(field, &pos);
      if (pos == field.size() && val > 0) return std::to_string(val - 1);
    } catch (const std::exception&) {
    }
    return field;
  };

  std::string out;
  size_t dim_begin = 0;
  while (true) {
    size_t dim_end = spec.find(',', dim_begin);
    std::string dim = spec.substr(dim_begin, dim_end == std::string::npos
                                                 ? std::string::npos
                                                 : dim_end - dim_begin);
    size_t field_begin = 0;
    int field_idx = 0;
    while (true) {
      size_t field_end = dim.find(':', field_begin);
      std::string field =
          trim(dim.substr(field_begin, field_end == std::string::npos
                                           ? std::string::npos
                                           : field_end - field_begin));
      // Fields 0 (start) and 1 (stop) are 1-based; field 2 is the step.
      out += (field_idx < 2) ? to_zero_based(field) : field;
      if (field_end == std::string::npos) break;
      out += ':';
      field_begin = field_end + 1;
      field_idx++;
    }
    if (dim_end == std::string::npos) break;
    out += ',';
    dim_begin = dim_end + 1;
  }
  return out;
}

/* Expose the converter as core.slice_spec(spec) -> 0-based string.
 * Pure string transform, usable by P5 without touching the C ABI. */
void register_slice(sol::table& mod) {
  mod.set_function("slice_spec",
                   [](const std::string& spec) { return lua_spec_to_cpp(spec); });
}

}  // namespace pd