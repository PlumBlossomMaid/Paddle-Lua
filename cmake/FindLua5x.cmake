# cmake/FindLua5x.cmake
#
# Finds ONE Lua version's headers + library. The build is version-locked
# by -DLUA_VERSION (51/52/53/54/jit), matching how luarocks installs per
# Lua version (plan/modules/04-packaging.md sec.3.4): each version gets
# its own paddle_core binary, and this module resolves just the one.
#
# Input:
#   LUA_VERSION  "51"|"52"|"53"|"54"|"jit"  (default: 51)
#   LUA_ROOT     optional explicit prefix (luarocks passes the path)
# Output:
#   Lua5x_FOUND        bool
#   Lua5x_INCLUDE_DIR  dir containing lua.h (luajit.h for jit)
#   Lua5x_LIBRARY      full path of the import/static lib
#   Lua5x_VERSION      "5.1" .. "5.4" / "LuaJIT"
#
# Windows layout seen on the dev machine (2026-10-25):
#   Lua 5.1   C:\Program Files (x86)\Lua\5.1\{include,lib\lua5.1.lib}
#   LuaJIT    vcpkg installed/x64-windows (lua51.dll + lua51.lib)

if(NOT DEFINED LUA_VERSION)
  set(LUA_VERSION "51")
endif()

# dotted version for path layout ("51" -> "5.1", "jit" -> "jit")
string(SUBSTRING "${LUA_VERSION}" 0 1 _lua_major)
string(SUBSTRING "${LUA_VERSION}" 1 1 _lua_minor)
if(LUA_VERSION STREQUAL "jit")
  set(_lua_ver_dotted "jit")
else()
  set(_lua_ver_dotted "5.${_lua_minor}")
endif()

set(Lua5x_VERSION "")
set(Lua5x_INCLUDE_DIR "")
set(Lua5x_LIBRARY "")

# ---- candidate include dirs ------------------------------------------
set(_lua_inc_candidates "")
set(_lua_lib_hints "")

if(LUA_VERSION STREQUAL "jit")
  set(Lua5x_VERSION "LuaJIT")
  set(_lua_inc_names luajit.h lua.h)
  set(_lua_lib_names luajit-5.1 libluajit-5.1 lua51)
else()
  set(Lua5x_VERSION "5.${_lua_minor}")
  set(_lua_inc_names lua.h)
  set(_lua_lib_names
    "lua${LUA_VERSION}" "lua5.${_lua_minor}" "lua${_lua_minor}")
endif()

# explicit LUA_ROOT first
if(DEFINED LUA_ROOT)
  list(APPEND _lua_inc_candidates "${LUA_ROOT}/include" "${LUA_ROOT}")
  list(APPEND _lua_lib_hints "${LUA_ROOT}/lib" "${LUA_ROOT}")
endif()

# known install prefixes
if(WIN32)
  # vcpkg LuaJIT (x64, no versioned subdir) -- candidate for EVERY
  # version: on the dev machine it is the only x64 5.1-ABI Lua, while
  # the PUC install under Program Files (x86) is 32-bit and its .lib is
  # dropped by the x64 linker (LNK4272 -> unresolved lua_* symbols,
  # caught 2026-10-25). VS's integrated VCPKG_ROOT points at its own
  # tree, which may lack Lua -- probe existence, don't trust the
  # variable, then fall back to the known dev-machine path (same lesson
  # as SOL2_INCLUDE_DIR).
  foreach(_vk
      "$ENV{VCPKG_ROOT}/installed/x64-windows"
      "E:/code/vcpkg/installed/x64-windows")
    if(EXISTS "${_vk}/include/lua.h" OR EXISTS "${_vk}/include/luajit/lua.h")
      # LuaJIT headers live in the luajit/ SUBDIRECTORY; the include root
      # may hold a different Lua's lua.h (vcpkg lua port = 5.5 on the dev
      # machine, which sol2 does not support -- C5). Prefer the subdir so
      # sol2 compiles against 5.1 headers and emits only 5.1-ABI calls;
      # otherwise it references lua_isinteger etc (5.3+) and fails to
      # link against the 5.1 LuaJIT lib (caught 2026-10-25, LNK2019).
      list(APPEND _lua_inc_candidates "${_vk}/include/luajit")
      list(APPEND _lua_inc_candidates "${_vk}/include")
      list(APPEND _lua_lib_hints "${_vk}/lib")
      break()
    endif()
  endforeach()
  # PUC install prefixes
  list(APPEND _lua_inc_candidates
    "C:/Program Files (x86)/Lua/${_lua_ver_dotted}/include"
    "C:/Program Files/Lua/${_lua_ver_dotted}/include")
  list(APPEND _lua_lib_hints
    "C:/Program Files (x86)/Lua/${_lua_ver_dotted}/lib"
    "C:/Program Files/Lua/${_lua_ver_dotted}/lib")
else()
  list(APPEND _lua_inc_candidates
    "/usr/include/lua5.${_lua_minor}" "/usr/include/luajit-2.1")
  list(APPEND _lua_lib_hints
    "/usr/lib/x86_64-linux-gnu" "/usr/lib")
endif()

# Probe candidates explicitly one at a time. Do NOT expand a list into
# find_path() PATHS: a path with spaces (C:/Program Files (x86)/...) then
# splits into several arguments and silently never matches (caught by
# probe 2026-10-25).
foreach(_cand IN LISTS _lua_inc_candidates)
  foreach(_n IN LISTS _lua_inc_names)
    if(NOT Lua5x_INCLUDE_DIR AND EXISTS "${_cand}/${_n}")
      set(Lua5x_INCLUDE_DIR "${_cand}")
    endif()
  endforeach()
endforeach()
if(NOT Lua5x_INCLUDE_DIR)
  find_path(Lua5x_INCLUDE_DIR NAMES ${_lua_inc_names})
endif()

foreach(_hint IN LISTS _lua_lib_hints)
  foreach(_n IN LISTS _lua_lib_names)
    if(NOT Lua5x_LIBRARY AND WIN32 AND EXISTS "${_hint}/${_n}.lib")
      set(Lua5x_LIBRARY "${_hint}/${_n}.lib")
    elseif(NOT Lua5x_LIBRARY AND EXISTS "${_hint}/lib${_n}.so")
      set(Lua5x_LIBRARY "${_hint}/lib${_n}.so")
    elseif(NOT Lua5x_LIBRARY AND EXISTS "${_hint}/${_n}.so")
      set(Lua5x_LIBRARY "${_hint}/${_n}.so")
    elseif(NOT Lua5x_LIBRARY AND EXISTS "${_hint}/${_n}")
      set(Lua5x_LIBRARY "${_hint}/${_n}")
    endif()
  endforeach()
endforeach()
if(NOT Lua5x_LIBRARY)
  find_library(Lua5x_LIBRARY NAMES ${_lua_lib_names})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Lua5x
  REQUIRED_VARS Lua5x_INCLUDE_DIR Lua5x_LIBRARY
  FAIL_MESSAGE
    "Lua ${LUA_VERSION} not found. Set -DLUA_ROOT=<prefix> (luarocks does this) or install it. (04-packaging.md sec.3.4)")
