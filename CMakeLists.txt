# paddle-lua top-level CMakeLists.txt
#
# Builds the paddle_core C extension (sol2 binding over the pure-C ABI
# layer). Layout / stage: plan/layout.md (P1 capi, P2 sol, P4 packaging).
#
# One source tree, five binaries (02-binding.md sec.3.1): capi is pure C
# ABI compiled ONCE (5 Lua versions share it); csrc/sol is sol2 heavy
# templates compiled once PER Lua version. luarocks invokes this with
# -DLUA_VERSION=51/52/53/54/jit and gets that version's paddle_core
# (04-packaging.md sec.3.4).
#
# Targets:
#   paddle_capi     static lib: csrc/capi + csrc/capi_gen (generated ops),
#                   links against libpaddle (via FindLibPaddle)
#   paddle_core     module: csrc/sol, links paddle_capi + libluaX
#   smoke_capi      EXE: csrc/capi test (test_c_only.c) -- compile gate
#
# Windows specifics (m0-report sec.2/16.4/4.1): link baseline = the 483
# component libs from build.ninja (FindLibPaddle); /LIBPATH:build-nopy;
# /FORCE:MULTIPLE only for standalone EXEs, NOT the binding DLL.

cmake_minimum_required(VERSION 3.15)
project(paddle-lua CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  # Paddle's component libs are built with /MT (static CRT, m0-report
  # sec.16.4 / T-M1-05 pit 2). Mixing /MD here pulls libcmt+msvcrt both
  # into the DLL -> LNK2038 RuntimeLibrary mismatch + LNK1169 duplicate
  # symbols (caught 2026-10-25). vcpkg LuaJIT's lua51.lib is only an
  # import stub (no CRT objects) so /MT is safe against it too.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(LibPaddle REQUIRED)
find_package(Lua5x REQUIRED)

# ---- sol2: header-only, locate its include dir -----------------------
# sol2 ships via vcpkg as installed/<triplet>/include/sol/sol.hpp.
# Discovery order: -DSOL2_INCLUDE_DIR wins; then the VCPKG_ROOT-derived
# path, but ONLY if sol.hpp actually exists there (VS's integrated vcpkg
# sets VCPKG_ROOT to its own tree, which may lack sol -- probe the file,
# don't trust the variable); then the known dev-machine vcpkg path.
# (T-M1-06 pit 3: the Lua 5.1 headers MUST stay first on the include
# path; the vcpkg include dir is used only for sol itself, otherwise
# sol2 picks up Lua 5.5 headers and emits APIs that do not exist in 5.1
# -> LNK2019.)
if(NOT DEFINED SOL2_INCLUDE_DIR OR NOT EXISTS "${SOL2_INCLUDE_DIR}/sol/sol.hpp")
  set(SOL2_INCLUDE_DIR "" CACHE PATH "dir containing sol/sol.hpp" FORCE)
  foreach(_cand
      "$ENV{VCPKG_ROOT}/installed/x64-windows/include"
      "E:/code/vcpkg/installed/x64-windows/include")
    if(EXISTS "${_cand}/sol/sol.hpp")
      set(SOL2_INCLUDE_DIR "${_cand}" CACHE PATH
        "dir containing sol/sol.hpp" FORCE)
      break()
    endif()
  endforeach()
endif()

# ---- sources ---------------------------------------------------------
file(GLOB _capi_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi/*.cc")
file(GLOB _capi_gen_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi_gen/*.cc")
file(GLOB _sol_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sol/*.cc")

# ---- paddle_capi: the pure-C-ABI middle layer + generated ops --------
add_library(paddle_capi STATIC ${_capi_srcs} ${_capi_gen_srcs})
target_include_directories(paddle_capi PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi")
target_include_directories(paddle_capi SYSTEM PUBLIC
  "${LibPaddle_INCLUDE_DIRS}")
target_compile_definitions(paddle_capi PUBLIC ${LibPaddle_DEFINES})

if(MSVC)
  target_compile_options(paddle_capi PRIVATE /bigobj /W0)
endif()

# ---- paddle_core: sol2 binding, one per Lua version ------------------
add_library(paddle_core MODULE ${_sol_srcs})
target_include_directories(paddle_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi")
target_include_directories(paddle_core SYSTEM PUBLIC
  "${Lua5x_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sol"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi_gen")
target_compile_definitions(paddle_core PUBLIC ${LibPaddle_DEFINES})
# sol2 arg type-checking for op lambdas (add(1,2) -> Lua error, not a
# 0xC0000005): SOL_SAFE_FUNCTION_CALLS alone is NOT enough under LuaJIT.
# The actual gate for the pre-call argument check is
#   stack.hpp:198  #if SOL_IS_OFF(SOL_PROPAGATE_EXCEPTIONS)
# and LuaJIT 2.1+ forces that macro to SOL_DEFAULT_ON (compatibility/
# lua_version.hpp:142-150), which reads SOL_ON only if the user defines
# SOL_EXCEPTIONS_SAFE_PROPAGATION. So we must turn exception propagation
# OFF explicitly, or number->const LuaTensor& reaches
# stack_get_unqualified.hpp's get_no_lua_nil, which dereferences the NULL
# from lua_touserdata (0xC0000005). Bad arg types must surface as Lua
# errors (modules/02-binding.md sec.3.3).
target_compile_definitions(paddle_core PRIVATE
  SOL_SAFE_FUNCTION_CALLS=1
  SOL_EXCEPTIONS_SAFE_PROPAGATION=0)

if(MSVC)
  target_compile_options(paddle_core PRIVATE /bigobj /W0)
  # sol2 needs the Lua headers first; vcpkg include only for sol itself
  # (T-M1-06 pit 3)
  target_include_directories(paddle_core SYSTEM PRIVATE
    "${Lua5x_INCLUDE_DIR}"
    "${SOL2_INCLUDE_DIR}")
endif()

target_link_libraries(paddle_core PRIVATE
  paddle_capi
  "${Lua5x_LIBRARY}"
  ${LibPaddle_LINK_LIBRARIES})

if(WIN32)
  # libonednn.lib / libcblas.lib sit at the build root (m0-report sec.2)
  target_link_directories(paddle_core PRIVATE "${LibPaddle_BUILD_DIR}")
  # module name must be paddle_core.dll (require "paddle_core")
  set_target_properties(paddle_core PROPERTIES PREFIX "" OUTPUT_NAME paddle_core)
endif()

# ---- smoke_capi: pure-C compile gate (test_c_only.c) -----------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi/test_c_only.c")
  add_executable(smoke_capi "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi/test_c_only.c")
  target_include_directories(smoke_capi PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi")
  target_link_libraries(smoke_capi PRIVATE paddle_capi)
endif()
