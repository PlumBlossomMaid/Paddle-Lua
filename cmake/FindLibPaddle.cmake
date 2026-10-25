# cmake/FindLibPaddle.cmake
#
# Locates the Paddle build to link against. Discovery order
# (plan/modules/04-packaging.md sec.3.2):
#   1. env PADDLE_LUA_LIBPADDLE -- explicit override (a single .lib/.so)
#   2. env PADDLE_ROOT / -DPADDLE_ROOT -- find the build dir under it
#   3. system library paths
#   4. actionable error (what to set, not just "not found")
#
# Reality (m0-report sec.2): the WITH_PYTHON=OFF + ON_INFER=OFF training
# build does NOT produce one aggregate libpaddle -- it produces component
# static libs + phi.dll. The smoke programs linked the same baseline as
# paddle_inference_c.dll: its LINK_LIBRARIES list in build.ninja (483
# libs). This module therefore exposes:
#   LibPaddle_FOUND
#   LibPaddle_INCLUDE_DIRS   (PADDLE_ROOT source root -- headers are
#                             included as "paddle/phi/api/include/...")
#   LibPaddle_LINK_LIBRARIES (absolute .lib paths for MSVC)
#   LibPaddle_DEFINES        (the Paddle compile-time macros the C ABI
#                             layer needs, e.g. PHI_INNER/STATIC_IR)
#   LibPaddle_BUILD_DIR      (for /LIBPATH and runtime DLL search)
#   LibPaddle_SOURCE_DIR
#
# Windows specifics (m0-report sec.16.4): must add
# /LIBPATH:<build-dir> (libonednn.lib/libcblas.lib sit at the build
# root) and /FORCE:MULTIPLE is needed only for standalone EXEs -- the
# binding DLL path does NOT hit LNK2005 (m0-report sec.4.1).

include(FindPackageHandleStandardArgs)

set(LibPaddle_FOUND FALSE)

# ---- 1. explicit override -------------------------------------------
set(_lp_paddle_lua_lib $ENV{PADDLE_LUA_LIBPADDLE})
if(_lp_paddle_lua_lib)
  get_filename_component(LibPaddle_BUILD_DIR "${_lp_paddle_lua_lib}" DIRECTORY)
  set(LibPaddle_LINK_LIBRARIES "${_lp_paddle_lua_lib}")
  set(LibPaddle_FOUND TRUE)
  # include dirs unknown from a bare lib -- caller must pass -DPADDLE_ROOT
  if(DEFINED PADDLE_ROOT)
    set(LibPaddle_SOURCE_DIR "${PADDLE_ROOT}")
    set(LibPaddle_INCLUDE_DIRS "${PADDLE_ROOT}")
  endif()
endif()

# ---- 2. PADDLE_ROOT (env wins, -DPADDLE_ROOT fallback) ---------------
if(NOT LibPaddle_FOUND)
  if(DEFINED ENV{PADDLE_ROOT})
    set(LibPaddle_SOURCE_DIR "$ENV{PADDLE_ROOT}")
  elseif(DEFINED PADDLE_ROOT)
    set(LibPaddle_SOURCE_DIR "${PADDLE_ROOT}")
  endif()
  if(LibPaddle_SOURCE_DIR)
    # find the build dir (build-nopy or any build-* with build.ninja)
    file(GLOB _lp_build_dirs "${LibPaddle_SOURCE_DIR}/build-*")
    foreach(_d ${_lp_build_dirs})
      if(EXISTS "${_d}/build.ninja")
        set(LibPaddle_BUILD_DIR "${_d}")
        break()
      endif()
    endforeach()
    if(NOT LibPaddle_BUILD_DIR)
      foreach(_d "${LibPaddle_SOURCE_DIR}/build" "${LibPaddle_SOURCE_DIR}/build-nopy")
        if(EXISTS "${_d}/build.ninja")
          set(LibPaddle_BUILD_DIR "${_d}")
          break()
        endif()
      endforeach()
    endif()
  endif()
endif()

# ---- 3. extract the link baseline from build.ninja (Windows) ---------
# paddle_inference_c.dll is linked against the full component set; reuse
# its LINK_LIBRARIES as the baseline (m0-report sec.2). Locate the .dll
# build statement by name, then read a window after it: the LINK_LIBRARIES
# property line follows the build line but OTHER properties (LINK_FLAGS,
# DEFINES...) sit between, so a plain adjacent-line regex fails (caught
# by probe 2026-10-25). Values are relative to the build root or bare
# system lib names; convert the former to absolute.
if(LibPaddle_BUILD_DIR AND WIN32)
  file(READ "${LibPaddle_BUILD_DIR}/build.ninja" _lp_ninja)
  string(FIND "${_lp_ninja}" "paddle_inference_c.dll" _lp_dll_pos)
  if(_lp_dll_pos GREATER -1)
    string(SUBSTRING "${_lp_ninja}" ${_lp_dll_pos} 131072 _lp_win)
    string(REGEX MATCH "LINK_LIBRARIES = ([^\n]*)\n" _lp_m "${_lp_win}")
    if(_lp_m)
      set(_lp_link_libs "${CMAKE_MATCH_1}")
      separate_arguments(_lp_link_libs)
      set(LibPaddle_LINK_LIBRARIES "")
      foreach(_l ${_lp_link_libs})
        if(_l MATCHES "\\.lib$")
          if(IS_ABSOLUTE "${_l}")
            list(APPEND LibPaddle_LINK_LIBRARIES "${_l}")
          elseif(EXISTS "${LibPaddle_BUILD_DIR}/${_l}")
            list(APPEND LibPaddle_LINK_LIBRARIES "${LibPaddle_BUILD_DIR}/${_l}")
          else()
            # bare name (system lib like shlwapi.lib) -- resolve via /LIBPATH
            list(APPEND LibPaddle_LINK_LIBRARIES "${_l}")
          endif()
        endif()
      endforeach()
      set(LibPaddle_FOUND TRUE)
    endif()
  endif()
endif()

# ---- 4. Unix: find libpaddle.so / component pattern ------------------
if(NOT LibPaddle_FOUND AND LibPaddle_SOURCE_DIR)
  find_library(_lp_fluid_lib paddle_fluid
    PATHS "${LibPaddle_BUILD_DIR}" "${LibPaddle_SOURCE_DIR}" NO_DEFAULT_PATH)
  if(_lp_fluid_lib)
    set(LibPaddle_LINK_LIBRARIES "${_lp_fluid_lib}")
    set(LibPaddle_FOUND TRUE)
  endif()
endif()

# ---- include dirs + defines ------------------------------------------
if(LibPaddle_SOURCE_DIR)
  set(LibPaddle_INCLUDE_DIRS "${LibPaddle_SOURCE_DIR}")
endif()
# Third-party headers (glog / onednn / eigen / mklml ...) live under the
# build tree, NOT the source root -- capi/capi_gen fail with C1083
# (glog/logging.h, dnnl.hpp) without them (caught 2026-10-25). Extract
# the -I list from build.ninja's INCLUDES line (same provenance as the
# link baseline); fall back to the few known dirs if not extractable.
set(_lp_third_incs "")
if(LibPaddle_BUILD_DIR AND WIN32 AND _lp_ninja)
  string(REGEX MATCH "\n  INCLUDES = ([^\n]*)\n" _lp_inc "${_lp_ninja}")
  if(_lp_inc)
    set(_lp_incs "${CMAKE_MATCH_1}")
    string(REGEX MATCHALL "-I([^ ]+)" _lp_inc_hits "${_lp_incs}")
    foreach(_h IN LISTS _lp_inc_hits)
      string(REGEX REPLACE "^-I" "" _p "${_h}")
      list(APPEND _lp_third_incs "${_p}")
    endforeach()
  endif()
endif()
if(_lp_third_incs)
  list(APPEND LibPaddle_INCLUDE_DIRS ${_lp_third_incs})
elseif(LibPaddle_BUILD_DIR)
  list(APPEND LibPaddle_INCLUDE_DIRS
    "${LibPaddle_BUILD_DIR}"
    "${LibPaddle_BUILD_DIR}/third_party/install"
    "${LibPaddle_BUILD_DIR}/third_party/install/glog/include"
    "${LibPaddle_BUILD_DIR}/third_party/install/onednn/include"
    "${LibPaddle_BUILD_DIR}/third_party/install/mklml/include"
    "${LibPaddle_BUILD_DIR}/third_party/install/gflags/include")
endif()

# The Paddle compile-time macros come from build.ninja itself (the CXX
# rule for a representative TU) -- the smoke needed the full 35, missing
# PHI_INNER/STATIC_IR/PADDLE_NO_PYTHON fails with C2236 (m0-report
# sec.16.4). Fall back to a conservative subset if not extractable.
set(LibPaddle_DEFINES "")
if(LibPaddle_BUILD_DIR AND WIN32 AND _lp_ninja)
  string(REGEX MATCH "\n  DEFINES = ([^\n]*)\n" _lp_def "${_lp_ninja}")
  if(_lp_def)
    set(_lp_defs "${CMAKE_MATCH_1}")
    string(STRIP "${_lp_defs}" _lp_defs)
    # GOOGLE_GLOG_DLL_DECL arrives as a quoted empty literal ("") from
    # build.ninja ($-escaping). glog's DECLARE_bool expands to
    # `extern GOOGLE_GLOG_DLL_DECL bool FLAGS_x` -- a literal "" is an
    # illegal linkage spec (C2537). Normalize to a true empty define.
    string(REPLACE
      "GOOGLE_GLOG_DLL_DECL=\"\"" "GOOGLE_GLOG_DLL_DECL=" _lp_defs
      "${_lp_defs}")
    separate_arguments(_lp_defs)
    set(LibPaddle_DEFINES ${_lp_defs})
  endif()
endif()
if(NOT LibPaddle_DEFINES)
  set(LibPaddle_DEFINES
    -DEIGEN_STRONG_INLINE=inline
    -DGLOG_NO_ABBREVIATED_SEVERITIES
    -DGOOGLE_GLOG_DLL_DECL=
    -DNOMINMAX
    -DPADDLE_DISABLE_PROFILER
    -DPADDLE_NO_PYTHON
    -DPADDLE_WITH_AVX
    -DPADDLE_WITH_MKLML
    -DPADDLE_WITH_DNNL
    -DPHI_INNER
    -DSTATIC_IR
  )
endif()

find_package_handle_standard_args(LibPaddle
  REQUIRED_VARS LibPaddle_FOUND LibPaddle_INCLUDE_DIRS LibPaddle_LINK_LIBRARIES
  FAIL_MESSAGE
    "Paddle build not found. Set PADDLE_ROOT to the Paddle source dir that has a build-*/build.ninja (e.g. PADDLE_ROOT=E:\\code\\Paddle), or PADDLE_LUA_LIBPADDLE to an explicit libpaddle lib. (04-packaging.md sec.3.2)")
