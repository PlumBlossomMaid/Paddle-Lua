-- lua/paddle/utils/cpp_extension.lua -- paddle.utils.cpp_extension (P16)
-- JIT-compile a user C++ custom-op into a DLL and hand back a table of
-- callables, one per registered operator. Contract: api/utils.md sec 2.x
-- design: plan/modules/16-cpp-extension.md sec 3.
--
--   local ops = paddle.utils.cpp_extension.load{
--     name = "my_relu",
--     sources = { "my_relu.cc" },        -- absolute or CWD-relative
--     verbose = false,                   -- stream the compiler log
--   }
--   local y = ops.my_relu(x)             -- positional inputs -> {out...}
--
-- Pipeline (mirrors upstream cpp_extension.load, python/paddle/utils/
-- cpp_extension/cpp_extension.py:1637-1785; toolchain/hash parts:
-- extension_utils.py):
--   1. probe the host toolchain (MSVC vs clang++) and the Paddle build
--      tree (include/lib baseline, the SAME one paddle_core.dll links)
--   2. compute a cache key = md5(source contents + paddle version +
--      compiler path + flags); skip the compile on a cache hit
--   3. clang++/cl.exe JIT-compile sources into a shared object
--   4. core.load_custom_op(dso)  -- registers custom_op.<name> + legacy
--   5. return { <op_name> = callable } for every newly registered op;
--      each callable forwards to core.custom_op_run (the C ABI executor,
--      custom_op.cc, 16-cpp-extension.md sec 3.2 step 6)
--
-- Host-only linking rule (measured 2026-11-02): the produced DLL links
-- against the ALREADY-LOADED host Paddle (phi.lib + paddle_core.lib),
-- NEVER a second full Paddle -- a second copy re-registers common.dll's
-- shared flag registry and dies with "flag multiple definition io.cc"
-- (flags_native.cc:309).
--
-- Lua 5.1 subset (C3). Compile-side uses os.execute (cmd), the same path
-- manual build_test_custom_relu.cmd takes.

local rule = require "argrule.rule"
local md5 = require "paddle.utils.md5"

local M = {}

-- ---------------------------------------------------------------------
-- 1. Environment probes
-- ---------------------------------------------------------------------

local function exists(p)
  local f = io.open(p, "rb")
  if f then f:close() return true end
  return false
end

-- io.open cannot open a DIRECTORY (LuaJIT returns nil); probe existence
-- via the shell instead (cmd /c if exist works for both files and dirs).
-- cmd's echo output trails a space AND CRLF; trim all whitespace before
-- comparing.
local function exists_dir(p)
  local probe = io.popen('cmd /c if exist "' .. p .. '" echo 1 2>nul')
  if not probe then return false end
  local line = probe:read("*l")
  probe:close()
  return line ~= nil and line:gsub("%s", "") == "1"
end

-- Paddle source root (PADDLE_ROOT env, fallback common dev location).
local function paddle_root()
  local r = os.getenv("PADDLE_ROOT")
  if r and r ~= "" and exists_dir(r) then return r end
  if exists_dir("E:\\code\\Paddle") then return "E:\\code\\Paddle" end
  error("paddle.utils.cpp_extension: set PADDLE_ROOT to the Paddle "
        .. "source root (it has a build-*/build.ninja)", 3)
end

-- Paddle build dir: PADDLE_BUILD_DIR env, else the first build-* under
-- the root that has build.ninja (FindLibPaddle.cmake sec.2 discovery).
local function paddle_build_dir(root)
  local b = os.getenv("PADDLE_BUILD_DIR")
  if b and b ~= "" and exists(b .. "\\build.ninja") then return b end
  local probe = io.popen(
    "cmd /c dir /b /ad " .. root .. "\\build-* 2>nul")
  if probe then
    for line in probe:lines() do
      local d = root .. "\\" .. line:gsub("\r$", "")
      if exists(d .. "\\build.ninja") then
        probe:close()
        return d
      end
    end
    probe:close()
  end
  error("paddle.utils.cpp_extension: no build-*/build.ninja under "
        .. root .. " (set PADDLE_BUILD_DIR to override)", 3)
end

-- Locate the MSVC vcvars64.bat. Order: VCVARS64 env override -> vswhere
-- -> common VS install paths (16/17/18 Community/Professional/Enterprise).
local function find_vcvars()
  local v = os.getenv("VCVARS64")
  if v and v ~= "" and exists(v) then return v end
  local candidates = {
    "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer"
      .. "\\vswhere.exe",
    "C:\\Program Files\\Microsoft Visual Studio\\Installer\\vswhere.exe",
  }
  for _, vswhere in ipairs(candidates) do
    if exists(vswhere) then
      local p = io.popen('"' .. vswhere
        .. '" -latest -products * -property installationPath 2>nul')
      if p then
        local line = p:read("*l")
        p:close()
        if line and line ~= "" then
          local bat = line .. "\\VC\\Auxiliary\\Build\\vcvars64.bat"
          if exists(bat) then return bat end
        end
      end
    end
  end
  for _, year in ipairs({ "18", "17", "16" }) do
    for _, flavor in ipairs({ "Community", "Professional", "Enterprise" }) do
      local base = "C:\\Program Files\\Microsoft Visual Studio\\"
        .. year .. "\\" .. flavor
      local bat = base .. "\\VC\\Auxiliary\\Build\\vcvars64.bat"
      if exists(bat) then return bat end
    end
  end
  error("paddle.utils.cpp_extension: MSVC not found (needs Visual Studio "
        .. "2017+; set VCVARS64 to its vcvars64.bat)", 3)
end

-- The include baseline that paddle_capi itself compiles with (extracted
-- from build.ninja INCLUDES, FindLibPaddle.cmake sec.3). Mirrors
-- build_test_custom_relu.cmd; paths verified 2026-11-02.
local function paddle_include_dirs(root, build)
  return {
    root,                                                   -- paddle/
    build,                                                  -- build root
    build .. "\\third_party\\install\\gflags\\include",
    build .. "\\third_party\\install\\glog\\include",
    root .. "\\third_party\\eigen3",
    root .. "\\third_party\\threadpool",
    root .. "\\third_party\\dlpack\\include",
    build .. "\\third_party\\install\\xxhash\\include",
    build .. "\\third_party\\install\\warpctc\\include",
    build .. "\\third_party\\install\\warprnnt\\include",
    build .. "\\third_party\\install\\utf8proc\\include",
    build .. "\\third_party\\install\\mklml\\include",
    build .. "\\third_party\\install\\onednn\\include",
    build .. "\\third_party\\install\\protobuf\\include",
    root .. "\\third_party\\nlohmann_json\\include",
    build .. "\\third_party\\install\\yaml-cpp\\include",
    build .. "\\third_party\\pybind\\src\\extern_pybind\\include",
    build .. "\\third_party\\install\\libuv\\include",
    build .. "\\third_party\\install\\cryptopp\\include",
    build .. "\\third_party\\pocketfft\\src",
    build .. "\\third_party\\dirent\\src\\extern_dirent\\include",
  }
end

-- Compile-time defines (build.ninja DEFINES for paddle_capi, the same
-- macro set the kernel/host requires). `interface`-macro hazard dodged by
-- WIN32_LEAN_AND_MEAN (msvcrt's #define interface struct vs the pir
-- interface() method, measured 2026-11-02).
local function paddle_defines()
  return {
    "-DEIGEN_STRONG_INLINE=inline",
    "-DGLOG_NO_ABBREVIATED_SEVERITIES",
    "-DGOOGLE_GLOG_DLL_DECL=",
    "-DNOMINMAX",
    "-DPADDLE_DISABLE_PROFILER",
    "-DPADDLE_DLL_INFERENCE",
    "-DPADDLE_NO_PYTHON",
    "-DPADDLE_VERSION=3.5.0.dev20260803",
    "-DPADDLE_WITH_AVX",
    "-DPADDLE_WITH_CRYPTO",
    "-DPADDLE_WITH_DNNL",
    "-DPADDLE_WITH_MKLML",
    "-DPADDLE_WITH_POCKETFFT",
    "-DPADDLE_WITH_SSE3",
    "-DPHI_SHARED",
    "-DSTATIC_IR",
    "-DUTF8PROC_STATIC",
    "-DWIN32_LEAN_AND_MEAN",
  }
end

-- Link baseline: the ALREADY-LOADED host Paddle (phi.dll import lib +
-- paddle_core.dll import lib). Never paddle_inference.lib / a second
-- full Paddle (flags_native.cc:309 double registration, measured).
local function paddle_link_libs(root, build)
  local phi = build .. "\\paddle\\phi\\phi.lib"
  if not exists(phi) then
    error("paddle.utils.cpp_extension: phi.lib not found at " .. phi, 3)
  end
  -- paddle_core.lib lives in OUR build dir (the paddle-lua repo's build):
  -- PADDLE_LUA_BUILD env, else the CWD (smokes run from the repo root),
  -- else common dev names. ipairs dies on a leading nil (t[1]=nil ends
  -- the walk), so build the candidate list with add() -- a nil first
  -- element would silently iterate zero candidates.
  local cwd = (io.popen("cd"):read("*l") or ""):gsub("%s", "")
  local pd_core_lib
  local cands = {}
  local function add(c)
    if c and c ~= "" then cands[#cands + 1] = c end
  end
  add(os.getenv("PADDLE_LUA_BUILD"))
  add(cwd .. "\\build-test\\paddle_core.lib")
  add(cwd .. "\\build\\paddle_core.lib")
  add("build-test\\paddle_core.lib")
  add("build\\paddle_core.lib")
  for _, cand in ipairs(cands) do
    if exists(cand) then pd_core_lib = cand break end
  end
  if not pd_core_lib then
    error("paddle.utils.cpp_extension: paddle_core.lib not found "
          .. "(set PADDLE_LUA_BUILD to your paddle-lua build dir)", 3)
  end
  return { phi, pd_core_lib }
end

-- Build directory for JIT products: PADDLE_EXTENSION_DIR env, else the
-- paddle-lua build dir (cache lives alongside the host .dll).
local function extension_dir()
  local d = os.getenv("PADDLE_EXTENSION_DIR")
  if d and d ~= "" then return d end
  local cwd = (io.popen("cd"):read("*l") or ""):gsub("%s", "")
  local cands = {}
  local function add(c)
    if c and c ~= "" then cands[#cands + 1] = c end
  end
  add(os.getenv("PADDLE_LUA_BUILD"))
  add(cwd .. "\\build-test")
  add("build-test")
  for _, cand in ipairs(cands) do
    if exists(cand .. "\\paddle_core.dll") then return cand end
  end
  -- fall back to CWD (creates the cache next to wherever we run)
  return cwd
end

-- ---------------------------------------------------------------------
-- 2. Cache key
-- ---------------------------------------------------------------------

local function read_file(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local content = f:read("*a")
  f:close()
  return content
end

-- Hash the sources + everything that changes the product (compiler path,
-- paddle version, defines). Source paths resolve to absolute so the key
-- is location-independent.
local _core_version
local function paddle_version()
  if not _core_version then
    local c = require "paddle_core"
    _core_version = tostring(c.version())
  end
  return _core_version
end

local function cache_key(name, sources, toolchain)
  local parts = { name, paddle_version(), toolchain }
  for _, p in ipairs(sources) do
    local content = read_file(p)
    if not content then
      error("paddle.utils.cpp_extension: source not found: " .. p, 3)
    end
    parts[#parts + 1] = md5.sumhexa(content)
  end
  for _, d in ipairs(paddle_defines()) do parts[#parts + 1] = d end
  return md5.sumhexa(table.concat(parts))
end

-- ---------------------------------------------------------------------
-- 3. Compile driver
-- ---------------------------------------------------------------------

-- Resolve CWD-relative source paths to absolute.
local function absolute_cwd(p)
  if p:match("^[A-Za-z]:[/\\]") or p:sub(1, 1) == "/" then return p end
  local cwd = io.popen("cd"):read("*l") or "."
  return cwd .. "\\" .. p
end

-- Compile `name` sources -> extension DLL. Returns (dll_path, error).
local function compile(name, sources, extra_cxx_cflags, verbose)
  local root = paddle_root()
  local build = paddle_build_dir(root)
  local incs, libs = paddle_include_dirs(root, build), paddle_link_libs(root, build)

  local out_dir = extension_dir()
  local key = cache_key(name, sources, root .. "\\" .. build)
  local dll = out_dir .. "\\" .. name .. "_" .. key:sub(1, 12) .. ".dll"
  if exists(dll) then
    -- cache hit: nothing to compile
    return dll, nil
  end

  -- Drop stale products with the same name prefix (sources changed).
  local stale = io.popen("cmd /c dir /b " .. out_dir .. "\\" .. name
                         .. "_*.dll 2>nul")
  if stale then
    for line in stale:lines() do
      os.remove(out_dir .. "\\" .. line)
    end
    stale:close()
  end

  local vcvars = find_vcvars()
  local defs = table.concat(paddle_defines(), " ")
  local inc_args = {}
  for _, d in ipairs(incs) do inc_args[#inc_args + 1] = "-I" .. d end
  for i, p in ipairs(sources) do
    sources[i] = absolute_cwd(p)
  end
  local cflags_extra = extra_cxx_cflags and table.concat(extra_cxx_cflags, " ")
                         or ""
  local lib_args = {}
  for _, l in ipairs(libs) do lib_args[#lib_args + 1] = l end

  local log = out_dir .. "\\" .. name .. "_" .. key:sub(1, 12) .. ".log"
  local obj_prefix = out_dir .. "\\" .. name .. "_" .. key:sub(1, 12)

  -- The SAME recipe build_test_custom_relu.cmd uses (measured): vcvars
  -- for the MSVC env (std headers/CRT), each source .cc -> .obj, then
  -- link the .objs into one DLL. /Fo cannot be shared across multiple
  -- sources (D8036), so per-source .obj paths.
  local vc = 'call "' .. vcvars .. '" >nul 2>nul && '
  local compile_lines = {}
  for _, p in ipairs(sources) do
    local base = p:match("([^/\\]+)$"):gsub("%.", "_")
    compile_lines[#compile_lines + 1] =
      vc .. "cl /nologo /c /O2 /EHsc /std:c++17 /MT /bigobj "
      .. defs .. " " .. table.concat(inc_args, " ")
      .. cflags_extra .. " " .. p
      .. ' /Fo:' .. obj_prefix .. "_" .. base .. ".obj"
  end
  local objs = {}
  for _, p in ipairs(sources) do
    local base = p:match("([^/\\]+)$"):gsub("%.", "_")
    objs[#objs + 1] = obj_prefix .. "_" .. base .. ".obj"
  end
  local out_cmd = (table.concat(compile_lines, " && ")
    .. " && " .. vc .. "cl /nologo /LD "
    .. table.concat(objs, " ")
    .. ' /Fe:"' .. dll .. '" ' .. table.concat(lib_args, " "))
  local wrapped = 'cmd /c "' .. out_cmd .. ' 1>"' .. log .. '" 2>&1"'
  if verbose then
    print("paddle.utils.cpp_extension: compiling " .. #sources
          .. " source(s) -> " .. dll)
  end
  local rc = os.execute(wrapped)
  if rc ~= 0 and rc ~= true then
    local tail = read_file(log) or ""
    -- last ~1.5KB of the log is the actionable part
    if #tail > 1500 then tail = tail:sub(-1500) end
    return nil, "compile failed (exit " .. tostring(rc)
      .. "); log tail:\n" .. tail
  end
  if not exists(dll) then
    return nil, "compile produced no " .. dll .. "\n" .. (read_file(log) or "")
  end
  return dll, nil
end

-- ---------------------------------------------------------------------
-- 4. load{} face
-- ---------------------------------------------------------------------

-- core.custom_op_run is resolvable only after paddle is initialized; do
-- it lazily so require "paddle.utils.cpp_extension" never forces the
-- runtime up. NOTE: must be declared BEFORE op_callable below -- a local
-- is only visible from its declaration onward (Lua scoping), and a
-- function defined earlier would see the GLOBAL op_run instead.
local core
local function op_run(op_type, inputs)
  if not core then core = require "paddle_core" end
  return core.custom_op_run(op_type, inputs)
end

-- Build a callable for one registered op. Positional tensors today
-- (custom_relu reads inputs by declaration order); attr-aware signatures
-- land with the per-op signature query (16-cpp-extension.md sec 3.2
-- step 5, next tick). One output -> Tensor; many -> array (matches
-- layer:forward).
local function op_callable(op_type)
  return function(...)
    local n = select("#", ...)
    local inputs = {}
    for i = 1, n do inputs[i] = select(i, ...) end
    local out = op_run(op_type, inputs)
    if #out == 1 then return out[1] end
    return out
  end
end

-- paddle.utils.cpp_extension.load{ name, sources, extra_cxx_cflags,
-- verbose } -> { <op_name> = callable }.
local argrule_mod = require "argrule"

M.load = rule {
  { name = "name", type = argrule_mod.string,
    doc = "shared library base name (no extension)" },
  { name = "sources", type = argrule_mod.List { argrule_mod.string },
    doc = "C++/CUDA source files of the custom op (.cc/.cpp/.cu)" },
  { name = "extra_cxx_cflags",
    type = argrule_mod.List { argrule_mod.string },
    opt = true, default = nil,
    doc = "extra compiler flags appended after the baselines" },
  { name = "verbose", type = argrule_mod.boolean, opt = true, default = false,
    doc = "stream the compiler command and log" },
  call = function(name, sources, extra_cxx_cflags, verbose)
    local dll, err = compile(name, sources, extra_cxx_cflags, verbose)
    if not dll then error("paddle.utils.cpp_extension: " .. err, 3) end
    if not core then core = require "paddle_core" end
    local ops = core.load_custom_op(dll)
    local out = {}
    for op_name in pairs(ops) do
      out[op_name] = op_callable(op_name)
    end
    out._dll = dll
    return out
  end,
}

-- get_include_paths() -> table of include dirs (for a user's own build
-- system, mirrors find_paddle_includes semantic).
function M.get_include_paths()
  local root = paddle_root()
  local build = paddle_build_dir(root)
  local dirs = {}
  for _, d in ipairs(paddle_include_dirs(root, build)) do
    dirs[#dirs + 1] = d
  end
  return dirs
end

-- get_link_flags() -> table of lib paths (host-ready: phi.lib +
-- paddle_core.lib only, see header comment).
function M.get_link_flags()
  local root = paddle_root()
  local build = paddle_build_dir(root)
  local flags = {}
  for _, l in ipairs(paddle_link_libs(root, build)) do
    flags[#flags + 1] = l
  end
  return flags
end

return M