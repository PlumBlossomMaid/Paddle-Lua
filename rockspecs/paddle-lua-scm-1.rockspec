package = "paddle-lua"
version = "scm-1"

source = {
    url = "git+https://github.com/PlumBlossomMaid/Paddle-Lua.git",
    branch = "main",
}

description = {
    summary = "PaddlePaddle front-end for Lua, no Python runtime",
    detailed = [[
        paddle-lua binds PaddlePaddle's eager C++ API (paddle/phi/api) for
        Lua 5.1-5.4 / LuaJIT through a pure-C ABI middle layer (D2/C7).
        The binding DLL links a local Paddle build (WITH_PYTHON=OFF +
        ON_INFER=OFF); Paddle itself is NOT packaged (04-packaging.md
        sec.1). Lives in the same ecosystem as Insight7 (numpy stand-in),
        Penlight (class/List base), argrule (argument parsing) and Lanes
        (multi-worker DataLoader, D8).
    ]],
    homepage = "https://github.com/PlumBlossomMaid/Paddle-Lua",
    license = "Apache-2.0",
}

-- Hard-required: Lanes (D8), Penlight (R30/D34; pulls luafilesystem
-- transitively), argrule (D31/R27). Insight7 is SOFT-required (D26):
-- only paddle.np / from_insight / vision.transforms load it, lazily,
-- with an actionable error if missing (04-packaging.md sec.3.3).
dependencies = {
    "lua >= 5.1",
    "lanes >= 3.16",
    "penlight >= 1.13, < 2.0",
    "argrule >= 0.1, < 0.2",
}

-- LUA_VERSION tag for CMake (FindLua5x: 51/52/53/54/jit). luarocks is
-- installed per Lua version, so this rockspec is shared; the tag lets
-- the build pick that version's headers/lib. Default 51; the five-version
-- install matrix is exercised at P4 acceptance (04-packaging.md sec.3.4/
-- sec.5), at which point this is parameterized per luaversion install.
-- NOTE: do NOT use $(LUA_VERSION) here -- that built-in is the dotted
-- "5.1" form, while FindLua5x wants the tag form.
rockspec_format = "3.0"

build = {
    type = "cmake",
    variables = {
        LUA_VERSION = "51",
    },
    -- luarocks' cmake driver runs `cmake --build --target install`;
    -- CMakeLists.txt must provide an install() rule (add_packaging, P4).
}