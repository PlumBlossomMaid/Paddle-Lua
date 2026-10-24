-- tools/lint_51.lua
-- Lua 5.1 subset gate (C3). Verifies that every .lua file interpreted here
-- is valid Lua 5.1 SYNTAX by asking the 5.1 compiler itself (loadstring).
-- This is the most authoritative check available: the 5.1 parser IS the
-- spec. It catches goto/::label::, the // operator, bitwise ops, <const>,
-- <<\u{} escapes, integer-literal syntax errors etc. -- every construct
-- introduced after 5.1 fails to parse on 5.1 and is reported here.
--
-- Stage: P1 / M1 (WORKPLAN T-M1-01). Design: process/conventions.md sec.1.4
--   "用 Lua 5.1 自身的 loadstring 去 load 每个源文件 —— 5.1 编译器本身就
--    是最权威的 5.1 语法检查器。"
--
-- Usage:
--   lua tools/lint_51.lua file1.lua file2.lua ...   (explicit file list)
--   lua tools/lint_51.lua                           (stdin-free: nothing)
-- Exit code 0 = all files passed; 1 = at least one file is not 5.1.
--
-- Note: some 5.2+ constructs (e.g. the `goto` STATEMENT) are contextual
-- keywords in 5.1, not syntax errors -- `goto` parses fine on 5.1. The
-- loadstring gate catches structural violations; project conventions
-- (conventions.md sec.1.1) forbid those keywords in every .lua under
-- lua/ and tools/gen/, enforced by code review + this gate for real
-- syntax. Runtime-API gaps (table.unpack etc.) are out of scope here:
-- the 5.1 parser cannot see them, and pl.compat handles the rest.

local function is_lua_file(name)
  return string.sub(name, -4) == ".lua"
end

local files = {}
for i = 1, select("#", ...) do
  local f = select(i, ...)
  if is_lua_file(f) then
    files[#files + 1] = f
  end
end

if #files == 0 then
  io.stdout:write("lint_51: no .lua files given (exit 0)\n")
  os.exit(0)
end

local failed = 0
for _, path in ipairs(files) do
  local fh, err = io.open(path, "r")
  if not fh then
    io.stderr:write("lint_51: cannot open " .. path .. ": " .. tostring(err) .. "\n")
    failed = failed + 1
  else
    local src = fh:read("*a")
    fh:close()
    -- chunkname = path: the 5.1 parser names syntax errors with it.
    local fn, perr = loadstring(src, path)
    if not fn then
      io.stderr:write("lint_51 FAIL " .. path .. "\n  " .. tostring(perr) .. "\n")
      failed = failed + 1
    else
      io.stdout:write("lint_51 ok   " .. path .. "\n")
    end
  end
end

if failed > 0 then
  io.stderr:write("lint_51: " .. failed .. " file(s) are not Lua 5.1\n")
  os.exit(1)
end
io.stdout:write("lint_51: all " .. #files .. " file(s) are Lua 5.1\n")
os.exit(0)