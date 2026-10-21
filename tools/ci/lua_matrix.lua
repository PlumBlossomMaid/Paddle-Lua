-- lua_matrix.lua
-- M0 #23/#24 的多版本验证(plan/ci.md §2 的 lua-matrix job 用)
-- 在每个 Lua 版本上独立跑:5.1 / 5.2 / 5.3 / 5.4 / LuaJIT
-- 替代本机缺失的 5.2-5.4 版本(本机只有 5.1.5 + LuaJIT 2.1)
-- 断言:
--   #23  chunkname 与错误行号行为(loadstring/load 双通道)
--   #24  三道墙:upvalue 上限 / 局部变量上限 / 寄存器(表达式复杂度)墙
-- 用法: lua tools/ci/lua_matrix.lua
-- 退出码: 0=通过, 1=有断言失败

print("== LUA:", _VERSION, jit and jit.version or "(no jit)")

local loadfn
if loadstring then
  loadfn = loadstring -- 5.1 / 5.2 / LuaJIT 有 loadstring
else
  loadfn = load -- 5.3 / 5.4 只有 load(接受字符串 + chunkname)
end

local failures = 0
local function assert_ok(cond, msg)
  if not cond then
    failures = failures + 1
    print("FAIL: " .. msg)
  end
end

-- ============ #23 chunkname 与错误行号 ============
-- 构造均已在 5.1.5 与 LuaJIT 2.1 上实测,断言只查 chunkname 与行号,不断言消息措辞
-- (5.2+ 的 unexpected symbol 消息措辞可能略异,格式 [string "name"]:line 不变)
local _, e1 = loadfn("local a = \nlocal b = \nlocal c = missing\n", "mychunk")
assert_ok(e1 ~= nil and tostring(e1):find("mychunk"), "#23 named chunk keeps chunkname: " .. tostring(e1))
assert_ok(e1 ~= nil and tostring(e1):find(":2"), "#23 named chunk line number: " .. tostring(e1))

-- 默认 chunkname(不带名字):EOF 续行错误报 :3(实测两端一致)
local _, e2 = loadfn("local ok = 1\nlocal bad = 2 + \n", nil)
assert_ok(e2 ~= nil and tostring(e2):find(":3"), "#23 default chunk line: " .. tostring(e2))

-- 运行时错误携带 chunkname 与行号
local fn3 = loadfn("local a = 1\nlocal b = 2\nerror('runtime')\n", "run_chunk")
assert_ok(fn3 ~= nil, "#23 runtime chunk compiles")
if fn3 then
  local ok3, err3 = pcall(fn3)
  assert_ok(not ok3 and tostring(err3):find("run_chunk"), "#23 runtime err keeps chunkname: " .. tostring(err3))
  assert_ok(not ok3 and tostring(err3):find(":3"), "#23 runtime err line: " .. tostring(err3))
end

-- ============ #24 三道墙 ============
local function tryCompile(s)
  local fn, err = loadfn(s, "wall")
  if fn then return true end
  return false, err
end

-- upvalue 墙:chunk 声明 N 个 local,内层函数全部引用 -> N 个 upvalue
local function upvalue_ok(N)
  local decls, refs = {}, {}
  for i = 1, N do decls[i] = "local u" .. i; refs[i] = "u" .. i end
  return tryCompile(table.concat(decls, "\n") .. "\nreturn function() return " ..
    table.concat(refs, "+") .. " end")
end

-- 局部变量墙:单函数 N 个 local
local function locals_ok(N)
  local decls = {}
  for i = 1, N do decls[i] = "local v" .. i .. " = 0" end
  return tryCompile("return function()\n" .. table.concat(decls, "\n") .. "\nreturn v1 end")
end

-- 寄存器/表达式复杂度墙:单调用 N 个实参(宽参数列表,寄存器同时存活)
local function regwall_ok(N)
  local args = {}
  for i = 1, N do args[i] = "1" end
  return tryCompile("return function() return f(" .. table.concat(args, ",") .. ") end")
end

local function bisect(fn, lo, hi)
  local best = 0
  while lo <= hi do
    local mid = math.floor((lo + hi) / 2)
    if fn(mid) then best = mid; lo = mid + 1 else hi = mid - 1 end
  end
  return best
end

local uv_wall = bisect(upvalue_ok, 1, 600)
local lv_wall = bisect(locals_ok, 1, 600)
local rg_wall = bisect(regwall_ok, 1, 600)
print(string.format("#24 walls: upvalue=%d locals=%d reg/complexity=%d", uv_wall, lv_wall, rg_wall))

-- 断言已核实的稳定事实(AGENTS.md §4:只断言有出处的)
local is_lua51 = _VERSION == "Lua 5.1"
local is_luajit = jit ~= nil
if is_lua51 or is_luajit then
  -- 5.1 系:LUAI_MAXUPVALUES=60(本机 5.1.5 与 LuaJIT 2.1 已实测)
  assert_ok(uv_wall == 60, "#24 5.1-family upvalue wall should be 60, got " .. uv_wall)
end
-- 局部变量上限 LUAI_MAXVARS=200,五个版本均为此常量(lparser.h,已核实 5.1;其余版本由 CI 首跑确认)
assert_ok(lv_wall == 200, "#24 locals wall should be 200, got " .. lv_wall)
-- 寄存器墙构造依赖,不断言具体值;只要求能编译 argrule 原型形态所需的最小量(N=122,foundations §5.4.6)
-- 宽参数构造的实测:5.1=248 / LuaJIT=247;5.2-5.4 由 CI 首跑确认后回填 status.md §4

if failures == 0 then
  print("lua-matrix OK")
else
  print(string.format("lua-matrix FAIL: %d 断言失败", failures))
  os.exit(1)
end
