-- tools/ci/lua_close_check.lua
-- M0 #9: verify __close metamethod behavior on userdata in Lua 5.4.
--
-- Background: Lua 5.4 added <close> variables and the __close metamethod.
-- sol2 maps C++ RAII types onto userdata; *when* __close fires (scope exit
-- vs GC) decides how we wire resource release in the 5.4 bindings.
--
-- Verified semantics (running against Lua 5.4.7, see m0-report §13):
--   1. A <close> variable fires __close once on scope exit — even for plain
--      tables (manual §3.3.8: __close applies to any object with the
--      metamethod; the to-be-closed mechanism is what invokes it).
--   2. The garbage collector runs __gc, NOT __close, on a userdata that was
--      never declared <close>. There is no GC fallback for __close.
--   3. Consequence for our bindings: a C++ RAII usertype (e.g. Tensor holding
--      a device allocation) must expose BOTH __close (deterministic release
--      via <close> variables on 5.4) and __gc (last-resort release when the
--      user drops the last reference without <close>). Both handlers must be
--      idempotent — a shared_ptr drop is naturally idempotent; a manual
--      release flag must be checked in both.
--
-- Runs on Lua 5.1..5.4/LuaJIT: on <5.4 there is no <close> syntax (Case A
-- is skipped); Case B re-verifies the same GC fact across all versions.

local failures = 0
local function assert_ok(cond, msg)
  if not cond then
    failures = failures + 1
    print("FAIL: " .. msg)
  end
end

local has_close = string.find((loadstring or load)("return 1", "check") and _VERSION or "x", "5.4", 1, true) ~= nil

-- Detect syntax support for <close> by trying to compile a chunk.
local close_supported = false
local ok = pcall(load, "local x <close> = 0; return x")
close_supported = ok

print("== LUA: " .. _VERSION .. "  <close> supported: " .. tostring(close_supported))

-- Case A: <close> fires on scope exit (5.4 only)
if close_supported then
  local closed = 0
  local mt = { __close = function() closed = closed + 1 end }
  local chunk = [[
    local closed = 0
    local mt = { __close = function() closed = closed + 1 end }
    do
      local u <close> = setmetatable({}, mt)
      -- scope exit here must call __close
    end
    return closed
  ]]
  local f, err = load(chunk, "close_scope_test")
  assert_ok(f ~= nil, "failed to load scope test: " .. tostring(err))
  if f then
    local n = f()
    assert_ok(n == 1, "expected __close once on scope exit, got " .. tostring(n))
  end
end

-- Case B: GC runs __gc, NOT __close, on a userdata never declared <close>.
--
-- Lua 5.4 manual §3.3.8: __close belongs to the to-be-closed mechanism
-- (<close> variables). A userdata that reaches the collector through an
-- ordinary reference is finalized via __gc (manual §2.5.4); there is no GC
-- fallback that calls __close. This is exactly the fact our bindings must
-- rely on: a C++ RAII usertype needs BOTH __close (deterministic release
-- via <close>) and __gc (last-resort release), not just __close.
--
-- Pure Lua cannot allocate a fresh userdata (newproxy was removed in 5.4),
-- so we borrow a real one — the io.tmpfile() handle — and replace its
-- metatable with one carrying BOTH __gc and __close. Whichever fires tells
-- us which path the collector actually took.
local tmp = assert(io.tmpfile())
local closed_by = nil
debug.setmetatable(tmp, {
  __gc    = function() closed_by = "gc" end,
  __close = function() closed_by = "close" end,
})
tmp = nil
collectgarbage("collect")
collectgarbage("collect")
-- On every supported version the collector runs __gc; __close must NOT fire
-- on the plain-GC path (it is only invoked at <close> scope exit, Case A).
-- If a future Lua changed that, RAII usertypes would release twice — pin it.
assert_ok(closed_by == "gc",
          "expected __gc only at GC (no __close), got " .. tostring(closed_by))

if failures == 0 then
  print("lua-close OK: __close semantics verified on " .. _VERSION)
else
  print("lua-close FAIL: " .. failures .. " assertion(s) failed")
  os.exit(1)
end