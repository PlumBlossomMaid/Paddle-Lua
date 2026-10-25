-- tools/ci/regen_diff.lua
-- regen-diff gate (layout.md sec.5 / ci.md sec.6 item 3): the committed
-- generated products (csrc/capi_gen/, lua/paddle/_ops/, csrc/sol/gen_*)
-- must be byte-identical to what the generators produce from the yaml
-- RIGHT NOW. Someone hand-editing generated code, or a generator
-- regression, shows up here as a non-empty `git status --porcelain` on
-- those dirs -- the products are regenerated IN PLACE (overwriting any
-- hand-edit), then compared against the committed versions. Pure Lua
-- 5.1, zero Python (C9 / R42).
--
-- Usage:
--   lua tools/ci/regen_diff.lua [--paddle-root <dir>] [--sha <short-sha>]
-- Exit 0 = products up-to-date. Non-zero = drift; the diffs are printed.
--
-- PADDLE_ROOT: env PADDLE_ROOT wins, else --paddle-root, else the
--   recorded path in docs/migration/process/status.md (AGENTS.md sec.4.3).
-- SHA: default = the sha embedded in csrc/capi_gen/ops_gen.h -- the
--   version the committed products were generated with. This makes the
--   check hermetic: regen with the SAME sha, diff must be empty. Pass
--   --sha to override (e.g. after bumping Paddle, regenerate + commit).
--
-- Input yaml set: ops.yaml + inconsistent/dygraph_ops.yaml (surveyed
--   2026-10-24: add/matmul live in dygraph_ops.yaml, 03-codegen.md sec.2;
--   the 529 forward ops come from exactly these two files).
--
-- Product dirs: csrc/capi_gen (emit_capi), lua/paddle/_ops (emit_lua),
--   csrc/sol/gen_ops_*.cc + gen_ops_all.cc (emit_sol, 2.3.4 closing
--   item 3). csrc/sol's HAND-written files (module.cc, tensor_binding.cc,
--   sol_common.hpp, enums_place.cc, slice.cc) are NOT checked here --
--   only the generator-owned gen_ops_* set.
package.path = "tools/gen/?.lua;" .. package.path

local function slurp(p)
  local f = io.open(p, "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  return s
end

-- sha embedded in the committed ops_gen.h header ("@ <short-sha>")
local function embedded_sha()
  local h = slurp("csrc/capi_gen/ops_gen.h")
  if not h then
    io.stderr:write("regen_diff: csrc/capi_gen/ops_gen.h not found -- "
      .. "products not generated/committed yet?\n")
    os.exit(2)
  end
  local sha = string.match(h, "@ (%x+)")
  if not sha then
    io.stderr:write("regen_diff: cannot find sha in ops_gen.h header\n")
    os.exit(2)
  end
  return sha
end

local paddle_root, sha = nil, nil
local n = select("#", ...)
local i = 1
while i <= n do
  local a = select(i, ...)
  if a == "--paddle-root" then
    i = i + 1
    paddle_root = select(i, ...)
  elseif a == "--sha" then
    i = i + 1
    sha = select(i, ...)
  else
    io.stderr:write("regen_diff: unknown arg '" .. tostring(a) .. "'\n")
    os.exit(2)
  end
  i = i + 1
end
paddle_root = os.getenv("PADDLE_ROOT") or paddle_root
if not paddle_root then
  -- fallback: the path recorded in status.md (AGENTS.md sec.4.3 step 2)
  local st = slurp("docs/migration/process/status.md")
  local m = st and string.match(st, "`%$PADDLE_ROOT`%s*|%s*`?([^`|]+)`?")
  paddle_root = m or "E:/code/paddle"
end
sha = sha or embedded_sha()

local y1 = paddle_root .. "/paddle/phi/ops/yaml/ops.yaml"
local y2 = paddle_root .. "/paddle/phi/ops/yaml/inconsistent/dygraph_ops.yaml"
for _, y in ipairs({ y1, y2 }) do
  if not slurp(y) then
    io.stderr:write("regen_diff: missing yaml " .. y .. " (is PADDLE_ROOT "
      .. "correct / at the recorded version?)\n")
    os.exit(2)
  end
end

-- run both generators IN PLACE (this is the point: hand-edits get
-- overwritten and then compared away)
local function run(cmd)
  local p = io.popen(cmd .. " 2>&1")
  local out = p:read("*a")
  p:close()
  return out
end

local out = run("lua tools/gen/emit_capi.lua csrc/capi_gen " .. y1 .. " "
  .. y2 .. " --sha " .. sha)
if string.find(out, "emit_capi FAIL") then
  io.stderr:write("regen_diff: emit_capi failed:\n" .. out)
  os.exit(1)
end
out = run("lua tools/gen/emit_lua.lua lua/paddle/_ops " .. y1 .. " "
  .. y2 .. " --sha " .. sha)
if string.find(out, "emit_lua FAIL") then
  io.stderr:write("regen_diff: emit_lua failed:\n" .. out)
  os.exit(1)
end
-- emit_sol (2.3.4 closing item 3): csrc/sol/gen_ops_*.cc + gen_ops_all.cc.
-- Full emission (no --ops whitelist): the committed products are the
-- complete 529-op set. csrc/sol's hand-written files are NOT touched by
-- the generator and are not part of this check.
out = run("lua tools/gen/emit_sol.lua csrc/sol " .. y1 .. " "
  .. y2 .. " --sha " .. sha)
if string.find(out, "emit_sol FAIL") then
  io.stderr:write("regen_diff: emit_sol failed:\n" .. out)
  os.exit(1)
end

-- drift = any modified/untracked/deleted file in the product dirs
local st = run("git status --porcelain -- csrc/capi_gen lua/paddle/_ops "
  .. "csrc/sol/gen_ops_*")
if st ~= "" then
  io.stderr:write("regen_diff FAIL: products drifted from committed "
    .. "versions:\n" .. st
    .. "Regenerate and commit the new products (do NOT hand-edit "
    .. "csrc/capi_gen/ lua/paddle/_ops/ or csrc/sol/gen_ops_*).\n")
  os.exit(1)
end
io.stdout:write("regen_diff: OK -- generated products are byte-identical "
  .. "to the committed ones (sha " .. sha .. ")\n")
os.exit(0)
