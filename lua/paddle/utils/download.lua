-- lua/paddle/utils/download.lua -- paddle.utils.download
-- Stage: 3.1 download base (WORKPLAN 3.1). Contract: api/utils.md sec 2.1
--   paddle.utils.download.download{ url, path, md5 }
--   -- download to path (cached / MD5-verified); if the file already
--   -- exists and MD5 matches, do NOT re-download.
-- Upstream: python/paddle/utils/download.py (download.py:142-218; stream
-- via httpx, write to "<path>_tmp" then move -- :172-186, retry limit 3
-- -- :24/:194-208, _md5check -- :224-237).
-- Transport: luasocket + luasec (ssl.https), the user-confirmed stack
-- (2026-10-27, twice; plan/modules/12-dataset-vision.md sec 3.1).
-- NOTE: luasec 1.3.2 https.lua rejects proxies ("proxy not supported",
-- https.lua:155-156) -- bcebos (dataset source) is direct-connect.
-- Lua 5.1 subset (C3).

local argrule = require "argrule"
local rule = require "argrule.rule"
local md5 = require "paddle.utils.md5"

local M = {}

-- Soft-dependency probe: luasocket + luasec are runtime rocks. Fail with
-- an actionable message at call time, not at require time.
local SOCKET, HTTPS, LTN12, SOCKET_SLEEP
local function transport()
  if SOCKET == nil then
    local ok_s, s = pcall(require, "socket")
    local ok_h, h = pcall(require, "ssl.https")
    local ok_l, l = pcall(require, "ltn12")
    if not (ok_s and ok_h and ok_l) then
      SOCKET = false
    else
      SOCKET, HTTPS, LTN12 = s, h, l
      SOCKET_SLEEP = s.sleep
    end
  end
  if SOCKET == false then
    error("paddle.utils.download: needs luarocks rocks 'luasocket' and "
          .. "'luasec' (install with: luarocks install luasocket luasec)",
          3)
  end
  return HTTPS, LTN12, SOCKET_SLEEP
end

local RETRY_LIMIT = 3          -- mirrors download.py:24 DOWNLOAD_RETRY_LIMIT

local function sleep(seconds)
  SOCKET_SLEEP(seconds)
end

-- exists() mirrors os.path.exists; keep it local and explicit.
local function exists(p)
  local f = io.open(p, "rb")
  if f then f:close() return true end
  return false
end

-- Create parent directories of path (mkdir -p semantics).
local function ensure_parent(path)
  local dir = path:match("^(.*)[/\\][^/\\]*$")
  if not dir or dir == "" then return end
  -- reuse Penlight (C11: one ecosystem base) if present
  local ok, dirs = pcall(require, "pl.dir")
  if ok then
    dirs.makepath(dir)
    return
  end
  -- fallback: walk up creating each level
  local sep = path:match("^.*\\") and "\\" or "/"
  local acc = ""
  for part in (dir .. sep):gmatch("([^" .. sep .. "]+)" .. sep) do
    acc = acc == "" and part or (acc .. sep .. part)
    local f = io.open(acc, "rb")
    local had = f ~= nil
    if f then f:close() end
    if not had then
      os.execute("mkdir \"" .. acc .. "\"")
    end
  end
end

-- One download attempt. Streams the response body directly to disk
-- (ltn12.sink.file => no full file in memory), then atomically renames
-- "<path>_tmp" -> path (mirrors download.py:172-186).
local function download_once(https, ltn12, url, path)
  local tmp = path .. "_tmp"
  local f, err = io.open(tmp, "wb")
  if not f then
    return nil, "cannot open temp file " .. tmp .. ": " .. tostring(err)
  end
  local res, code, headers, status =
      https.request{ url = url, sink = ltn12.sink.file(f) }
  pcall(f.close, f)   -- ltn12.sink.file closes the handle at stream end;
  if not res or code ~= 200 then
    os.remove(tmp)
    return nil, "HTTP " .. tostring(code or (res and "?" or status
             or "connection failed")) .. " from " .. url
  end
  -- os.rename fails if the destination already exists on Windows
  -- (upstream shutil.move overwrites -- download.py:184); drop it first.
  os.remove(path)
  local ok, rerr = os.rename(tmp, path)
  if not ok then
    os.remove(tmp)
    return nil, "rename " .. tmp .. " -> " .. path .. ": "
             .. tostring(rerr)
  end
  return true
end

-- MD5 gate: nil expected => skip verification (mirrors _md5check
-- download.py:224-226).
local function md5_ok(path, expected)
  if expected == nil then return true end
  local got = md5.sumhexa_file(path)
  if got ~= expected then
    return nil, ("md5 mismatch: calc %s != base %s (file %s)"):format(
      got, expected, path)
  end
  return true
end

-- paddle.utils.download.download{ url, path, md5 }
M.download = rule {
  { name = "url", type = argrule.string,
    doc = "remote URL (http/https)" },
  { name = "path", type = argrule.string,
    doc = "local file path to write" },
  { name = "md5", type = argrule.string, opt = true, default = nil,
    doc = "expected MD5 hex; nil skips verification" },
  call = function(url, path, expected)
    local https, ltn12, _ = transport()

    -- Cache hit: file already there and MD5 matches -> done, mirrors
    -- get_path_from_url check_exist (download.py:59-63).
    if exists(path) then
      local ok, msg = md5_ok(path, expected)
      if ok then return path end
      -- stale/corrupt file: fall through and re-download
    end

    ensure_parent(path)
    for attempt = 1, RETRY_LIMIT do
      local ok, err = download_once(https, ltn12, url, path)
      if ok then
        local mok, mmsg = md5_ok(path, expected)
        if mok then return path end
        -- md5 failed: discard and retry (mirrors download.py:194-207 loop)
        os.remove(path)
        err = mmsg
      end
      if attempt < RETRY_LIMIT then
        sleep(1)                      -- download.py:206 time.sleep(1)
      else
        -- 12-dataset-vision.md sec 5: failure message must include a
        -- manual-download pointer.
        error("paddle.utils.download: failed to fetch " .. url
              .. " after " .. RETRY_LIMIT .. " attempts -- "
              .. tostring(err)
              .. "\n  Manual download: " .. url, 2)
      end
    end
    return path                        -- unreachable, keeps Lua quiet
  end,
}

return M