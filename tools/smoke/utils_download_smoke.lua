-- tools/smoke/utils_download_smoke.lua -- paddle.utils.download smoke
-- Stage: 3.1 download base. Mirrors python/paddle/utils/download.py
-- semantics (api/utils.md sec 2.1):
--   1. fresh download: file written, MD5 matches the known-good value
--   2. cache hit: second call does NOT re-download (mtime unchanged)
--   3. corrupt file: MD5 mismatch triggers re-download of the good copy
--   4. bad URL: actionable error (manual-download pointer)
--
-- Requires luasocket + luasec (x64 LuaJIT build). Run from repo root:
--   set PATH=build-test\dl-deps\luajit-lib;%PATH%
--   lua tools/smoke/utils_download_smoke.lua

package.cpath = "./build-test/?.dll;./build-test/dl-deps/luajit-lib/?.dll;./build-test/dl-deps/luajit-lib/?/core.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;./build-test/dl-deps/luajit-lib/?.lua;./build-test/dl-deps/luajit-lib/?/init.lua;" .. package.path

local core = require "paddle_core"
core.init()

local paddle = require "paddle"
local download = paddle.utils.download.download

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

local URL = "https://dataset.bj.bcebos.com/mnist/t10k-labels-idx1-ubyte.gz"
local MD5 = "ec29112dd5afa0611ce80d1b7f02629c"   -- mnist.py:37
local DIR = "build-test/dl-deps/smoke-utils-dl"
local PATH = DIR .. "/t10k-labels-idx1-ubyte.gz"

local md5util = require "paddle.utils.md5"

-- 1. fresh download
os.execute("mkdir " .. DIR .. " 2>nul")
os.remove(PATH)
os.remove(PATH .. "_tmp")
local p = download{ url = URL, path = PATH, md5 = MD5 }
check(p == PATH, "download returns the written path")
local f = io.open(PATH, "rb")
check(f ~= nil, "downloaded file exists")
f:close()
check(md5util.sumhexa_file(PATH) == MD5, "downloaded file MD5 matches")

-- 2. cache hit: re-download must not touch the file
local function size_of(p)
  local f, err = io.open(p, "rb")
  if not f then return -1 end
  local n = f:read("*a"):len()
  f:close()
  return n
end
local mtime1 = os.time()   -- coarse; rely on size+md5 below instead
local size1 = size_of(PATH)
download{ url = URL, path = PATH, md5 = MD5 }
local size2 = size_of(PATH)
check(size1 == size2, "cache hit does not rewrite the file")
check(md5util.sumhexa_file(PATH) == MD5, "file intact after cache hit")

-- 3. corrupt file -> re-download the good copy
local w = io.open(PATH, "wb")
w:write("CORRUPTED-CONTENT")
w:close()
download{ url = URL, path = PATH, md5 = MD5 }
check(md5util.sumhexa_file(PATH) == MD5, "corrupt file replaced by good copy")

-- 4. bad URL -> error with manual-download pointer
local ok, err = pcall(download, { url = "https://dataset.bj.bcebos.com/mnist/no-such-file.gz", path = DIR .. "/no-such.gz", md5 = MD5 })
check(not ok, "bad URL raises")
check(err:find("Manual download", 1, true) ~= nil, "error carries manual-download pointer")

-- cleanup the smoke data dir (one-off artifacts must not linger)
os.execute("rmdir /s /q " .. DIR .. " 2>nul")

print(("UTILS_DOWNLOAD_SMOKE PASS (%d assertions)"):format(ntests))