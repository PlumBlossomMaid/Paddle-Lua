-- lua/paddle/utils/md5.lua -- pure-Lua MD5 (RFC 1321), internal to paddle.utils
-- Stage: 3.1 download base. Upstream hashlib.md5 equivalent
-- (python/paddle/utils/download.py:224-237). Lua stdlib has no MD5; the
-- download contract (api/utils.md sec 2.1) requires md5 verification,
-- so we carry a portable implementation.
--
-- Bit ops: probe LuaJIT `bit` first, then 5.2+ `bit32` (runtime feature,
-- allowed -- AGENTS.md sec 8.1; syntax stays Lua 5.1 subset, C3). Pure
-- PUC 5.1 with neither library gets a clear error (install LuaJIT or a
-- bit library).

local M = {}

-- ---- bit layer -------------------------------------------------------
local band, bor, bxor, bnot, lshift, rshift, tobit

local ok_bit, bit = pcall(require, "bit")
if ok_bit and type(bit.band) == "function" then
  -- LuaJIT / luabitop: band etc all normalise to UINT32 (0..2^32-1)
  band, bor, bxor = bit.band, bit.bor, bit.bxor
  bnot, lshift, rshift = bit.bnot, bit.lshift, bit.rshift
  tobit = function(x) return x end
else
  local ok32, bit32 = pcall(require, "bit32")
  if not (ok32 and type(bit32.band) == "function") then
    error("paddle.utils.md5: need a bit library (LuaJIT `bit`, 5.2+ "
          .. "`bit32`, or luarocks bitop) to compute MD5", 2)
  end
  band, bor, bxor = bit32.band, bit32.bor, bit32.bxor
  bnot, lshift = bit32.bnot, bit32.lshift
  -- byte extract: bit32.rshift(x, n) keeps x UINT32 already
  rshift = bit32.rshift
  tobit = function(x) return x % 4294967296 end
end

-- ---- core transforms (RFC 1321) --------------------------------------

local F = function(x, y, z) return bor(band(x, y), band(bnot(x), z)) end
local G = function(x, y, z) return bor(band(x, z), band(y, bnot(z))) end
local H = function(x, y, z) return bxor(bxor(x, y), z) end
local I = function(x, y, z) return bxor(y, bor(x, bnot(z))) end

local T = {}
for i = 1, 64 do
  T[i] = tobit(math.floor(math.abs(math.sin(i)) * 4294967296))
end

-- R[i]: (a, b, c, d, x, s, t) -> transform round via table of funcs.
local R = {
  F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F,
  G, G, G, G, G, G, G, G, G, G, G, G, G, G, G, G,
  H, H, H, H, H, H, H, H, H, H, H, H, H, H, H, H,
  I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I,
}

local S = {
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
  5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
}

-- X index per round (RFC 1321 message schedule).
local XIDX = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12,
  5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2,
  0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9,
}

local function transform(state, block)
  -- block: 64-byte string. Returns new state {a,b,c,d}.
  local x = {}
  for j = 0, 15 do
    local o = j * 4
    x[j + 1] = tobit(string.byte(block, o + 1)
                     + string.byte(block, o + 2) * 256
                     + string.byte(block, o + 3) * 65536
                     + string.byte(block, o + 4) * 16777216)
  end
  local a, b, c, d = state[1], state[2], state[3], state[4]
  -- RFC 1321 rounds: update a, then rotate register roles so the next
  -- round updates d (FF(a,b,c,d) -> FF(d,a,b,c) -> FF(c,d,a,b) -> FF(b,c,d,a)).
  -- All additions are mod 2^32: LuaJIT bit.* returns SIGNED int32
  -- (band(0xFFFFFFFF,..) == -1), so any intermediate > 2^31 flips to
  -- negative and the high-bit rotation breaks. Wrap each sum first.
  for i = 1, 64 do
    local f = R[i](b, c, d)
    local sum = (a + f + x[XIDX[i] + 1] + T[i]) % 4294967296
    local rot = bor(lshift(sum, S[i]), rshift(sum, 32 - S[i]))
    a = (b + rot) % 4294967296
    a, b, c, d = d, a, b, c
  end
  return { (state[1] + a) % 4294967296, (state[2] + b) % 4294967296,
           (state[3] + c) % 4294967296, (state[4] + d) % 4294967296 }
end

-- ---- public: hex digest of a string -----------------------------------

function M.sumhexa(s)
  local len = #s
  -- RFC 1321 padding: 0x80, zeros, 8-byte little-endian bit length.
  -- padlen counts ALL padding bytes (0x80 + zeros + 8 length bytes) and
  -- must satisfy (len + padlen) % 64 == 0, with padlen >= 9.
  local r = len % 64
  local padlen = 64 - r
  if padlen < 9 then padlen = padlen + 64 end
  local bits = tobit(len * 8)
  local bitshi = tobit(math.floor(len / 536870912))  -- len*8 / 2^32
  local pad = string.char(0x80) .. string.rep("\0", padlen - 9)
  pad = pad .. string.char(bits % 256, math.floor(bits / 256) % 256,
                           math.floor(bits / 65536) % 256,
                           math.floor(bits / 16777216) % 256,
                           bitshi % 256, math.floor(bitshi / 256) % 256,
                           math.floor(bitshi / 65536) % 256,
                           math.floor(bitshi / 16777216) % 256)
  local msg = s .. pad
  local state = { tobit(0x67452301), tobit(0xefcdab89),
                  tobit(0x98badcfe), tobit(0x10325476) }
  for o = 0, #msg - 1, 64 do
    state = transform(state, msg:sub(o + 1, o + 64))
  end
  local out = {}
  for i = 1, 4 do
    local v = state[i]
    out[#out + 1] = string.format("%02x%02x%02x%02x",
      v % 256, math.floor(v / 256) % 256,
      math.floor(v / 65536) % 256, math.floor(v / 16777216) % 256)
  end
  return table.concat(out)
end

-- ---- public: streamed MD5 of a file (read 4096-byte chunks like
-- python/paddle/utils/download.py:224-237) ------------------------------

function M.sumhexa_file(path)
  local f = io.open(path, "rb")
  if not f then
    error("paddle.utils.md5: cannot open " .. tostring(path), 2)
  end
  local bufs = {}
  while true do
    local chunk = f:read(4096)
    if not chunk then break end
    bufs[#bufs + 1] = chunk
  end
  f:close()
  return M.sumhexa(table.concat(bufs))
end

return M