-- check_doctree.lua
-- 校验 docs/migration/WORKPLAN.md 工程树的链接完整性(plan/ci.md §5)
-- 用法: lua tools/ci/check_doctree.lua [workplan 路径]
-- 规则:
--   ✅/🔵 节点:指针目标文件必须存在(相对 docs/migration/,含花括号展开)
--   ⬜ /⛔ 节点:不校验存在性(目标是"待创建的产出")
--   指针支持 `+` 多目标与 {a,b}.md 花括号展开;`->` 后是动作描述的行跳过
-- 退出码: 0=通过, 1=有断链
--
-- ⚠️ 已知边界:ci.md §5 的「叶节点必须有 完工判据/前置 小节」检查未实现 ——
-- 模块文档(plan/modules/*.md)骨架是六节(modules/README.md),无这两个字段名,
-- 一刀切检查会从第一天全红。等模块文档骨架统一后再补。

local path = arg[1] or "docs/migration/WORKPLAN.md"
local f, err = io.open(path, "r")
if not f then
  io.stderr:write("cannot open " .. path .. ": " .. tostring(err) .. "\n")
  os.exit(1)
end
local content = f:read("*a")
f:close()

local tree_header = content:find("## 4%. 工程树")
if not tree_header then
  io.stderr:write("no '## 4. 工程树' section\n")
  os.exit(1)
end
local fence_open = content:find("```", tree_header)
if not fence_open then
  io.stderr:write("no tree fence\n")
  os.exit(1)
end
local fence_close = content:find("```", fence_open + 3)
if not fence_close then
  io.stderr:write("tree fence not closed\n")
  os.exit(1)
end
local block = content:sub(fence_open + 3, fence_close - 1)

-- 展开 {a,b}.md 花括号 -> a.md b.md
local function expand_braces_all(s)
  local results = { s }
  while true do
    local changed = false
    local next_results = {}
    for _, item in ipairs(results) do
      local start_pos = item:find("{", 1, true)
      if start_pos then
        local close_pos = item:find("}", start_pos + 1, true)
        if close_pos then
          changed = true
          local prefix = item:sub(1, start_pos - 1)
          local inner = item:sub(start_pos + 1, close_pos - 1)
          local suffix = item:sub(close_pos + 1)
          for part in inner:gmatch("[^,]+") do
            next_results[#next_results + 1] = prefix .. part .. suffix
          end
        else
          next_results[#next_results + 1] = item
        end
      else
        next_results[#next_results + 1] = item
      end
    end
    results = next_results
    if not changed then break end
  end
  return results
end

-- 从指针文本提取所有 .md 路径(自动截断行尾注释)
local function md_paths(ptr)
  local out = {}
  for _, expanded in ipairs(expand_braces_all(ptr)) do
    for p in expanded:gmatch("[%w_./%-]+%.md") do
      out[#out + 1] = p
    end
  end
  return out
end

local STATUS = { ["✅"] = true, ["🔵"] = true, ["⬜"] = true, ["⛔"] = true }
local function node_status(line)
  for st in pairs(STATUS) do
    if line:find("[" .. st .. "]", 1, true) then return st end
  end
  return nil
end

local errors = {}
local checked, skipped = 0, 0
local lineno = 0
for line in block:gmatch("[^\n]+") do
  lineno = lineno + 1
  local st = node_status(line)
  local arrow = line:find("->", 1, true)
  if st and arrow then
    local ptr = line:sub(arrow + 2)
    local paths = md_paths(ptr)
    if #paths == 0 then
      skipped = skipped + 1 -- 指针指向动作描述(如 1.7.0 的"去 luarocks 占位"),跳过
    else
      for _, p in ipairs(paths) do
        if st == "✅" or st == "🔵" then
          checked = checked + 1
          local full = p
          local prefix = "docs/migration/"
          if p:sub(1, #prefix) ~= prefix then full = prefix .. p end
          local fp = io.open(full, "r")
          if not fp then
            errors[#errors + 1] = string.format("行 %d [%s] 断链 -> %s", lineno, st, full)
          else
            fp:close()
          end
        else
          skipped = skipped + 1
        end
      end
    end
  end
end

if #errors > 0 then
  for _, e in ipairs(errors) do io.stderr:write(e, "\n") end
  io.stderr:write(string.format("doctree FAIL: %d 处断链 (✅/🔵 checked=%d)\n", #errors, checked))
  os.exit(1)
end
io.write(string.format("doctree OK: %d 个 ✅/🔵 指针全部存在, %d 个 ⬜/跳过\n", checked, skipped))
