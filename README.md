[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Lua](https://img.shields.io/badge/Lua-5.1%20|%205.2%20|%205.3%20|%205.4%20|%20JIT-blue.svg)](https://www.lua.org/)
[![Upstream](https://img.shields.io/badge/Paddle-3.5.0%20unmodified-green.svg)](https://github.com/PaddlePaddle/Paddle)
[![Status](https://img.shields.io/badge/status-design%20phase-orange.svg)](docs/migration/README.md)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Paddle-Lua

**A Lua frontend for PaddlePaddle. `local paddle = require "paddle"` — no Python runtime required.**

> ⚠️ **Design phase. No code yet.**
> This repository currently holds a complete technical plan.
> See **[docs/migration/](docs/migration/README.md)** — 13 documents, ~3000 lines.

---

## Why this is possible

PaddlePaddle's training engine is **pure C++**. Automatic differentiation (`egr::Backward`),
the tensor type (`paddle::Tensor`) and every kernel (phi) have no dependency on Python
whatsoever. Python is only a *shell*: argument parsing, `nn.Layer` bookkeeping, the optimizer
loop, the data pipeline.

So this project does not reimplement a deep learning framework. It replaces the shell.

```
             Python shell                    Lua shell        ← we write this
  ─────────────────────────────   ─────────────────────────
   pybind11 bindings               sol2 + a plain C ABI       ← we write this
  ═══════════════════════════════════════════════════════════
   generated forward API  ·  eager autograd  ·  phi kernels   ← unchanged, pure C++
```

**Zero upstream modification.** Paddle is consumed as-is.

---

## What it looks like

```lua
local paddle = require "paddle"
local class  = require "pl.class"
local nn, F  = paddle.nn, paddle.nn.functional

local Net = class(nn.Layer)
function Net:_init()
  self:super()
  self.fc = nn.Linear(784, 10)     -- registered automatically, as in Python
end
function Net:forward(x) return self.fc(x) end

local net = Net()
local opt = paddle.optimizer.Adam{ learning_rate = 1e-3, parameters = net:parameters() }

for _, batch in paddle.io.DataLoader{ ds, batch_size = 64, num_workers = 4 } do
  local loss = F.cross_entropy(net(batch[1]), batch[2])
  loss:backward()
  opt:step()
  opt:clear_grad()
end
```

Indexing is **1-based throughout** — including `axis` arguments. Slicing borrows
Python's syntax through a string, a technique proven in
[Insight7](https://github.com/PlumBlossomMaid):

```lua
x["1:3, :"]     -- rows 1 and 2, all columns
x["::-1"]       -- reversed
x[{1, 2}]       -- element access
```

---

## Design at a glance

| | |
|---|---|
| Host languages | Lua 5.1 / 5.2 / 5.3 / 5.4 / LuaJIT |
| Python runtime | **not required** |
| Binding layer | sol2 on top of a plain C ABI shim |
| Upstream changes | **zero** |
| Library source dialect | Lua 5.1 syntax only (so one source tree serves all five VMs) |
| Multi-worker data loading | [Lua Lanes](https://github.com/LuaLanes/lanes), a hard dependency — threads, not processes |
| Indexing | 1-based everywhere |
| Op bindings | generated from Paddle's own `ops.yaml` / `backward.yaml`, not hand-written |
| Classes and collections | [Penlight](https://github.com/lunarmodules/Penlight) — `pl.class`, `pl.List`. Vendored, pure Lua |
| The numpy slot | [Insight7](https://github.com/PlumBlossomMaid) — arrays without autograd, for data prep |

### Three decisions worth explaining

**sol2 for LuaJIT too, not FFI.** Binding overhead is 50–200 ns; a kernel launch is
1–10 µs. Maintaining two binding paths to save 1–2% is not worth it.

**Threads, not processes, for `num_workers`.** Python's DataLoader forks because of the
GIL. Lua has no GIL — separate `lua_State`s run genuinely in parallel in one process,
which means workers can hand tensors to the main thread by **pointer**, with no
serialization at all. `paddle::Tensor` already refcounts its buffer atomically through a
`shared_ptr`, so crossing a thread boundary costs one increment.

**Penlight is a first-class citizen, not a nod to the ecosystem.** Lua ships almost no
standard library: no classes, no list type, no cross-version compatibility layer. Rather
than invent our own — which only we would know how to debug — `nn.Layer` *is* a
`pl.class`, `parameters()` *returns* a `pl.List`, and version shims come from
`pl.compat`. Whatever you already know about Penlight applies here directly.

**Reuse Paddle's C++ executor for static graphs.** `StandaloneExecutor` /
`InterpreterCore` already provide the IR, scheduling, garbage collection, passes and
serialization; `run_program_ad_func` already makes the traced graph participate in eager
autograd. The Lua side only needs a tracer.

---

## The one risk that decides everything

`cmake/configure.cmake:15` guards the no-Python path with `(NOT WITH_PYTHON) AND ON_INFER`.
Upstream's expected Python-free scenario is **inference**. The combination we need —
no Python, *with* training — may never have been compiled by anyone.

So the very first task is not writing code. It is:

```bash
cmake .. -DWITH_PYTHON=OFF -DON_INFER=OFF -DWITH_GPU=ON
```

If that cannot produce a `libpaddle` whose `egr::Backward` actually runs, the entire
project needs re-estimating. Everything else in the plan is downstream of this.

---

## Scope

**In:** `paddle.Tensor`, `paddle.nn`, `paddle.optimizer`, `paddle.io`, `paddle.vision`,
`paddle.jit` (trace first, script later), `paddle.dataset`, `paddle.static`,
`paddle.distributed` (written but untestable for now).

**Out:** `paddle.Model` and `paddle.metric.*` — deliberately. They are superseded by
[PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean) (Lightning + Keras style
trainer) and [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics)
(torchmetrics style), whose designs we follow instead.

Also out: `paddle.incubate`, `paddle.utils.unique_name` internals, `paddle.fluid`, and
anything whose only purpose is to serve Python's packaging ecosystem.

---

## Roadmap

| | Milestone | Deliverable |
|---|---|---|
| **M0** | Feasibility gate | Python-free build; backward pass verified |
| **M1** | MVP | MNIST trains end-to-end; `jit::Layer` inference of Python-trained models |
| **M2** | Usable | `paddle.vision`, DataLoader with Lanes workers, full GC machinery |
| **M3** | Complete | trace-mode dynamic-to-static; `metrics-lua` |
| **M4** | Beyond | script-mode with control flow (vendored luacheck parser) |

Rough estimate: MVP 4–6 person-months.

---

## Related projects

```
paddle-lua        C++ bindings + core Lua layer     ← this repo
  ├─ metrics-lua     pure Lua, torchmetrics style
  └─ ocean-lua       pure Lua, Lightning/Keras style
```

| | |
|---|---|
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | Upstream. Used as-is, **never modified** |
| [PaddleOcean](https://github.com/PlumBlossomMaid/PaddleOcean) | Trainer design we follow instead of `paddle.Model` |
| [PaddleMetrics](https://github.com/PlumBlossomMaid/PaddleMetrics) | Metrics design we follow instead of `paddle.metric.*` |
| [Lua Lanes](https://github.com/LuaLanes/lanes) | Multithreading, required for `num_workers > 0` |
| [sol2](https://github.com/ThePhD/sol2) | C++/Lua binding library |
| [Penlight](https://github.com/lunarmodules/Penlight) | Classes, collections, compatibility layer. Vendored subset, MIT |
| [Insight7](https://github.com/PlumBlossomMaid) | The numpy of this stack. Also the origin of our string slicing and keyword-argument conventions |

---

## Documentation

The full technical plan lives in **[`docs/`](docs/migration/README.md)**.

---

## License

Apache 2.0, matching PaddlePaddle.
