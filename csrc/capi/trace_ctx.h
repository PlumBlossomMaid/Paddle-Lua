/* trace_ctx.h -- internal C++ surface of the static-graph trace recorder
 * (P15-A, eager-record). NOT part of the C ABI: it returns pir::Value /
 * pir::Builder* and is included by trace.cc and (later) by the generated
 * op wrappers' emitted trace branch. Never include it from paddle_c.h.
 *
 * Design (15-to-static.md sec.2.2/3.1, eager-record decision):
 *   While a trace is active, the generated pd_op_* wrapper executes the
 *   op eagerly AND, in the same call, builds the matching pir op via the
 *   typed <Op>Op::Build wrapper. The typed Build owns per-op structure
 *   (scalar attrs become FullOp inputs, pd_op4.cc:4360; output types come
 *   from <Op>Op::InferMeta, so we never hand-compute shapes). The helpers
 *   below only manage the tensor-handle <-> pir::Value bookkeeping, the
 *   feed/DataOp emission, and the run; they do NOT assemble ops by name
 *   with a bare Operation::Create (that mis-shapes ops like scale).
 */
#ifndef PADDLE_LUA_TRACE_CTX_H
#define PADDLE_LUA_TRACE_CTX_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

#include "capi_impl.h"  // pd_tensor (opaque), pd_tensor_t

/* One in-flight trace. Created by pd_trace_begin, filled by the emitted
 * trace branches, consumed by pd_trace_run, released by pd_trace_free. */
struct PdTraceCtx {
  pir::IrContext* ir = nullptr;
  std::shared_ptr<pir::Program> program;
  std::unique_ptr<pir::Builder> builder;
  /* tensor -> the pir::Value that produced it (feed or op result).
   * KEYED BY IMPL POINTER, not the handle: sol passes LuaTensor BY VALUE
   * into op lambdas, and the LuaTensor copy ctor retains a NEW handle
   * shell sharing the same impl (sol_common.hpp); handle addresses also
   * get malloc-reused after Lua GC frees a temporary. Both break a
   * handle-keyed map (miss -> duplicate feed, or stale-hit -> wrong
   * graph, measured as nondeterministic wrong outputs on repeated
   * traces). The impl (shared_ptr target of paddle::Tensor) is the one
   * stable identity; feed/op retain-holders below keep it alive. */
  std::unordered_map<const void*, pir::Value> value_of;
  /* (name, tensor) feeds in declaration order; drives fx_names + run's x.
   * feeds[0 .. n_declared-1] are the pd_trace_input-declared inputs;
   * the rest are auto-declared weights/constants (pd_trace_value_or_feed)
   * that always keep their captured tensors on re-run. */
  std::vector<std::pair<std::string, pd_tensor>> feeds;
  size_t n_declared = 0;
  int feed_counter = 0;
  /* fetch list: parallel fo_values / fo_names for run_program_ad_func. */
  std::vector<pir::Value> fo_values;
  std::vector<std::string> fo_names;
  /* The run scope, created lazily and STABLE across pd_trace_run2 calls.
   * run_program_impl's InterpreterCoreInfoCache keys the interpreter by
   * (program_id, scope ptr, place...): a fresh scope per call frees the
   * previous one, malloc reuses the address, the cache then serves a
   * stale interpreter bound to a FREED scope -- garbage tensor shapes
   * (nondeterministic matmul shape scramble). Holding it here keeps the
   * cached interpreter + its re-shared feeds consistent. */
  std::shared_ptr<paddle::framework::Scope> run_scope;
  /* Every pd_trace_run2 replaces run_scope with a FRESH Scope and keeps
   * the previous ones alive here. Two reasons, both measured:
   *   1. InterpreterCoreInfoCache keys on (program_id, scope ptr, ...):
   *      a freed scope's address gets malloc-reused, the cache then
   *      serves an interpreter bound to a FREED scope (garbage tensor
   *      shapes -- the nondeterministic matmul scramble).
   *   2. A REUSED scope makes the interpreter's GC empty the input vars
   *      after run 1 (holders released, scope["x1"] becomes a shells);
   *      run 2 then re-feeds the SAME var object and the cached
   *      interpreter keeps reading the emptied/stale binding. A fresh
   *      scope forces a fresh interpreter that re-reads all feeds. */
  std::vector<std::shared_ptr<paddle::framework::Scope>> retired_scopes;
};

/* True while a trace is active on this thread. The generated wrappers
 * gate their trace branch on this. */
bool            pd_trace_active();

/* IrContext of the active trace (for Attribute::get calls). */
pir::IrContext* pd_trace_ir();

/* Builder bound to the active trace's program block (insertion at end). */
pir::Builder*   pd_trace_builder();

/* Map an input tensor handle to its pir::Value. An unseen tensor (a
 * weight/constant met during eager trace, not produced by a traced op)
 * becomes an auto-declared pd_op.data feed named "feed_<k>". */
pir::Value      pd_trace_value_or_feed(pd_tensor tensor);

/* Bind an eager output handle to the recorded op result Value, so the
 * next op consuming that handle finds its producer. */
void            pd_trace_map_output(pd_tensor tensor, pir::Value value);

#endif  /* PADDLE_LUA_TRACE_CTX_H */
