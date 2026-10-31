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
  /* tensor handle -> the pir::Value that produced it (feed or op result). */
  std::unordered_map<pd_tensor, pir::Value> value_of;
  /* (name, tensor) feeds in declaration order; drives fx_names + run's x. */
  std::vector<std::pair<std::string, pd_tensor>> feeds;
  int feed_counter = 0;
  /* fetch list: parallel fo_values / fo_names for run_program_ad_func. */
  std::vector<pir::Value> fo_values;
  std::vector<std::string> fo_names;
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
