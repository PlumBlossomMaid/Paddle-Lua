// trace.cc -- static-graph trace recorder (P15-A, eager-record).
//
// Implements the pd_trace_* C ABI (paddle_c.h Group 12) plus the internal
// helpers the generated op wrappers' trace branch will use (trace_ctx.h).
//
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, HEAD 2bfeb115b0):
//   paddle/pir/include/core/program.h:44/66        Program(IrContext*) / block()
//   paddle/pir/include/core/builder.h:70-71/139-141 Builder(ctx,block) / Build
//   paddle/pir/include/core/ir_context.h:124        GetRegisteredOpInfo(name)
//   paddle/fluid/pir/dialect/operator/ir/op_dialect.h:31
//                                                  OperatorDialect (pd_op)
//   paddle/fluid/pybind/pir.cc:2606-2610           GetOrRegisterDialect
//   paddle/fluid/pybind/pir.cc:2247-2265           AppendShadowOutput shape
//                                                  (builtin.shadow_output via
//                                                  output_name attr)
//   paddle/fluid/pybind/pir.cc:2621 (docstring)    pd_op.data attr set:
//                                                  name/shape/dtype/place
//   paddle/fluid/pir/dialect/operator/utils/utils.h:75
//                                                  TransToIrDataType
//   paddle/pir/include/core/builtin_type.h:78-86   DenseTensorType::get
//   paddle/fluid/eager/to_static/run_program_func.h:23-28
//                                                  run_program_ad_func
//   paddle/fluid/eager/to_static/run_program_impl.cc:448-500
//                                                  prog_attrs key set
//   python/paddle/jit/dy2static/partial_program.py:427
//                                                  empty backward for eval

#include "trace_ctx.h"

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/to_static/run_program_func.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/op_info.h"

/* The opaque handle's real shape (pd_trace in paddle_c.h). */
struct pd_trace_t {
  PdTraceCtx ctx;
};

/* One active trace per thread (P15-A traces on the main lane only). */
static thread_local PdTraceCtx* g_trace = nullptr;

bool pd_trace_active() { return g_trace != nullptr; }
pir::IrContext* pd_trace_ir() { return g_trace->ir; }
pir::Builder* pd_trace_builder() { return g_trace->builder.get(); }

namespace {

/* DDim -> std::vector<int64_t> (for the pd_op.data shape attr). */
std::vector<int64_t> dims_to_vec(const phi::DDim& d) {
  std::vector<int64_t> v(static_cast<size_t>(d.size()));
  for (int i = 0; i < d.size(); ++i) v[static_cast<size_t>(i)] = d[i];
  return v;
}

/* DenseTensorType for a real eager tensor (feed/output typing). */
pir::Type dense_type_of(PdTraceCtx& c, pd_tensor h) {
  const paddle::Tensor& t = reinterpret_cast<const pd_tensor_t*>(h)->value;
  auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());
  PADDLE_ENFORCE_NOT_NULL(
      dense.get(),
      common::errors::InvalidArgument(
          "pd_trace: tensor has no DenseTensor impl (trace types only)"));
  pir::Type elem =
      paddle::dialect::TransToIrDataType(dense->dtype(), c.ir);  // utils.h:75
  return pir::DenseTensorType::get(c.ir, elem, dense->dims());  // builtin_type.h:78
}

/* Emit pd_op.data for `h` under `name` and register it as a feed. Mirrors
 * the feed op a translated program carries (pir.cc:2621 docstring). */
void add_feed(PdTraceCtx& c, pd_tensor h, const std::string& name) {
  const paddle::Tensor& t = reinterpret_cast<const pd_tensor_t*>(h)->value;
  auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());
  PADDLE_ENFORCE_NOT_NULL(
      dense.get(),
      common::errors::InvalidArgument("pd_trace_input: not a DenseTensor"));
  pir::AttributeMap attrs = {
      {"name", pir::StrAttribute::get(c.ir, name)},
      {"shape", paddle::dialect::IntArrayAttribute::get(
                    c.ir, phi::IntArray(dims_to_vec(dense->dims())))},
      {"dtype", paddle::dialect::DataTypeAttribute::get(c.ir, dense->dtype())},
      {"place", paddle::dialect::PlaceAttribute::get(c.ir, t.place())},
  };
  pir::Operation* op = c.builder->Build(
      {}, attrs, {dense_type_of(c, h)}, c.ir->GetRegisteredOpInfo("pd_op.data"));
  c.value_of[h] = op->result(0);
  c.feeds.emplace_back(name, h);
}

}  // namespace

pir::Value pd_trace_value_or_feed(pd_tensor tensor) {
  PdTraceCtx& c = *g_trace;
  auto it = c.value_of.find(tensor);
  if (it != c.value_of.end()) return it->second;
  // Unseen tensor (weight/constant): auto-declare as a feed.
  std::string name = "feed_" + std::to_string(c.feed_counter++);
  add_feed(c, tensor, name);
  return c.value_of[tensor];
}

void pd_trace_map_output(pd_tensor tensor, pir::Value value) {
  g_trace->value_of[tensor] = value;
}

extern "C" {

pd_trace pd_trace_begin(void) {
  PD_GUARD_BEGIN;
  auto* t = new pd_trace_t();
  PdTraceCtx& c = t->ctx;
  c.ir = pir::IrContext::Instance();
  // Register the pd_op dialect in this process (pir.cc:2606-2610).
  c.ir->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  c.program = std::make_shared<pir::Program>(c.ir);   // program.h:44
  c.builder = std::make_unique<pir::Builder>(c.ir, c.program->block());
  g_trace = &c;
  return reinterpret_cast<pd_trace>(t);
  PD_GUARD_END(nullptr);
}

void pd_trace_free(pd_trace t) {
  PD_GUARD_BEGIN;
  if (t == nullptr) return;
  pd_trace_t* real = reinterpret_cast<pd_trace_t*>(t);
  if (g_trace == &real->ctx) g_trace = nullptr;
  delete real;
  PD_GUARD_END_VOID;
}

pd_status pd_trace_input(pd_trace t, pd_tensor tensor, const char* name) {
  PD_GUARD_BEGIN;
  if (t == nullptr || tensor == nullptr || name == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_trace_input: null arg");
    return PD_INVALID_ARG;
  }
  PdTraceCtx& c = reinterpret_cast<pd_trace_t*>(t)->ctx;
  if (c.value_of.find(tensor) == c.value_of.end()) {
    add_feed(c, tensor, name);
  }
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

pd_status pd_trace_output(pd_trace t, pd_tensor tensor, const char* name) {
  PD_GUARD_BEGIN;
  if (t == nullptr || tensor == nullptr || name == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_trace_output: null arg");
    return PD_INVALID_ARG;
  }
  PdTraceCtx& c = reinterpret_cast<pd_trace_t*>(t)->ctx;
  auto it = c.value_of.find(tensor);
  if (it == c.value_of.end()) {
    pd_set_error(PD_INVALID_ARG,
                 "pd_trace_output: tensor is not a recorded op output");
    return PD_INVALID_ARG;
  }
  // builtin.shadow_output mirrors AppendShadowOutput (pir.cc:2250-2260).
  pir::AttributeMap attrs = {
      {"output_name", pir::StrAttribute::get(c.ir, name)},
  };
  c.builder->Build({it->second}, attrs, {},
                   c.ir->GetRegisteredOpInfo("builtin.shadow_output"));
  c.fo_values.push_back(it->second);
  c.fo_names.emplace_back(name);
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

int pd_trace_run(pd_trace t, pd_tensor* outs, int nout) {
  PD_GUARD_BEGIN;
  if (t == nullptr || (nout > 0 && outs == nullptr)) {
    pd_set_error(PD_INVALID_ARG, "pd_trace_run: null arg");
    return -1;
  }
  PdTraceCtx& c = reinterpret_cast<pd_trace_t*>(t)->ctx;

  // prog_attrs key set per run_program_impl.cc:448-500.
  paddle::framework::AttributeMap prog_attrs;
  prog_attrs["program_id"] = static_cast<int64_t>(c.program->id());
  prog_attrs["is_test"] = true;  // inference (partial_program.py:427)
  prog_attrs["in_sot_mode"] = false;
  std::vector<std::string> fx_names;
  std::vector<paddle::Tensor> x;
  fx_names.reserve(c.feeds.size());
  x.reserve(c.feeds.size());
  for (auto& pr : c.feeds) {
    fx_names.push_back(pr.first);
    x.push_back(reinterpret_cast<pd_tensor_t*>(pr.second)->value);
  }
  prog_attrs["fx_names"] = fx_names;
  prog_attrs["fp_names"] = std::vector<std::string>{};
  prog_attrs["fo_names"] = c.fo_names;
  prog_attrs["no_need_buffers_names"] = std::vector<std::string>{};
  prog_attrs["fo_values"] = c.fo_values;
  prog_attrs["forward_program"] = c.program;
  prog_attrs["backward_program"] = std::make_shared<pir::Program>(c.ir);

  paddle::framework::AttributeMap cuda_graph_attrs;
  cuda_graph_attrs["cuda_graph_state"] = static_cast<int64_t>(0);
  cuda_graph_attrs["cuda_graph_dispatch_key"] = static_cast<int64_t>(0);

  // Single outer scope (run_program_impl.cc:466-473).
  auto scope = std::make_shared<paddle::framework::Scope>();
  std::vector<paddle::framework::Scope*> step_scope{scope.get()};

  std::vector<paddle::Tensor> out = egr::to_static::run_program_ad_func(
      x, {}, step_scope, prog_attrs, cuda_graph_attrs);
  int n = static_cast<int>(out.size());
  int write = n < nout ? n : nout;
  for (int i = 0; i < write; ++i) {
    outs[i] = reinterpret_cast<pd_tensor>(
        new pd_tensor_t{std::move(out[static_cast<size_t>(i)])});
  }
  return n;
  PD_GUARD_END(-1);
}

}  // extern "C"
