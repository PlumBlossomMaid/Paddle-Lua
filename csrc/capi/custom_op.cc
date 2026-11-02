// custom_op.cc -- custom-operator loading AND execution for the C ABI layer
// (P16, 16-cpp-extension.md). Implements pd_load_op_meta_info (paddle_c.h
// Group 13) and pd_custom_op_run (Group 14).
//
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, HEAD 2bfeb115b0):
//   paddle/fluid/framework/custom_operator.h:316
//                                                  LoadOpMetaInfoAndRegisterOp
//   paddle/fluid/framework/custom_operator.cc:1279-1315
//                                                  RegisterOperatorWithMetaInfoMap
//     -- registers BOTH the PIR dialect op (CustomOpDialect::RegisterCustomOp
//        via custom_dialect, op_dialect.cc:1180, name = "custom_op." + op,
//        custom_operator_utils.h:29) so the op participates in pir program
//        assembly, AND the legacy fluid OpInfo table (RegisterOperatorWithMetaInfo,
//        custom_operator.cc:962) which owns the executable kernel
//        (RegisterOperatorKernel -> OperatorWithKernel, custom_operator.cc:898-961)
//   paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:1180-1229
//                                                  CustomOpDialect::RegisterCustomOp
//   paddle/fluid/pir/transforms/pd_op_to_kernel_pass.cc:3744-3751
//                                                  IsCustomOp -> HandleForCustomOp
//   paddle/phi/api/ext/op_meta_info.h:1164         PD_BUILD_OP macro (user side)
//   paddle/fluid/pybind/manual_static_op_function.h:561-966
//                                                  static_api_run_custom_op
//     -- the execution blueprint: OpMetaInfoMap lookup -> OperationArgument
//        -> input feeds -> attrs -> RunInferShape/RunInferDtype -> output
//        DenseTensorType -> Build -> shadow_output -> run_program_ad_func
//   paddle/fluid/framework/custom_operator_utils.h:462/536/230
//                                                  RunInferShape / RunInferDtype /
//                                                  CheckDefaultInferShapeDtype
//   paddle/fluid/framework/custom_operator_utils.h:53  IsDuplicableVar
//   paddle/fluid/pir/dialect/operator/utils/utils.h:75
//                                                  TransToIrDataType
//   paddle/pir/include/core/operation_utils.h:56-100
//                                                  OperationArgument
//   paddle/pir/include/core/builder.h:139          Build(OperationArgument&&)
//   paddle/pir/include/core/builtin_op.h:260       PassStopGradientsDefaultly
//   paddle/fluid/eager/to_static/run_program_func.h:23-28
//                                                  run_program_ad_func
//
// The PIR path is what makes a loaded custom op usable from our Lua trace /
// program-assembly machinery (P15): once registered as "custom_op.<name>",
// a builder can look it up via IrContext::GetRegisteredOpInfo and run it
// through run_program_ad_func exactly like a generated op.

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/eager/to_static/run_program_func.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/init_phi.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

#include "capi_impl.h"
#include "paddle_c.h"

// The P16 custom-op DLL needs `paddle::InitPhi::InitPhi()`: paddle/extension.h
// carries `static paddle::InitPhi g_init_phi;` (extension.h:51), and every
// user op includes that header. Upstream defines the ctor in
// paddle/fluid/platform/init_phi.cc:27 as the ONE-LINE
// `paddle::framework::InitMemoryMethod()`, compiled into libinit_phi.lib.
// paddle_core does NOT link that lib, so the symbol would be missing from the
// process. The host (this DLL) defines it instead: the process already
// initialized phi via pd_init_once (mutable_data etc. work from a bare
// pd_tensor_from_host_buffer), and InitMemoryMethod is idempotent
// (init.cc:443 sets singleton flags). PADDLE_DLL_EXPORT (compile define)
// makes PADDLE_API = dllexport, so this definition is EXPORTED from
// paddle_core.dll and a user custom-op DLL links against paddle_core.lib +
// phi.lib -- the single in-process Paddle -- instead of dragging in a second
// full Paddle (which would re-register common.dll's flag registry and die
// with "flag multiple definition", flags_native.cc:309).
namespace paddle {
InitPhi::InitPhi() {}
}  // namespace paddle

extern "C" {

int pd_load_op_meta_info(const char* dso_path, char* names, int cap) {
  PD_GUARD_BEGIN;
  if (dso_path == nullptr || dso_path[0] == '\0') {
    pd_set_error(PD_INVALID_ARG, "pd_load_op_meta_info: empty dso_path");
    return -1;
  }
  if (cap > 0 && names == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_load_op_meta_info: names null with cap>0");
    return -1;
  }
  // LoadOpMetaInfoAndRegisterOp (custom_operator.h:316) returns the DIFF of
  // newly-registered ops: loading the same dso twice yields [] on the second
  // call, so the sol layer's probe-then-fill (two calls) would see count 0.
  // Cache the names per dso on first (successful) load; later calls replay
  // the cache, making the two-phase protocol idempotent.
  static std::unordered_map<std::string, std::vector<std::string>> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto& names_list = cache[std::string(dso_path)];
  if (names_list.empty()) {
    const auto& diff =
        paddle::framework::LoadOpMetaInfoAndRegisterOp(std::string(dso_path));
    for (const auto& pair : diff) {
      names_list.push_back(pair.first);
    }
  }
  size_t written = 0;
  int total = 0;
  for (const auto& name : names_list) {
    ++total;
    if (written + name.size() + 1 <= static_cast<size_t>(cap)) {
      std::memcpy(names + written, name.c_str(), name.size() + 1);
      written += name.size() + 1;
    }
  }
  return total;
  PD_GUARD_END(-1);
}

namespace {

/* DDim -> std::vector<int64_t> (same helper as trace.cc). */
std::vector<int64_t> dims_to_vec(const phi::DDim& d) {
  std::vector<int64_t> v(static_cast<size_t>(d.size()));
  for (int i = 0; i < d.size(); ++i) v[static_cast<size_t>(i)] = d[i];
  return v;
}

/* The pir op name for a custom op: "custom_op." + op (+ "_" if inplace).
 * Mirrors static_api_run_custom_op (manual_static_op_function.h:617-620). */
std::string pir_op_name_of(const std::string& op_type,
                           const std::unordered_map<std::string, std::string>&
                               inplace_map) {
  std::string n = paddle::framework::kCustomDialectPrefix + op_type;
  if (!inplace_map.empty()) n += "_";
  return n;
}

/* Emit pd_op.data for `t` under `name` and return its result Value.
 * Mirrors trace.cc add_feed (pir.cc:2621 docstring). */
pir::Value emit_feed(pir::Builder& builder, pir::IrContext* ctx,
                     pd_tensor h, const std::string& name) {
  const paddle::Tensor& t = reinterpret_cast<const pd_tensor_t*>(h)->value;
  auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());
  PADDLE_ENFORCE_NOT_NULL(
      dense.get(),
      common::errors::InvalidArgument(
          "pd_custom_op_run: input '%s' has no DenseTensor impl", name));
  pir::AttributeMap attrs = {
      {"name", pir::StrAttribute::get(ctx, name)},
      {"shape", paddle::dialect::IntArrayAttribute::get(
                    ctx, phi::IntArray(dims_to_vec(dense->dims())))},
      {"dtype", paddle::dialect::DataTypeAttribute::get(ctx, dense->dtype())},
      {"place", paddle::dialect::PlaceAttribute::get(ctx, t.place())},
  };
  pir::Type elem = paddle::dialect::TransToIrDataType(dense->dtype(), ctx);
  pir::Type dt = pir::DenseTensorType::get(ctx, elem, dense->dims());
  pir::Operation* op = builder.Build(
      {}, attrs, {dt}, ctx->GetRegisteredOpInfo("pd_op.data"));
  return op->result(0);
}

/* RunInferShape / RunInferDtype expect custom_attrs as paddle::any. The
 * C ABI hands us raw typed pointers (parallel arrays). Translate each into
 * the right paddle::any and, in the same pass, add the matching pir
 * attribute to the OperationArgument. */
void add_attr(pir::OperationArgument& argument, pir::IrContext* ctx,
              const std::string& name, const std::string& type_str,
              int type_code, const void* value, int len,
              std::vector<paddle::any>& custom_attrs) {
  if (type_str == "bool" && type_code == PD_ATTR_BOOL) {
    bool b = (*static_cast<const int*>(value)) != 0;
    custom_attrs.push_back(b);
    argument.AddAttribute(name, pir::BoolAttribute::get(ctx, b));
  } else if (type_str == "int" && type_code == PD_ATTR_INT) {
    int v = *static_cast<const int*>(value);
    custom_attrs.push_back(v);
    argument.AddAttribute(name, pir::Int32Attribute::get(ctx, v));
  } else if (type_str == "float" && type_code == PD_ATTR_FLOAT) {
    float v = *static_cast<const float*>(value);
    custom_attrs.push_back(v);
    argument.AddAttribute(name, pir::FloatAttribute::get(ctx, v));
  } else if (type_str == "double" && type_code == PD_ATTR_DOUBLE) {
    double v = *static_cast<const double*>(value);
    custom_attrs.push_back(v);
    argument.AddAttribute(name, pir::DoubleAttribute::get(ctx, v));
  } else if (type_str == "int64_t" && type_code == PD_ATTR_INT64) {
    int64_t v = *static_cast<const int64_t*>(value);
    custom_attrs.push_back(v);
    argument.AddAttribute(name, pir::Int64Attribute::get(ctx, v));
  } else if (type_str == "std::string" && type_code == PD_ATTR_STRING) {
    const char* s = *static_cast<const char* const*>(value);
    std::string str(s ? s : "");
    custom_attrs.push_back(str);
    argument.AddAttribute(name, pir::StrAttribute::get(ctx, str));
  } else if (type_str == "std::vector<int>" && type_code == PD_ATTR_VEC_INT) {
    const int* arr = static_cast<const int*>(value);
    std::vector<int> vec(arr, arr + len);
    custom_attrs.push_back(vec);
    std::vector<pir::Attribute> attr_arr;
    for (int v : vec)
      attr_arr.push_back(pir::Int32Attribute::get(ctx, v));
    argument.AddAttribute(name,
                          pir::ArrayAttribute::get(ctx, attr_arr));
  } else if (type_str == "std::vector<float>" &&
             type_code == PD_ATTR_VEC_FLOAT) {
    const float* arr = static_cast<const float*>(value);
    std::vector<float> vec(arr, arr + len);
    custom_attrs.push_back(vec);
    std::vector<pir::Attribute> attr_arr;
    for (float v : vec)
      attr_arr.push_back(pir::FloatAttribute::get(ctx, v));
    argument.AddAttribute(name,
                          pir::ArrayAttribute::get(ctx, attr_arr));
  } else if (type_str == "std::vector<int64_t>" &&
             type_code == PD_ATTR_VEC_INT64) {
    const int64_t* arr = static_cast<const int64_t*>(value);
    std::vector<int64_t> vec(arr, arr + len);
    custom_attrs.push_back(vec);
    std::vector<pir::Attribute> attr_arr;
    for (int64_t v : vec)
      attr_arr.push_back(pir::Int64Attribute::get(ctx, v));
    argument.AddAttribute(name,
                          pir::ArrayAttribute::get(ctx, attr_arr));
  } else if (type_str == "std::vector<std::string>" &&
             type_code == PD_ATTR_VEC_STRING) {
    const char* const* arr = static_cast<const char* const*>(value);
    std::vector<std::string> vec;
    for (int i = 0; i < len; ++i)
      vec.emplace_back(arr[i] ? arr[i] : "");
    custom_attrs.push_back(vec);
    std::vector<pir::Attribute> attr_arr;
    for (const std::string& s : vec)
      attr_arr.push_back(pir::StrAttribute::get(ctx, s));
    argument.AddAttribute(name,
                          pir::ArrayAttribute::get(ctx, attr_arr));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "pd_custom_op_run: attribute '%s' type mismatch (declared %s, "
        "provided code %d)",
        name, type_str, type_code));
  }
}

}  // namespace

int pd_custom_op_run(const char* op_type, const pd_tensor* inputs,
                     int ninputs, const int* attr_types,
                     const void* const* attr_values, const int* attr_lens,
                     int nattrs, pd_tensor* outs, int nout) {
  PD_GUARD_BEGIN;
  if (op_type == nullptr || op_type[0] == '\0') {
    pd_set_error(PD_INVALID_ARG, "pd_custom_op_run: empty op_type");
    return -1;
  }
  if (ninputs > 0 && inputs == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_custom_op_run: null inputs");
    return -1;
  }
  if (nattrs > 0 && (attr_types == nullptr || attr_values == nullptr)) {
    pd_set_error(PD_INVALID_ARG, "pd_custom_op_run: null attrs");
    return -1;
  }
  if (nout > 0 && outs == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_custom_op_run: null outs");
    return -1;
  }

  // ---- 1. OpMetaInfoMap lookup (manual_static_op_function.h:566-572) ----
  const auto& meta_info_map = paddle::OpMetaInfoMap::Instance().GetMap();
  auto it = meta_info_map.find(op_type);
  if (it == meta_info_map.end()) {
    pd_set_error(
        PD_INVALID_ARG,
        ("pd_custom_op_run: op '" + std::string(op_type)
         + "' not registered (load the dso first via pd_load_op_meta_info)")
            .c_str());
    return -1;
  }
  const auto& vec_map = it->second;
  const auto& inputs_meta = paddle::OpMetaInfoHelper::GetInputs(vec_map[0]);
  const auto& attrs_meta = paddle::OpMetaInfoHelper::GetAttrs(vec_map[0]);
  const auto& outputs_meta = paddle::OpMetaInfoHelper::GetOutputs(vec_map[0]);
  const auto& inplace_map =
      paddle::OpMetaInfoHelper::GetInplaceMap(vec_map[0]);
  const auto& inplace_reverse_map =
      paddle::OpMetaInfoHelper::GetInplaceReverseMap(vec_map[0]);
  auto infershape_func =
      paddle::OpMetaInfoHelper::GetInferShapeFn(vec_map[0]);
  auto inferdtype_func =
      paddle::OpMetaInfoHelper::GetInferDtypeFn(vec_map[0]);

  // ---- 2. Arity / duplicable checks (custom_relu: 1 in, 1 out, no vec) ----
  for (const auto& in : inputs_meta) {
    if (paddle::framework::detail::IsDuplicableVar(in)) {
      pd_set_error(
          PD_INVALID_ARG,
          ("pd_custom_op_run: vector input '" + std::string(in)
           + "' not supported yet").c_str());
      return -1;
    }
  }
  for (const auto& out : outputs_meta) {
    if (paddle::framework::detail::IsDuplicableVar(out)) {
      pd_set_error(
          PD_INVALID_ARG,
          ("pd_custom_op_run: vector output '" + std::string(out)
           + "' not supported yet").c_str());
      return -1;
    }
  }
  if (static_cast<size_t>(ninputs) != inputs_meta.size()) {
    pd_set_error(
        PD_INVALID_ARG,
        ("pd_custom_op_run: op '" + std::string(op_type)
         + "' expects " + std::to_string(inputs_meta.size()) + " inputs, got "
         + std::to_string(ninputs) + ")")
            .c_str());
    return -1;
  }
  if (static_cast<size_t>(nattrs) != attrs_meta.size()) {
    pd_set_error(
        PD_INVALID_ARG,
        ("pd_custom_op_run: op '" + std::string(op_type)
         + "' expects " + std::to_string(attrs_meta.size()) + " attrs, got "
         + std::to_string(nattrs) + ")")
            .c_str());
    return -1;
  }

  // ---- 3. Program + builder (same shape as pd_trace_begin) ----
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  std::shared_ptr<pir::Program> program =
      std::make_shared<pir::Program>(ctx);          // program.h:44
  pir::Builder builder(ctx, program->block());      // builder.h:139

  // ---- 4. OpInfo lookup (manual_static_op_function.h:617-620) ----
  std::string pir_op_name = pir_op_name_of(op_type, inplace_map);
  pir::OpInfo pir_info = ctx->GetRegisteredOpInfo(pir_op_name);
  if (!pir_info) {
    pd_set_error(PD_INVALID_ARG,
                 ("pd_custom_op_run: no pir op '" + pir_op_name
                  + "' (custom dialect not registered for '" + op_type
                  + "')").c_str());
    return -1;
  }

  // ---- 5. Inputs -> feeds (manual_static_op_function.h:624-665) ----
  pir::OperationArgument argument(pir_info);           // operation_utils.h:52
  std::vector<pir::Value> argument_inputs;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<phi::DataType> input_dtypes;
  std::unordered_map<std::string, int> input_name2id_map;
  std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;
  std::vector<std::vector<phi::DataType>> vec_input_dtypes;
  std::unordered_map<std::string, int> vec_input_name2id_map;
  int input_index = 0;
  for (size_t i = 0; i < inputs_meta.size(); ++i) {
    const auto& iname = inputs_meta.at(i);
    if (paddle::framework::detail::IsDuplicableVar(iname)) continue;
    pir::Value v = emit_feed(builder, ctx, inputs[i], iname);
    paddle::dialect::DenseTensorType in_type =
        v.type().dyn_cast<paddle::dialect::DenseTensorType>();
    input_shapes.push_back(phi::vectorize(in_type.dims()));
    input_dtypes.push_back(
        paddle::dialect::TransToPhiDataType(in_type.dtype()));
    input_name2id_map[iname] = input_index;
    ++input_index;
    argument_inputs.push_back(v);
  }
  argument.AddInputs(argument_inputs);

  // ---- 6. Attrs (manual_static_op_function.h:672-810) ----
  std::vector<paddle::any> custom_attrs;
  for (size_t i = 0; i < attrs_meta.size(); ++i) {
    const auto& attr = attrs_meta.at(i);
    std::vector<std::string> attr_name_and_type = paddle::ParseAttrStr(attr);
    const std::string& attr_name = attr_name_and_type[0];
    const std::string& attr_type_str = attr_name_and_type[1];
    add_attr(argument, ctx, attr_name, attr_type_str, attr_types[i],
             attr_values[i], attr_lens[i], custom_attrs);
  }

  // ---- 7. Infer shape/dtype (manual_static_op_function.h:813-816) ----
  paddle::framework::CheckDefaultInferShapeDtype(
      infershape_func, inferdtype_func, vec_map[0]);
  std::vector<std::vector<int64_t>> output_shapes =
      paddle::framework::RunInferShape(
          infershape_func, vec_map[0], input_shapes, input_name2id_map,
          vec_input_shapes, vec_input_name2id_map, custom_attrs);
  std::vector<phi::DataType> output_dtypes =
      paddle::framework::RunInferDtype(
          inferdtype_func, vec_map[0], input_dtypes, input_name2id_map,
          vec_input_dtypes, vec_input_name2id_map, custom_attrs);

  // ---- 8. Output types (manual_static_op_function.h:828-930) ----
  // output_name2value_num logic, reduced to non-duplicable outputs:
  //   value_num = 1 normally, 0 when an inplace output maps to an
  //   optional-None input (input_shapes[idx] empty). RunInferShape /
  //   RunInferDtype ALREADY include inplace outputs in their result (in
  //   declaration order), so value_index advances once per emitted type.
  size_t value_index = 0;
  std::vector<pir::Type> argument_outputs;
  for (size_t i = 0; i < outputs_meta.size(); ++i) {
    const auto& output = outputs_meta.at(i);
    bool optional_none = false;
    if (inplace_reverse_map.find(output) != inplace_reverse_map.end()) {
      const auto& input = inplace_reverse_map.at(output);
      auto idx = input_name2id_map.at(input);
      optional_none = input_shapes[idx].empty();
    }
    if (optional_none) {
      argument_outputs.push_back(pir::Type());
      continue;
    }
    auto ddims = phi::make_ddim(output_shapes[value_index]);
    auto dtype = output_dtypes[value_index];
    argument_outputs.push_back(paddle::dialect::DenseTensorType::get(
        ctx, paddle::dialect::TransToIrDataType(dtype, ctx), ddims,
        phi::DataLayout::NCHW, {}, 0));
    ++value_index;
  }
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());

  // ---- 9. Build (manual_static_op_function.h:933-939) ----
  pir::PassStopGradientsDefaultly(argument);
  pir::Operation* op = builder.Build(std::move(argument));

  // ---- 10. Fetch each output via shadow_output (like pd_trace_output) ----
  // op->result(i) aligns with outputs_meta[i] (optional-None outputs still
  // occupy a result slot; the blueprint pushes a (null) Type for them).
  std::vector<pir::Value> fo_values;
  std::vector<std::string> fo_names;
  for (size_t i = 0; i < outputs_meta.size(); ++i) {
    const auto& output = outputs_meta.at(i);
    pir::Value out_v = op->result(i);
    // shadow_output (mirror AppendShadowOutput, pir.cc:2250-2260).
    pir::AttributeMap attrs = {
        {"output_name", pir::StrAttribute::get(ctx, output)},
    };
    builder.Build({out_v}, attrs, {},
                  ctx->GetRegisteredOpInfo("builtin.shadow_output"));
    fo_values.push_back(out_v);
    fo_names.push_back(output);
  }

  // ---- 11. run_program_ad_func inference (trace.cc pattern) ----
  paddle::framework::AttributeMap prog_attrs;
  prog_attrs["program_id"] = static_cast<int64_t>(program->id());
  prog_attrs["is_test"] = true;
  prog_attrs["in_sot_mode"] = false;
  std::vector<std::string> fx_names;
  std::vector<paddle::Tensor> x;
  for (size_t i = 0; i < inputs_meta.size(); ++i) {
    fx_names.push_back(inputs_meta[i]);
    x.push_back(reinterpret_cast<const pd_tensor_t*>(inputs[i])->value);
  }
  prog_attrs["fx_names"] = fx_names;
  prog_attrs["fp_names"] = std::vector<std::string>{};
  prog_attrs["fo_names"] = fo_names;
  prog_attrs["no_need_buffers_names"] = std::vector<std::string>{};
  prog_attrs["fo_values"] = fo_values;
  prog_attrs["forward_program"] = program;
  prog_attrs["backward_program"] = std::make_shared<pir::Program>(ctx);

  paddle::framework::AttributeMap cuda_graph_attrs;
  cuda_graph_attrs["cuda_graph_state"] = static_cast<int64_t>(0);
  cuda_graph_attrs["cuda_graph_dispatch_key"] = static_cast<int64_t>(0);

  auto step_scope = std::make_shared<paddle::framework::Scope>();
  std::vector<paddle::framework::Scope*> step_scope_vec{step_scope.get()};

  std::vector<paddle::Tensor> out = egr::to_static::run_program_ad_func(
      x, {}, step_scope_vec, prog_attrs, cuda_graph_attrs);
  int n = static_cast<int>(out.size());
  int write = n < nout ? n : nout;
  for (int i = 0; i < write; ++i) {
    outs[i] = reinterpret_cast<pd_tensor>(
        new pd_tensor_t{std::move(out[static_cast<size_t>(i)])});
  }
  return n;
  PD_GUARD_END(-1);
}

int pd_custom_op_meta(const char* op_type, char* buf, int cap) {
  PD_GUARD_BEGIN;
  if (op_type == nullptr || op_type[0] == '\0') {
    pd_set_error(PD_INVALID_ARG, "pd_custom_op_meta: empty op_type");
    return -1;
  }
  const auto& meta_info_map = paddle::OpMetaInfoMap::Instance().GetMap();
  auto it = meta_info_map.find(op_type);
  if (it == meta_info_map.end()) {
    pd_set_error(PD_INVALID_ARG,
                 ("pd_custom_op_meta: op '" + std::string(op_type)
                  + "' not registered (load the dso first)")
                     .c_str());
    return -1;
  }
  const auto& vec_map = it->second;
  // Schema: three newline-terminated lines, each a comma-joined list:
  //   line 1: inputs  (names, in declaration order)
  //   line 2: outputs (names, in declaration order)
  //   line 3: attrs   ("name:type", in declaration order; type strings
  //                    are ParseAttrStr's second field, op_meta_info.h)
  // Values carry no NULs or newlines (PD_BUILD_OP identifiers), so the
  // Lua side splits on ',' / '\n' safely.
  auto join = [](const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) out += ",";
      out += v[i];
    }
    out += "\n";
    return out;
  };
  std::string schema;
  schema += join(paddle::OpMetaInfoHelper::GetInputs(vec_map[0]));
  schema += join(paddle::OpMetaInfoHelper::GetOutputs(vec_map[0]));
  schema += join(paddle::OpMetaInfoHelper::GetAttrs(vec_map[0]));
  if (cap <= 0 || buf == nullptr) {
    return static_cast<int>(schema.size());
  }
  int n = static_cast<int>(schema.size());
  if (n >= cap) n = cap - 1;  // keep room for the terminator
  std::memcpy(buf, schema.data(), static_cast<size_t>(n));
  buf[n] = '\0';
  return n;
  PD_GUARD_END(-1);
}

}  // extern "C"