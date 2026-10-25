// jit.cc -- jit::Layer load & inference for the C ABI layer.
// Stage: P1 / M1 (WORKPLAN 2.1.6). Follows device.cc (layout.md sec.3).
//
// Design: docs/migration/plan/modules/08-jit-infer.md sec.3.2.
// Every Paddle symbol below was read from $PADDLE_ROOT sources
// (E:\code\Paddle, branch windows_dataloader_multiprocess, v3.5.0):
//   paddle/fluid/jit/serializer.h:78           jit::Load(path, place)
//   paddle/fluid/jit/layer.h:57                Layer::forward(vector<Tensor>)
//   paddle/fluid/jit/layer.h:63                Layer::to(place)
//   paddle/fluid/jit/layer.h:73                Layer::FunctionNames()
//   paddle/fluid/jit/layer.h:78                Layer::Clone(void* stream)
//
// Link gap (WITH_PYTHON=OFF): libfluid_jit's PirFunctionInfo ctor calls
// GetFeedTargetNames/GetFetchTargetNames, defined in paddle/fluid/pybind
// (not built without Python). smoke_jit_load.cc ships the same shim for the
// smoke EXE; a DSO linking libfluid_jit needs it too -- see
// process/m0-report.md T-M0-17 before declaring a link error a code bug.

#include <cstring>
#include <string>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"

#include "capi_impl.h"

// Runtime config for jit::Load, matching the T-M0-17 verified setup
// (smoke_jit_load.cc RunMain): the WITH_PYTHON=OFF link does not pull in
// PIR's parametric-storage registration (storage_manager.cc:85 throws when
// the type is unregistered), so Load must read the NON-PIR format
// (smoke_jit_load.cc RunMain): the WITH_PYTHON=OFF link does not pull in
// PIR's parametric-storage registration (storage_manager.cc:85 throws when
// the type is unregistered), so Load must read the NON-PIR format
// (serializer.cc:48 branches on FLAGS_enable_pir_api) -- enable_pir_api
// defaults to true (flags.cc:1888), which would select the PIR path.
// The "New" engine is our target anyway (D16 InterpreterCore, serializer.cc:
// 73-76), and use_mkldnn=false avoids the upstream fc/oneDNN bug (Q-24,
// onednn_context.cc:347). These are set once here, before the first Load.
COMMON_DECLARE_bool(enable_pir_api);
COMMON_DECLARE_string(jit_engine_type);
COMMON_DECLARE_bool(use_mkldnn);
namespace {
struct JitFlagsOnce {
  JitFlagsOnce() {
    FLAGS_enable_pir_api = false;
    FLAGS_jit_engine_type = "New";
    FLAGS_use_mkldnn = false;
  }
};
const JitFlagsOnce kJitFlagsOnce;
}  // namespace

// ---------------------------------------------------------------------------
// Upstream link-gap shim (WITH_PYTHON=OFF only), same source as the T-M0-17
// smoke (smoke_jit_load.cc): libfluid_jit's PirFunctionInfo ctor
// (function_schema.cc:112/115) calls paddle::GetFeedTargetNames /
// GetFetchTargetNames unconditionally, but those free functions live in
// paddle/fluid/pybind/pir_utils.cc:20/36 -- the pybind directory, which is
// NOT built under WITH_PYTHON=OFF. The implementations are pure PIR (DataOp/
// FeedOp/FetchOp/ShadowOutputOp + StrAttribute) with zero Python dependency,
// so we provide equivalents here, in the same TU as pd_jit_layer_load -- a
// static library pulls in jit.cc.obj (and with it this shim) only when a
// pd_jit_layer_* symbol is referenced, which is exactly when libfluid_jit
// needs the names. See process/m0-report.md T-M0-17 / api/jit.md sec.6 ①.
namespace paddle {
std::vector<std::string> GetFeedTargetNames(pir::Program* prog) {
  std::vector<std::string> feed_target;
  for (auto& op : *(prog->block())) {
    if (op.isa<paddle::dialect::DataOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      feed_target.push_back(name);
      continue;
    } else if (op.isa<paddle::dialect::FeedOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      feed_target.push_back(name);
      continue;
    }
  }
  return feed_target;
}

std::vector<std::string> GetFetchTargetNames(pir::Program* prog) {
  std::vector<std::string> fetch_target;
  for (auto& op : *(prog->block())) {
    if (op.isa<paddle::dialect::FetchOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      fetch_target.push_back(name);
      continue;
    } else if (op.isa<pir::ShadowOutputOp>()) {
      auto name = op.attribute<pir::StrAttribute>("output_name").AsString();
      fetch_target.push_back(name);
      continue;
    }
  }
  return fetch_target;
}
}  // namespace paddle
// ---------------------------------------------------------------------------

extern "C" {

pd_layer pd_jit_layer_load(const char* path, pd_place place) {
  PD_GUARD_BEGIN;
  if (path == nullptr || place == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_jit_layer_load: path/place is null");
    return nullptr;
  }
  // serializer.h:78 -- Load(file_path, place) -> jit::Layer (by value)
  paddle::jit::Layer layer = paddle::jit::Load(
      std::string(path), reinterpret_cast<pd_place_t*>(place)->value);
  (void)layer;
  return reinterpret_cast<pd_layer>(new pd_layer_t{std::move(layer)});
  PD_GUARD_END(nullptr);
}

void pd_jit_layer_free(pd_layer layer) {
  PD_GUARD_BEGIN;
  delete reinterpret_cast<pd_layer_t*>(layer);
  PD_GUARD_END_VOID;
}

int pd_jit_layer_forward(pd_layer layer,
                         const pd_tensor* inputs, int ntensors,
                         pd_tensor* out, int nout) {
  PD_GUARD_BEGIN;
  if (layer == nullptr || (ntensors > 0 && inputs == nullptr) ||
      (nout > 0 && out == nullptr)) {
    pd_set_error(PD_INVALID_ARG, "pd_jit_layer_forward: null arg");
    return -1;
  }
  std::vector<paddle::Tensor> ins;
  ins.reserve(static_cast<size_t>(ntensors));
  for (int i = 0; i < ntensors; ++i) {
    ins.push_back(reinterpret_cast<const pd_tensor_t*>(inputs[i])->value);
  }
  // layer.h:57 -- forward(vector<Tensor>) -> vector<Tensor>
  std::vector<paddle::Tensor> outs =
      reinterpret_cast<pd_layer_t*>(layer)->value.forward(ins);
  int n = static_cast<int>(outs.size());
  int write = n < nout ? n : nout;
  for (int i = 0; i < write; ++i) {
    out[i] = reinterpret_cast<pd_tensor>(
        new pd_tensor_t{std::move(outs[static_cast<size_t>(i)])});
  }
  return n;
  PD_GUARD_END(-1);
}

pd_status pd_jit_layer_to(pd_layer layer, pd_place place) {
  PD_GUARD_BEGIN;
  if (layer == nullptr || place == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_jit_layer_to: layer/place is null");
    return PD_INVALID_ARG;
  }
  // layer.h:63 -- to(place)
  reinterpret_cast<pd_layer_t*>(layer)->value.to(
      reinterpret_cast<pd_place_t*>(place)->value);
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

int pd_jit_layer_function_names(pd_layer layer,
                                char* buf, size_t nbuf, size_t* len) {
  PD_GUARD_BEGIN;
  if (layer == nullptr || (buf == nullptr && nbuf > 0) || len == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_jit_layer_function_names: null arg");
    return -1;
  }
  // layer.h:73 -- FunctionNames() -> vector<string>
  std::vector<std::string> names =
      reinterpret_cast<pd_layer_t*>(layer)->value.FunctionNames();
  size_t need = 0;
  for (const std::string& s : names) need += s.size() + 1;  // + NUL each
  *len = need;
  if (buf == nullptr) return static_cast<int>(names.size());  // probe
  size_t off = 0;
  for (const std::string& s : names) {
    size_t take = s.size() + 1;
    if (off + take > nbuf) break;
    std::memcpy(buf + off, s.c_str(), s.size());
    buf[off + s.size()] = '\0';
    off += take;
  }
  return static_cast<int>(names.size());
  PD_GUARD_END(-1);
}

pd_layer pd_jit_layer_clone(pd_layer layer) {
  PD_GUARD_BEGIN;
  if (layer == nullptr) {
    pd_set_error(PD_INVALID_ARG, "pd_jit_layer_clone: layer is null");
    return nullptr;
  }
  // layer.h:76 -- Clone(void* stream = nullptr) -> shared_ptr<Layer>
  auto cloned = reinterpret_cast<pd_layer_t*>(layer)->value.Clone(nullptr);
  if (cloned == nullptr) {
    pd_set_error(PD_ERROR, "pd_jit_layer_clone: Clone returned null");
    return nullptr;
  }
  return reinterpret_cast<pd_layer>(new pd_layer_t{*cloned});
  PD_GUARD_END(nullptr);
}

}  // extern "C"