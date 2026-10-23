// tools/smoke/smoke_save_load.cc
// T-M0-19: verify SaveTensor/LoadTensor round-trip format stability in a
// WITH_PYTHON=OFF + ON_INFER=OFF build.
//
// Question under test: paddle.save/load for a single tensor goes through
// framework::SaveTensor/LoadTensor (decision D19). Is the on-disk format
// stable -- versioned, lossless round-trip, and byte-identical on re-save
// (no timestamps/randomness)? Can the Lua frontend rely on it as the wire
// format for paddle.save/load of tensors?
//
// References:
//   paddle/fluid/framework/io/save_load_tensor.h:20-24   -> SaveTensor/LoadTensor decl
//   paddle/fluid/framework/io/save_load_tensor.cc:24-45  -> impl: fopen + phi::
//                                                           SerializeToStream /
//                                                           DeserializeFromStream
//   paddle/phi/core/framework/dense_tensor_serialize.h:34-48 -> SerializeToStream
//                                                           decl (no DeviceContext
//                                                           overload)
//   paddle/phi/core/framework/dense_tensor_serialize.cc:69-82/165-210
//                                                        -> format: field1 uint32
//                                                           version(=0), field2 LoD,
//                                                           field3 TensorToStream
//   paddle/phi/core/framework/dense_tensor_tostream.cc:97-228
//                                                        -> field3 layout: uint32
//                                                           version(0) + int32 proto
//                                                           size + TensorDesc proto
//                                                           (dtype+dims) + raw data
//   paddle/fluid/pybind/io.cc:140                        -> python load_dense_tensor
//                                                           = framework::LoadTensor
//                                                           (D19 holds)
//
// Build: same 483-lib baseline as tools/smoke/smoke.cc (m0-report.md section 2),
//   /LIBPATH:E:\code\Paddle\build-nopy, /FORCE:MULTIPLE, /bigobj, 35 DEFINES.
// Run: smoke_save_load.exe   (cwd: repo root; files written to .qwen/tmp/)

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/io/save_load_tensor.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace {

const char* kPath1 = ".qwen/tmp/smoke_save_load_1.bin";
const char* kPath2 = ".qwen/tmp/smoke_save_load_2.bin";

bool ReadFileBytes(const std::string& path, std::vector<char>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamoff len = f.tellg();
  f.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(len));
  f.read(out->data(), len);
  return static_cast<bool>(f);
}

}  // namespace

int main() {
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();

  // Build a 2x3 float32 tensor with known data.
  phi::DenseTensor t;
  t.Resize(common::make_ddim({2, 3}));
  auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
  float* data = dev_ctx->Alloc<float>(&t);
  for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i) * 1.5f;

  // --- Round trip ---
  paddle::framework::SaveTensor(t, kPath1, /*overwrite=*/true);
  phi::DenseTensor out;
  paddle::framework::LoadTensor(kPath1, &out);

  if (!(out.dims() == t.dims())) {
    std::cout << "SMOKE FAIL: dims mismatch" << std::endl;
    return 1;
  }
  if (!(out.dtype() == t.dtype())) {
    std::cout << "SMOKE FAIL: dtype mismatch" << std::endl;
    return 1;
  }
  const float* od = out.data<float>();
  for (int i = 0; i < 6; ++i) {
    if (od[i] != data[i]) {
      std::cout << "SMOKE FAIL: data[" << i << "] " << od[i] << " != "
                << data[i] << std::endl;
      return 1;
    }
  }

  // --- Version field (field1 = uint32 version, must be 0) ---
  std::vector<char> bytes;
  if (!ReadFileBytes(kPath1, &bytes) || bytes.size() < 4) {
    std::cout << "SMOKE FAIL: cannot read back file" << std::endl;
    return 1;
  }
  uint32_t version = 0;
  std::memcpy(&version, bytes.data(), sizeof(version));
  if (version != 0U) {
    std::cout << "SMOKE FAIL: unexpected version field " << version
              << std::endl;
    return 1;
  }

  // --- Re-save byte-identical (no timestamp / randomness in the format) ---
  paddle::framework::SaveTensor(t, kPath2, /*overwrite=*/true);
  std::vector<char> b2;
  if (!ReadFileBytes(kPath2, &b2) || b2.size() != bytes.size() ||
      std::memcmp(b2.data(), bytes.data(), b2.size()) != 0) {
    std::cout << "SMOKE FAIL: re-save is not byte-identical" << std::endl;
    return 1;
  }

  std::cout << "SMOKE PASS: round-trip ok (dims/dtype/data), version="
            << version << ", file_size=" << bytes.size()
            << ", re-save byte-identical" << std::endl;
  return 0;
}
