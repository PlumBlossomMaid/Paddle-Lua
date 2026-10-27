// gzip.cc -- in-memory gzip decompression for the C ABI layer.
// Stage: M2 / 3.1 (WORKPLAN 3.1.2, MNIST dataset download+parse).
//
// Design: docs/migration/plan/modules/12-dataset-vision.md sec.4.2
//   -- "解压优先看能不能在 C ABI 层解决" (AGENTS.md sec.9.1 step 1:
//   our own .so adding a function is ~0 marginal cost vs a new rock).
//   Paddle ships no bindable inflate (grepped $PADDLE_ROOT/paddle for
//   gzopen/inflateInit/uncompress/zlib.h: zero hits), but the build tree
//   bundles zlib (third_party/install/zlib: zlib.h + zlibstatic.lib),
//   which we link and expose as one pure-C function.
//
//   Two-pass protocol (mirrors pd_jit_layer_function_names):
//     pass 1: out=NULL, cap=0  -> *out_len = decompressed size
//     pass 2: out=buffer, cap>=*out_len -> bytes written to *out_len
//   zlib sources (read 2026-10-27, vcpkg/Paddle bundled zlib 1.3):
//     zlib.h:1113-1116  inflateInit2(zstrm, 15+16) auto-detects gzip
//     zlib.h:1125-1130  inflate() returns Z_STREAM_END on completion
//     zlib.h:1185-1186  Z_FINISH flushes all pending output

#include <cstring>
#include <zlib.h>

#include "capi_impl.h"

extern "C" {

pd_status pd_gzip_inflate(const unsigned char* src, size_t src_len,
                          unsigned char* out, size_t cap, size_t* out_len) {
  PD_GUARD_BEGIN;
  if (src == nullptr || out_len == nullptr) {
    pd_set_error(PD_INVALID_ARG, "src/out_len is null");
    return PD_INVALID_ARG;
  }
  z_stream zs;
  std::memset(&zs, 0, sizeof(zs));
  // windowBits = 15 + 16: zlib auto-detects the gzip header (zlib.h:1113).
  if (inflateInit2(&zs, 15 + 16) != Z_OK) {
    pd_set_error(PD_ERROR, "inflateInit2 failed");
    return PD_ERROR;
  }
  zs.next_in = const_cast<Bytef*>(src);
  zs.avail_in = static_cast<uInt>(src_len);   // uInt is 32-bit; MNIST .gz < 4G

  if (out == nullptr) {
    // ---- pass 1: drain into a scratch buffer, measure total size ----
    Bytef scratch[65536];
    size_t total = 0;
    int ret;
    do {
      zs.next_out = scratch;
      zs.avail_out = sizeof(scratch);
      ret = inflate(&zs, Z_NO_FLUSH);
      if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) {
        inflateEnd(&zs);
        pd_set_error(PD_ERROR, "inflate failed (corrupt gzip?)");
        return PD_ERROR;
      }
      total += sizeof(scratch) - zs.avail_out;
    } while (ret != Z_STREAM_END && zs.avail_out == 0);
    inflateEnd(&zs);
    *out_len = total;
    return PD_OK;
  }

  // ---- pass 2: write into the caller's buffer -----------------------
  zs.next_out = out;
  zs.avail_out = static_cast<uInt>(cap);
  int ret = inflate(&zs, Z_FINISH);   // Z_FINISH: drain everything
  size_t written = cap - zs.avail_out;
  if (ret != Z_STREAM_END) {
    inflateEnd(&zs);
    pd_set_error(PD_ERROR,
                 ret == Z_BUF_ERROR ? "output buffer too small"
                                    : "inflate failed (corrupt gzip?)");
    *out_len = written;
    return PD_ERROR;
  }
  inflateEnd(&zs);
  *out_len = written;
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

}  // extern "C"
