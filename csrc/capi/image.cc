// image.cc -- 2D image resize for the C ABI layer.
// Stage: M2 / 3.1 (WORKPLAN 3.1.3, paddle.vision.transforms.Resize).
//
// Design: docs/migration/plan/modules/12-dataset-vision.md sec.4.2 --
// "解压/图像缩放优先在 C ABI 层解决" (AGENTS.md sec.9.1 step 1: our own
// .so adding a function is ~0 marginal cost vs a new rock). Insight7's
// core has no 2D resize (grepped src/: zero hits), and Paddle's interp
// operators are Tensor-level -- transforms must stay on insight.Array
// (12-dataset-vision.md sec.3.2), so the interpolation lives HERE, as a
// pure byte transform.
//
// Mirrors opencv resize coordinate mapping (functional_cv2.py:59-120 is
// a thin cv2 wrapper, 977-line file; the reference mapping below is the
// classic OpenCV INTER_LINEAR/INTER_NEAREST formula family, read from
// opencv modules/imgproc/src/resize.cpp):
//   scale = src / dst
//   nearest:  sx = floor(dx * scale)                 (INTER_NEAREST)
//   bilinear: fx = (dx + 0.5) * scale - 0.5          (INTER_LINEAR)
//             x0 = floor(fx); frac = fx - x0; clamp
//   dst = (1-fy)(1-fx)p00 + fx(1-fy)p10 + (1-fx)fyp01 + fxfyp11
//
// dtype support: U8 and F32 only (the two dtype families cv2 resize
// handles; everything else returns PD_INVALID_ARG -- no silent cast).
// Channel order is passthrough (H,W,C stays H,W,C; grayscale H,W stays
// H,W). 1-byte and 4-byte items from same base.

#include <cstdint>
#include <cstring>

#include "capi_impl.h"

namespace {

/* Per-channel linear interpolation on a uint8 image. Clamps the source
 * coordinate to [0, h-1] / [0, w-1] (opencv resize.cpp INTER_LINEAR
 * inside-loop clamp). */
inline uint8_t sample_clamp_u8(const uint8_t* src, int h, int w, int ch,
                               int src_h, int src_w) {
  return src[((src_h < 0 ? 0 : (src_h > h - 1 ? h - 1 : src_h)) * w +
              (src_w < 0 ? 0 : (src_w > w - 1 ? w - 1 : src_w))) *
                 ch];
}

inline float sample_clamp_f32(const float* src, int h, int w, int ch,
                              int src_h, int src_w) {
  return src[((src_h < 0 ? 0 : (src_h > h - 1 ? h - 1 : src_h)) * w +
              (src_w < 0 ? 0 : (src_w > w - 1 ? w - 1 : src_w))) *
                 ch];
}

}  // namespace

extern "C" {

/* Resize an image buffer. Two-pass protocol NOT needed: the output size
 * is computable (dst_h*dst_w*channels*itemsize), so a single call with
 * dst_cap >= that is sufficient; *dst_len reports the byte count.
 *
 * src/dst byte layouts are row-major (H,W,C) -- identical to cv2's,
 * so the Lua side can feed insight.Array bytes straight through.
 *
 * interp: 0 = nearest, 1 = bilinear. Anything else -> PD_INVALID_ARG.
 * dtype:  PD_DTYPE_UINT8 or PD_DTYPE_FLOAT32. Else PD_INVALID_ARG.
 */
pd_status pd_image_resize(const unsigned char* src, size_t src_len,
                          int src_h, int src_w, int channels, int dtype,
                          int dst_h, int dst_w, int interpolation,
                          unsigned char* out, size_t dst_cap,
                          size_t* dst_len) {
  PD_GUARD_BEGIN;
  if (src == nullptr || out == nullptr || dst_len == nullptr || src_h <= 0 ||
      src_w <= 0 || channels <= 0 || dst_h <= 0 || dst_w <= 0) {
    pd_set_error(PD_INVALID_ARG, "image_resize: bad dims/null ptr");
    return PD_INVALID_ARG;
  }
  if (interpolation != 0 && interpolation != 1) {
    pd_set_error(PD_INVALID_ARG, "image_resize: interp must be 0/1");
    return PD_INVALID_ARG;
  }
  if (dtype != PD_DTYPE_UINT8 && dtype != PD_DTYPE_FLOAT32) {
    pd_set_error(PD_INVALID_ARG,
                 "image_resize: dtype must be UINT8 or FLOAT32");
    return PD_INVALID_ARG;
  }
  const size_t itemsize = (dtype == PD_DTYPE_UINT8) ? 1 : 4;
  const size_t need =
      static_cast<size_t>(dst_h) * dst_w * channels * itemsize;
  if (dst_cap < need) {
    pd_set_error(PD_ERROR, "image_resize: output buffer too small");
    *dst_len = need;
    return PD_ERROR;
  }
  // src_len sanity: the caller's byte count must match H*W*C*itemsize.
  if (src_len != static_cast<size_t>(src_h) * src_w * channels * itemsize) {
    pd_set_error(PD_INVALID_ARG, "image_resize: src_len mismatch");
    return PD_INVALID_ARG;
  }

  if (dtype == PD_DTYPE_UINT8) {
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(src);
    uint8_t* dp = reinterpret_cast<uint8_t*>(out);
    const float sy_scale = static_cast<float>(src_h) / dst_h;
    const float sx_scale = static_cast<float>(src_w) / dst_w;
    if (interpolation == 0) {
      for (int dy = 0; dy < dst_h; ++dy) {
        const int sy = static_cast<int>(dy * sy_scale);
        const int sy_c = sy < 0 ? 0 : (sy > src_h - 1 ? src_h - 1 : sy);
        for (int dx = 0; dx < dst_w; ++dx) {
          const int sx = static_cast<int>(dx * sx_scale);
          const int sx_c = sx < 0 ? 0 : (sx > src_w - 1 ? src_w - 1 : sx);
          std::memcpy(out + (dy * dst_w + dx) * channels,
                      sp + (sy_c * src_w + sx_c) * channels,
                      static_cast<size_t>(channels));
        }
      }
    } else {
      for (int dy = 0; dy < dst_h; ++dy) {
        float fy = (dy + 0.5f) * sy_scale - 0.5f;
        if (fy < 0.f) fy = 0.f;
        const int y0 = static_cast<int>(fy);
        const int y1 = (y0 + 1 > src_h - 1) ? y0 : y0 + 1;
        const float wy1 = fy - y0;
        const float wy0 = 1.f - wy1;
        for (int dx = 0; dx < dst_w; ++dx) {
          float fx = (dx + 0.5f) * sx_scale - 0.5f;
          if (fx < 0.f) fx = 0.f;
          const int x0 = static_cast<int>(fx);
          const int x1 = (x0 + 1 > src_w - 1) ? x0 : x0 + 1;
          const float wx1 = fx - x0;
          const float wx0 = 1.f - wx1;
          uint8_t* dstpx = out + (dy * dst_w + dx) * channels;
          for (int c = 0; c < channels; ++c) {
            const float v =
                wy0 * (wx0 * sp[(y0 * src_w + x0) * channels + c] +
                       wx1 * sp[(y0 * src_w + x1) * channels + c]) +
                wy1 * (wx0 * sp[(y1 * src_w + x0) * channels + c] +
                       wx1 * sp[(y1 * src_w + x1) * channels + c]);
            // round-half-up -> int (opencv saturate_cast<uchar> rounds)
            dstpx[c] = static_cast<uint8_t>(v + 0.5f);
          }
        }
      }
    }
  } else {
    const float* sp = reinterpret_cast<const float*>(src);
    float* dp = reinterpret_cast<float*>(out);
    const float sy_scale = static_cast<float>(src_h) / dst_h;
    const float sx_scale = static_cast<float>(src_w) / dst_w;
    if (interpolation == 0) {
      for (int dy = 0; dy < dst_h; ++dy) {
        const int sy = static_cast<int>(dy * sy_scale);
        const int sy_c = sy < 0 ? 0 : (sy > src_h - 1 ? src_h - 1 : sy);
        for (int dx = 0; dx < dst_w; ++dx) {
          const int sx = static_cast<int>(dx * sx_scale);
          const int sx_c = sx < 0 ? 0 : (sx > src_w - 1 ? src_w - 1 : sx);
          for (int c = 0; c < channels; ++c) {
            dp[(dy * dst_w + dx) * channels + c] =
                sp[(sy_c * src_w + sx_c) * channels + c];
          }
        }
      }
    } else {
      for (int dy = 0; dy < dst_h; ++dy) {
        float fy = (dy + 0.5f) * sy_scale - 0.5f;
        if (fy < 0.f) fy = 0.f;
        const int y0 = static_cast<int>(fy);
        const int y1 = (y0 + 1 > src_h - 1) ? y0 : y0 + 1;
        const float wy1 = fy - y0;
        const float wy0 = 1.f - wy1;
        for (int dx = 0; dx < dst_w; ++dx) {
          float fx = (dx + 0.5f) * sx_scale - 0.5f;
          if (fx < 0.f) fx = 0.f;
          const int x0 = static_cast<int>(fx);
          const int x1 = (x0 + 1 > src_w - 1) ? x0 : x0 + 1;
          const float wx1 = fx - x0;
          const float wx0 = 1.f - wx1;
          float* dstpx = dp + (dy * dst_w + dx) * channels;
          for (int c = 0; c < channels; ++c) {
            dstpx[c] =
                wy0 * (wx0 * sp[(y0 * src_w + x0) * channels + c] +
                       wx1 * sp[(y0 * src_w + x1) * channels + c]) +
                wy1 * (wx0 * sp[(y1 * src_w + x0) * channels + c] +
                       wx1 * sp[(y1 * src_w + x1) * channels + c]);
          }
        }
      }
    }
  }
  *dst_len = need;
  return PD_OK;
  PD_GUARD_END(PD_ERROR);
}

}  // extern "C"