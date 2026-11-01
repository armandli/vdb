#ifndef SQ8_H
#define SQ8_H

#include <span>

#include <vdb/types.h>
#include <vdb/vector_store.h>

namespace vdb {

// Scalar 8-bit quantizer: per-dimension [lo, hi] mapped to 0..255.
// 4x smaller than float32; used to traverse/shortlist, always followed by
// full-precision rescoring of the shortlist.
struct SQ8Quantizer {
  // Learns per-dimension bounds from `store`. `quantile` < 1 clips the
  // bounds to the [1-q, q] quantiles so outliers don't waste levels;
  // quantile == 1 uses the exact min / max.
  static Result<SQ8Quantizer> train(const VectorStore& store,
                                    float quantile = 1.0f);

  uint32_t dim() const;
  s::span<const float> lo() const;
  s::span<const float> hi() const;

  // code[i] = round((clamp(v[i], lo, hi) - lo) / (hi - lo) * 255).
  // A dimension with hi == lo always encodes to 0 and decodes to lo.
  void encode(s::span<const float> v, s::span<uint8_t> code) const;
  void decode(s::span<const uint8_t> code, s::span<float> out) const;

  // Asymmetric distance: full-precision query vs. decoded code, with the
  // same semantics as distance_prenormalized(). Must not allocate.
  float distance(Metric metric,
                 s::span<const float> query,
                 s::span<const uint8_t> code) const;

private:
  vec<float> mLo;
  vec<float> mHi;
};

}  // namespace vdb

#endif  // SQ8_H
