#ifndef FUSION_H
#define FUSION_H

#include <span>

#include <vdb/types.h>

namespace vdb {

struct Fused {
  str id;
  float score;

  bool operator==(const Fused&) const = default;
};

// Reciprocal rank fusion of ranked id lists (best first):
//   score(d) = sum_i weight_i / (k + rank_i(d)),   rank is 1-based
// A list that does not contain d contributes nothing. Missing weights
// default to 1; a duplicate id within one list counts at its best rank.
// Output: score desc, ties by id asc, truncated to `limit` (0 = no limit).
vec<Fused> rrf(s::span<const vec<str>> ranked_lists,
               size_t limit = 0,
               float k = 60.0f,
               s::span<const float> weights = {});

}  // namespace vdb

#endif  // FUSION_H
