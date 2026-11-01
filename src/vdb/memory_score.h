#ifndef MEMORY_SCORE_H
#define MEMORY_SCORE_H

#include <span>

#include <vdb/types.h>

namespace vdb {

// Generative-Agents style retrieval score for agent memory:
//   score = w_rel * norm(similarity) + w_rec * norm(decay^hours)
//         + w_imp * norm(importance)
// where norm() is min-max normalization over the candidate set.
struct MemoryScoreWeights {
  float relevance = 1.0f;
  float recency = 1.0f;
  float importance = 1.0f;
  double decay_per_hour = 0.995;

  bool operator==(const MemoryScoreWeights&) const = default;
};

struct MemoryCandidate {
  float similarity;     // higher is more similar
  double hours_since;   // since last access / creation; negative clamps to 0
  float importance;     // caller's scale, e.g. 1..10
};

// One score per candidate, same order. A term whose max == min over the
// candidates normalizes to 0 for everyone (it cannot affect the ranking),
// so the result never contains NaN.
vec<float> memory_scores(s::span<const MemoryCandidate> candidates,
                         const MemoryScoreWeights& weights);

}  // namespace vdb

#endif  // MEMORY_SCORE_H
