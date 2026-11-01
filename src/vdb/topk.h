#ifndef TOPK_H
#define TOPK_H

#include <vdb/types.h>

namespace vdb {

// Bounded max-heap keeping the k smallest distances seen.
struct TopK {
  explicit TopK(size_t k);

  // Inserts if fewer than k are held or distance < worst(). Ties with the
  // current worst are rejected so results are stable in insertion order.
  void push(InternalId id, float distance);

  // True when push(_, distance) would be kept. Lets callers skip work
  // (e.g. full-precision rescoring) for hopeless candidates.
  bool accepts(float distance) const;

  // Largest held distance; +infinity while size() < k.
  float worst() const;

  size_t size() const;
  size_t capacity() const;
  bool empty() const;

  // Hits ordered by ascending distance, ties by ascending id. Leaves the
  // heap empty.
  vec<Hit> take_sorted();

private:
  size_t mK;
  vec<Hit> mHeap;
};

}  // namespace vdb

#endif  // TOPK_H
