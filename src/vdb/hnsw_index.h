#ifndef HNSW_INDEX_H
#define HNSW_INDEX_H

#include <span>

#include <vdb/types.h>
#include <vdb/vector_store.h>

namespace vdb {

struct HnswParams {
  uint32_t m = 16;                 // max links per node above layer 0
  uint32_t ef_construction = 100;  // build beam width
  uint32_t ef_search = 100;        // default query beam width
  uint64_t seed = 42;              // level generator seed (deterministic)

  bool operator==(const HnswParams&) const = default;
};

// Hierarchical navigable small world graph over a VectorStore.
//
// Layer 0 allows 2*m links, upper layers m. Levels are drawn as
// floor(-ln(U) / ln(m)). Neighbor selection uses the HNSW heuristic
// (keep a candidate only if it is closer to the base than to every
// already-kept neighbor).
//
// The index stores ids and links only; vectors are read from `store`,
// which must outlive the index. Inserts are single-writer; searches may run
// concurrently with each other but not with insert (segments guarantee
// this by indexing only sealed, immutable data).
struct HnswIndex {
  HnswIndex(const VectorStore& store, Metric metric, HnswParams params = {});
  ~HnswIndex();
  HnswIndex(HnswIndex&&) noexcept;
  HnswIndex& operator=(HnswIndex&&) noexcept;

  // Links store row `id` into the graph. ids may be inserted in any order
  // but each at most once.
  void insert(InternalId id);

  // Inserts rows [0, store.size()) using `threads` workers (0 = hardware
  // concurrency). Result quality must match sequential insert.
  void build(size_t threads = 0);

  // Approximate k-NN. ef is raised to at least k; ef == 0 uses
  // params().ef_search. Points rejected by `allow` are still traversed
  // (as routing waypoints) but never returned. Ordered by (distance, id).
  vec<Hit> search(
      s::span<const float> query,
      size_t k,
      size_t ef = 0,
      const IdFilter& allow = {}) const;

  size_t size() const;
  const HnswParams& params() const;
  Metric metric() const;
  opt<InternalId> entry_point() const;
  uint32_t max_level() const;
  uint32_t level_of(InternalId id) const;
  // Neighbors of `id` on `level` (empty when level > level_of(id)).
  s::span<const InternalId> neighbors(InternalId id, uint32_t level) const;

  // File format: magic "VDBH", u32 format version, params, metric, entry
  // point, then per-node level + links, crc32c trailer.
  Status save(const sf::path& path) const;
  static Result<HnswIndex> load(const sf::path& path, const VectorStore& store);

private:
  struct Impl;
  uptr<Impl> mImpl;
};

}  // namespace vdb

#endif  // HNSW_INDEX_H
