#ifndef FLAT_SEARCH_H
#define FLAT_SEARCH_H

#include <span>

#include <roaring/roaring.hh>

#include <vdb/types.h>
#include <vdb/vector_store.h>

namespace vdb {

// Exact k-NN by scanning every row. This is the ground-truth oracle for all
// approximate paths and must stay simple. Vectors in `store` (and `query`)
// are assumed prenormalized for Cosine.
//
// Returns min(k, #accepted) hits ordered by (distance, id).
vec<Hit> flat_search(
    const VectorStore& store,
    Metric metric,
    s::span<const float> query,
    size_t k,
    const IdFilter& allow = {});

// Exact k-NN over only the ids in `candidates` (pre-filter path).
vec<Hit> flat_search_candidates(
    const VectorStore& store,
    Metric metric,
    s::span<const float> query,
    size_t k,
    const roaring::Roaring& candidates);

}  // namespace vdb

#endif  // FLAT_SEARCH_H
