#include <vdb/flat_search.h>

namespace vdb {

vec<Hit> flat_search(
    const VectorStore& /*store*/,
    Metric /*metric*/,
    s::span<const float> /*query*/,
    size_t /*k*/,
    const IdFilter& /*allow*/)
{
  // TODO(M1)
  return {};
}

vec<Hit> flat_search_candidates(
    const VectorStore& /*store*/,
    Metric /*metric*/,
    s::span<const float> /*query*/,
    size_t /*k*/,
    const roaring::Roaring& /*candidates*/)
{
  // TODO(M1)
  return {};
}

}  // namespace vdb
