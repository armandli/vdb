#ifndef MLX_BACKEND_H
#define MLX_BACKEND_H

#include <span>

#include <vdb/types.h>
#include <vdb/vector_store.h>

namespace vdb {

// Apple MLX (Metal GPU, unified memory) batch backend. Only built when
// VDB_HAS_MLX is defined. It wins on throughput, not single-query latency:
// use it for many queries at once (ground truth, bulk evaluation, batch
// search API) and k-means style training, never in the per-query path.

// True when a Metal GPU device is usable at runtime.
bool mlx_available();

// Exact k-NN for nq = queries.size() / store.dim() row-major queries,
// computed as one GEMM (||q||^2 + ||x||^2 - 2 q.x for L2) plus a
// per-row top-k. Results must equal flat_search() up to float rounding
// ties. Prenormalized semantics for Cosine.
Result<vec<vec<Hit>>> mlx_batch_search(const VectorStore& store,
                                       Metric metric,
                                       s::span<const float> queries,
                                       size_t k);

}  // namespace vdb

#endif  // MLX_BACKEND_H
