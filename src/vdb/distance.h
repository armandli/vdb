#ifndef DISTANCE_H
#define DISTANCE_H

#include <span>

#include <vdb/types.h>

namespace vdb {

// SIMD kernels (Google Highway, runtime dispatched). All spans of a call
// must have equal length; checked only by assert.

float l2_sqr(s::span<const float> a, s::span<const float> b);
float dot(s::span<const float> a, s::span<const float> b);
float norm(s::span<const float> a);

// Scales v to unit length in place. Returns false (v unchanged) when the
// norm is zero, since a zero vector has no direction for cosine.
bool normalize(s::span<float> v);

// Distance under `metric`, smaller is closer:
//   L2           -> l2_sqr(a, b)
//   InnerProduct -> -dot(a, b)
//   Cosine       -> 1 - dot(a, b) / (|a| |b|); 1 when either norm is zero
float distance(Metric metric, s::span<const float> a, s::span<const float> b);

// Same as distance() but assumes both inputs are already unit length for
// Cosine (1 - dot). This is what segments call after normalize-on-insert.
float distance_prenormalized(
    Metric metric,
    s::span<const float> a,
    s::span<const float> b);

// One query against `rows.size() / dim` contiguous row-major vectors.
// out.size() must equal the row count. Prenormalized semantics.
void distance_batch(
    Metric metric,
    s::span<const float> query,
    s::span<const float> rows,
    size_t dim,
    s::span<float> out);

}  // namespace vdb

#endif  // DISTANCE_H
