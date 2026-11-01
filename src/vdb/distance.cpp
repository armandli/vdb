#include <vdb/distance.h>

// TODO(M1): implement with Google Highway runtime dispatch
// (hwy/foreach_target.h + HWY_EXPORT / HWY_DYNAMIC_DISPATCH).

namespace vdb {

float l2_sqr(s::span<const float> /*a*/, s::span<const float> /*b*/) {
  // TODO(M1)
  return 0.0f;
}

float dot(s::span<const float> /*a*/, s::span<const float> /*b*/) {
  // TODO(M1)
  return 0.0f;
}

float norm(s::span<const float> /*a*/) {
  // TODO(M1)
  return 0.0f;
}

bool normalize(s::span<float> /*v*/) {
  // TODO(M1)
  return false;
}

float distance(
    Metric /*metric*/,
    s::span<const float> /*a*/,
    s::span<const float> /*b*/)
{
  // TODO(M1)
  return 0.0f;
}

float distance_prenormalized(
    Metric /*metric*/,
    s::span<const float> /*a*/,
    s::span<const float> /*b*/)
{
  // TODO(M1)
  return 0.0f;
}

void distance_batch(
    Metric /*metric*/,
    s::span<const float> /*query*/,
    s::span<const float> /*rows*/,
    size_t /*dim*/,
    s::span<float> /*out*/)
{
  // TODO(M1)
}

}  // namespace vdb
