#include <vdb/topk.h>

namespace vdb {

TopK::TopK(size_t k)
  : mK(k)
{
  // TODO(M1)
}

void TopK::push(InternalId /*id*/, float /*distance*/) {
  // TODO(M1)
}

bool TopK::accepts(float /*distance*/) const {
  // TODO(M1)
  return false;
}

float TopK::worst() const {
  // TODO(M1)
  return 0.0f;
}

size_t TopK::size() const {
  return mHeap.size();
}

size_t TopK::capacity() const {
  return mK;
}

bool TopK::empty() const {
  return mHeap.empty();
}

vec<Hit> TopK::take_sorted() {
  // TODO(M1)
  return {};
}

}  // namespace vdb
