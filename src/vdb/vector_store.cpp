#include <vdb/vector_store.h>

namespace vdb {

VectorStore::VectorStore(uint32_t dim)
  : mDim(dim)
{
  // TODO(M1)
}

uint32_t VectorStore::dim() const {
  return mDim;
}

size_t VectorStore::size() const {
  return mSize;
}

bool VectorStore::empty() const {
  return mSize == 0;
}

size_t VectorStore::bytes() const {
  // TODO(M1)
  return 0;
}

Result<InternalId> VectorStore::append(s::span<const float> /*v*/) {
  // TODO(M1)
  return make_error(ErrorCode::Unsupported, "VectorStore::append");
}

Status VectorStore::set(InternalId /*id*/, s::span<const float> /*v*/) {
  // TODO(M1)
  return make_error(ErrorCode::Unsupported, "VectorStore::set");
}

s::span<const float> VectorStore::get(InternalId /*id*/) const {
  // TODO(M1)
  return {};
}

s::span<float> VectorStore::get_mut(InternalId /*id*/) {
  // TODO(M1)
  return {};
}

s::span<const float> VectorStore::rows() const {
  // TODO(M1)
  return {};
}

const float* VectorStore::data() const {
  return mData.get();
}

void VectorStore::reserve(size_t /*n*/) {
  // TODO(M1): grow mData / mCapacity, keep VectorStore::ALIGNMENT
  (void)mCapacity;
}

Status VectorStore::save(const sf::path& /*path*/) const {
  // TODO(M1)
  return make_error(ErrorCode::Unsupported, "VectorStore::save");
}

Result<VectorStore> VectorStore::load(const sf::path& /*path*/) {
  // TODO(M1)
  return make_error(ErrorCode::Unsupported, "VectorStore::load");
}

}  // namespace vdb
