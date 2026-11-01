#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <span>

#include <vdb/types.h>

namespace vdb {

// Row-major N x dim float32 storage. Row i is InternalId i.
//
// Invariants:
//  - data() is 64-byte aligned and rows are contiguous with stride dim(),
//    so distance_batch() can stream any id range directly.
//  - get() spans stay valid until the next append() (growth may move data).
struct VectorStore {
  static constexpr size_t ALIGNMENT = 64;

  explicit VectorStore(uint32_t dim);

  uint32_t dim() const;
  size_t size() const;
  bool empty() const;
  size_t bytes() const;

  // Copies v (size must equal dim()) and returns its id (== old size()).
  Result<InternalId> append(s::span<const float> v);
  Status set(InternalId id, s::span<const float> v);

  s::span<const float> get(InternalId id) const;
  s::span<float> get_mut(InternalId id);
  s::span<const float> rows() const;
  const float* data() const;

  void reserve(size_t n);

  // File format: magic "VDBV", u32 format version, u32 dim, u64 count,
  // u32 crc32c of the row bytes, then the rows. load() rejects bad magic,
  // version, truncation and checksum mismatch.
  Status save(const sf::path& path) const;
  static Result<VectorStore> load(const sf::path& path);

private:
  uint32_t mDim;
  size_t mSize = 0;
  size_t mCapacity = 0;
  uptr<float[], void (*)(float*)> mData{nullptr, nullptr};
};

}  // namespace vdb

#endif  // VECTOR_STORE_H
