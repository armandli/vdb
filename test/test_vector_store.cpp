#include <cstdint>
#include <fstream>
#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/vector_store.h>

namespace vt = vdb::test;

TEST(VectorStore, EmptyStore) {
  vdb::VectorStore store(8);
  EXPECT_EQ(store.dim(), 8u);
  EXPECT_EQ(store.size(), 0u);
  EXPECT_TRUE(store.empty());
  EXPECT_TRUE(store.rows().empty());
}

TEST(VectorStore, AppendAssignsDenseIds) {
  vdb::VectorStore store(3);
  for (vdb::InternalId i = 0; i < 5; ++i) {
    vec<float> v = {float(i), float(i) + 0.5f, -float(i)};
    auto id = store.append(v);
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(*id, i);
  }
  EXPECT_EQ(store.size(), 5u);
  for (vdb::InternalId i = 0; i < 5; ++i) {
    s::span<const float> v = store.get(i);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], float(i));
    EXPECT_EQ(v[1], float(i) + 0.5f);
    EXPECT_EQ(v[2], -float(i));
  }
}

TEST(VectorStore, RejectsWrongDimension) {
  vdb::VectorStore store(4);
  vec<float> v = {1.0f, 2.0f};
  auto id = store.append(v);
  ASSERT_FALSE(id.has_value());
  EXPECT_EQ(id.error().code, vdb::ErrorCode::DimensionMismatch);
  EXPECT_EQ(store.size(), 0u);
}

TEST(VectorStore, RowsAreContiguousAndAligned) {
  s::mt19937_64 rng(1);
  vdb::VectorStore store(17);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(store.append(vt::random_vector(rng, 17)).has_value());
  }
  EXPECT_EQ(reinterpret_cast<uintptr_t>(store.data()) %
                vdb::VectorStore::ALIGNMENT,
            0u);
  ASSERT_EQ(store.rows().size(), 1000u * 17u);
  for (vdb::InternalId i = 0; i < 1000; ++i) {
    EXPECT_EQ(store.get(i).data(), store.data() + size_t(i) * 17);
  }
}

TEST(VectorStore, GrowthPreservesData) {
  s::mt19937_64 rng(2);
  vdb::VectorStore store(5);
  vec<vec<float>> kept;
  for (int i = 0; i < 5000; ++i) {
    kept.push_back(vt::random_vector(rng, 5));
    ASSERT_TRUE(store.append(kept.back()).has_value());
  }
  for (vdb::InternalId i = 0; i < kept.size(); ++i) {
    s::span<const float> v = store.get(i);
    EXPECT_TRUE(s::equal(v.begin(), v.end(), kept[i].begin()));
  }
}

TEST(VectorStore, SetOverwrites) {
  vdb::VectorStore store(2);
  ASSERT_TRUE(store.append(vec<float>{1.0f, 2.0f}).has_value());
  ASSERT_TRUE(store.set(0, vec<float>{3.0f, 4.0f}).has_value());
  EXPECT_EQ(store.get(0)[0], 3.0f);
  EXPECT_EQ(store.get(0)[1], 4.0f);
  EXPECT_FALSE(store.set(1, vec<float>{0.0f, 0.0f}).has_value());
}

TEST(VectorStore, BytesTracksPayload) {
  vdb::VectorStore store(10);
  for (int i = 0; i < 7; ++i) {
    ASSERT_TRUE(store.append(vec<float>(10, 1.0f)).has_value());
  }
  EXPECT_EQ(store.bytes(), 7u * 10u * sizeof(float));
}

TEST(VectorStore, SaveLoadRoundTrip) {
  vt::TempDir dir;
  s::mt19937_64 rng(3);
  vdb::VectorStore store(33);
  for (int i = 0; i < 321; ++i) {
    ASSERT_TRUE(store.append(vt::random_vector(rng, 33)).has_value());
  }
  sf::path file = dir.path() / "vectors.bin";
  ASSERT_TRUE(store.save(file).has_value());

  auto loaded = vdb::VectorStore::load(file);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  ASSERT_EQ(loaded->dim(), 33u);
  ASSERT_EQ(loaded->size(), 321u);
  EXPECT_TRUE(s::equal(store.rows().begin(), store.rows().end(),
                       loaded->rows().begin()));
}

TEST(VectorStore, SaveLoadEmpty) {
  vt::TempDir dir;
  vdb::VectorStore store(4);
  ASSERT_TRUE(store.save(dir.path() / "v.bin").has_value());
  auto loaded = vdb::VectorStore::load(dir.path() / "v.bin");
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->size(), 0u);
  EXPECT_EQ(loaded->dim(), 4u);
}

TEST(VectorStore, LoadMissingFile) {
  vt::TempDir dir;
  auto loaded = vdb::VectorStore::load(dir.path() / "nope.bin");
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, vdb::ErrorCode::IOError);
}

namespace {

vec<char> read_all(const sf::path& p) {
  s::ifstream in(p, s::ios::binary);
  return vec<char>(s::istreambuf_iterator<char>(in), {});
}

void write_all(const sf::path& p, const vec<char>& bytes) {
  s::ofstream out(p, s::ios::binary | s::ios::trunc);
  out.write(bytes.data(), s::streamsize(bytes.size()));
}

}  // namespace

TEST(VectorStore, LoadRejectsBadMagic) {
  vt::TempDir dir;
  vdb::VectorStore store(4);
  ASSERT_TRUE(store.append(vec<float>(4, 1.0f)).has_value());
  sf::path file = dir.path() / "v.bin";
  ASSERT_TRUE(store.save(file).has_value());
  vec<char> bytes = read_all(file);
  bytes[0] ^= 0x5A;
  write_all(file, bytes);
  auto loaded = vdb::VectorStore::load(file);
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, vdb::ErrorCode::Corruption);
}

TEST(VectorStore, LoadRejectsFutureVersion) {
  vt::TempDir dir;
  vdb::VectorStore store(4);
  sf::path file = dir.path() / "v.bin";
  ASSERT_TRUE(store.save(file).has_value());
  vec<char> bytes = read_all(file);
  // Bytes 4..7 hold the little-endian u32 format version.
  bytes[7] = char(0x7F);
  write_all(file, bytes);
  auto loaded = vdb::VectorStore::load(file);
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, vdb::ErrorCode::VersionMismatch);
}

TEST(VectorStore, LoadRejectsFlippedDataByte) {
  vt::TempDir dir;
  s::mt19937_64 rng(4);
  vdb::VectorStore store(16);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(store.append(vt::random_vector(rng, 16)).has_value());
  }
  sf::path file = dir.path() / "v.bin";
  ASSERT_TRUE(store.save(file).has_value());
  vec<char> bytes = read_all(file);
  bytes[bytes.size() - 3] ^= 0x01;
  write_all(file, bytes);
  auto loaded = vdb::VectorStore::load(file);
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, vdb::ErrorCode::Corruption);
}

// Truncating anywhere must yield a clean error, never a crash or a store
// with garbage rows.
TEST(VectorStore, LoadRejectsEveryTruncation) {
  vt::TempDir dir;
  s::mt19937_64 rng(5);
  vdb::VectorStore store(8);
  for (int i = 0; i < 20; ++i) {
    ASSERT_TRUE(store.append(vt::random_vector(rng, 8)).has_value());
  }
  sf::path file = dir.path() / "v.bin";
  ASSERT_TRUE(store.save(file).has_value());
  vec<char> full = read_all(file);
  for (size_t len = 0; len < full.size(); len += 7) {
    write_all(file, vec<char>(full.begin(), full.begin() + long(len)));
    auto loaded = vdb::VectorStore::load(file);
    EXPECT_FALSE(loaded.has_value()) << "len=" << len;
  }
}
