#include <deque>
#include <fstream>
#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/flat_search.h>
#include <vdb/hnsw_index.h>

namespace vt = vdb::test;

namespace {

constexpr uint32_t DIM = 32;

vdb::VectorStore clustered_store(size_t n, uint64_t seed, bool unit) {
  s::mt19937_64 rng(seed);
  vec<float> rows = vt::clustered_dataset(rng, n, DIM, 20);
  vdb::VectorStore store(DIM);
  for (size_t i = 0; i < n; ++i) {
    s::span<float> v(rows.data() + i * DIM, DIM);
    if (unit) {
      vt::normalize_ref(v);
    }
    EXPECT_TRUE(store.append(v).has_value());
  }
  return store;
}

vec<vec<float>> queries(size_t n, uint64_t seed, bool unit) {
  s::mt19937_64 rng(seed);
  vec<float> rows = vt::clustered_dataset(rng, n, DIM, 20);
  vec<vec<float>> qs;
  for (size_t i = 0; i < n; ++i) {
    vec<float> q(rows.begin() + long(i * DIM), rows.begin() + long((i + 1) * DIM));
    if (unit) {
      vt::normalize_ref(q);
    }
    qs.push_back(s::move(q));
  }
  return qs;
}

vec<vdb::InternalId> ids_of(const vec<vdb::Hit>& hits) {
  vec<vdb::InternalId> ids;
  for (const vdb::Hit& h : hits) {
    ids.push_back(h.id);
  }
  return ids;
}

double mean_recall(
    const vdb::HnswIndex& index,
    const vdb::VectorStore& store,
    vdb::Metric metric,
    const vec<vec<float>>& qs,
    size_t k,
    size_t ef,
    const vdb::IdFilter& allow = {})
{
  double sum = 0.0;
  for (const vec<float>& q : qs) {
    vec<vdb::Hit> truth = vdb::flat_search(store, metric, q, k, allow);
    vec<vdb::Hit> got = index.search(q, k, ef, allow);
    sum += vt::recall(ids_of(got), ids_of(truth));
  }
  return sum / double(qs.size());
}

vdb::HnswIndex build_sequential(
    const vdb::VectorStore& store,
    vdb::Metric metric,
    vdb::HnswParams params = {})
{
  vdb::HnswIndex index(store, metric, params);
  for (vdb::InternalId i = 0; i < store.size(); ++i) {
    index.insert(i);
  }
  return index;
}

}  // namespace

TEST(Hnsw, EmptyIndex) {
  vdb::VectorStore store(DIM);
  vdb::HnswIndex index(store, vdb::Metric::L2);
  EXPECT_EQ(index.size(), 0u);
  EXPECT_FALSE(index.entry_point().has_value());
  vec<float> q(DIM, 0.5f);
  EXPECT_TRUE(index.search(q, 10).empty());
}

TEST(Hnsw, SinglePoint) {
  vdb::VectorStore store(DIM);
  ASSERT_TRUE(store.append(vec<float>(DIM, 1.0f)).has_value());
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  EXPECT_EQ(index.entry_point(), 0u);
  vec<vdb::Hit> hits = index.search(vec<float>(DIM, 0.0f), 5);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
}

TEST(Hnsw, FewerPointsThanDegree) {
  vdb::VectorStore store = clustered_store(5, 1, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  for (const vec<float>& q : queries(10, 2, false)) {
    EXPECT_EQ(ids_of(index.search(q, 5)),
              ids_of(vdb::flat_search(store, vdb::Metric::L2, q, 5)));
  }
}

TEST(Hnsw, KLargerThanSize) {
  vdb::VectorStore store = clustered_store(50, 3, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vec<vdb::Hit> hits = index.search(queries(1, 4, false)[0], 100, 200);
  EXPECT_EQ(hits.size(), 50u);
}

TEST(Hnsw, ResultsSortedAndDistancesExact) {
  vdb::VectorStore store = clustered_store(2000, 5, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vec<float> q = queries(1, 6, false)[0];
  vec<vdb::Hit> hits = index.search(q, 20);
  ASSERT_EQ(hits.size(), 20u);
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_NEAR(hits[i].distance, vt::l2_sqr_ref(store.get(hits[i].id), q),
                1e-2);
    if (i > 0) {
      EXPECT_LE(hits[i - 1].distance, hits[i].distance);
    }
  }
}

// The milestone criterion: recall@10 >= 0.95 against brute force.
TEST(Hnsw, RecallAgainstBruteForceL2) {
  vdb::VectorStore store = clustered_store(5000, 7, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  double r = mean_recall(index, store, vdb::Metric::L2,
                         queries(200, 8, false), 10, 100);
  EXPECT_GE(r, 0.95);
}

TEST(Hnsw, RecallAgainstBruteForceCosine) {
  vdb::VectorStore store = clustered_store(5000, 9, true);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::Cosine);
  double r = mean_recall(index, store, vdb::Metric::Cosine,
                         queries(200, 10, true), 10, 100);
  EXPECT_GE(r, 0.95);
}

TEST(Hnsw, RecallAtOneForStoredVectors) {
  vdb::VectorStore store = clustered_store(3000, 11, true);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::Cosine);
  size_t found = 0;
  for (vdb::InternalId i = 0; i < 3000; i += 30) {
    vec<vdb::Hit> hits = index.search(store.get(i), 1, 64);
    found += (not hits.empty() and hits[0].distance <= 1e-5f);
  }
  EXPECT_GE(found, 99u);
}

// Recall must not drop as the beam widens; a drop means the candidate heap
// or visited set is wrong. This test catches most search-loop bugs.
TEST(Hnsw, RecallMonotoneInEf) {
  vdb::VectorStore store = clustered_store(4000, 12, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vec<vec<float>> qs = queries(150, 13, false);
  double prev = 0.0;
  for (size_t ef : {10u, 20u, 40u, 80u, 160u, 320u}) {
    double r = mean_recall(index, store, vdb::Metric::L2, qs, 10, ef);
    EXPECT_GE(r + 0.005, prev) << "ef=" << ef;
    prev = r;
  }
  EXPECT_GE(prev, 0.99);
}

TEST(Hnsw, DegreeAndLevelInvariants) {
  vdb::HnswParams params;
  params.m = 8;
  vdb::VectorStore store = clustered_store(3000, 14, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2, params);
  ASSERT_EQ(index.size(), 3000u);
  for (vdb::InternalId id = 0; id < 3000; ++id) {
    uint32_t top = index.level_of(id);
    EXPECT_LE(top, index.max_level());
    for (uint32_t level = 0; level <= top; ++level) {
      s::span<const vdb::InternalId> nbrs = index.neighbors(id, level);
      EXPECT_LE(nbrs.size(), level == 0 ? 2 * params.m : params.m);
      uset<vdb::InternalId> seen;
      for (vdb::InternalId n : nbrs) {
        EXPECT_LT(n, 3000u);
        EXPECT_NE(n, id) << "self loop";
        EXPECT_TRUE(seen.insert(n).second) << "duplicate link";
        EXPECT_GE(index.level_of(n), level) << "link to absent level";
      }
    }
    EXPECT_TRUE(index.neighbors(id, top + 1).empty());
  }
}

TEST(Hnsw, LayerZeroReachableFromEntryPoint) {
  vdb::VectorStore store = clustered_store(3000, 15, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  ASSERT_TRUE(index.entry_point().has_value());
  vec<bool> seen(3000, false);
  s::deque<vdb::InternalId> todo{*index.entry_point()};
  seen[*index.entry_point()] = true;
  size_t reached = 1;
  while (not todo.empty()) {
    vdb::InternalId id = todo.front();
    todo.pop_front();
    for (vdb::InternalId n : index.neighbors(id, 0)) {
      if (not seen[n]) {
        seen[n] = true;
        ++reached;
        todo.push_back(n);
      }
    }
  }
  EXPECT_EQ(reached, 3000u);
}

TEST(Hnsw, LevelDistributionIsGeometric) {
  vdb::VectorStore store = clustered_store(5000, 16, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  size_t upper = 0;
  for (vdb::InternalId id = 0; id < 5000; ++id) {
    upper += index.level_of(id) >= 1;
  }
  // P(level >= 1) = 1/m = 6.25% for m = 16.
  double frac = double(upper) / 5000.0;
  EXPECT_GT(frac, 0.04);
  EXPECT_LT(frac, 0.09);
}

TEST(Hnsw, SequentialBuildIsDeterministic) {
  vdb::VectorStore store = clustered_store(1000, 17, false);
  vdb::HnswIndex a = build_sequential(store, vdb::Metric::L2);
  vdb::HnswIndex b = build_sequential(store, vdb::Metric::L2);
  EXPECT_EQ(a.entry_point(), b.entry_point());
  for (vdb::InternalId id = 0; id < 1000; ++id) {
    ASSERT_EQ(a.level_of(id), b.level_of(id));
    s::span<const vdb::InternalId> na = a.neighbors(id, 0);
    s::span<const vdb::InternalId> nb = b.neighbors(id, 0);
    EXPECT_TRUE(s::equal(na.begin(), na.end(), nb.begin(), nb.end()));
  }
}

TEST(Hnsw, ParallelBuildRecall) {
  vdb::VectorStore store = clustered_store(5000, 18, false);
  vdb::HnswIndex index(store, vdb::Metric::L2);
  index.build(4);
  EXPECT_EQ(index.size(), 5000u);
  double r = mean_recall(index, store, vdb::Metric::L2,
                         queries(200, 19, false), 10, 100);
  EXPECT_GE(r, 0.95);
}

TEST(Hnsw, IdenticalVectors) {
  vdb::VectorStore store(DIM);
  for (int i = 0; i < 300; ++i) {
    ASSERT_TRUE(store.append(vec<float>(DIM, 0.25f)).has_value());
  }
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vec<vdb::Hit> hits = index.search(vec<float>(DIM, 0.25f), 10);
  EXPECT_EQ(hits.size(), 10u);
  for (const vdb::Hit& h : hits) {
    EXPECT_FLOAT_EQ(h.distance, 0.0f);
  }
}

TEST(Hnsw, FarOutlierIsFound) {
  vdb::VectorStore store = clustered_store(2000, 20, false);
  vec<float> outlier(DIM, 1000.0f);
  auto id = store.append(outlier);
  ASSERT_TRUE(id.has_value());
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vec<vdb::Hit> hits = index.search(outlier, 1);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, *id);
}

TEST(Hnsw, AllowFilterInline) {
  vdb::VectorStore store = clustered_store(4000, 21, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vdb::IdFilter half = [](vdb::InternalId id) { return id % 2 == 0; };
  vec<vec<float>> qs = queries(100, 22, false);
  for (const vec<float>& q : qs) {
    for (const vdb::Hit& h : index.search(q, 10, 100, half)) {
      EXPECT_EQ(h.id % 2, 0u);
    }
  }
  EXPECT_GE(mean_recall(index, store, vdb::Metric::L2, qs, 10, 100, half),
            0.9);
}

TEST(Hnsw, AllowFilterRejectingAll) {
  vdb::VectorStore store = clustered_store(500, 23, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  vdb::IdFilter none = [](vdb::InternalId) { return false; };
  EXPECT_TRUE(index.search(queries(1, 24, false)[0], 10, 50, none).empty());
}

TEST(Hnsw, SaveLoadRoundTrip) {
  vt::TempDir dir;
  vdb::VectorStore store = clustered_store(2000, 25, true);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::Cosine);
  sf::path file = dir.path() / "graph.bin";
  ASSERT_TRUE(index.save(file).has_value());
  auto loaded = vdb::HnswIndex::load(file, store);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_EQ(loaded->size(), index.size());
  EXPECT_EQ(loaded->params(), index.params());
  EXPECT_EQ(loaded->metric(), vdb::Metric::Cosine);
  for (const vec<float>& q : queries(50, 26, true)) {
    EXPECT_EQ(loaded->search(q, 10), index.search(q, 10));
  }
}

TEST(Hnsw, LoadRejectsCorruption) {
  vt::TempDir dir;
  vdb::VectorStore store = clustered_store(500, 27, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  sf::path file = dir.path() / "graph.bin";
  ASSERT_TRUE(index.save(file).has_value());
  vec<char> bytes;
  {
    s::ifstream in(file, s::ios::binary);
    bytes.assign(s::istreambuf_iterator<char>(in), {});
  }
  bytes[bytes.size() / 2] ^= 0x10;
  {
    s::ofstream out(file, s::ios::binary | s::ios::trunc);
    out.write(bytes.data(), s::streamsize(bytes.size()));
  }
  auto loaded = vdb::HnswIndex::load(file, store);
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, vdb::ErrorCode::Corruption);
}

TEST(Hnsw, LoadRejectsMismatchedStore) {
  vt::TempDir dir;
  vdb::VectorStore store = clustered_store(500, 28, false);
  vdb::HnswIndex index = build_sequential(store, vdb::Metric::L2);
  sf::path file = dir.path() / "graph.bin";
  ASSERT_TRUE(index.save(file).has_value());
  vdb::VectorStore smaller = clustered_store(100, 28, false);
  EXPECT_FALSE(vdb::HnswIndex::load(file, smaller).has_value());
}
