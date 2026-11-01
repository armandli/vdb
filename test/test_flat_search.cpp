#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/distance.h>
#include <vdb/flat_search.h>

namespace vt = vdb::test;

namespace {

vdb::VectorStore make_store(
    s::mt19937_64& rng,
    size_t n,
    uint32_t dim,
    bool unit)
{
  vdb::VectorStore store(dim);
  for (size_t i = 0; i < n; ++i) {
    vec<float> v =
        unit ? vt::random_unit_vector(rng, dim) : vt::random_vector(rng, dim);
    EXPECT_TRUE(store.append(v).has_value());
  }
  return store;
}

vec<vdb::InternalId> ids_of(const vec<vdb::Hit>& hits) {
  vec<vdb::InternalId> ids;
  for (const vdb::Hit& h : hits) {
    ids.push_back(h.id);
  }
  return ids;
}

}  // namespace

TEST(FlatSearch, EmptyStore) {
  vdb::VectorStore store(4);
  vec<float> q(4, 1.0f);
  EXPECT_TRUE(vdb::flat_search(store, vdb::Metric::L2, q, 10).empty());
}

TEST(FlatSearch, MatchesReferenceForEveryMetric) {
  s::mt19937_64 rng(1);
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    vdb::VectorStore store = make_store(rng, 2000, 48, true);
    for (int trial = 0; trial < 10; ++trial) {
      vec<float> q = vt::random_unit_vector(rng, 48);
      vec<vdb::Hit> hits = vdb::flat_search(store, m, q, 10);
      EXPECT_EQ(ids_of(hits),
                vt::brute_force_ids(m, store.rows(), 48, q, 10))
          << "metric=" << int(m);
    }
  }
}

TEST(FlatSearch, ResultsSortedWithCorrectDistances) {
  s::mt19937_64 rng(2);
  vdb::VectorStore store = make_store(rng, 500, 16, false);
  vec<float> q = vt::random_vector(rng, 16);
  vec<vdb::Hit> hits = vdb::flat_search(store, vdb::Metric::L2, q, 50);
  ASSERT_EQ(hits.size(), 50u);
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_NEAR(hits[i].distance,
                vt::l2_sqr_ref(store.get(hits[i].id), q), 1e-3);
    if (i > 0) {
      EXPECT_LE(hits[i - 1].distance, hits[i].distance);
    }
  }
}

TEST(FlatSearch, KLargerThanN) {
  s::mt19937_64 rng(3);
  vdb::VectorStore store = make_store(rng, 7, 4, false);
  vec<float> q = vt::random_vector(rng, 4);
  vec<vdb::Hit> hits = vdb::flat_search(store, vdb::Metric::L2, q, 100);
  EXPECT_EQ(hits.size(), 7u);
}

TEST(FlatSearch, KZero) {
  s::mt19937_64 rng(4);
  vdb::VectorStore store = make_store(rng, 7, 4, false);
  vec<float> q = vt::random_vector(rng, 4);
  EXPECT_TRUE(vdb::flat_search(store, vdb::Metric::L2, q, 0).empty());
}

TEST(FlatSearch, ExactMatchComesFirst) {
  s::mt19937_64 rng(5);
  vdb::VectorStore store = make_store(rng, 1000, 32, true);
  vec<float> q(store.get(123).begin(), store.get(123).end());
  vec<vdb::Hit> hits = vdb::flat_search(store, vdb::Metric::Cosine, q, 1);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 123u);
  EXPECT_NEAR(hits[0].distance, 0.0f, 1e-5f);
}

TEST(FlatSearch, AllowFilterExcludesIds) {
  s::mt19937_64 rng(6);
  vdb::VectorStore store = make_store(rng, 1000, 16, false);
  vec<float> q = vt::random_vector(rng, 16);
  vdb::IdFilter even = [](vdb::InternalId id) { return id % 2 == 0; };
  vec<vdb::Hit> hits = vdb::flat_search(store, vdb::Metric::L2, q, 20, even);
  ASSERT_EQ(hits.size(), 20u);
  for (const vdb::Hit& h : hits) {
    EXPECT_EQ(h.id % 2, 0u);
  }
}

TEST(FlatSearch, FilterRejectingEverything) {
  s::mt19937_64 rng(7);
  vdb::VectorStore store = make_store(rng, 100, 8, false);
  vec<float> q = vt::random_vector(rng, 8);
  vdb::IdFilter none = [](vdb::InternalId) { return false; };
  EXPECT_TRUE(vdb::flat_search(store, vdb::Metric::L2, q, 5, none).empty());
}

TEST(FlatSearch, CandidatesMatchesFilteredScan) {
  s::mt19937_64 rng(8);
  vdb::VectorStore store = make_store(rng, 3000, 24, true);
  roaring::Roaring candidates;
  for (uint32_t i = 0; i < 3000; i += 37) {
    candidates.add(i);
  }
  vdb::IdFilter allow = [&](vdb::InternalId id) {
    return candidates.contains(id);
  };
  for (int trial = 0; trial < 5; ++trial) {
    vec<float> q = vt::random_unit_vector(rng, 24);
    EXPECT_EQ(vdb::flat_search_candidates(store, vdb::Metric::Cosine, q, 10,
                                          candidates),
              vdb::flat_search(store, vdb::Metric::Cosine, q, 10, allow));
  }
}

TEST(FlatSearch, CandidatesOutOfRangeIgnored) {
  s::mt19937_64 rng(9);
  vdb::VectorStore store = make_store(rng, 10, 4, false);
  roaring::Roaring candidates;
  candidates.add(3);
  candidates.add(1000);
  vec<float> q = vt::random_vector(rng, 4);
  vec<vdb::Hit> hits =
      vdb::flat_search_candidates(store, vdb::Metric::L2, q, 5, candidates);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 3u);
}

TEST(FlatSearch, DuplicateVectorsTieBrokenById) {
  vdb::VectorStore store(2);
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(store.append(vec<float>{1.0f, 0.0f}).has_value());
  }
  vec<float> q = {1.0f, 0.0f};
  vec<vdb::Hit> hits = vdb::flat_search(store, vdb::Metric::L2, q, 3);
  EXPECT_EQ(ids_of(hits), (vec<vdb::InternalId>{0, 1, 2}));
}
