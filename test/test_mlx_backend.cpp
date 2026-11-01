#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/flat_search.h>
#include <vdb/mlx_backend.h>

namespace vt = vdb::test;

TEST(MlxBackend, Available) {
  EXPECT_TRUE(vdb::mlx_available());
}

TEST(MlxBackend, BatchMatchesFlatSearch) {
  s::mt19937_64 rng(1);
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    vdb::VectorStore store(64);
    for (int i = 0; i < 4000; ++i) {
      ASSERT_TRUE(store.append(vt::random_unit_vector(rng, 64)).has_value());
    }
    vec<float> queries;
    for (int q = 0; q < 32; ++q) {
      vec<float> v = vt::random_unit_vector(rng, 64);
      queries.insert(queries.end(), v.begin(), v.end());
    }
    auto batch = vdb::mlx_batch_search(store, m, queries, 10);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->size(), 32u);
    for (size_t q = 0; q < 32; ++q) {
      s::span<const float> query(queries.data() + q * 64, 64);
      vec<vdb::Hit> want = vdb::flat_search(store, m, query, 10);
      ASSERT_EQ((*batch)[q].size(), want.size());
      vec<vdb::InternalId> got_ids;
      vec<vdb::InternalId> want_ids;
      for (size_t i = 0; i < want.size(); ++i) {
        got_ids.push_back((*batch)[q][i].id);
        want_ids.push_back(want[i].id);
        EXPECT_NEAR((*batch)[q][i].distance, want[i].distance, 1e-3f);
      }
      // GPU float accumulation order differs; allow near-tie swaps.
      EXPECT_GE(vt::recall(got_ids, want_ids), 0.9) << "metric " << int(m);
    }
  }
}

TEST(MlxBackend, KLargerThanStore) {
  s::mt19937_64 rng(2);
  vdb::VectorStore store(8);
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(store.append(vt::random_unit_vector(rng, 8)).has_value());
  }
  vec<float> q = vt::random_unit_vector(rng, 8);
  auto batch = vdb::mlx_batch_search(store, vdb::Metric::L2, q, 50);
  ASSERT_TRUE(batch.has_value());
  ASSERT_EQ(batch->size(), 1u);
  EXPECT_EQ((*batch)[0].size(), 5u);
}

TEST(MlxBackend, RejectsRaggedQueries) {
  vdb::VectorStore store(8);
  ASSERT_TRUE(store.append(vec<float>(8, 1.0f)).has_value());
  vec<float> ragged(12, 1.0f);
  auto batch = vdb::mlx_batch_search(store, vdb::Metric::L2, ragged, 1);
  ASSERT_FALSE(batch.has_value());
  EXPECT_EQ(batch.error().code, vdb::ErrorCode::DimensionMismatch);
}
