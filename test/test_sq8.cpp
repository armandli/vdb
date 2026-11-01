#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/distance.h>
#include <vdb/sq8.h>

namespace vt = vdb::test;

namespace {

vdb::VectorStore random_store(size_t n, uint32_t dim, uint64_t seed) {
  s::mt19937_64 rng(seed);
  vdb::VectorStore store(dim);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(store.append(vt::random_unit_vector(rng, dim)).has_value());
  }
  return store;
}

}  // namespace

TEST(SQ8, TrainOnEmptyStoreFails) {
  vdb::VectorStore store(8);
  EXPECT_FALSE(vdb::SQ8Quantizer::train(store).has_value());
}

TEST(SQ8, BoundsAreExactMinMax) {
  vdb::VectorStore store(2);
  ASSERT_TRUE(store.append(vec<float>{-1.0f, 5.0f}).has_value());
  ASSERT_TRUE(store.append(vec<float>{3.0f, 2.0f}).has_value());
  ASSERT_TRUE(store.append(vec<float>{0.0f, 4.0f}).has_value());
  auto q = vdb::SQ8Quantizer::train(store);
  ASSERT_TRUE(q.has_value());
  EXPECT_EQ(q->dim(), 2u);
  EXPECT_FLOAT_EQ(q->lo()[0], -1.0f);
  EXPECT_FLOAT_EQ(q->hi()[0], 3.0f);
  EXPECT_FLOAT_EQ(q->lo()[1], 2.0f);
  EXPECT_FLOAT_EQ(q->hi()[1], 5.0f);
}

TEST(SQ8, EndpointsEncodeTo0And255) {
  vdb::VectorStore store(1);
  ASSERT_TRUE(store.append(vec<float>{-2.0f}).has_value());
  ASSERT_TRUE(store.append(vec<float>{6.0f}).has_value());
  auto q = vdb::SQ8Quantizer::train(store);
  ASSERT_TRUE(q.has_value());
  uint8_t code = 0;
  q->encode(vec<float>{-2.0f}, s::span(&code, 1));
  EXPECT_EQ(code, 0);
  q->encode(vec<float>{6.0f}, s::span(&code, 1));
  EXPECT_EQ(code, 255);
  q->encode(vec<float>{100.0f}, s::span(&code, 1));
  EXPECT_EQ(code, 255) << "values above hi clamp";
  q->encode(vec<float>{-100.0f}, s::span(&code, 1));
  EXPECT_EQ(code, 0) << "values below lo clamp";
}

TEST(SQ8, ReconstructionErrorWithinHalfStep) {
  vdb::VectorStore store = random_store(2000, 64, 1);
  auto q = vdb::SQ8Quantizer::train(store);
  ASSERT_TRUE(q.has_value());
  vec<uint8_t> code(64);
  vec<float> back(64);
  for (vdb::InternalId i = 0; i < store.size(); ++i) {
    s::span<const float> v = store.get(i);
    q->encode(v, code);
    q->decode(code, back);
    for (uint32_t d = 0; d < 64; ++d) {
      float step = (q->hi()[d] - q->lo()[d]) / 255.0f;
      EXPECT_LE(s::abs(back[d] - v[d]), step / 2 + 1e-6f)
          << "row " << i << " dim " << d;
    }
  }
}

TEST(SQ8, ConstantDimension) {
  vdb::VectorStore store(2);
  ASSERT_TRUE(store.append(vec<float>{7.0f, 1.0f}).has_value());
  ASSERT_TRUE(store.append(vec<float>{7.0f, 2.0f}).has_value());
  auto q = vdb::SQ8Quantizer::train(store);
  ASSERT_TRUE(q.has_value());
  vec<uint8_t> code(2);
  vec<float> back(2);
  q->encode(vec<float>{7.0f, 1.5f}, code);
  EXPECT_EQ(code[0], 0);
  q->decode(code, back);
  EXPECT_FLOAT_EQ(back[0], 7.0f);
}

TEST(SQ8, QuantileClipsOutliers) {
  s::mt19937_64 rng(2);
  vdb::VectorStore store(1);
  s::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(store.append(vec<float>{u(rng)}).has_value());
  }
  ASSERT_TRUE(store.append(vec<float>{1000.0f}).has_value());
  auto exact = vdb::SQ8Quantizer::train(store, 1.0f);
  auto clipped = vdb::SQ8Quantizer::train(store, 0.99f);
  ASSERT_TRUE(exact.has_value() and clipped.has_value());
  EXPECT_FLOAT_EQ(exact->hi()[0], 1000.0f);
  EXPECT_LT(clipped->hi()[0], 1.5f);
}

TEST(SQ8, AsymmetricDistanceApproximatesExact) {
  vdb::VectorStore store = random_store(1000, 128, 3);
  auto q = vdb::SQ8Quantizer::train(store);
  ASSERT_TRUE(q.has_value());
  s::mt19937_64 rng(4);
  vec<uint8_t> code(128);
  vec<float> back(128);
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    vec<float> query = vt::random_unit_vector(rng, 128);
    for (vdb::InternalId i = 0; i < 100; ++i) {
      q->encode(store.get(i), code);
      q->decode(code, back);
      float exact = vdb::distance_prenormalized(m, query, store.get(i));
      float approx = q->distance(m, query, code);
      // The asymmetric distance is exactly the distance to the decoded
      // vector, and close to the true one.
      EXPECT_NEAR(approx, vdb::distance_prenormalized(m, query, back), 1e-4f);
      EXPECT_NEAR(approx, exact, 0.05f) << "metric " << int(m);
    }
  }
}

// Ranking on codes then rescoring the shortlist must recover nearly all of
// the exact top-10: the "quantize to shortlist, rerank to rank" rule.
TEST(SQ8, ShortlistPlusRescoreRecall) {
  vdb::VectorStore store = random_store(5000, 64, 5);
  auto q = vdb::SQ8Quantizer::train(store);
  ASSERT_TRUE(q.has_value());
  vec<uint8_t> codes(store.size() * 64);
  for (vdb::InternalId i = 0; i < store.size(); ++i) {
    q->encode(store.get(i), s::span(codes.data() + size_t(i) * 64, 64));
  }
  s::mt19937_64 rng(6);
  double total = 0.0;
  for (int trial = 0; trial < 50; ++trial) {
    vec<float> query = vt::random_unit_vector(rng, 64);
    vec<s::pair<float, vdb::InternalId>> approx;
    for (vdb::InternalId i = 0; i < store.size(); ++i) {
      approx.emplace_back(
          q->distance(vdb::Metric::Cosine, query,
                      s::span(codes.data() + size_t(i) * 64, 64)),
          i);
    }
    s::partial_sort(approx.begin(), approx.begin() + 40, approx.end());
    vec<s::pair<float, vdb::InternalId>> rescored;
    for (int j = 0; j < 40; ++j) {
      vdb::InternalId id = approx[j].second;
      rescored.emplace_back(
          vdb::distance_prenormalized(vdb::Metric::Cosine, query,
                                      store.get(id)),
          id);
    }
    s::sort(rescored.begin(), rescored.end());
    vec<vdb::InternalId> got;
    for (int j = 0; j < 10; ++j) {
      got.push_back(rescored[j].second);
    }
    total += vt::recall(got, vt::brute_force_ids(vdb::Metric::Cosine,
                                                 store.rows(), 64, query, 10));
  }
  EXPECT_GE(total / 50.0, 0.98);
}
