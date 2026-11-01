#include <cmath>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/distance.h>

namespace vt = vdb::test;

namespace {

// Dimensions chosen to hit every SIMD tail case: below one lane, exact
// multiples of 4/8/16 lanes, and odd remainders.
const vec<uint32_t> DIMS = {1, 3, 4, 7, 8, 15, 16, 17, 31, 64, 100, 128,
                            384, 768, 1024, 1536};

float rel_tol(double expected) {
  return float(1e-4 * s::max(1.0, s::abs(expected)));
}

}  // namespace

TEST(Distance, L2MatchesReference) {
  s::mt19937_64 rng(1);
  for (uint32_t dim : DIMS) {
    vec<float> a = vt::random_vector(rng, dim);
    vec<float> b = vt::random_vector(rng, dim);
    double want = vt::l2_sqr_ref(a, b);
    EXPECT_NEAR(vdb::l2_sqr(a, b), want, rel_tol(want)) << "dim=" << dim;
  }
}

TEST(Distance, DotMatchesReference) {
  s::mt19937_64 rng(2);
  for (uint32_t dim : DIMS) {
    vec<float> a = vt::random_vector(rng, dim);
    vec<float> b = vt::random_vector(rng, dim);
    double want = vt::dot_ref(a, b);
    EXPECT_NEAR(vdb::dot(a, b), want, rel_tol(want)) << "dim=" << dim;
  }
}

TEST(Distance, NormMatchesReference) {
  s::mt19937_64 rng(3);
  for (uint32_t dim : DIMS) {
    vec<float> a = vt::random_vector(rng, dim);
    double want = s::sqrt(vt::dot_ref(a, a));
    EXPECT_NEAR(vdb::norm(a), want, rel_tol(want)) << "dim=" << dim;
  }
}

TEST(Distance, MetricDispatchMatchesReference) {
  s::mt19937_64 rng(4);
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    for (uint32_t dim : DIMS) {
      vec<float> a = vt::random_vector(rng, dim);
      vec<float> b = vt::random_vector(rng, dim);
      double want = vt::distance_ref(m, a, b);
      EXPECT_NEAR(vdb::distance(m, a, b), want, rel_tol(want))
          << "metric=" << int(m) << " dim=" << dim;
    }
  }
}

TEST(Distance, SelfDistanceIsMinimal) {
  s::mt19937_64 rng(5);
  vec<float> a = vt::random_vector(rng, 128);
  EXPECT_FLOAT_EQ(vdb::l2_sqr(a, a), 0.0f);
  EXPECT_NEAR(vdb::distance(vdb::Metric::Cosine, a, a), 0.0f, 1e-6f);
}

TEST(Distance, Symmetric) {
  s::mt19937_64 rng(6);
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    vec<float> a = vt::random_vector(rng, 77);
    vec<float> b = vt::random_vector(rng, 77);
    EXPECT_FLOAT_EQ(vdb::distance(m, a, b), vdb::distance(m, b, a));
  }
}

TEST(Distance, TriangleInequalityForEuclidean) {
  s::mt19937_64 rng(7);
  for (int trial = 0; trial < 100; ++trial) {
    vec<float> a = vt::random_vector(rng, 32);
    vec<float> b = vt::random_vector(rng, 32);
    vec<float> c = vt::random_vector(rng, 32);
    float ab = s::sqrt(vdb::l2_sqr(a, b));
    float bc = s::sqrt(vdb::l2_sqr(b, c));
    float ac = s::sqrt(vdb::l2_sqr(a, c));
    EXPECT_LE(ac, ab + bc + 1e-4f);
  }
}

TEST(Distance, NormalizeProducesUnitVector) {
  s::mt19937_64 rng(8);
  for (uint32_t dim : DIMS) {
    vec<float> a = vt::random_vector(rng, dim);
    ASSERT_TRUE(vdb::normalize(a));
    EXPECT_NEAR(vdb::norm(a), 1.0f, 1e-5f) << "dim=" << dim;
  }
}

TEST(Distance, NormalizeRejectsZeroVector) {
  vec<float> z(16, 0.0f);
  EXPECT_FALSE(vdb::normalize(z));
  for (float x : z) {
    EXPECT_EQ(x, 0.0f);
  }
}

TEST(Distance, CosineWithZeroVectorIsOne) {
  vec<float> z(8, 0.0f);
  vec<float> a(8, 1.0f);
  EXPECT_FLOAT_EQ(vdb::distance(vdb::Metric::Cosine, z, a), 1.0f);
}

TEST(Distance, CosineIsScaleInvariant) {
  s::mt19937_64 rng(9);
  vec<float> a = vt::random_vector(rng, 64);
  vec<float> b = vt::random_vector(rng, 64);
  vec<float> b10 = b;
  for (float& x : b10) {
    x *= 10.0f;
  }
  EXPECT_NEAR(vdb::distance(vdb::Metric::Cosine, a, b),
              vdb::distance(vdb::Metric::Cosine, a, b10), 1e-5f);
}

TEST(Distance, PrenormalizedCosineMatchesCosine) {
  s::mt19937_64 rng(10);
  vec<float> a = vt::random_vector(rng, 300);
  vec<float> b = vt::random_vector(rng, 300);
  float want = vdb::distance(vdb::Metric::Cosine, a, b);
  vdb::normalize(a);
  vdb::normalize(b);
  EXPECT_NEAR(vdb::distance_prenormalized(vdb::Metric::Cosine, a, b), want,
              1e-5f);
}

// For unit vectors ||a-b||^2 = 2 - 2 a.b, so L2 and cosine rank identically.
TEST(Distance, UnitVectorL2EqualsTwiceCosine) {
  s::mt19937_64 rng(11);
  vec<float> a = vt::random_unit_vector(rng, 128);
  vec<float> b = vt::random_unit_vector(rng, 128);
  EXPECT_NEAR(vdb::l2_sqr(a, b),
              2.0f * vdb::distance_prenormalized(vdb::Metric::Cosine, a, b),
              1e-5f);
}

TEST(Distance, BatchMatchesSingle) {
  s::mt19937_64 rng(12);
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    for (uint32_t dim : {1u, 7u, 128u, 769u}) {
      size_t n = 37;
      vec<float> rows;
      for (size_t i = 0; i < n; ++i) {
        vec<float> v = vt::random_unit_vector(rng, dim);
        rows.insert(rows.end(), v.begin(), v.end());
      }
      vec<float> q = vt::random_unit_vector(rng, dim);
      vec<float> out(n);
      vdb::distance_batch(m, q, rows, dim, out);
      for (size_t i = 0; i < n; ++i) {
        s::span<const float> row(rows.data() + i * dim, dim);
        EXPECT_NEAR(out[i], vdb::distance_prenormalized(m, q, row), 1e-5f)
            << "metric=" << int(m) << " dim=" << dim << " row=" << i;
      }
    }
  }
}

TEST(Distance, BatchWithNoRows) {
  vec<float> q(4, 1.0f);
  vec<float> out;
  vdb::distance_batch(vdb::Metric::L2, q, {}, 4, out);
  SUCCEED();
}

TEST(Distance, UnalignedInputs) {
  s::mt19937_64 rng(13);
  vec<float> buf_a = vt::random_vector(rng, 130);
  vec<float> buf_b = vt::random_vector(rng, 130);
  // Offset by one float so the data is never 16/32/64-byte aligned.
  s::span<const float> a(buf_a.data() + 1, 129);
  s::span<const float> b(buf_b.data() + 1, 129);
  double want = vt::l2_sqr_ref(a, b);
  EXPECT_NEAR(vdb::l2_sqr(a, b), want, rel_tol(want));
}

TEST(Distance, LargeMagnitudesStayFinite) {
  vec<float> a(256, 1e15f);
  vec<float> b(256, -1e15f);
  EXPECT_TRUE(s::isfinite(vdb::distance(vdb::Metric::Cosine, a, b)));
  EXPECT_NEAR(vdb::distance(vdb::Metric::Cosine, a, b), 2.0f, 1e-4f);
}
