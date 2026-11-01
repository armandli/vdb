#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <vdb/memory_score.h>

TEST(MemoryScore, EmptyInput) {
  EXPECT_TRUE(vdb::memory_scores({}, {}).empty());
}

TEST(MemoryScore, SingleCandidateIsZeroNotNan) {
  vec<vdb::MemoryCandidate> c = {{0.9f, 3.0, 5.0f}};
  vec<float> out = vdb::memory_scores(c, {});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(MemoryScore, MinMaxNormalizedSum) {
  // similarity:  0.2 .. 1.0 -> 0, 0.5, 1
  // recency:     0.995^h for h = 0, 100, 1000 -> normalized 1, ~0.6, 0
  // importance:  1, 10, 5.5 -> 0, 1, 0.5
  vec<vdb::MemoryCandidate> c = {
      {0.2f, 0.0, 1.0f},
      {0.6f, 100.0, 10.0f},
      {1.0f, 1000.0, 5.5f},
  };
  vec<float> out = vdb::memory_scores(c, {});
  ASSERT_EQ(out.size(), 3u);
  double r0 = 1.0;
  double r1 = s::pow(0.995, 100.0);
  double r2 = s::pow(0.995, 1000.0);
  double rec1 = (r1 - r2) / (r0 - r2);
  EXPECT_NEAR(out[0], 0.0 + 1.0 + 0.0, 1e-5);
  EXPECT_NEAR(out[1], 0.5 + rec1 + 1.0, 1e-5);
  EXPECT_NEAR(out[2], 1.0 + 0.0 + 0.5, 1e-5);
}

TEST(MemoryScore, WeightsScaleTerms) {
  vec<vdb::MemoryCandidate> c = {
      {1.0f, 500.0, 1.0f},
      {0.0f, 0.0, 1.0f},
  };
  vdb::MemoryScoreWeights relevance_only{1.0f, 0.0f, 0.0f};
  vdb::MemoryScoreWeights recency_only{0.0f, 1.0f, 0.0f};
  vec<float> a = vdb::memory_scores(c, relevance_only);
  vec<float> b = vdb::memory_scores(c, recency_only);
  EXPECT_GT(a[0], a[1]);
  EXPECT_GT(b[1], b[0]);
}

TEST(MemoryScore, EqualTermContributesNothing) {
  vec<vdb::MemoryCandidate> c = {
      {0.1f, 5.0, 7.0f},
      {0.9f, 5.0, 7.0f},
  };
  vec<float> out = vdb::memory_scores(c, {});
  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_FLOAT_EQ(out[1], 1.0f);
}

TEST(MemoryScore, NegativeAgeClampsToNow) {
  vec<vdb::MemoryCandidate> c = {
      {0.5f, -10.0, 1.0f},
      {0.5f, 0.0, 1.0f},
      {0.5f, 50.0, 1.0f},
  };
  vec<float> out = vdb::memory_scores(c, {});
  EXPECT_FLOAT_EQ(out[0], out[1]);
  EXPECT_GT(out[1], out[2]);
}

TEST(MemoryScore, InfiniteAgeIsFinite) {
  vec<vdb::MemoryCandidate> c = {
      {0.5f, s::numeric_limits<double>::infinity(), 1.0f},
      {0.5f, 1.0, 1.0f},
  };
  vec<float> out = vdb::memory_scores(c, {});
  for (float x : out) {
    EXPECT_TRUE(s::isfinite(x));
  }
  EXPECT_GT(out[1], out[0]);
}

TEST(MemoryScore, CustomDecay) {
  vec<vdb::MemoryCandidate> c = {
      {0.5f, 0.0, 1.0f},
      {0.5f, 1.0, 1.0f},
      {0.5f, 2.0, 1.0f},
  };
  vdb::MemoryScoreWeights w;
  w.decay_per_hour = 0.5;
  vec<float> out = vdb::memory_scores(c, w);
  // recency 1, 0.5, 0.25 -> normalized 1, 1/3, 0
  EXPECT_NEAR(out[0], 1.0f, 1e-6f);
  EXPECT_NEAR(out[1], 1.0f / 3.0f, 1e-6f);
  EXPECT_NEAR(out[2], 0.0f, 1e-6f);
}
