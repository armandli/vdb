#include <algorithm>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <vdb/topk.h>

TEST(TopK, EmptyHeap) {
  vdb::TopK top(5);
  EXPECT_TRUE(top.empty());
  EXPECT_EQ(top.size(), 0u);
  EXPECT_EQ(top.capacity(), 5u);
  EXPECT_EQ(top.worst(), s::numeric_limits<float>::infinity());
  EXPECT_TRUE(top.accepts(1e30f));
  EXPECT_TRUE(top.take_sorted().empty());
}

TEST(TopK, KeepsSmallestK) {
  vdb::TopK top(3);
  for (vdb::InternalId i = 0; i < 10; ++i) {
    top.push(i, float(10 - i));
  }
  vec<vdb::Hit> hits = top.take_sorted();
  ASSERT_EQ(hits.size(), 3u);
  EXPECT_EQ(hits[0], (vdb::Hit{9, 1.0f}));
  EXPECT_EQ(hits[1], (vdb::Hit{8, 2.0f}));
  EXPECT_EQ(hits[2], (vdb::Hit{7, 3.0f}));
}

TEST(TopK, FewerThanK) {
  vdb::TopK top(10);
  top.push(3, 0.5f);
  top.push(1, 0.25f);
  EXPECT_EQ(top.size(), 2u);
  EXPECT_EQ(top.worst(), s::numeric_limits<float>::infinity());
  vec<vdb::Hit> hits = top.take_sorted();
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].id, 1u);
  EXPECT_EQ(hits[1].id, 3u);
}

TEST(TopK, WorstAndAccepts) {
  vdb::TopK top(2);
  top.push(0, 1.0f);
  top.push(1, 3.0f);
  EXPECT_FLOAT_EQ(top.worst(), 3.0f);
  EXPECT_TRUE(top.accepts(2.0f));
  EXPECT_FALSE(top.accepts(3.0f));
  EXPECT_FALSE(top.accepts(4.0f));
  top.push(2, 2.0f);
  EXPECT_FLOAT_EQ(top.worst(), 2.0f);
}

TEST(TopK, TieWithWorstIsRejected) {
  vdb::TopK top(2);
  top.push(5, 1.0f);
  top.push(6, 1.0f);
  top.push(7, 1.0f);
  vec<vdb::Hit> hits = top.take_sorted();
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].id, 5u);
  EXPECT_EQ(hits[1].id, 6u);
}

TEST(TopK, SortedByDistanceThenId) {
  vdb::TopK top(4);
  top.push(9, 2.0f);
  top.push(4, 1.0f);
  top.push(2, 2.0f);
  top.push(7, 1.0f);
  vec<vdb::Hit> hits = top.take_sorted();
  ASSERT_EQ(hits.size(), 4u);
  EXPECT_EQ(hits[0].id, 4u);
  EXPECT_EQ(hits[1].id, 7u);
  EXPECT_EQ(hits[2].id, 2u);
  EXPECT_EQ(hits[3].id, 9u);
}

TEST(TopK, TakeSortedEmptiesHeap) {
  vdb::TopK top(3);
  top.push(1, 1.0f);
  top.take_sorted();
  EXPECT_TRUE(top.empty());
  top.push(2, 5.0f);
  EXPECT_EQ(top.take_sorted().size(), 1u);
}

TEST(TopK, ZeroCapacityAcceptsNothing) {
  vdb::TopK top(0);
  EXPECT_FALSE(top.accepts(-1e30f));
  top.push(1, 0.0f);
  EXPECT_TRUE(top.take_sorted().empty());
}

TEST(TopK, RandomizedAgainstSort) {
  s::mt19937_64 rng(42);
  s::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (size_t k : {1u, 5u, 64u, 1000u}) {
    vdb::TopK top(k);
    vec<vdb::Hit> all;
    for (vdb::InternalId i = 0; i < 500; ++i) {
      float d = dist(rng);
      top.push(i, d);
      all.push_back({i, d});
    }
    s::sort(all.begin(), all.end(), [](const vdb::Hit& a, const vdb::Hit& b) {
      return a.distance < b.distance or
          (a.distance == b.distance and a.id < b.id);
    });
    all.resize(s::min(k, all.size()));
    EXPECT_EQ(top.take_sorted(), all) << "k=" << k;
  }
}

// top-k(5) must be a prefix of top-k(10) for the same stream.
TEST(TopK, SmallerKIsPrefix) {
  s::mt19937_64 rng(7);
  s::uniform_real_distribution<float> dist(0.0f, 1.0f);
  vdb::TopK top5(5);
  vdb::TopK top10(10);
  for (vdb::InternalId i = 0; i < 200; ++i) {
    float d = dist(rng);
    top5.push(i, d);
    top10.push(i, d);
  }
  vec<vdb::Hit> a = top5.take_sorted();
  vec<vdb::Hit> b = top10.take_sorted();
  ASSERT_EQ(a.size(), 5u);
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b[i]);
  }
}
