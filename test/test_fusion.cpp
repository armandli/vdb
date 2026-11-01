#include <gtest/gtest.h>

#include <vdb/fusion.h>

TEST(Rrf, SingleListKeepsOrder) {
  vec<vec<str>> lists = {{"a", "b", "c"}};
  vec<vdb::Fused> out = vdb::rrf(lists);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].id, "a");
  EXPECT_EQ(out[1].id, "b");
  EXPECT_EQ(out[2].id, "c");
  EXPECT_FLOAT_EQ(out[0].score, 1.0f / 61.0f);
  EXPECT_FLOAT_EQ(out[2].score, 1.0f / 63.0f);
}

TEST(Rrf, ScoresSumAcrossLists) {
  vec<vec<str>> lists = {{"a", "b"}, {"b", "c"}};
  vec<vdb::Fused> out = vdb::rrf(lists);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].id, "b");
  EXPECT_FLOAT_EQ(out[0].score, 1.0f / 62.0f + 1.0f / 61.0f);
}

TEST(Rrf, ListOrderDoesNotMatter) {
  vec<vec<str>> ab = {{"x", "y", "z"}, {"z", "w"}};
  vec<vec<str>> ba = {{"z", "w"}, {"x", "y", "z"}};
  EXPECT_EQ(vdb::rrf(ab), vdb::rrf(ba));
}

TEST(Rrf, EmptyListIsNeutral) {
  vec<vec<str>> one = {{"a", "b", "c"}};
  vec<vec<str>> with_empty = {{"a", "b", "c"}, {}};
  EXPECT_EQ(vdb::rrf(one), vdb::rrf(with_empty));
  vec<vec<str>> none;
  EXPECT_TRUE(vdb::rrf(none).empty());
}

TEST(Rrf, TiesBreakById) {
  vec<vec<str>> lists = {{"b", "a"}, {"a", "b"}};
  vec<vdb::Fused> out = vdb::rrf(lists);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0].score, out[1].score);
  EXPECT_EQ(out[0].id, "a");
  EXPECT_EQ(out[1].id, "b");
}

TEST(Rrf, Limit) {
  vec<vec<str>> lists = {{"a", "b", "c", "d"}};
  EXPECT_EQ(vdb::rrf(lists, 2).size(), 2u);
  EXPECT_EQ(vdb::rrf(lists, 0).size(), 4u);
  EXPECT_EQ(vdb::rrf(lists, 100).size(), 4u);
}

TEST(Rrf, CustomK) {
  vec<vec<str>> lists = {{"a"}};
  EXPECT_FLOAT_EQ(vdb::rrf(lists, 0, 1.0f)[0].score, 0.5f);
}

TEST(Rrf, Weights) {
  vec<vec<str>> lists = {{"a", "b"}, {"b", "a"}};
  vec<float> weights = {1.0f, 3.0f};
  vec<vdb::Fused> out = vdb::rrf(lists, 0, 60.0f, weights);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].id, "b");
  EXPECT_FLOAT_EQ(out[0].score, 1.0f / 62.0f + 3.0f / 61.0f);
}

TEST(Rrf, MissingWeightsDefaultToOne) {
  vec<vec<str>> lists = {{"a"}, {"b"}};
  vec<float> weights = {2.0f};
  vec<vdb::Fused> out = vdb::rrf(lists, 0, 60.0f, weights);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].id, "a");
  EXPECT_FLOAT_EQ(out[1].score, 1.0f / 61.0f);
}

TEST(Rrf, DuplicateInListCountsBestRank) {
  vec<vec<str>> lists = {{"a", "b", "a"}};
  vec<vdb::Fused> out = vdb::rrf(lists);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0].score, 1.0f / 61.0f);
}

// Agreement between retrievers should outrank a single first place.
TEST(Rrf, ConsensusBeatsSingleWin) {
  vec<vec<str>> lists = {{"solo", "both"}, {"x", "both"}, {"y", "both"}};
  EXPECT_EQ(vdb::rrf(lists)[0].id, "both");
}
