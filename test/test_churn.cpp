#include <random>

#include <gtest/gtest.h>

#include <db_util.h>

namespace vt = vdb::test;

namespace {

constexpr uint32_t DIM = 24;

vec<vec<float>> random_queries(size_t n, uint64_t seed) {
  s::mt19937_64 rng(seed);
  vec<vec<float>> qs;
  for (size_t i = 0; i < n; ++i) {
    qs.push_back(vt::random_vector(rng, DIM));
  }
  return qs;
}

}  // namespace

// The test that separates toys from databases: repeated delete/insert
// rounds must not erode recall, and deleted points must never come back.
TEST(Churn, RecallStableAcrossRounds) {
  vdb::Database db = vt::open_memory();
  vdb::CollectionConfig cfg = vt::config(DIM);
  cfg.vacuum_min_points = 100;
  auto c = vt::must(db.create_collection("c", cfg));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 5000, DIM)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());

  s::mt19937_64 rng(9);
  vec<size_t> live(5000);
  s::iota(live.begin(), live.end(), 0);
  uset<str> deleted;
  size_t next = 5000;
  vec<vec<float>> qs = random_queries(50, 10);

  for (int round = 0; round < 10; ++round) {
    s::shuffle(live.begin(), live.end(), rng);
    vec<str> doomed;
    for (size_t j = 0; j < 500; ++j) {
      doomed.push_back(vt::id_for(live.back()));
      live.pop_back();
    }
    auto removed = c->remove(doomed);
    ASSERT_TRUE(removed.has_value());
    EXPECT_EQ(*removed, 500u);
    deleted.insert(doomed.begin(), doomed.end());

    ASSERT_TRUE(c->upsert(vt::numbered_records(next, next + 500, DIM))
                    .has_value());
    for (size_t j = next; j < next + 500; ++j) {
      live.push_back(j);
    }
    next += 500;
    ASSERT_TRUE(c->optimize().has_value());

    EXPECT_EQ(c->count(), 5000u) << "round " << round;
    double r = vt::collection_recall(*c, qs, 10);
    EXPECT_GE(r, 0.95) << "round " << round;

    for (const vec<float>& q : qs) {
      vdb::SearchRequest req;
      req.vector = q;
      req.k = 50;
      auto hits = c->search(req);
      ASSERT_TRUE(hits.has_value());
      for (const vdb::SearchHit& h : *hits) {
        EXPECT_FALSE(deleted.contains(h.id)) << h.id << " round " << round;
      }
    }
  }
  // Vacuum keeps tombstones bounded by the configured ratio.
  vdb::CollectionStats st = c->stats();
  EXPECT_LE(double(st.deleted_points),
            cfg.vacuum_deleted_ratio *
                double(st.live_points + st.deleted_points) + 1.0);
}

TEST(Churn, SearchForDeletedVectorFindsSomethingElse) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(DIM)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 2000, DIM)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());
  vec<str> doomed = {vt::id_for(123)};
  ASSERT_TRUE(c->remove(doomed).has_value());
  vdb::SearchRequest req;
  req.vector = vt::vector_for(123, DIM);
  req.k = 5;
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value());
  EXPECT_EQ(hits->size(), 5u);
  for (const vdb::SearchHit& h : *hits) {
    EXPECT_NE(h.id, vt::id_for(123));
  }
}

TEST(Churn, DeleteEverything) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(DIM)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 1500, DIM)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());
  vec<str> all;
  for (size_t i = 0; i < 1500; ++i) {
    all.push_back(vt::id_for(i));
  }
  auto removed = c->remove(all);
  ASSERT_TRUE(removed.has_value());
  EXPECT_EQ(*removed, 1500u);
  EXPECT_EQ(c->count(), 0u);
  vdb::SearchRequest req;
  req.vector = vt::vector_for(0, DIM);
  req.k = 10;
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value());
  EXPECT_TRUE(hits->empty());
  ASSERT_TRUE(c->optimize().has_value());
  EXPECT_EQ(c->stats().deleted_points, 0u)
      << "a fully deleted segment must be vacuumed";
}

TEST(Churn, ReinsertDeletedId) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(DIM)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 1000, DIM)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());
  vec<str> doomed = {vt::id_for(7)};
  ASSERT_TRUE(c->remove(doomed).has_value());
  ASSERT_TRUE(c->upsert(vt::numbered_records(7, 8, DIM)).has_value());
  EXPECT_EQ(c->count(), 1000u);
  vdb::SearchRequest req;
  req.vector = vt::vector_for(7, DIM);
  req.k = 1;
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value() and not hits->empty());
  EXPECT_EQ((*hits)[0].id, vt::id_for(7));
}

TEST(Churn, DeleteEntryPointRegion) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(DIM)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 3000, DIM)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());
  // The first inserted points are the most likely HNSW entry points.
  vec<str> doomed;
  for (size_t i = 0; i < 300; ++i) {
    doomed.push_back(vt::id_for(i));
  }
  ASSERT_TRUE(c->remove(doomed).has_value());
  EXPECT_GE(vt::collection_recall(*c, random_queries(50, 11), 10), 0.95);
}
