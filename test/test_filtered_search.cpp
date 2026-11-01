#include <random>

#include <gtest/gtest.h>

#include <db_util.h>

namespace vt = vdb::test;

namespace {

constexpr uint32_t DIM = 32;
constexpr size_t N = 20000;
constexpr size_t CLUSTERS = 20;

// Each point carries:
//   r       = i % 10000, indexed Integer (selectivity sweep)
//   r_raw   = same value, NOT indexed (forces the payload-scan path)
//   cluster = which gaussian center it came from, indexed Keyword
//   tenant  = "t<i % 50>", indexed Keyword with is_tenant
struct FilteredFixture : ::testing::Test {
  static void SetUpTestSuite() {
    db = new vdb::Database(vt::open_memory());
    vdb::CollectionConfig cfg = vt::config(DIM);
    cfg.indexes = {
        {"r", vdb::FieldType::Integer},
        {"cluster", vdb::FieldType::Keyword},
        {"tenant", vdb::FieldType::Keyword, true},
    };
    // Small threshold so the 0.1%..100% band really exercises the graph.
    cfg.full_scan_threshold_bytes = DIM * sizeof(float) * 500;
    col = vt::must(db->create_collection("f", cfg)).get();

    s::mt19937_64 rng(77);
    vec<vec<float>> centers;
    for (size_t c = 0; c < CLUSTERS; ++c) {
      centers.push_back(vt::random_vector(rng, DIM));
    }
    s::normal_distribution<float> noise(0.0f, 0.25f);
    vec<vdb::Record> records;
    for (size_t i = 0; i < N; ++i) {
      size_t c = i % CLUSTERS;
      vec<float> v(DIM);
      for (uint32_t d = 0; d < DIM; ++d) {
        v[d] = centers[c][d] + noise(rng);
      }
      vdb::Payload p;
      p["r"] = int64_t(i % 10000);
      p["r_raw"] = int64_t(i % 10000);
      p["cluster"] = "c" + s::to_string(c);
      p["tenant"] = "t" + s::to_string(i % 50);
      records.push_back({vt::id_for(i), s::move(v), {}, s::move(p)});
    }
    // Seal 90% into an indexed segment and leave a growing tail, so every
    // query spans both segment kinds.
    size_t sealed = N * 9 / 10;
    ASSERT_TRUE(col->upsert(s::span(records).first(sealed)).has_value());
    ASSERT_TRUE(col->optimize(true).has_value());
    ASSERT_TRUE(col->upsert(s::span(records).subspan(sealed)).has_value());
    cluster_centers = centers;
  }

  static void TearDownTestSuite() {
    delete db;
    db = nullptr;
    col = nullptr;
  }

  static vec<float> query_near(size_t cluster, uint64_t seed) {
    s::mt19937_64 rng(seed);
    s::normal_distribution<float> noise(0.0f, 0.25f);
    vec<float> q = cluster_centers[cluster];
    for (float& x : q) {
      x += noise(rng);
    }
    return q;
  }

  static vdb::Database* db;
  static vdb::Collection* col;
  static vec<vec<float>> cluster_centers;
};

vdb::Database* FilteredFixture::db = nullptr;
vdb::Collection* FilteredFixture::col = nullptr;
vec<vec<float>> FilteredFixture::cluster_centers;

void check_sweep(vdb::Collection& col, strv field, size_t k) {
  for (int64_t limit : {1, 10, 100, 1000, 5000, 10000}) {
    str json = R"({")" + str(field) + R"(":{"$lt":)" + s::to_string(limit) +
        "}}";
    vdb::Filter f = vt::must(vdb::parse_filter(json));
    size_t matching = size_t(limit) * (N / 10000);
    double recall_sum = 0.0;
    const int queries = 20;
    for (int q = 0; q < queries; ++q) {
      vdb::SearchRequest req;
      req.vector = FilteredFixture::query_near(size_t(q) % CLUSTERS,
                                               uint64_t(q) + 1000);
      req.k = k;
      req.filter = f;
      auto got = col.search(req);
      ASSERT_TRUE(got.has_value()) << json;
      // The criterion every naive filtered search fails: exactly
      // min(k, matches) results at every selectivity.
      EXPECT_EQ(got->size(), s::min(k, matching)) << json;
      for (const vdb::SearchHit& h : *got) {
        EXPECT_LT(s::get<int64_t>(h.payload.at("r")), limit) << json;
      }
      req.exact = true;
      auto want = col.search(req);
      ASSERT_TRUE(want.has_value());
      recall_sum += vt::recall(vt::ids_of(*got), vt::ids_of(*want));
    }
    EXPECT_GE(recall_sum / queries, 0.95) << json;
  }
}

}  // namespace

TEST_F(FilteredFixture, SelectivitySweepIndexedField) {
  check_sweep(*col, "r", 10);
}

TEST_F(FilteredFixture, SelectivitySweepUnindexedField) {
  check_sweep(*col, "r_raw", 10);
}

TEST_F(FilteredFixture, SelectivitySweepLargeK) {
  check_sweep(*col, "r", 100);
}

// Anti-correlated: the query sits in cluster 0 but only cluster 7 matches,
// so every match is far away. Inline graph filtering tends to strand here.
TEST_F(FilteredFixture, AntiCorrelatedFilter) {
  vdb::Filter f = vt::must(vdb::parse_filter(R"({"cluster":"c7"})"));
  double recall_sum = 0.0;
  for (int q = 0; q < 20; ++q) {
    vdb::SearchRequest req;
    req.vector = query_near(0, uint64_t(q) + 5000);
    req.k = 10;
    req.filter = f;
    auto got = col->search(req);
    ASSERT_TRUE(got.has_value());
    ASSERT_EQ(got->size(), 10u);
    for (const vdb::SearchHit& h : *got) {
      EXPECT_EQ(s::get<str>(h.payload.at("cluster")), "c7");
    }
    req.exact = true;
    auto want = col->search(req);
    ASSERT_TRUE(want.has_value());
    recall_sum += vt::recall(vt::ids_of(*got), vt::ids_of(*want));
  }
  EXPECT_GE(recall_sum / 20, 0.95);
}

TEST_F(FilteredFixture, CorrelatedFilter) {
  vdb::Filter f = vt::must(vdb::parse_filter(R"({"cluster":"c3"})"));
  vdb::SearchRequest req;
  req.vector = query_near(3, 42);
  req.k = 10;
  req.filter = f;
  auto got = col->search(req);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(got->size(), 10u);
}

TEST_F(FilteredFixture, TenantIsolation) {
  for (int t = 0; t < 50; t += 7) {
    str tenant = "t" + s::to_string(t);
    vdb::SearchRequest req;
    req.vector = query_near(size_t(t) % CLUSTERS, uint64_t(t));
    req.k = 25;
    req.filter = vdb::f_cmp(vdb::FilterOp::Eq, "tenant", tenant);
    auto got = col->search(req);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->size(), 25u);
    for (const vdb::SearchHit& h : *got) {
      EXPECT_EQ(s::get<str>(h.payload.at("tenant")), tenant);
    }
  }
}

TEST_F(FilteredFixture, NoMatchesReturnsEmpty) {
  vdb::SearchRequest req;
  req.vector = query_near(0, 1);
  req.k = 10;
  req.filter = vt::must(vdb::parse_filter(R"({"cluster":"nope"})"));
  auto got = col->search(req);
  ASSERT_TRUE(got.has_value());
  EXPECT_TRUE(got->empty());
}

TEST_F(FilteredFixture, EmptyFilterEqualsUnfiltered) {
  vdb::SearchRequest plain;
  plain.vector = query_near(4, 9);
  plain.k = 20;
  plain.exact = true;
  vdb::SearchRequest filtered = plain;
  filtered.filter = vdb::f_and({});
  auto a = col->search(plain);
  auto b = col->search(filtered);
  ASSERT_TRUE(a.has_value() and b.has_value());
  EXPECT_EQ(vt::ids_of(*a), vt::ids_of(*b));
}

// Metamorphic: results for A AND B all satisfy A; A OR B ⊇ exact A at k.
TEST_F(FilteredFixture, MetamorphicFilterLaws) {
  vdb::Filter a = vt::must(vdb::parse_filter(R"({"r":{"$lt":3000}})"));
  vdb::Filter b = vt::must(vdb::parse_filter(R"({"cluster":"c2"})"));
  vdb::SearchRequest req;
  req.vector = query_near(2, 3);
  req.k = 30;
  req.exact = true;
  req.filter = vdb::f_and({a, b});
  auto ab = col->search(req);
  ASSERT_TRUE(ab.has_value());
  for (const vdb::SearchHit& h : *ab) {
    EXPECT_TRUE(vdb::matches(a, h.payload));
    EXPECT_TRUE(vdb::matches(b, h.payload));
  }
  req.filter = vdb::f_not(a);
  auto not_a = col->search(req);
  ASSERT_TRUE(not_a.has_value());
  for (const vdb::SearchHit& h : *not_a) {
    EXPECT_FALSE(vdb::matches(a, h.payload));
  }
  auto na = col->count_where(a);
  auto nn = col->count_where(vdb::f_not(a));
  ASSERT_TRUE(na.has_value() and nn.has_value());
  EXPECT_EQ(*na + *nn, N);
}

TEST_F(FilteredFixture, PlannerUsesCandidatesForTinyFilters) {
  vdb::SearchRequest req;
  req.vector = query_near(0, 1);
  req.filter = vt::must(vdb::parse_filter(R"({"r":{"$lt":2}})"));
  auto plan = col->explain(req);
  ASSERT_TRUE(plan.has_value());
  for (const vdb::SegmentPlan& p : *plan) {
    EXPECT_EQ(p.strategy, vdb::SearchStrategy::PlainFromCandidates);
  }
}
