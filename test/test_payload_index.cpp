#include <random>

#include <gtest/gtest.h>

#include <vdb/payload_index.h>

namespace {

struct Corpus {
  vec<vdb::Payload> docs;
  roaring::Roaring universe;
};

// tenant: 10 values with skewed frequency, year: 2000..2029, score: double,
// flag: bool, tags: lists, note: an unindexed string.
Corpus make_corpus(size_t n, uint64_t seed) {
  s::mt19937_64 rng(seed);
  s::geometric_distribution<int> tenant(0.3);
  s::uniform_int_distribution<int64_t> year(2000, 2029);
  s::uniform_real_distribution<double> score(0.0, 1.0);
  s::bernoulli_distribution flag(0.2);
  s::uniform_int_distribution<int> tag(0, 5);
  Corpus c;
  for (size_t i = 0; i < n; ++i) {
    vdb::Payload p;
    p["tenant"] = "t" + s::to_string(s::min(tenant(rng), 9));
    p["year"] = year(rng);
    p["score"] = score(rng);
    p["flag"] = flag(rng);
    p["tags"] = vdb::StrList{"g" + s::to_string(tag(rng)),
                             "g" + s::to_string(tag(rng))};
    p["note"] = str(i % 2 == 0 ? "even" : "odd");
    if (i % 7 == 0) {
      p.erase("year");
    }
    c.docs.push_back(s::move(p));
    c.universe.add(uint32_t(i));
  }
  return c;
}

vdb::PayloadIndex make_index(const Corpus& c) {
  vdb::PayloadIndex index({
      {"tenant", vdb::FieldType::Keyword, true},
      {"year", vdb::FieldType::Integer},
      {"score", vdb::FieldType::Float},
      {"flag", vdb::FieldType::Bool},
      {"tags", vdb::FieldType::Keyword},
  });
  for (uint32_t i = 0; i < c.docs.size(); ++i) {
    index.add(i, c.docs[i]);
  }
  return index;
}

roaring::Roaring scan(const Corpus& c, const vdb::Filter& f) {
  roaring::Roaring out;
  for (uint32_t i : c.universe) {
    if (vdb::matches(f, c.docs[i])) {
      out.add(i);
    }
  }
  return out;
}

vdb::Filter parse(strv json) {
  auto f = vdb::parse_filter(json);
  EXPECT_TRUE(f.has_value()) << json;
  return f.value_or(vdb::f_and({}));
}

// Filters touching only indexed fields: evaluate() must be exact.
const vec<strv> INDEXED_FILTERS = {
    R"({"tenant":"t0"})",
    R"({"tenant":"t9"})",
    R"({"tenant":"missing"})",
    R"({"tenant":{"$in":["t1","t2"]}})",
    R"({"tenant":{"$ne":"t0"}})",
    R"({"tenant":{"$nin":["t0","t1"]}})",
    R"({"year":{"$gte":2010,"$lt":2020}})",
    R"({"year":2015})",
    R"({"year":{"$gt":2029}})",
    R"({"year":{"$exists":false}})",
    R"({"score":{"$lt":0.1}})",
    R"({"score":{"$gte":0.5}})",
    R"({"year":{"$gte":2020.5}})",
    R"({"flag":true})",
    R"({"tags":"g3"})",
    R"({"tags":{"$nin":["g0","g1"]}})",
    R"({"tenant":"t0","flag":true})",
    R"({"$or":[{"tenant":"t3"},{"year":{"$lt":2002}}]})",
    R"({"$not":{"tenant":"t0"}})",
    R"({"$not":{"$or":[{"flag":true},{"tags":"g1"}]}})",
};

}  // namespace

TEST(PayloadIndex, DeclaredFields) {
  vdb::PayloadIndex index({{"a", vdb::FieldType::Keyword}});
  EXPECT_TRUE(index.is_indexed("a"));
  EXPECT_FALSE(index.is_indexed("b"));
  ASSERT_EQ(index.specs().size(), 1u);
  EXPECT_EQ(index.specs()[0].name, "a");
}

TEST(PayloadIndex, EvaluateMatchesScan) {
  Corpus c = make_corpus(5000, 1);
  vdb::PayloadIndex index = make_index(c);
  for (strv json : INDEXED_FILTERS) {
    vdb::Filter f = parse(json);
    auto got = index.evaluate(f, c.universe);
    ASSERT_TRUE(got.has_value()) << json;
    EXPECT_EQ(*got, scan(c, f)) << json;
  }
}

TEST(PayloadIndex, UnindexedFieldIsUnresolvable) {
  Corpus c = make_corpus(100, 2);
  vdb::PayloadIndex index = make_index(c);
  EXPECT_FALSE(index.evaluate(parse(R"({"note":"even"})"), c.universe));
  EXPECT_FALSE(
      index.evaluate(parse(R"({"tenant":"t0","note":"odd"})"), c.universe));
}

TEST(PayloadIndex, EmptyFilterIsUniverse) {
  Corpus c = make_corpus(300, 3);
  vdb::PayloadIndex index = make_index(c);
  auto got = index.evaluate(parse("{}"), c.universe);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, c.universe);
}

TEST(PayloadIndex, RemoveDropsFromAllFields) {
  Corpus c = make_corpus(1000, 4);
  vdb::PayloadIndex index = make_index(c);
  for (uint32_t i = 0; i < 1000; i += 3) {
    index.remove(i, c.docs[i]);
    c.universe.remove(i);
  }
  for (strv json : INDEXED_FILTERS) {
    vdb::Filter f = parse(json);
    auto got = index.evaluate(f, c.universe);
    ASSERT_TRUE(got.has_value()) << json;
    EXPECT_EQ(*got, scan(c, f)) << json;
  }
}

TEST(PayloadIndex, WrongTypedValuesIgnoredByIndex) {
  vdb::PayloadIndex index({{"year", vdb::FieldType::Integer}});
  vdb::Payload p;
  p["year"] = str("2020");
  index.add(0, p);
  roaring::Roaring universe;
  universe.add(0);
  auto got = index.evaluate(parse(R"({"year":2020})"), universe);
  ASSERT_TRUE(got.has_value());
  EXPECT_TRUE(got->isEmpty());
}

TEST(PayloadIndex, IndexedLeafEstimateIsExact) {
  Corpus c = make_corpus(4000, 5);
  vdb::PayloadIndex index = make_index(c);
  for (strv json : {R"({"tenant":"t0"})", R"({"flag":true})",
                    R"({"tags":"g2"})", R"({"year":{"$lt":2005}})"}) {
    vdb::Filter f = parse(json);
    size_t truth = scan(c, f).cardinality();
    vdb::Cardinality est = index.estimate(f, 4000);
    EXPECT_EQ(est, (vdb::Cardinality{truth, truth, truth})) << json;
  }
}

TEST(PayloadIndex, UnindexedLeafEstimate) {
  vdb::PayloadIndex index({});
  vdb::Cardinality est = index.estimate(parse(R"({"note":"x"})"), 1000);
  EXPECT_EQ(est, (vdb::Cardinality{0, 500, 1000}));
}

// The interval must always contain the truth and be ordered, for every
// filter shape including unindexed leaves.
TEST(PayloadIndex, EstimateIntervalContainsTruth) {
  Corpus c = make_corpus(3000, 6);
  vdb::PayloadIndex index = make_index(c);
  vec<strv> filters = INDEXED_FILTERS;
  filters.push_back(R"({"note":"even"})");
  filters.push_back(R"({"tenant":"t0","note":"odd"})");
  filters.push_back(R"({"$or":[{"note":"even"},{"flag":true}]})");
  filters.push_back(R"({"$not":{"note":"even","tenant":"t1"}})");
  for (strv json : filters) {
    vdb::Filter f = parse(json);
    size_t truth = scan(c, f).cardinality();
    vdb::Cardinality est = index.estimate(f, 3000);
    EXPECT_LE(est.min, est.exp) << json;
    EXPECT_LE(est.exp, est.max) << json;
    EXPECT_LE(est.max, 3000u) << json;
    EXPECT_LE(est.min, truth) << json;
    EXPECT_GE(est.max, truth) << json;
  }
}

TEST(PayloadIndex, AndEstimateUsesIndependence) {
  Corpus c = make_corpus(10000, 7);
  vdb::PayloadIndex index = make_index(c);
  vdb::Filter a = parse(R"({"flag":true})");
  vdb::Filter b = parse(R"({"tenant":"t0"})");
  size_t na = scan(c, a).cardinality();
  size_t nb = scan(c, b).cardinality();
  vdb::Cardinality est = index.estimate(vdb::f_and({a, b}), 10000);
  double want = 10000.0 * (double(na) / 10000.0) * (double(nb) / 10000.0);
  EXPECT_NEAR(double(est.exp), want, 1.0);
  EXPECT_EQ(est.max, s::min(na, nb));
}
