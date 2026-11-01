#include <random>

#include <gtest/gtest.h>

#include <vdb/filter.h>

namespace {

vdb::Payload doc() {
  vdb::Payload p;
  p["tenant"] = str("acme");
  p["year"] = int64_t(2024);
  p["score"] = 0.75;
  p["draft"] = false;
  p["tags"] = vdb::StrList{"ml", "db"};
  p["ids"] = vdb::IntList{3, 5, 8};
  return p;
}

bool eval(strv json, const vdb::Payload& p = doc()) {
  auto f = vdb::parse_filter(json);
  EXPECT_TRUE(f.has_value()) << json << ": "
                             << (f.has_value() ? "" : f.error().message);
  return f.has_value() and vdb::matches(*f, p);
}

}  // namespace

TEST(Filter, EmptyMatchesEverything) {
  EXPECT_TRUE(eval("{}"));
  EXPECT_TRUE(eval("{}", {}));
}

TEST(Filter, ShorthandEquality) {
  EXPECT_TRUE(eval(R"({"tenant":"acme"})"));
  EXPECT_FALSE(eval(R"({"tenant":"other"})"));
  EXPECT_TRUE(eval(R"({"year":2024})"));
  EXPECT_TRUE(eval(R"({"draft":false})"));
}

TEST(Filter, ImplicitAndAcrossFields) {
  EXPECT_TRUE(eval(R"({"tenant":"acme","year":2024})"));
  EXPECT_FALSE(eval(R"({"tenant":"acme","year":2023})"));
}

TEST(Filter, ComparisonOperators) {
  EXPECT_TRUE(eval(R"({"year":{"$gt":2023}})"));
  EXPECT_FALSE(eval(R"({"year":{"$gt":2024}})"));
  EXPECT_TRUE(eval(R"({"year":{"$gte":2024}})"));
  EXPECT_TRUE(eval(R"({"year":{"$lt":2025}})"));
  EXPECT_FALSE(eval(R"({"year":{"$lt":2024}})"));
  EXPECT_TRUE(eval(R"({"year":{"$lte":2024}})"));
  EXPECT_TRUE(eval(R"({"year":{"$ne":1999}})"));
  EXPECT_FALSE(eval(R"({"year":{"$ne":2024}})"));
  EXPECT_TRUE(eval(R"({"year":{"$gte":2020,"$lt":2030}})"));
  EXPECT_FALSE(eval(R"({"year":{"$gte":2020,"$lt":2024}})"));
}

TEST(Filter, IntAndDoubleCompareNumerically) {
  EXPECT_TRUE(eval(R"({"year":2024.0})"));
  EXPECT_TRUE(eval(R"({"score":{"$gt":0}})"));
  EXPECT_TRUE(eval(R"({"score":{"$lt":1}})"));
  EXPECT_FALSE(eval(R"({"score":1})"));
}

TEST(Filter, CrossTypeComparisonFails) {
  EXPECT_FALSE(eval(R"({"year":"2024"})"));
  EXPECT_FALSE(eval(R"({"tenant":{"$gt":5}})"));
  EXPECT_FALSE(eval(R"({"draft":0})"));
}

TEST(Filter, StringRangeIsLexicographic) {
  EXPECT_TRUE(eval(R"({"tenant":{"$gt":"abc"}})"));
  EXPECT_TRUE(eval(R"({"tenant":{"$lt":"b"}})"));
  EXPECT_FALSE(eval(R"({"tenant":{"$gt":"acme"}})"));
}

TEST(Filter, InAndNin) {
  EXPECT_TRUE(eval(R"({"tenant":{"$in":["x","acme"]}})"));
  EXPECT_FALSE(eval(R"({"tenant":{"$in":["x","y"]}})"));
  EXPECT_TRUE(eval(R"({"tenant":{"$nin":["x","y"]}})"));
  EXPECT_FALSE(eval(R"({"tenant":{"$nin":["acme"]}})"));
  EXPECT_FALSE(eval(R"({"tenant":{"$in":[]}})"));
  EXPECT_TRUE(eval(R"({"tenant":{"$nin":[]}})"));
}

TEST(Filter, ListFieldsMatchAnyElement) {
  EXPECT_TRUE(eval(R"({"tags":"ml"})"));
  EXPECT_FALSE(eval(R"({"tags":"web"})"));
  EXPECT_TRUE(eval(R"({"tags":{"$in":["web","db"]}})"));
  EXPECT_FALSE(eval(R"({"tags":{"$ne":"ml"}})"));
  EXPECT_TRUE(eval(R"({"tags":{"$ne":"web"}})"));
  EXPECT_FALSE(eval(R"({"tags":{"$nin":["db"]}})"));
  EXPECT_TRUE(eval(R"({"ids":5})"));
  EXPECT_TRUE(eval(R"({"ids":{"$gt":7}})"));
  EXPECT_FALSE(eval(R"({"ids":{"$gt":8}})"));
}

TEST(Filter, MissingField) {
  EXPECT_FALSE(eval(R"({"nope":1})"));
  EXPECT_FALSE(eval(R"({"nope":{"$gt":1}})"));
  EXPECT_FALSE(eval(R"({"nope":{"$in":[1]}})"));
  EXPECT_TRUE(eval(R"({"nope":{"$ne":1}})"));
  EXPECT_TRUE(eval(R"({"nope":{"$nin":[1]}})"));
  EXPECT_TRUE(eval(R"({"nope":{"$exists":false}})"));
  EXPECT_FALSE(eval(R"({"nope":{"$exists":true}})"));
  EXPECT_TRUE(eval(R"({"tenant":{"$exists":true}})"));
}

TEST(Filter, LogicalOperators) {
  EXPECT_TRUE(eval(R"({"$or":[{"tenant":"x"},{"year":2024}]})"));
  EXPECT_FALSE(eval(R"({"$or":[{"tenant":"x"},{"year":1}]})"));
  EXPECT_TRUE(eval(R"({"$and":[{"tenant":"acme"},{"year":2024}]})"));
  EXPECT_FALSE(eval(R"({"$and":[{"tenant":"acme"},{"year":1}]})"));
  EXPECT_TRUE(eval(R"({"$not":{"tenant":"x"}})"));
  EXPECT_FALSE(eval(R"({"$not":{"tenant":"acme"}})"));
  EXPECT_TRUE(eval(
      R"({"tenant":"acme","$or":[{"draft":true},{"tags":"db"}]})"));
}

TEST(Filter, ParseErrors) {
  for (strv bad : {
           R"([])",
           R"({"a":{"$bogus":1}})",
           R"({"$and":[]})",
           R"({"$or":[]})",
           R"({"$and":{"a":1}})",
           R"({"a":{"$in":5}})",
           R"({"a":{"$exists":"yes"}})",
           R"({"a":[1,2]})",
           R"({"a":{"$gt":[1]}})",
           R"({"$not":[{"a":1}]})",
           R"({"a":)",
           R"()",
       }) {
    auto f = vdb::parse_filter(bad);
    EXPECT_FALSE(f.has_value()) << bad;
    if (not f.has_value()) {
      EXPECT_EQ(f.error().code, vdb::ErrorCode::ParseError) << bad;
    }
  }
}

TEST(Filter, DeepNestingIsRejectedNotCrashed) {
  str json;
  for (int i = 0; i < 10000; ++i) {
    json += R"({"$not":)";
  }
  json += R"({"a":1})";
  for (int i = 0; i < 10000; ++i) {
    json += "}";
  }
  EXPECT_FALSE(vdb::parse_filter(json).has_value());
}

TEST(Filter, NestingAtLimitIsAccepted) {
  str json;
  for (int i = 0; i < 31; ++i) {
    json += R"({"$not":)";
  }
  json += R"({"a":1})";
  for (int i = 0; i < 31; ++i) {
    json += "}";
  }
  EXPECT_TRUE(vdb::parse_filter(json).has_value());
}

TEST(Filter, BuildersMatchParser) {
  vdb::Filter built = vdb::f_and({
      vdb::f_cmp(vdb::FilterOp::Eq, "tenant", str("acme")),
      vdb::f_or({
          vdb::f_cmp(vdb::FilterOp::Gte, "year", int64_t(2025)),
          vdb::f_in("tags", {str("db")}),
      }),
      vdb::f_not(vdb::f_exists("deleted")),
  });
  EXPECT_TRUE(vdb::matches(built, doc()));
  vdb::Payload other = doc();
  other["deleted"] = true;
  EXPECT_FALSE(vdb::matches(built, other));
}

// Metamorphic laws over random payloads.
TEST(Filter, MetamorphicLaws) {
  s::mt19937_64 rng(3);
  s::uniform_int_distribution<int64_t> year(2000, 2030);
  s::uniform_int_distribution<int> tenant(0, 4);
  auto a = vdb::parse_filter(R"({"year":{"$gte":2015}})");
  auto b = vdb::parse_filter(R"({"tenant":{"$in":["t1","t3"]}})");
  ASSERT_TRUE(a.has_value() and b.has_value());
  vdb::Filter a_and_b = vdb::f_and({*a, *b});
  vdb::Filter a_or_b = vdb::f_or({*a, *b});
  vdb::Filter not_a = vdb::f_not(*a);
  for (int i = 0; i < 1000; ++i) {
    vdb::Payload p;
    p["year"] = year(rng);
    p["tenant"] = "t" + s::to_string(tenant(rng));
    bool ma = vdb::matches(*a, p);
    bool mb = vdb::matches(*b, p);
    EXPECT_EQ(vdb::matches(a_and_b, p), ma and mb);
    EXPECT_EQ(vdb::matches(a_or_b, p), ma or mb);
    EXPECT_NE(vdb::matches(not_a, p), ma);
  }
}
