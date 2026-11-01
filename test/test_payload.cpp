#include <gtest/gtest.h>

#include <vdb/payload.h>

TEST(Payload, ParsesScalars) {
  auto p = vdb::parse_payload(
      R"({"s":"hi","i":42,"neg":-7,"d":1.5,"t":true,"f":false,"n":null})");
  ASSERT_TRUE(p.has_value()) << p.error().message;
  EXPECT_EQ(s::get<str>(p->at("s")), "hi");
  EXPECT_EQ(s::get<int64_t>(p->at("i")), 42);
  EXPECT_EQ(s::get<int64_t>(p->at("neg")), -7);
  EXPECT_DOUBLE_EQ(s::get<double>(p->at("d")), 1.5);
  EXPECT_EQ(s::get<bool>(p->at("t")), true);
  EXPECT_EQ(s::get<bool>(p->at("f")), false);
  EXPECT_EQ(vdb::type_of(p->at("n")), vdb::ValueType::Null);
}

TEST(Payload, IntegersOutsideInt64BecomeDouble) {
  auto p = vdb::parse_payload(R"({"big":18446744073709551615,"e":1e3})");
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(vdb::type_of(p->at("big")), vdb::ValueType::Double);
  EXPECT_EQ(vdb::type_of(p->at("e")), vdb::ValueType::Double);
}

TEST(Payload, ParsesHomogeneousArrays) {
  auto p = vdb::parse_payload(R"({"tags":["a","b"],"ids":[1,2,3],"e":[]})");
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(s::get<vdb::StrList>(p->at("tags")), (vdb::StrList{"a", "b"}));
  EXPECT_EQ(s::get<vdb::IntList>(p->at("ids")), (vdb::IntList{1, 2, 3}));
  EXPECT_EQ(vdb::type_of(p->at("e")), vdb::ValueType::IntList);
}

TEST(Payload, RejectsMixedArrays) {
  EXPECT_FALSE(vdb::parse_payload(R"({"x":[1,"a"]})").has_value());
  EXPECT_FALSE(vdb::parse_payload(R"({"x":[1.5,2.5]})").has_value());
  EXPECT_FALSE(vdb::parse_payload(R"({"x":[{"a":1}]})").has_value());
}

TEST(Payload, FlattensNestedObjects) {
  auto p = vdb::parse_payload(R"({"user":{"name":"al","age":{"y":3}}})");
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(p->size(), 2u);
  EXPECT_EQ(s::get<str>(p->at("user.name")), "al");
  EXPECT_EQ(s::get<int64_t>(p->at("user.age.y")), 3);
}

TEST(Payload, RejectsNonObjectAndMalformed) {
  EXPECT_FALSE(vdb::parse_payload("[1,2]").has_value());
  EXPECT_FALSE(vdb::parse_payload("42").has_value());
  EXPECT_FALSE(vdb::parse_payload("{\"a\":").has_value());
  EXPECT_FALSE(vdb::parse_payload("").has_value());
  auto bad = vdb::parse_payload("{oops}");
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().code, vdb::ErrorCode::ParseError);
}

TEST(Payload, EmptyObject) {
  auto p = vdb::parse_payload("{}");
  ASSERT_TRUE(p.has_value());
  EXPECT_TRUE(p->empty());
}

TEST(Payload, UnicodeAndEscapes) {
  auto p = vdb::parse_payload(R"({"k":"café \"q\" \n 日本"})");
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(s::get<str>(p->at("k")), "café \"q\" \n 日本");
}

TEST(Payload, JsonRoundTrip) {
  vdb::Payload p;
  p["s"] = str("he said \"hi\"\n");
  p["i"] = int64_t(-5);
  p["d"] = 0.25;
  p["b"] = true;
  p["tags"] = vdb::StrList{"x", "y"};
  p["ids"] = vdb::IntList{9, 8};
  p["n"] = s::monostate{};
  auto back = vdb::parse_payload(vdb::to_json(p));
  ASSERT_TRUE(back.has_value()) << vdb::to_json(p);
  EXPECT_EQ(*back, p);
}

TEST(Payload, JsonIsDeterministic) {
  vdb::Payload a;
  a["z"] = int64_t(1);
  a["a"] = int64_t(2);
  vdb::Payload b;
  b["a"] = int64_t(2);
  b["z"] = int64_t(1);
  EXPECT_EQ(vdb::to_json(a), vdb::to_json(b));
  EXPECT_EQ(vdb::to_json(a), R"({"a":2,"z":1})");
}

TEST(Payload, BinaryRoundTrip) {
  vdb::Payload p;
  p["s"] = str("text");
  p["i"] = int64_t(1) << 60;
  p["d"] = -3.75;
  p["b"] = false;
  p["tags"] = vdb::StrList{"", "long tag value"};
  p["ids"] = vdb::IntList{};
  p["n"] = s::monostate{};
  vec<uint8_t> bytes = vdb::encode_payload(p);
  auto back = vdb::decode_payload(bytes);
  ASSERT_TRUE(back.has_value());
  EXPECT_EQ(*back, p);
}

TEST(Payload, BinaryEmpty) {
  vec<uint8_t> bytes = vdb::encode_payload({});
  auto back = vdb::decode_payload(bytes);
  ASSERT_TRUE(back.has_value());
  EXPECT_TRUE(back->empty());
}

TEST(Payload, BinaryTruncationIsCorruption) {
  vdb::Payload p;
  p["s"] = str("some string value");
  p["ids"] = vdb::IntList{1, 2, 3};
  vec<uint8_t> bytes = vdb::encode_payload(p);
  for (size_t len = 0; len < bytes.size(); ++len) {
    auto back = vdb::decode_payload(s::span(bytes.data(), len));
    if (back.has_value()) {
      // A strict prefix may only decode if it is itself a valid encoding
      // of a smaller payload; it must never equal the original.
      EXPECT_NE(*back, p) << "len=" << len;
    } else {
      EXPECT_EQ(back.error().code, vdb::ErrorCode::Corruption);
    }
  }
}
