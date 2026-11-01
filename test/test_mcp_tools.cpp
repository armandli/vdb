#include <cctype>
#include <functional>

#include <gtest/gtest.h>
#include <simdjson.h>

#include <db_util.h>
#include <vdb/mcp_tools.h>

namespace vt = vdb::test;
namespace sj = simdjson;

namespace {

constexpr uint32_t DIM = 64;

// Deterministic bag-of-words embedder: each token hashes to a dimension,
// so texts sharing words are close under cosine.
struct FakeEmbedder : vdb::Embedder {
  strv model_id() const override { return "fake-bow-64"; }
  uint32_t dim() const override { return DIM; }

  vdb::Result<vec<vec<float>>> embed(s::span<const str> texts) override {
    vec<vec<float>> out;
    for (const str& text : texts) {
      vec<float> v(DIM, 0.0f);
      str word;
      auto flush = [&] {
        if (not word.empty()) {
          v[s::hash<str>{}(word) % DIM] += 1.0f;
          word.clear();
        }
      };
      for (char ch : text) {
        if (s::isalnum(static_cast<unsigned char>(ch))) {
          word += char(s::tolower(static_cast<unsigned char>(ch)));
        } else {
          flush();
        }
      }
      flush();
      v[DIM - 1] += 0.01f;  // never the zero vector
      out.push_back(s::move(v));
    }
    ++calls;
    return out;
  }

  size_t calls = 0;
};

struct Json {
  sj::dom::parser parser;
  sj::dom::element root;
};

uptr<Json> parse(const str& text) {
  auto j = s::make_unique<Json>();
  auto r = j->parser.parse(text);
  EXPECT_EQ(r.error(), sj::SUCCESS) << text;
  if (r.error() == sj::SUCCESS) {
    j->root = r.value();
  }
  return j;
}

uptr<Json> call_ok(vdb::McpTools& tools, strv tool, strv args) {
  auto r = tools.call(tool, args);
  EXPECT_TRUE(r.has_value()) << tool << " " << args << ": "
                             << (r.has_value() ? "" : r.error().message);
  return parse(r.value_or("{}"));
}

vec<str> hit_ids(const Json& j) {
  vec<str> ids;
  for (sj::dom::element hit : j.root["hits"].get_array()) {
    ids.emplace_back(strv(hit["id"].get_string()));
  }
  return ids;
}

str vector_json(const vec<float>& v) {
  str out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    out += (i ? "," : "") + s::to_string(v[i]);
  }
  return out + "]";
}

}  // namespace

TEST(McpTools, ListsAllToolsWithSchemas) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db, s::make_shared<FakeEmbedder>());
  uset<str> names;
  for (const vdb::ToolSpec& t : tools.list_tools()) {
    names.insert(t.name);
    EXPECT_FALSE(t.description.empty()) << t.name;
    uptr<Json> schema = parse(t.input_schema);
    EXPECT_EQ(strv(schema->root["type"].get_string()), "object") << t.name;
  }
  for (strv want :
       {"vdb_list_collections", "vdb_create_collection", "vdb_upsert",
        "vdb_search", "vdb_get", "vdb_delete", "memory_remember",
        "memory_recall", "memory_forget"}) {
    EXPECT_TRUE(names.contains(str(want))) << want;
  }
}

TEST(McpTools, UnknownToolAndBadArguments) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db);
  EXPECT_FALSE(tools.call("no_such_tool", "{}").has_value());
  auto bad_json = tools.call("vdb_search", "{not json");
  ASSERT_FALSE(bad_json.has_value());
  EXPECT_EQ(bad_json.error().code, vdb::ErrorCode::ParseError);
  auto missing = tools.call("vdb_search", R"({"k":3})");
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error().code, vdb::ErrorCode::InvalidArgument);
  auto unknown_col =
      tools.call("vdb_search", R"({"collection":"nope","vector":[1]})");
  ASSERT_FALSE(unknown_col.has_value());
  EXPECT_EQ(unknown_col.error().code, vdb::ErrorCode::NotFound);
}

TEST(McpTools, VectorCrud) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db);
  call_ok(tools, "vdb_create_collection",
          R"({"name":"docs","dim":3,"metric":"l2",
              "indexes":[{"field":"tag","type":"keyword"}]})");
  uptr<Json> list = call_ok(tools, "vdb_list_collections", "{}");
  EXPECT_EQ(list->root["collections"].get_array().size(), 1u);

  uptr<Json> up = call_ok(tools, "vdb_upsert", R"({"collection":"docs",
      "records":[
        {"id":"a","vector":[1,0,0],"document":"alpha","payload":{"tag":"x"}},
        {"id":"b","vector":[0,1,0],"document":"beta","payload":{"tag":"y"}},
        {"id":"c","vector":[0,0,1],"document":"gamma","payload":{"tag":"x"}}
      ]})");
  EXPECT_EQ(int64_t(up->root["upserted"].get_int64()), 3);

  uptr<Json> hits = call_ok(tools, "vdb_search",
      R"({"collection":"docs","vector":[0.9,0.1,0],"k":2})");
  EXPECT_EQ(hit_ids(*hits), (vec<str>{"a", "b"}));
  sj::dom::element first = hits->root["hits"].get_array().at(0);
  EXPECT_EQ(strv(first["document"].get_string()), "alpha");
  EXPECT_EQ(strv(first["payload"]["tag"].get_string()), "x");
  EXPECT_TRUE(first["score"].is_number());

  uptr<Json> filtered = call_ok(tools, "vdb_search",
      R"({"collection":"docs","vector":[0,1,0],"k":5,"filter":{"tag":"x"}})");
  vec<str> ids = hit_ids(*filtered);
  s::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids, (vec<str>{"a", "c"}));

  uptr<Json> got = call_ok(tools, "vdb_get",
      R"({"collection":"docs","ids":["b","missing"]})");
  sj::dom::array records = got->root["records"].get_array();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(strv(records.at(0)["id"].get_string()), "b");

  uptr<Json> del = call_ok(tools, "vdb_delete",
      R"({"collection":"docs","filter":{"tag":"x"}})");
  EXPECT_EQ(int64_t(del->root["deleted"].get_int64()), 2);
  uptr<Json> del2 = call_ok(tools, "vdb_delete",
      R"({"collection":"docs","ids":["b"]})");
  EXPECT_EQ(int64_t(del2->root["deleted"].get_int64()), 1);
  EXPECT_EQ(vt::must(db.get_collection("docs"))->count(), 0u);
}

TEST(McpTools, UpsertRejectsWrongDimension) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db);
  call_ok(tools, "vdb_create_collection", R"({"name":"c","dim":2})");
  auto r = tools.call("vdb_upsert",
      R"({"collection":"c","records":[{"id":"a","vector":[1,2,3]}]})");
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, vdb::ErrorCode::DimensionMismatch);
}

TEST(McpTools, TextInTextOutWithEmbedder) {
  vdb::Database db = vt::open_memory();
  auto embedder = s::make_shared<FakeEmbedder>();
  vdb::McpTools tools(db, embedder);
  // dim omitted: taken from the embedder, model id recorded.
  call_ok(tools, "vdb_create_collection", R"({"name":"notes"})");
  auto col = vt::must(db.get_collection("notes"));
  EXPECT_EQ(col->config().dim, DIM);
  EXPECT_EQ(col->config().embedding_model, "fake-bow-64");

  call_ok(tools, "vdb_upsert", R"({"collection":"notes","records":[
      {"id":"1","document":"peanut allergy is severe"},
      {"id":"2","document":"likes ramen and sushi"},
      {"id":"3","document":"works on vector databases"}]})");
  EXPECT_GE(embedder->calls, 1u);

  uptr<Json> hits = call_ok(tools, "vdb_search",
      R"({"collection":"notes","query":"what about the allergy","k":1})");
  EXPECT_EQ(hit_ids(*hits), (vec<str>{"1"}));
}

TEST(McpTools, TextWithoutEmbedderNeedsVectors) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db);
  call_ok(tools, "vdb_create_collection", R"({"name":"c","dim":4})");
  auto r = tools.call("vdb_upsert",
      R"({"collection":"c","records":[{"id":"a","document":"text only"}]})");
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, vdb::ErrorCode::InvalidArgument);
  auto create = tools.call("vdb_create_collection", R"({"name":"nodim"})");
  ASSERT_FALSE(create.has_value());
  EXPECT_EQ(create.error().code, vdb::ErrorCode::InvalidArgument);
}

TEST(McpTools, MemoryRememberRecallForget) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db, s::make_shared<FakeEmbedder>());

  uptr<Json> a = call_ok(tools, "memory_remember",
      R"({"namespace":"user42","content":"user is allergic to peanuts",
          "kind":"semantic","importance":9})");
  str allergy_id(strv(a->root["id"].get_string()));
  EXPECT_FALSE(allergy_id.empty());
  call_ok(tools, "memory_remember",
          R"({"namespace":"user42","content":"user enjoys hiking trips"})");
  call_ok(tools, "memory_remember",
          R"({"namespace":"other","content":"other user allergic to cats"})");

  uptr<Json> recall = call_ok(tools, "memory_recall",
      R"({"namespace":"user42","query":"allergic food","k":5})");
  vec<str> ids = hit_ids(*recall);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids[0], allergy_id);
  EXPECT_EQ(ids.size(), 2u) << "namespaces are isolated";
  sj::dom::element top = recall->root["hits"].get_array().at(0);
  EXPECT_EQ(strv(top["document"].get_string()), "user is allergic to peanuts");
  EXPECT_EQ(strv(top["payload"]["kind"].get_string()), "semantic");

  uptr<Json> kinds = call_ok(tools, "memory_recall",
      R"({"namespace":"user42","query":"anything","kinds":["semantic"]})");
  EXPECT_EQ(hit_ids(*kinds), (vec<str>{allergy_id}));

  str forget_args = R"({"namespace":"other","id":")" + allergy_id + R"("})";
  uptr<Json> wrong_ns = call_ok(tools, "memory_forget", forget_args);
  EXPECT_EQ(int64_t(wrong_ns->root["deleted"].get_int64()), 0)
      << "cannot forget another namespace's memory";

  forget_args = R"({"namespace":"user42","id":")" + allergy_id + R"("})";
  uptr<Json> forgot = call_ok(tools, "memory_forget", forget_args);
  EXPECT_EQ(int64_t(forgot->root["deleted"].get_int64()), 1);
  uptr<Json> after = call_ok(tools, "memory_recall",
      R"({"namespace":"user42","query":"allergic food","k":5})");
  EXPECT_EQ(hit_ids(*after).size(), 1u);
}

TEST(McpTools, MemoryNeedsEmbedder) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db);
  auto r = tools.call("memory_remember",
                      R"({"namespace":"n","content":"something"})");
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, vdb::ErrorCode::Unsupported);
}

TEST(McpTools, SearchWithExplicitVectorAndK) {
  vdb::Database db = vt::open_memory();
  vdb::McpTools tools(db);
  call_ok(tools, "vdb_create_collection", R"({"name":"c","dim":8})");
  auto col = vt::must(db.get_collection("c"));
  ASSERT_TRUE(col->upsert(vt::numbered_records(0, 50, 8)).has_value());
  str args = R"({"collection":"c","k":7,"vector":)" +
      vector_json(vt::vector_for(3, 8)) + "}";
  uptr<Json> hits = call_ok(tools, "vdb_search", args);
  vec<str> ids = hit_ids(*hits);
  ASSERT_EQ(ids.size(), 7u);
  EXPECT_EQ(ids[0], vt::id_for(3));
}
