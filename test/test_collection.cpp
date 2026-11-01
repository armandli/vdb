#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <db_util.h>

namespace vt = vdb::test;

TEST(Database, CreateGetListDrop) {
  vdb::Database db = vt::open_memory();
  EXPECT_TRUE(db.list_collections().empty());
  auto a = vt::must(db.create_collection("alpha", vt::config(8)));
  auto b = vt::must(db.create_collection("beta", vt::config(4)));
  ASSERT_TRUE(a and b);
  EXPECT_EQ(a->name(), "alpha");
  EXPECT_EQ(a->config().dim, 8u);
  vec<str> names = db.list_collections();
  s::sort(names.begin(), names.end());
  EXPECT_EQ(names, (vec<str>{"alpha", "beta"}));

  auto again = db.get_collection("alpha");
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(again->get(), a.get());

  ASSERT_TRUE(db.drop_collection("alpha").has_value());
  EXPECT_EQ(db.list_collections(), (vec<str>{"beta"}));
  auto gone = db.get_collection("alpha");
  ASSERT_FALSE(gone.has_value());
  EXPECT_EQ(gone.error().code, vdb::ErrorCode::NotFound);
}

TEST(Database, CreateDuplicateFails) {
  vdb::Database db = vt::open_memory();
  ASSERT_TRUE(db.create_collection("c", vt::config(4)).has_value());
  auto dup = db.create_collection("c", vt::config(4));
  ASSERT_FALSE(dup.has_value());
  EXPECT_EQ(dup.error().code, vdb::ErrorCode::AlreadyExists);
}

TEST(Database, InvalidCollectionNamesAndConfigs) {
  vdb::Database db = vt::open_memory();
  vec<str> bad_names = {"", "has space", "slash/name", "..", "dot.name",
                        str(65, 'x')};
  for (const str& bad : bad_names) {
    auto c = db.create_collection(bad, vt::config(4));
    ASSERT_FALSE(c.has_value()) << bad;
    EXPECT_EQ(c.error().code, vdb::ErrorCode::InvalidArgument) << bad;
  }
  EXPECT_TRUE(db.create_collection("ok_name-1", vt::config(4)).has_value());
  auto zero = db.create_collection("zero", vt::config(0));
  ASSERT_FALSE(zero.has_value());
  EXPECT_EQ(zero.error().code, vdb::ErrorCode::InvalidArgument);
}

TEST(Database, GetOrCreate) {
  vdb::Database db = vt::open_memory();
  auto a = vt::must(db.get_or_create_collection("c", vt::config(16)));
  auto b = vt::must(db.get_or_create_collection("c", vt::config(16)));
  EXPECT_EQ(a.get(), b.get());
  auto mismatch = db.get_or_create_collection("c", vt::config(32));
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_EQ(mismatch.error().code, vdb::ErrorCode::InvalidArgument);
  auto metric = db.get_or_create_collection(
      "c", vt::config(16, vdb::Metric::L2));
  EXPECT_FALSE(metric.has_value());
}

TEST(Database, DroppedHandleFails) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(4)));
  ASSERT_TRUE(db.drop_collection("c").has_value());
  auto r = c->upsert(vt::numbered_records(0, 1, 4));
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, vdb::ErrorCode::NotFound);
  EXPECT_FALSE(db.drop_collection("c").has_value());
}

TEST(Collection, UpsertAndGet) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(4, vdb::Metric::L2)));
  vdb::Payload p;
  p["tenant"] = str("u1");
  p["n"] = int64_t(3);
  vdb::Record r{"a", {1.0f, 2.0f, 3.0f, 4.0f}, "hello world", p};
  auto n = c->upsert(s::span(&r, 1));
  ASSERT_TRUE(n.has_value());
  EXPECT_EQ(*n, 1u);
  EXPECT_EQ(c->count(), 1u);

  auto got = c->get("a");
  ASSERT_TRUE(got.has_value());
  ASSERT_TRUE(got->has_value());
  EXPECT_EQ((*got)->id, "a");
  EXPECT_EQ((*got)->vector, r.vector);
  EXPECT_EQ((*got)->document, "hello world");
  EXPECT_EQ((*got)->payload, p);

  auto no_vec = c->get("a", false);
  ASSERT_TRUE(no_vec.has_value() and no_vec->has_value());
  EXPECT_TRUE((*no_vec)->vector.empty());

  auto missing = c->get("zzz");
  ASSERT_TRUE(missing.has_value());
  EXPECT_FALSE(missing->has_value());
}

TEST(Collection, CosineStoresNormalizedVectors) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(2)));
  vdb::Record r{"a", {3.0f, 4.0f}, {}, {}};
  ASSERT_TRUE(c->upsert(s::span(&r, 1)).has_value());
  auto got = c->get("a");
  ASSERT_TRUE(got.has_value() and got->has_value());
  EXPECT_NEAR((*got)->vector[0], 0.6f, 1e-6f);
  EXPECT_NEAR((*got)->vector[1], 0.8f, 1e-6f);
}

TEST(Collection, UpsertReplaces) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(4, vdb::Metric::L2)));
  vdb::Record r1{"a", {1, 0, 0, 0}, "v1", {}};
  vdb::Record r2{"a", {0, 1, 0, 0}, "v2", {}};
  ASSERT_TRUE(c->upsert(s::span(&r1, 1)).has_value());
  ASSERT_TRUE(c->upsert(s::span(&r2, 1)).has_value());
  EXPECT_EQ(c->count(), 1u);
  auto got = c->get("a");
  ASSERT_TRUE(got.has_value() and got->has_value());
  EXPECT_EQ((*got)->document, "v2");
  EXPECT_EQ((*got)->vector, r2.vector);

  vdb::SearchRequest req;
  req.vector = {1, 0, 0, 0};
  req.k = 10;
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value());
  ASSERT_EQ(hits->size(), 1u) << "old version must not be returned";
  EXPECT_EQ((*hits)[0].document, "v2");
}

TEST(Collection, BatchValidationIsAllOrNothing) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(3)));
  float nan = s::nanf("");
  struct Case {
    vec<vdb::Record> batch;
    vdb::ErrorCode code;
  };
  vec<Case> cases = {
      {{{"a", {1, 2, 3}, {}, {}}, {"b", {1, 2}, {}, {}}},
       vdb::ErrorCode::DimensionMismatch},
      {{{"a", {1, 2, 3}, {}, {}}, {"", {1, 2, 3}, {}, {}}},
       vdb::ErrorCode::InvalidArgument},
      {{{"a", {1, 2, 3}, {}, {}}, {"a", {3, 2, 1}, {}, {}}},
       vdb::ErrorCode::InvalidArgument},
      {{{"a", {1, 2, 3}, {}, {}}, {"b", {1, nan, 3}, {}, {}}},
       vdb::ErrorCode::InvalidArgument},
      {{{"a", {1, 2, 3}, {}, {}}, {"b", {0, 0, 0}, {}, {}}},
       vdb::ErrorCode::InvalidArgument},
  };
  for (const Case& t : cases) {
    auto r = c->upsert(t.batch);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, t.code) << r.error().message;
    EXPECT_EQ(c->count(), 0u) << "rejected batch must not be applied";
  }
}

TEST(Collection, ZeroVectorAllowedForL2) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(3, vdb::Metric::L2)));
  vdb::Record r{"z", {0, 0, 0}, {}, {}};
  EXPECT_TRUE(c->upsert(s::span(&r, 1)).has_value());
}

TEST(Collection, RemoveAndRemoveWhere) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(8)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 100, 8)).has_value());

  vec<str> ids = {vt::id_for(1), vt::id_for(2), "unknown"};
  auto removed = c->remove(ids);
  ASSERT_TRUE(removed.has_value());
  EXPECT_EQ(*removed, 2u);
  EXPECT_EQ(c->count(), 98u);
  auto got = c->get(vt::id_for(1));
  ASSERT_TRUE(got.has_value());
  EXPECT_FALSE(got->has_value());

  auto odd = c->remove_where(vt::must(vdb::parse_filter(R"({"parity":"odd"})")));
  ASSERT_TRUE(odd.has_value());
  EXPECT_EQ(*odd, 49u);
  EXPECT_EQ(c->count(), 49u);

  vdb::SearchRequest req;
  req.vector = vt::vector_for(3, 8);
  req.k = 100;
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value());
  EXPECT_EQ(hits->size(), 49u);
  for (const vdb::SearchHit& h : *hits) {
    EXPECT_EQ(s::get<str>(h.payload.at("parity")), "even");
    EXPECT_NE(h.id, vt::id_for(2));
  }
}

TEST(Collection, CountWhere) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(4)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 50, 4)).has_value());
  auto n = c->count_where(vt::must(vdb::parse_filter(R"({"i":{"$lt":10}})")));
  ASSERT_TRUE(n.has_value());
  EXPECT_EQ(*n, 10u);
  auto all = c->count_where(vdb::f_and({}));
  ASSERT_TRUE(all.has_value());
  EXPECT_EQ(*all, 50u);
}

TEST(Collection, ExactSearchMatchesReference) {
  vdb::Database db = vt::open_memory();
  for (vdb::Metric m :
       {vdb::Metric::L2, vdb::Metric::InnerProduct, vdb::Metric::Cosine}) {
    str name = "m" + s::to_string(int(m));
    auto c = vt::must(db.create_collection(name, vt::config(16, m)));
    ASSERT_TRUE(c->upsert(vt::numbered_records(0, 500, 16)).has_value());
    vec<float> rows;
    for (size_t i = 0; i < 500; ++i) {
      vec<float> v = vt::vector_for(i, 16);
      rows.insert(rows.end(), v.begin(), v.end());
    }
    s::mt19937_64 rng(1);
    for (int t = 0; t < 10; ++t) {
      vec<float> q = vt::random_vector(rng, 16);
      vdb::SearchRequest req;
      req.vector = q;
      req.k = 10;
      req.exact = true;
      auto hits = c->search(req);
      ASSERT_TRUE(hits.has_value());
      vec<str> want;
      for (vdb::InternalId id : vt::brute_force_ids(m, rows, 16, q, 10)) {
        want.push_back(vt::id_for(id));
      }
      EXPECT_EQ(vt::ids_of(*hits), want) << "metric " << int(m);
    }
  }
}

TEST(Collection, ScoreSemanticsAndOrdering) {
  vdb::Database db = vt::open_memory();
  auto cos = vt::must(db.create_collection("cos", vt::config(8)));
  auto l2 = vt::must(db.create_collection("l2", vt::config(8, vdb::Metric::L2)));
  ASSERT_TRUE(cos->upsert(vt::numbered_records(0, 200, 8)).has_value());
  ASSERT_TRUE(l2->upsert(vt::numbered_records(0, 200, 8)).has_value());
  vdb::SearchRequest req;
  req.vector = vt::vector_for(7, 8);
  req.k = 20;
  auto ch = cos->search(req);
  auto lh = l2->search(req);
  ASSERT_TRUE(ch.has_value() and lh.has_value());
  ASSERT_EQ(ch->size(), 20u);
  EXPECT_EQ((*ch)[0].id, vt::id_for(7));
  for (size_t i = 0; i < ch->size(); ++i) {
    EXPECT_NEAR((*ch)[i].score, 1.0f - (*ch)[i].distance, 1e-6f);
    EXPECT_NEAR((*lh)[i].score, -(*lh)[i].distance, 1e-6f);
    if (i > 0) {
      EXPECT_GE((*ch)[i - 1].score, (*ch)[i].score);
      EXPECT_GE((*lh)[i - 1].score, (*lh)[i].score);
    }
  }
}

TEST(Collection, SearchEdgeCases) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(4)));

  vdb::SearchRequest req;
  req.vector = {1, 0, 0, 0};
  auto empty = c->search(req);
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());

  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 5, 4)).has_value());
  req.k = 0;
  auto k0 = c->search(req);
  ASSERT_TRUE(k0.has_value());
  EXPECT_TRUE(k0->empty());

  req.k = 100;
  auto big = c->search(req);
  ASSERT_TRUE(big.has_value());
  EXPECT_EQ(big->size(), 5u);

  req.vector = {1, 0};
  auto dim = c->search(req);
  ASSERT_FALSE(dim.has_value());
  EXPECT_EQ(dim.error().code, vdb::ErrorCode::DimensionMismatch);

  req.vector.clear();
  auto nothing = c->search(req);
  ASSERT_FALSE(nothing.has_value());
  EXPECT_EQ(nothing.error().code, vdb::ErrorCode::InvalidArgument);

  req.text = "hello";
  auto no_text_index = c->search(req);
  ASSERT_FALSE(no_text_index.has_value());
  EXPECT_EQ(no_text_index.error().code, vdb::ErrorCode::InvalidArgument);
}

TEST(Collection, ResultFieldSelection) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(4)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 10, 4)).has_value());
  vdb::SearchRequest req;
  req.vector = vt::vector_for(0, 4);
  req.k = 3;
  req.with_vector = true;
  auto full = c->search(req);
  ASSERT_TRUE(full.has_value());
  EXPECT_EQ((*full)[0].vector.size(), 4u);
  EXPECT_FALSE((*full)[0].payload.empty());
  EXPECT_FALSE((*full)[0].document.empty());

  req.with_vector = false;
  req.with_payload = false;
  req.with_document = false;
  auto bare = c->search(req);
  ASSERT_TRUE(bare.has_value());
  EXPECT_TRUE((*bare)[0].vector.empty());
  EXPECT_TRUE((*bare)[0].payload.empty());
  EXPECT_TRUE((*bare)[0].document.empty());
}

TEST(Collection, OptimizeKeepsResultsAndIndexes) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(32)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 3000, 32)).has_value());
  vdb::SearchRequest req;
  req.vector = vt::vector_for(42, 32);
  req.k = 10;
  req.exact = true;
  auto before = c->search(req);
  ASSERT_TRUE(before.has_value());

  ASSERT_TRUE(c->optimize(true).has_value());
  vdb::CollectionStats st = c->stats();
  EXPECT_EQ(st.live_points, 3000u);
  EXPECT_GE(st.indexed_segments, 1u);

  auto after = c->search(req);
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(vt::ids_of(*after), vt::ids_of(*before));

  s::mt19937_64 rng(3);
  vec<vec<float>> qs;
  for (int i = 0; i < 50; ++i) {
    qs.push_back(vt::random_vector(rng, 32));
  }
  EXPECT_GE(vt::collection_recall(*c, qs, 10), 0.95);
}

TEST(Collection, WritesAfterOptimizeAreVisible) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(16)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 1000, 16)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());
  ASSERT_TRUE(c->upsert(vt::numbered_records(1000, 1100, 16)).has_value());
  EXPECT_EQ(c->count(), 1100u);
  vdb::SearchRequest req;
  req.vector = vt::vector_for(1050, 16);
  req.k = 1;
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value() and not hits->empty());
  EXPECT_EQ((*hits)[0].id, vt::id_for(1050));

  // Replacing a point that lives in a sealed segment must hide the old copy.
  vec<vdb::Record> replace = vt::numbered_records(5, 6, 16);
  replace[0].vector = vt::vector_for(99999, 16);
  ASSERT_TRUE(c->upsert(replace).has_value());
  req.vector = vt::vector_for(5, 16);
  req.k = 1100;
  hits = c->search(req);
  ASSERT_TRUE(hits.has_value());
  vec<str> ids = vt::ids_of(*hits);
  EXPECT_EQ(s::count(ids.begin(), ids.end(), vt::id_for(5)), 1);
  EXPECT_EQ(c->count(), 1100u);
}

TEST(Collection, ExplainReflectsPlanner) {
  vdb::Database db = vt::open_memory();
  vdb::CollectionConfig cfg = vt::config(64);
  cfg.indexes = {{"i", vdb::FieldType::Integer}};
  cfg.full_scan_threshold_bytes = 64 * 4 * 100;  // 100 vectors
  auto c = vt::must(db.create_collection("c", cfg));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 5000, 64)).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());

  vdb::SearchRequest req;
  req.vector = vt::vector_for(0, 64);
  auto plain = c->explain(req);
  ASSERT_TRUE(plain.has_value());
  ASSERT_FALSE(plain->empty());
  for (const vdb::SegmentPlan& p : *plain) {
    if (p.live_points > 100) {
      EXPECT_EQ(p.strategy, vdb::SearchStrategy::Graph);
    }
  }

  req.filter = vt::must(vdb::parse_filter(R"({"i":{"$lt":10}})"));
  auto narrow = c->explain(req);
  ASSERT_TRUE(narrow.has_value());
  size_t matched = 0;
  for (const vdb::SegmentPlan& p : *narrow) {
    EXPECT_EQ(p.strategy, vdb::SearchStrategy::PlainFromCandidates);
    matched += p.estimate.exp;
  }
  EXPECT_EQ(matched, 10u);

  req.exact = true;
  req.filter.reset();
  auto exact = c->explain(req);
  ASSERT_TRUE(exact.has_value());
  for (const vdb::SegmentPlan& p : *exact) {
    EXPECT_EQ(p.strategy, vdb::SearchStrategy::Plain);
  }
}

TEST(Collection, Sq8QuantizedRecall) {
  vdb::Database db = vt::open_memory();
  vdb::CollectionConfig cfg = vt::config(48);
  cfg.quantization = vdb::Quantization::SQ8;
  auto c = vt::must(db.create_collection("c", cfg));
  s::mt19937_64 rng(5);
  vec<float> rows = vt::clustered_dataset(rng, 5000, 48, 25);
  vec<vdb::Record> records;
  for (size_t i = 0; i < 5000; ++i) {
    records.push_back({vt::id_for(i),
                       vec<float>(rows.begin() + long(i * 48),
                                  rows.begin() + long((i + 1) * 48)),
                       {}, {}});
  }
  ASSERT_TRUE(c->upsert(records).has_value());
  ASSERT_TRUE(c->optimize(true).has_value());
  vec<vec<float>> qs;
  vec<float> qrows = vt::clustered_dataset(rng, 50, 48, 25);
  for (size_t i = 0; i < 50; ++i) {
    qs.emplace_back(qrows.begin() + long(i * 48),
                    qrows.begin() + long((i + 1) * 48));
  }
  EXPECT_GE(vt::collection_recall(*c, qs, 10), 0.95);
}

TEST(Collection, TextAndHybridSearch) {
  vdb::Database db = vt::open_memory();
  vdb::CollectionConfig cfg = vt::config(4);
  cfg.text_index = true;
  auto c = vt::must(db.create_collection("c", cfg));
  vec<vdb::Record> records = {
      {"sensor", {1, 0, 0, 0}, "error TS-999 means the sensor timed out", {}},
      {"pump", {0, 1, 0, 0}, "the pump overheats when the filter clogs", {}},
      {"fan", {0.9f, 0.1f, 0, 0}, "fan noise is normal at startup", {}},
      {"misc", {0, 0, 1, 0}, "unrelated text about gardening", {}},
  };
  ASSERT_TRUE(c->upsert(records).has_value());

  vdb::SearchRequest text;
  text.text = "TS-999 sensor";
  text.k = 10;
  auto th = c->search(text);
  ASSERT_TRUE(th.has_value());
  ASSERT_FALSE(th->empty());
  EXPECT_EQ((*th)[0].id, "sensor");
  EXPECT_TRUE(s::isnan((*th)[0].distance));
  for (const vdb::SearchHit& h : *th) {
    EXPECT_NE(h.id, "misc") << "documents without query terms are excluded";
  }

  // Vector alone prefers "fan"/"sensor" equally-ish; text pins "pump".
  vdb::SearchRequest hybrid;
  hybrid.vector = {0.7f, 0.7f, 0, 0};
  hybrid.text = "pump filter";
  hybrid.k = 4;
  auto hh = c->search(hybrid);
  ASSERT_TRUE(hh.has_value());
  ASSERT_FALSE(hh->empty());
  EXPECT_EQ((*hh)[0].id, "pump");
  EXPECT_FALSE(s::isnan((*hh)[0].distance));
}

TEST(Collection, MemoryScoringPrefersRecentImportant) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("mem", vt::config(4)));
  const int64_t now = 1'800'000'000;
  auto memory = [&](str id, vec<float> v, int64_t age_hours, int64_t imp) {
    vdb::Payload p;
    p["ts"] = now - age_hours * 3600;
    p["importance"] = imp;
    return vdb::Record{s::move(id), s::move(v), {}, s::move(p)};
  };
  vec<vdb::Record> records = {
      memory("old_trivial", {1.0f, 0.0f, 0, 0}, 24 * 300, 1),
      memory("new_important", {0.95f, 0.3f, 0, 0}, 1, 9),
      memory("unrelated", {0, 0, 1, 0}, 1, 1),
  };
  ASSERT_TRUE(c->upsert(records).has_value());

  vdb::SearchRequest plain;
  plain.vector = {1, 0, 0, 0};
  plain.k = 3;
  auto ph = c->search(plain);
  ASSERT_TRUE(ph.has_value());
  EXPECT_EQ((*ph)[0].id, "old_trivial");

  vdb::SearchRequest mem = plain;
  mem.memory = vdb::MemoryScoring{};
  mem.memory->now_unix = now;
  auto mh = c->search(mem);
  ASSERT_TRUE(mh.has_value());
  ASSERT_EQ(mh->size(), 3u);
  EXPECT_EQ((*mh)[0].id, "new_important");
  for (size_t i = 1; i < mh->size(); ++i) {
    EXPECT_GE((*mh)[i - 1].score, (*mh)[i].score);
  }
}

TEST(Collection, Stats) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(8)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 100, 8)).has_value());
  vec<str> ids = {vt::id_for(0), vt::id_for(1)};
  ASSERT_TRUE(c->remove(ids).has_value());
  vdb::CollectionStats st = c->stats();
  EXPECT_EQ(st.live_points, 98u);
  EXPECT_GE(st.segments, 1u);
  EXPECT_GE(st.vector_bytes, 98u * 8 * sizeof(float));
}
