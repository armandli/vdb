#ifndef DB_UTIL_H
#define DB_UTIL_H

#include <random>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/database.h>

// Helpers for tests that drive the public Database / Collection API.
namespace vdb::test {

inline vdb::Record make_record(
    s::mt19937_64& rng,
    str id,
    uint32_t dim,
    vdb::Payload payload = {},
    str document = {})
{
  return vdb::Record{s::move(id), random_vector(rng, dim), s::move(document),
                     s::move(payload)};
}

// Deterministic vector for id i, so a reopened database can be checked
// without keeping the inputs around.
inline vec<float> vector_for(size_t i, uint32_t dim) {
  s::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ i);
  return random_vector(rng, dim);
}

inline str id_for(size_t i) {
  return "doc-" + s::to_string(i);
}

inline vec<vdb::Record> numbered_records(
    size_t begin,
    size_t end,
    uint32_t dim)
{
  vec<vdb::Record> records;
  for (size_t i = begin; i < end; ++i) {
    vdb::Payload p;
    p["i"] = int64_t(i);
    p["parity"] = str(i % 2 == 0 ? "even" : "odd");
    records.push_back({id_for(i), vector_for(i, dim), "text " + id_for(i), p});
  }
  return records;
}

inline vec<str> ids_of(const vec<vdb::SearchHit>& hits) {
  vec<str> ids;
  for (const vdb::SearchHit& h : hits) {
    ids.push_back(h.id);
  }
  return ids;
}

inline vdb::Database open_memory() {
  auto db = vdb::Database::open_in_memory();
  EXPECT_TRUE(db.has_value());
  return s::move(*db);
}

inline sptr<vdb::Collection> must(Result<sptr<vdb::Collection>> c) {
  EXPECT_TRUE(c.has_value()) << (c.has_value() ? "" : c.error().message);
  return c.has_value() ? *c : nullptr;
}

inline vdb::Filter must(Result<vdb::Filter> f) {
  EXPECT_TRUE(f.has_value()) << (f.has_value() ? "" : f.error().message);
  return f.has_value() ? *f : vdb::f_and({});
}

inline vdb::CollectionConfig config(
    uint32_t dim,
    vdb::Metric metric = vdb::Metric::Cosine)
{
  vdb::CollectionConfig c;
  c.dim = dim;
  c.metric = metric;
  return c;
}

// Mean recall@k of the collection's default (approximate) search against
// its exact search, over `queries`.
inline double collection_recall(
    const vdb::Collection& col,
    const vec<vec<float>>& queries,
    size_t k,
    const opt<vdb::Filter>& filter = s::nullopt)
{
  double sum = 0.0;
  for (const vec<float>& q : queries) {
    vdb::SearchRequest approx;
    approx.vector = q;
    approx.k = k;
    approx.filter = filter;
    vdb::SearchRequest exact = approx;
    exact.exact = true;
    auto got = col.search(approx);
    auto want = col.search(exact);
    EXPECT_TRUE(got.has_value() and want.has_value());
    if (got.has_value() and want.has_value()) {
      sum += recall(ids_of(*got), ids_of(*want));
    }
  }
  return queries.empty() ? 1.0 : sum / double(queries.size());
}

}  // namespace vdb::test

#endif  // DB_UTIL_H
