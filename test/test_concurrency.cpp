#include <atomic>
#include <thread>

#include <gtest/gtest.h>

#include <db_util.h>

namespace vt = vdb::test;

namespace {

constexpr uint32_t DIM = 16;

}  // namespace

// Readers search, get and explain while one writer upserts, deletes and
// optimizes. Run under `make tsan` to catch races; here we check that no
// call fails and every read is consistent.
TEST(Concurrency, ReadersAndWriter) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(DIM)));
  ASSERT_TRUE(c->upsert(vt::numbered_records(0, 2000, DIM)).has_value());

  s::atomic<bool> stop{false};
  s::atomic<size_t> errors{0};
  s::atomic<size_t> searches{0};

  vec<s::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&, t] {
      s::mt19937_64 rng{uint64_t(t)};
      while (not stop.load()) {
        vdb::SearchRequest req;
        req.vector = vt::random_vector(rng, DIM);
        req.k = 10;
        if (t % 2 == 1) {
          req.filter = vdb::f_cmp(vdb::FilterOp::Eq, "parity", str("even"));
        }
        auto hits = c->search(req);
        if (not hits.has_value() or hits->size() != 10) {
          ++errors;
        }
        for (const vdb::SearchHit& h : hits.value_or(vec<vdb::SearchHit>{})) {
          if (req.filter and
              s::get<str>(h.payload.at("parity")) != "even") {
            ++errors;
          }
        }
        if (not c->get(vt::id_for(1)).has_value()) {
          ++errors;
        }
        ++searches;
      }
    });
  }

  for (size_t batch = 0; batch < 20; ++batch) {
    size_t begin = 2000 + batch * 100;
    ASSERT_TRUE(c->upsert(vt::numbered_records(begin, begin + 100, DIM))
                    .has_value());
    vec<str> doomed = {vt::id_for(begin + 1), vt::id_for(begin + 3)};
    ASSERT_TRUE(c->remove(doomed).has_value());
    if (batch % 5 == 4) {
      ASSERT_TRUE(c->optimize(true).has_value());
    }
  }
  stop = true;
  for (s::thread& r : readers) {
    r.join();
  }
  EXPECT_EQ(errors.load(), 0u);
  EXPECT_GT(searches.load(), 0u);
  EXPECT_EQ(c->count(), 2000u + 20 * 100 - 20 * 2);
}

// Once upsert() returns on one thread, every other thread must see it.
TEST(Concurrency, ReadYourWritesAcrossThreads) {
  vdb::Database db = vt::open_memory();
  auto c = vt::must(db.create_collection("c", vt::config(DIM)));
  s::atomic<int64_t> published{-1};
  s::atomic<size_t> misses{0};

  s::thread reader([&] {
    int64_t seen = -1;
    while (seen < 499) {
      int64_t p = published.load(s::memory_order_acquire);
      if (p == seen) {
        continue;
      }
      seen = p;
      auto got = c->get(vt::id_for(size_t(p)));
      if (not got.has_value() or not got->has_value()) {
        ++misses;
      }
      vdb::SearchRequest req;
      req.vector = vt::vector_for(size_t(p), DIM);
      req.k = 1;
      auto hits = c->search(req);
      if (not hits.has_value() or hits->empty() or
          (*hits)[0].id != vt::id_for(size_t(p))) {
        ++misses;
      }
    }
  });

  for (size_t i = 0; i < 500; ++i) {
    ASSERT_TRUE(c->upsert(vt::numbered_records(i, i + 1, DIM)).has_value());
    published.store(int64_t(i), s::memory_order_release);
    if (i % 100 == 99) {
      ASSERT_TRUE(c->optimize(true).has_value());
    }
  }
  reader.join();
  EXPECT_EQ(misses.load(), 0u);
}

TEST(Concurrency, ConcurrentWritersToDifferentCollections) {
  vdb::Database db = vt::open_memory();
  vec<sptr<vdb::Collection>> cols;
  for (int i = 0; i < 4; ++i) {
    cols.push_back(vt::must(
        db.create_collection("c" + s::to_string(i), vt::config(DIM))));
  }
  vec<s::thread> writers;
  s::atomic<size_t> errors{0};
  for (int i = 0; i < 4; ++i) {
    writers.emplace_back([&, i] {
      for (size_t b = 0; b < 10; ++b) {
        if (not cols[i]->upsert(vt::numbered_records(b * 50, b * 50 + 50, DIM))
                    .has_value()) {
          ++errors;
        }
      }
    });
  }
  for (s::thread& w : writers) {
    w.join();
  }
  EXPECT_EQ(errors.load(), 0u);
  for (const auto& c : cols) {
    EXPECT_EQ(c->count(), 500u);
  }
}
