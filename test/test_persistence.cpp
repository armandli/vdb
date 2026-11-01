#include <csignal>
#include <fstream>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <db_util.h>

namespace vt = vdb::test;

namespace {

constexpr uint32_t DIM = 16;

void expect_numbered(const vdb::Collection& c, size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    auto got = c.get(vt::id_for(i));
    ASSERT_TRUE(got.has_value());
    ASSERT_TRUE(got->has_value()) << vt::id_for(i) << " missing";
    vec<float> want = vt::vector_for(i, DIM);
    vt::normalize_ref(want);
    for (uint32_t d = 0; d < DIM; ++d) {
      EXPECT_NEAR((*got)->vector[d], want[d], 1e-6f);
    }
    EXPECT_EQ(s::get<int64_t>((*got)->payload.at("i")), int64_t(i));
  }
}

}  // namespace

TEST(Persistence, OpenCreatesDirectory) {
  vt::TempDir dir;
  sf::path path = dir.path() / "db";
  auto db = vdb::Database::open(path);
  ASSERT_TRUE(db.has_value()) << db.error().message;
  EXPECT_TRUE(sf::is_directory(path));
  EXPECT_EQ(db->path(), path);
}

TEST(Persistence, OpenMissingWithoutCreateFails) {
  vt::TempDir dir;
  vdb::DatabaseOptions options;
  options.create_if_missing = false;
  auto db = vdb::Database::open(dir.path() / "nope", options);
  ASSERT_FALSE(db.has_value());
  EXPECT_EQ(db.error().code, vdb::ErrorCode::NotFound);
}

TEST(Persistence, SecondOpenIsLockedOut) {
  vt::TempDir dir;
  auto first = vdb::Database::open(dir.path());
  ASSERT_TRUE(first.has_value());
  auto second = vdb::Database::open(dir.path());
  ASSERT_FALSE(second.has_value());
  EXPECT_EQ(second.error().code, vdb::ErrorCode::IOError);
  ASSERT_TRUE(first->close().has_value());
  auto third = vdb::Database::open(dir.path());
  EXPECT_TRUE(third.has_value()) << "close() must release the lock";
}

TEST(Persistence, ReopenAfterClose) {
  vt::TempDir dir;
  vdb::CollectionConfig cfg = vt::config(DIM);
  cfg.indexes = {{"i", vdb::FieldType::Integer}};
  cfg.text_index = true;
  cfg.embedding_model = "test-model-v1";
  {
    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value());
    auto c = vt::must(db->create_collection("docs", cfg));
    ASSERT_TRUE(c->upsert(vt::numbered_records(0, 500, DIM)).has_value());
    ASSERT_TRUE(db->close().has_value());
  }
  auto db = vdb::Database::open(dir.path());
  ASSERT_TRUE(db.has_value());
  EXPECT_EQ(db->list_collections(), (vec<str>{"docs"}));
  auto c = vt::must(db->get_collection("docs"));
  EXPECT_EQ(c->count(), 500u);
  EXPECT_EQ(c->config().dim, DIM);
  EXPECT_EQ(c->config().embedding_model, "test-model-v1");
  EXPECT_TRUE(c->config().text_index);
  EXPECT_EQ(c->config().indexes, cfg.indexes);
  expect_numbered(*c, 0, 500);
}

TEST(Persistence, DestructorPersists) {
  vt::TempDir dir;
  {
    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value());
    auto c = vt::must(db->create_collection("c", vt::config(DIM)));
    ASSERT_TRUE(c->upsert(vt::numbered_records(0, 50, DIM)).has_value());
  }
  auto db = vdb::Database::open(dir.path());
  ASSERT_TRUE(db.has_value());
  expect_numbered(*vt::must(db->get_collection("c")), 0, 50);
}

TEST(Persistence, DeletesAndDropsPersist) {
  vt::TempDir dir;
  {
    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value());
    auto keep = vt::must(db->create_collection("keep", vt::config(DIM)));
    auto gone = vt::must(db->create_collection("gone", vt::config(DIM)));
    ASSERT_TRUE(keep->upsert(vt::numbered_records(0, 100, DIM)).has_value());
    ASSERT_TRUE(gone->upsert(vt::numbered_records(0, 10, DIM)).has_value());
    vec<str> ids = {vt::id_for(3), vt::id_for(4)};
    ASSERT_TRUE(keep->remove(ids).has_value());
    ASSERT_TRUE(db->drop_collection("gone").has_value());
  }
  auto db = vdb::Database::open(dir.path());
  ASSERT_TRUE(db.has_value());
  EXPECT_EQ(db->list_collections(), (vec<str>{"keep"}));
  EXPECT_FALSE(sf::exists(dir.path() / "gone"));
  auto keep = vt::must(db->get_collection("keep"));
  EXPECT_EQ(keep->count(), 98u);
  auto got = keep->get(vt::id_for(3));
  ASSERT_TRUE(got.has_value());
  EXPECT_FALSE(got->has_value());
}

TEST(Persistence, OptimizedSegmentsSurviveReopen) {
  vt::TempDir dir;
  vdb::SearchRequest req;
  req.vector = vt::vector_for(17, DIM);
  req.k = 10;
  vec<str> before;
  {
    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value());
    auto c = vt::must(db->create_collection("c", vt::config(DIM)));
    ASSERT_TRUE(c->upsert(vt::numbered_records(0, 2000, DIM)).has_value());
    ASSERT_TRUE(c->optimize(true).has_value());
    ASSERT_TRUE(c->upsert(vt::numbered_records(2000, 2100, DIM)).has_value());
    auto hits = c->search(req);
    ASSERT_TRUE(hits.has_value());
    before = vt::ids_of(*hits);
  }
  auto db = vdb::Database::open(dir.path());
  ASSERT_TRUE(db.has_value());
  auto c = vt::must(db->get_collection("c"));
  EXPECT_EQ(c->count(), 2100u);
  EXPECT_GE(c->stats().indexed_segments, 1u);
  auto hits = c->search(req);
  ASSERT_TRUE(hits.has_value());
  EXPECT_EQ(vt::ids_of(*hits), before);
}

TEST(Persistence, DurabilityNoneStillPersistsOnClose) {
  vt::TempDir dir;
  vdb::DatabaseOptions options;
  options.durability = vdb::Durability::None;
  {
    auto db = vdb::Database::open(dir.path(), options);
    ASSERT_TRUE(db.has_value());
    auto c = vt::must(db->create_collection("c", vt::config(DIM)));
    ASSERT_TRUE(c->upsert(vt::numbered_records(0, 30, DIM)).has_value());
    ASSERT_TRUE(db->close().has_value());
  }
  auto db = vdb::Database::open(dir.path(), options);
  ASSERT_TRUE(db.has_value());
  expect_numbered(*vt::must(db->get_collection("c")), 0, 30);
}

TEST(Persistence, CorruptManifestIsAnErrorNotACrash) {
  vt::TempDir dir;
  {
    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value());
    ASSERT_TRUE(db->create_collection("c", vt::config(DIM)).has_value());
  }
  {
    s::ofstream out(dir.path() / "MANIFEST", s::ios::trunc);
    out << "\x01\x02 definitely not a manifest";
  }
  auto db = vdb::Database::open(dir.path());
  ASSERT_FALSE(db.has_value());
  EXPECT_EQ(db.error().code, vdb::ErrorCode::Corruption);
}

// With Durability::Sync, a write that returned is on disk even if the
// process dies before flush/close: the WAL replays it.
TEST(Persistence, AckedWritesSurviveProcessExit) {
  vt::TempDir dir;
  {
    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value());
    ASSERT_TRUE(db->create_collection("c", vt::config(DIM)).has_value());
  }
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    auto db = vdb::Database::open(dir.path());
    if (not db.has_value()) {
      _exit(2);
    }
    auto c = db->get_collection("c");
    if (not c.has_value()) {
      _exit(3);
    }
    for (size_t i = 0; i < 300; ++i) {
      if (not (*c)->upsert(vt::numbered_records(i, i + 1, DIM)).has_value()) {
        _exit(4);
      }
    }
    _exit(0);  // no destructor, no flush
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  auto db = vdb::Database::open(dir.path());
  ASSERT_TRUE(db.has_value()) << db.error().message;
  auto c = vt::must(db->get_collection("c"));
  EXPECT_EQ(c->count(), 300u);
  expect_numbered(*c, 0, 300);
}

// SIGKILL mid-ingest: every write the child reported as acknowledged must
// be present after reopen, and the database must open cleanly.
TEST(Persistence, KillDuringIngestKeepsAckedPrefix) {
  for (int round = 0; round < 5; ++round) {
    vt::TempDir dir;
    {
      auto db = vdb::Database::open(dir.path());
      ASSERT_TRUE(db.has_value());
      ASSERT_TRUE(db->create_collection("c", vt::config(DIM)).has_value());
    }
    int fds[2];
    ASSERT_EQ(pipe(fds), 0);
    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      close(fds[0]);
      auto db = vdb::Database::open(dir.path());
      if (not db.has_value()) {
        _exit(2);
      }
      auto c = db->get_collection("c");
      for (uint32_t i = 0;; ++i) {
        if (not (*c)->upsert(vt::numbered_records(i, i + 1, DIM))) {
          _exit(4);
        }
        if (i % 64 == 63) {
          (void)(*c)->optimize();
        }
        if (write(fds[1], &i, sizeof i) != sizeof i) {
          _exit(5);
        }
      }
    }
    close(fds[1]);
    uint32_t acked = 0;
    uint32_t value = 0;
    size_t reads = 0;
    while (reads < 100u + 37u * size_t(round) and
           read(fds[0], &value, sizeof value) == sizeof value) {
      acked = value;
      ++reads;
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    close(fds[0]);
    ASSERT_GT(reads, 0u) << "child died before acknowledging anything";

    auto db = vdb::Database::open(dir.path());
    ASSERT_TRUE(db.has_value()) << "round " << round << ": "
                                << db.error().message;
    auto c = vt::must(db->get_collection("c"));
    EXPECT_GE(c->count(), size_t(acked) + 1) << "round " << round;
    expect_numbered(*c, 0, size_t(acked) + 1);
  }
}
