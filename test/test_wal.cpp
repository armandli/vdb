#include <fstream>

#include <gtest/gtest.h>

#include <test_util.h>
#include <vdb/wal.h>

namespace vt = vdb::test;

namespace {

s::span<const s::byte> bytes_of(strv text) {
  return s::as_bytes(s::span(text.data(), text.size()));
}

struct Entry {
  vdb::OpNum op;
  str body;

  bool operator==(const Entry&) const = default;
};

vec<Entry> replay_all(const vdb::Wal& wal, vdb::OpNum from = 0) {
  vec<Entry> out;
  auto st = wal.replay(from, [&](vdb::OpNum op, s::span<const s::byte> b) {
    out.push_back({op, str(reinterpret_cast<const char*>(b.data()), b.size())});
    return vdb::Status{};
  });
  EXPECT_TRUE(st.has_value()) << st.error().message;
  return out;
}

vec<sf::path> wal_files(const sf::path& dir) {
  vec<sf::path> files;
  for (const auto& e : sf::directory_iterator(dir)) {
    if (e.path().extension() == ".log") {
      files.push_back(e.path());
    }
  }
  s::sort(files.begin(), files.end());
  return files;
}

void flip_byte(const sf::path& p, uintmax_t offset_from_end) {
  s::fstream f(p, s::ios::in | s::ios::out | s::ios::binary);
  f.seekg(-s::streamoff(offset_from_end), s::ios::end);
  char c = 0;
  f.read(&c, 1);
  c ^= 0x40;
  f.seekp(-s::streamoff(offset_from_end), s::ios::end);
  f.write(&c, 1);
}

}  // namespace

TEST(Wal, EmptyLog) {
  vt::TempDir dir;
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value()) << wal.error().message;
  EXPECT_EQ(wal->last_op(), 0u);
  EXPECT_EQ(wal->first_op(), 1u);
  EXPECT_TRUE(replay_all(*wal).empty());
}

TEST(Wal, AppendAssignsIncreasingOps) {
  vt::TempDir dir;
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  for (vdb::OpNum i = 1; i <= 5; ++i) {
    auto op = wal->append(bytes_of("rec" + s::to_string(i)));
    ASSERT_TRUE(op.has_value());
    EXPECT_EQ(*op, i);
  }
  EXPECT_EQ(wal->last_op(), 5u);
}

TEST(Wal, ReplayReturnsRecordsInOrder) {
  vt::TempDir dir;
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  vec<Entry> want;
  for (int i = 0; i < 100; ++i) {
    str body(size_t(i * 13 % 300), char('a' + i % 26));
    auto op = wal->append(bytes_of(body));
    ASSERT_TRUE(op.has_value());
    want.push_back({*op, body});
  }
  EXPECT_EQ(replay_all(*wal), want);
  EXPECT_EQ(replay_all(*wal, 50),
            vec<Entry>(want.begin() + 49, want.end()));
}

TEST(Wal, EmptyBodyIsValid) {
  vt::TempDir dir;
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  ASSERT_TRUE(wal->append({}).has_value());
  vec<Entry> got = replay_all(*wal);
  ASSERT_EQ(got.size(), 1u);
  EXPECT_TRUE(got[0].body.empty());
}

TEST(Wal, ReopenPreservesRecordsAndNumbering) {
  vt::TempDir dir;
  {
    auto wal = vdb::Wal::open(dir.path());
    ASSERT_TRUE(wal.has_value());
    ASSERT_TRUE(wal->append(bytes_of("one")).has_value());
    ASSERT_TRUE(wal->append(bytes_of("two")).has_value());
    ASSERT_TRUE(wal->sync().has_value());
  }
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  EXPECT_EQ(wal->last_op(), 2u);
  auto op = wal->append(bytes_of("three"));
  ASSERT_TRUE(op.has_value());
  EXPECT_EQ(*op, 3u);
  EXPECT_EQ(replay_all(*wal),
            (vec<Entry>{{1, "one"}, {2, "two"}, {3, "three"}}));
}

TEST(Wal, VisitorErrorStopsReplay) {
  vt::TempDir dir;
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(wal->append(bytes_of("x")).has_value());
  }
  int seen = 0;
  auto st = wal->replay(0, [&](vdb::OpNum op, s::span<const s::byte>) {
    ++seen;
    if (op == 3) {
      return vdb::Status{vdb::make_error(vdb::ErrorCode::Internal, "stop")};
    }
    return vdb::Status{};
  });
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().message, "stop");
  EXPECT_EQ(seen, 3);
}

TEST(Wal, RollsSegmentsBySize) {
  vt::TempDir dir;
  vdb::WalOptions options;
  options.segment_bytes = 1024;
  auto wal = vdb::Wal::open(dir.path(), options);
  ASSERT_TRUE(wal.has_value());
  str body(200, 'z');
  for (int i = 0; i < 50; ++i) {
    ASSERT_TRUE(wal->append(bytes_of(body)).has_value());
  }
  EXPECT_GT(wal_files(dir.path()).size(), 5u);
  EXPECT_EQ(replay_all(*wal).size(), 50u);
}

TEST(Wal, TruncateBeforeDropsOldFilesOnly) {
  vt::TempDir dir;
  vdb::WalOptions options;
  options.segment_bytes = 512;
  auto wal = vdb::Wal::open(dir.path(), options);
  ASSERT_TRUE(wal.has_value());
  for (int i = 0; i < 40; ++i) {
    ASSERT_TRUE(wal->append(bytes_of(str(100, 'q'))).has_value());
  }
  size_t files_before = wal_files(dir.path()).size();
  ASSERT_TRUE(wal->truncate_before(30).has_value());
  EXPECT_LT(wal_files(dir.path()).size(), files_before);
  EXPECT_LE(wal->first_op(), 30u);
  vec<Entry> rest = replay_all(*wal);
  ASSERT_FALSE(rest.empty());
  EXPECT_LE(rest.front().op, 30u);
  EXPECT_EQ(rest.back().op, 40u);
  // Numbering continues after truncation.
  auto op = wal->append(bytes_of("after"));
  ASSERT_TRUE(op.has_value());
  EXPECT_EQ(*op, 41u);
}

TEST(Wal, TruncateEverythingKeepsNumbering) {
  vt::TempDir dir;
  {
    vdb::WalOptions options;
    options.segment_bytes = 64;
    auto wal = vdb::Wal::open(dir.path(), options);
    ASSERT_TRUE(wal.has_value());
    for (int i = 0; i < 10; ++i) {
      ASSERT_TRUE(wal->append(bytes_of(str(80, 'r'))).has_value());
    }
    ASSERT_TRUE(wal->truncate_before(11).has_value());
    EXPECT_TRUE(replay_all(*wal).empty());
  }
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  auto op = wal->append(bytes_of("next"));
  ASSERT_TRUE(op.has_value());
  EXPECT_EQ(*op, 11u);
}

// A crash mid-append leaves a partial record at the tail. Reopen must keep
// every complete record, drop the partial one, and accept new appends.
TEST(Wal, TornTailIsTruncatedOnOpen) {
  vt::TempDir dir;
  {
    auto wal = vdb::Wal::open(dir.path());
    ASSERT_TRUE(wal.has_value());
    for (int i = 1; i <= 10; ++i) {
      ASSERT_TRUE(wal->append(bytes_of("record-" + s::to_string(i)))
                      .has_value());
    }
    ASSERT_TRUE(wal->sync().has_value());
  }
  vec<sf::path> files = wal_files(dir.path());
  ASSERT_FALSE(files.empty());
  sf::resize_file(files.back(), sf::file_size(files.back()) - 5);

  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value()) << wal.error().message;
  vec<Entry> got = replay_all(*wal);
  ASSERT_EQ(got.size(), 9u);
  EXPECT_EQ(got.back(), (Entry{9, "record-9"}));
  auto op = wal->append(bytes_of("fresh"));
  ASSERT_TRUE(op.has_value());
  EXPECT_EQ(*op, 10u);
  got = replay_all(*wal);
  EXPECT_EQ(got.back(), (Entry{10, "fresh"}));
}

TEST(Wal, EveryTailTruncationRecovers) {
  vt::TempDir dir;
  {
    auto wal = vdb::Wal::open(dir.path());
    ASSERT_TRUE(wal.has_value());
    for (int i = 0; i < 20; ++i) {
      ASSERT_TRUE(wal->append(bytes_of(str(size_t(i) + 1, 'k'))).has_value());
    }
  }
  sf::path file = wal_files(dir.path()).back();
  uintmax_t full = sf::file_size(file);
  vec<char> original(full);
  {
    s::ifstream in(file, s::ios::binary);
    in.read(original.data(), s::streamsize(full));
  }
  size_t prev = 20;
  for (uintmax_t len = full; len-- > 0;) {
    {
      s::ofstream out(file, s::ios::binary | s::ios::trunc);
      out.write(original.data(), s::streamsize(len));
    }
    auto wal = vdb::Wal::open(dir.path());
    ASSERT_TRUE(wal.has_value()) << "len=" << len;
    size_t n = replay_all(*wal).size();
    EXPECT_LE(n, prev) << "len=" << len;
    EXPECT_LT(n, 20u) << "len=" << len;
    prev = n;
  }
}

TEST(Wal, BadChecksumAtTailIsDropped) {
  vt::TempDir dir;
  {
    auto wal = vdb::Wal::open(dir.path());
    ASSERT_TRUE(wal.has_value());
    ASSERT_TRUE(wal->append(bytes_of("good")).has_value());
    ASSERT_TRUE(wal->append(bytes_of("will be corrupted")).has_value());
  }
  flip_byte(wal_files(dir.path()).back(), 2);
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  EXPECT_EQ(replay_all(*wal), (vec<Entry>{{1, "good"}}));
}

TEST(Wal, CorruptionInClosedFileIsReported) {
  vt::TempDir dir;
  vdb::WalOptions options;
  options.segment_bytes = 256;
  {
    auto wal = vdb::Wal::open(dir.path(), options);
    ASSERT_TRUE(wal.has_value());
    for (int i = 0; i < 30; ++i) {
      ASSERT_TRUE(wal->append(bytes_of(str(60, 'c'))).has_value());
    }
  }
  vec<sf::path> files = wal_files(dir.path());
  ASSERT_GE(files.size(), 3u);
  flip_byte(files.front(), 10);
  auto wal = vdb::Wal::open(dir.path(), options);
  if (wal.has_value()) {
    auto st = wal->replay(0, [](vdb::OpNum, s::span<const s::byte>) {
      return vdb::Status{};
    });
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().code, vdb::ErrorCode::Corruption);
  } else {
    EXPECT_EQ(wal.error().code, vdb::ErrorCode::Corruption);
  }
}

TEST(Wal, SyncOnAppend) {
  vt::TempDir dir;
  vdb::WalOptions options;
  options.sync_on_append = true;
  auto wal = vdb::Wal::open(dir.path(), options);
  ASSERT_TRUE(wal.has_value());
  ASSERT_TRUE(wal->append(bytes_of("durable")).has_value());
  EXPECT_EQ(replay_all(*wal), (vec<Entry>{{1, "durable"}}));
}

TEST(Wal, LargeRecord) {
  vt::TempDir dir;
  auto wal = vdb::Wal::open(dir.path());
  ASSERT_TRUE(wal.has_value());
  str big(5u << 20, 'L');
  ASSERT_TRUE(wal->append(bytes_of(big)).has_value());
  vec<Entry> got = replay_all(*wal);
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].body.size(), big.size());
}
