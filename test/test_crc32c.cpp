#include <cstring>
#include <random>

#include <gtest/gtest.h>

#include <vdb/crc32c.h>

namespace {

uint32_t crc_of(strv text, uint32_t seed = 0) {
  return vdb::crc32c(text.data(), text.size(), seed);
}

// Bitwise reference implementation of CRC-32C.
uint32_t crc_ref(const uint8_t* data, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int b = 0; b < 8; ++b) {
      crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

}  // namespace

TEST(Crc32c, KnownVectors) {
  EXPECT_EQ(crc_of(""), 0x00000000u);
  EXPECT_EQ(crc_of("a"), 0xC1D04330u);
  EXPECT_EQ(crc_of("123456789"), 0xE3069283u);
  EXPECT_EQ(crc_of("The quick brown fox jumps over the lazy dog"),
            0x22620404u);
}

// RFC 3720 (iSCSI) appendix B.4 test vectors.
TEST(Crc32c, Rfc3720Vectors) {
  vec<uint8_t> zeros(32, 0x00);
  vec<uint8_t> ones(32, 0xFF);
  vec<uint8_t> incr(32);
  vec<uint8_t> decr(32);
  for (int i = 0; i < 32; ++i) {
    incr[i] = uint8_t(i);
    decr[i] = uint8_t(31 - i);
  }
  EXPECT_EQ(vdb::crc32c(zeros.data(), 32), 0x8A9136AAu);
  EXPECT_EQ(vdb::crc32c(ones.data(), 32), 0x62A8AB43u);
  EXPECT_EQ(vdb::crc32c(incr.data(), 32), 0x46DD794Eu);
  EXPECT_EQ(vdb::crc32c(decr.data(), 32), 0x113FDB5Cu);
}

TEST(Crc32c, ChainingEqualsWhole) {
  strv text = "hello, write-ahead log";
  for (size_t cut = 0; cut <= text.size(); ++cut) {
    uint32_t first = crc_of(text.substr(0, cut));
    EXPECT_EQ(crc_of(text.substr(cut), first), crc_of(text)) << cut;
  }
}

TEST(Crc32c, SpanOverloadMatchesPointer) {
  str text = "span and pointer agree";
  auto bytes = s::as_bytes(s::span(text.data(), text.size()));
  EXPECT_EQ(vdb::crc32c(bytes), crc_of(text));
}

// Covers the hardware path's alignment prologue and 8-byte main loop.
TEST(Crc32c, RandomBuffersMatchReference) {
  s::mt19937_64 rng(1);
  vec<uint8_t> buf(4096 + 16);
  for (uint8_t& b : buf) {
    b = uint8_t(rng());
  }
  for (size_t offset = 0; offset < 8; ++offset) {
    for (size_t len : {0u, 1u, 3u, 7u, 8u, 9u, 63u, 64u, 65u, 1000u, 4096u}) {
      EXPECT_EQ(vdb::crc32c(buf.data() + offset, len),
                crc_ref(buf.data() + offset, len))
          << "offset=" << offset << " len=" << len;
    }
  }
}

TEST(Crc32c, DetectsSingleBitFlips) {
  vec<uint8_t> buf(256, 0xAB);
  uint32_t good = vdb::crc32c(buf.data(), buf.size());
  for (size_t i = 0; i < buf.size(); i += 17) {
    for (int bit = 0; bit < 8; ++bit) {
      buf[i] ^= uint8_t(1 << bit);
      EXPECT_NE(vdb::crc32c(buf.data(), buf.size()), good);
      buf[i] ^= uint8_t(1 << bit);
    }
  }
}
