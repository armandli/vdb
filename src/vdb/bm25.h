#ifndef BM25_H
#define BM25_H

#include <vdb/types.h>

namespace vdb {

// Splits on every ASCII byte that is not alphanumeric and lowercases ASCII
// letters. Bytes >= 0x80 count as token characters, so UTF-8 words stay
// intact (no Unicode case folding). No stopwords, no stemming.
vec<str> tokenize(strv text);

struct Bm25Params {
  float k1 = 1.2f;
  float b = 0.75f;
};

// Higher score is better (unlike Hit::distance).
struct ScoredId {
  InternalId id;
  float score;

  bool operator==(const ScoredId&) const = default;
};

// In-memory inverted index with BM25 ranking:
//   idf(t)   = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
//   score(d) = sum over DISTINCT query terms t of
//              idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
// with N = live documents, n_t = live documents containing t, |d| = token
// count of d, avgdl = mean |d| over live documents.
struct Bm25Index {
  explicit Bm25Index(Bm25Params params = {});
  ~Bm25Index();
  Bm25Index(Bm25Index&&) noexcept;
  Bm25Index& operator=(Bm25Index&&) noexcept;

  // Adding an id that already exists replaces its text.
  void add(InternalId id, strv text);
  void remove(InternalId id);

  size_t size() const;
  double avg_doc_len() const;
  // `term` is normalized like tokenize() (so "QUICK" == "quick").
  size_t doc_freq(strv term) const;

  float score(strv query, InternalId id) const;

  // Top-k documents with score > 0, ordered by score desc then id asc.
  vec<ScoredId> search(strv query, size_t k, const IdFilter& allow = {}) const;

private:
  struct Impl;
  uptr<Impl> mImpl;
};

}  // namespace vdb

#endif  // BM25_H
