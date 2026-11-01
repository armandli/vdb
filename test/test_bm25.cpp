#include <cmath>

#include <gtest/gtest.h>

#include <vdb/bm25.h>

namespace {

// Independent BM25 over pre-tokenized documents, per the formula in bm25.h.
double bm25_ref(
    const vec<vec<str>>& docs,
    const vec<str>& query,
    size_t d,
    double k1 = 1.2,
    double b = 0.75)
{
  double n = double(docs.size());
  double avgdl = 0.0;
  for (const vec<str>& doc : docs) {
    avgdl += double(doc.size());
  }
  avgdl /= n;
  uset<str> terms(query.begin(), query.end());
  double score = 0.0;
  for (const str& t : terms) {
    double df = 0.0;
    for (const vec<str>& doc : docs) {
      df += s::find(doc.begin(), doc.end(), t) != doc.end();
    }
    double tf = double(s::count(docs[d].begin(), docs[d].end(), t));
    if (tf == 0.0) {
      continue;
    }
    double idf = s::log(1.0 + (n - df + 0.5) / (df + 0.5));
    double len = double(docs[d].size());
    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl));
  }
  return score;
}

const vec<str> CORPUS = {
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog outpaces a quick red fox",
    "lazy afternoons are for sleeping dogs",
    "error TS-999 indicates the sensor timed out",
    "vector databases store embeddings for retrieval",
    "",
};

vdb::Bm25Index make_index() {
  vdb::Bm25Index index;
  for (vdb::InternalId i = 0; i < CORPUS.size(); ++i) {
    index.add(i, CORPUS[i]);
  }
  return index;
}

}  // namespace

TEST(Tokenize, LowercasesAndSplits) {
  EXPECT_EQ(vdb::tokenize("Hello, World! TS-999"),
            (vec<str>{"hello", "world", "ts", "999"}));
  EXPECT_EQ(vdb::tokenize("  multiple   spaces\tand\nlines "),
            (vec<str>{"multiple", "spaces", "and", "lines"}));
  EXPECT_TRUE(vdb::tokenize("").empty());
  EXPECT_TRUE(vdb::tokenize("!!! ... ---").empty());
}

TEST(Tokenize, KeepsUtf8Words) {
  EXPECT_EQ(vdb::tokenize("Café 日本語 naïve"),
            (vec<str>{"café", "日本語", "naïve"}));
}

TEST(Bm25, Statistics) {
  vdb::Bm25Index index = make_index();
  EXPECT_EQ(index.size(), CORPUS.size());
  EXPECT_EQ(index.doc_freq("quick"), 2u);
  EXPECT_EQ(index.doc_freq("fox"), 2u);
  EXPECT_EQ(index.doc_freq("QUICK"), 2u) << "terms are normalized";
  EXPECT_EQ(index.doc_freq("absent"), 0u);
  double total = 0.0;
  for (const str& d : CORPUS) {
    total += double(vdb::tokenize(d).size());
  }
  EXPECT_NEAR(index.avg_doc_len(), total / double(CORPUS.size()), 1e-9);
}

TEST(Bm25, ScoresMatchReference) {
  vdb::Bm25Index index = make_index();
  vec<vec<str>> docs;
  for (const str& d : CORPUS) {
    docs.push_back(vdb::tokenize(d));
  }
  for (strv q : {"quick fox", "lazy dog", "TS-999", "quick quick brown",
                 "embeddings retrieval", "nothing matches"}) {
    vec<str> terms = vdb::tokenize(q);
    for (vdb::InternalId d = 0; d < CORPUS.size(); ++d) {
      EXPECT_NEAR(index.score(q, d), bm25_ref(docs, terms, d), 1e-4)
          << "query '" << q << "' doc " << d;
    }
  }
}

TEST(Bm25, SearchRanksAndExcludesNonMatches) {
  vdb::Bm25Index index = make_index();
  vec<vdb::ScoredId> hits = index.search("quick fox", 10);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].id, 1u) << "two 'quick' beat one";
  EXPECT_EQ(hits[1].id, 0u);
  EXPECT_GT(hits[0].score, hits[1].score);
  EXPECT_TRUE(index.search("nothing matches", 10).empty());
  EXPECT_TRUE(index.search("", 10).empty());
}

TEST(Bm25, ExactTokenQuery) {
  vdb::Bm25Index index = make_index();
  vec<vdb::ScoredId> hits = index.search("ts-999", 3);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 3u);
}

TEST(Bm25, KLimitsAndTieBreak) {
  vdb::Bm25Index index;
  for (vdb::InternalId i = 0; i < 10; ++i) {
    index.add(9 - i, "same text");
  }
  vec<vdb::ScoredId> hits = index.search("same", 3);
  ASSERT_EQ(hits.size(), 3u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_EQ(hits[1].id, 1u);
  EXPECT_EQ(hits[2].id, 2u);
}

TEST(Bm25, AllowFilter) {
  vdb::Bm25Index index = make_index();
  vdb::IdFilter not_one = [](vdb::InternalId id) { return id != 1; };
  vec<vdb::ScoredId> hits = index.search("quick fox", 10, not_one);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
}

TEST(Bm25, RemoveUpdatesStatistics) {
  vdb::Bm25Index index = make_index();
  index.remove(1);
  EXPECT_EQ(index.size(), CORPUS.size() - 1);
  EXPECT_EQ(index.doc_freq("quick"), 1u);
  vec<vdb::ScoredId> hits = index.search("quick", 10);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
  index.remove(1);  // removing twice is harmless
  EXPECT_EQ(index.size(), CORPUS.size() - 1);
}

TEST(Bm25, ReAddReplacesText) {
  vdb::Bm25Index index = make_index();
  index.add(4, "now about foxes and dogs");
  EXPECT_EQ(index.size(), CORPUS.size());
  EXPECT_EQ(index.doc_freq("embeddings"), 0u);
  EXPECT_EQ(index.doc_freq("foxes"), 1u);
}

TEST(Bm25, CustomParams) {
  vdb::Bm25Index index({2.0f, 0.0f});
  vec<vec<str>> docs;
  for (vdb::InternalId i = 0; i < CORPUS.size(); ++i) {
    index.add(i, CORPUS[i]);
    docs.push_back(vdb::tokenize(CORPUS[i]));
  }
  EXPECT_NEAR(index.score("quick dog", 1),
              bm25_ref(docs, vdb::tokenize("quick dog"), 1, 2.0, 0.0), 1e-4);
}
