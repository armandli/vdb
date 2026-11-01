#ifndef COLLECTION_H
#define COLLECTION_H

#include <span>

#include <vdb/filter.h>
#include <vdb/hnsw_index.h>
#include <vdb/memory_score.h>
#include <vdb/payload_index.h>
#include <vdb/planner.h>

namespace vdb {

enum struct Quantization : uint8_t {
  None,
  SQ8,  // sealed segments traverse on uint8 codes, rescore with float32
};

struct CollectionConfig {
  uint32_t dim = 0;
  Metric metric = Metric::Cosine;
  HnswParams hnsw{};
  Quantization quantization = Quantization::None;
  // Declared payload indexes. Filters on undeclared fields still work, by
  // scanning payloads.
  vec<FieldIndexSpec> indexes;
  // BM25 index over Record::document, enables SearchRequest::text.
  bool text_index = false;
  // Recorded at creation; an Embedder with another model_id is refused.
  str embedding_model;

  // Planner break-even point, in bytes of vectors a brute-force scan reads.
  size_t full_scan_threshold_bytes = 10ull << 20;
  // A growing segment is sealed and indexed by optimize() once its vectors
  // exceed this many bytes.
  size_t indexing_threshold_bytes = 10ull << 20;
  // optimize() rebuilds a sealed segment whose deleted fraction exceeds
  // this ratio and that holds at least vacuum_min_points points.
  double vacuum_deleted_ratio = 0.2;
  size_t vacuum_min_points = 1000;
};

struct Record {
  str id;               // user key, unique within the collection
  // Exactly config().dim floats. Cosine collections normalize on insert
  // and get() returns the normalized vector.
  vec<float> vector;
  str document;         // optional text (returned, and BM25-indexed)
  Payload payload;
};

// Agent-memory re-ranking (see memory_score.h). Candidates are the top
// `candidate_pool` hits by similarity; their payloads supply the
// timestamp (unix seconds, Int) and importance (Int or Double) fields.
// Missing fields count as hours_since = +inf (recency 0) / importance 0.
struct MemoryScoring {
  str timestamp_field = "ts";
  str importance_field = "importance";
  int64_t now_unix = 0;  // 0 = wall clock; set explicitly in tests
  MemoryScoreWeights weights{};
  size_t candidate_pool = 100;
};

struct SearchRequest {
  vec<float> vector;  // dense query; may be empty when `text` is set
  str text;           // BM25 query; needs config().text_index
  size_t k = 10;
  opt<Filter> filter;
  size_t ef = 0;          // HNSW beam, 0 = config().hnsw.ef_search
  bool exact = false;     // force brute force (ground truth)
  // SQ8 shortlist size = ceil(k * oversampling) before float32 rescoring.
  float oversampling = 2.0f;
  bool with_vector = false;
  bool with_payload = true;
  bool with_document = true;
  opt<MemoryScoring> memory;
};

// `score` is the final ranking key, higher is better, results are sorted
// by it descending:
//  - vector only: Cosine -> 1 - distance, InnerProduct / L2 -> -distance
//  - text only:   BM25 score
//  - both:        RRF over the vector and text rankings (k = 60)
//  - memory set:  memory_scores() over the candidate pool
// `distance` is the raw metric distance, NaN when no vector query was given.
struct SearchHit {
  str id;
  float score;
  float distance;
  str document;
  Payload payload;
  vec<float> vector;
};

struct CollectionStats {
  size_t live_points = 0;
  size_t deleted_points = 0;  // tombstoned, not yet vacuumed
  size_t segments = 0;
  size_t indexed_segments = 0;
  size_t vector_bytes = 0;
  OpNum last_op = 0;
};

struct SegmentPlan {
  uint64_t segment_id;
  SearchStrategy strategy;
  Cardinality estimate;
  size_t live_points;
};

// A named set of points sharing dim / metric / indexes. Obtained from a
// Database; all methods are thread-safe. Reads see every write whose call
// has returned (read-your-writes), including from other threads.
struct Collection {
  struct Impl;
  explicit Collection(uptr<Impl> impl);
  ~Collection();
  Collection(const Collection&) = delete;
  Collection& operator=(const Collection&) = delete;

  const str& name() const;
  const CollectionConfig& config() const;

  // Inserts or replaces by id. The batch is validated first (dim, empty or
  // duplicate ids, non-finite values, zero vector under Cosine) and either
  // fully applied or rejected. Returns records written.
  Result<size_t> upsert(s::span<const Record> records);

  Result<opt<Record>> get(strv id, bool with_vector = true) const;

  // Returns how many of `ids` existed. Unknown ids are not an error.
  Result<size_t> remove(s::span<const str> ids);
  Result<size_t> remove_where(const Filter& filter);

  // Returns min(k, #matching live points) hits at every filter
  // selectivity: a Graph segment search that yields fewer than k hits
  // falls back to a scan of that segment. k == 0 returns no hits. Errors:
  // DimensionMismatch for a wrong-sized vector, InvalidArgument when
  // neither vector nor text is given or text is set without text_index.
  Result<vec<SearchHit>> search(const SearchRequest& request) const;

  // The per-segment plan search() would execute, without executing it.
  Result<vec<SegmentPlan>> explain(const SearchRequest& request) const;

  size_t count() const;
  Result<size_t> count_where(const Filter& filter) const;

  // Persist all segments and truncate the WAL up to what they hold.
  Status flush();

  // Synchronous maintenance: seal + index the growing segment when over
  // the indexing threshold (`force` seals regardless), vacuum segments over
  // the deleted ratio, merge small sealed segments. Searches keep working
  // during optimize() and see identical results before and after.
  Status optimize(bool force = false);

  CollectionStats stats() const;

private:
  uptr<Impl> mImpl;
};

}  // namespace vdb

#endif  // COLLECTION_H
