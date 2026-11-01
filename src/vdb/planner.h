#ifndef PLANNER_H
#define PLANNER_H

#include <vdb/payload_index.h>

namespace vdb {

enum struct SearchStrategy : uint8_t {
  Plain,                // scan every live point, test the filter per point
  PlainFromCandidates,  // resolve filter to a bitmap, scan only matches
  Graph,                // HNSW with the filter pushed in as `allow`
};

struct PlannerInput {
  bool exact_requested = false;  // SearchRequest::exact
  bool segment_indexed = false;  // segment has an HNSW graph
  bool has_filter = false;
  bool filter_resolvable = false;  // PayloadIndex::evaluate() can answer
  Cardinality estimate{};          // matches among live points
  size_t live_points = 0;
  size_t bytes_per_vector = 0;     // dim * sizeof(element) actually scanned
  size_t full_scan_threshold_bytes = 10ull << 20;
};

// Per-segment strategy choice, thresholds in vector BYTES (not selectivity
// percentages) because brute force is memory-bandwidth bound:
//
//  1. exact requested or segment not indexed:
//       PlainFromCandidates if has_filter and filter_resolvable, else Plain
//  2. no filter: Graph
//  3. est.max * bytes_per_vector < threshold: scan
//  4. est.min * bytes_per_vector >= threshold: Graph
//  5. ambiguous band: decide on est.exp the same way as 3 / 4
//
// "scan" is PlainFromCandidates when the filter is resolvable, otherwise
// Plain (test the payload per point, compute distances only for matches).
// For unresolvable filters the caller should refine `estimate.exp` by
// sampling payloads, since PayloadIndex can only say {0, total/2, total}.
SearchStrategy choose_strategy(const PlannerInput& in);

strv to_string(SearchStrategy s);

}  // namespace vdb

#endif  // PLANNER_H
