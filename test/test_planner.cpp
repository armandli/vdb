#include <gtest/gtest.h>

#include <vdb/planner.h>

namespace {

// 1 KiB per vector (256-dim f32) against the default 10 MiB threshold:
// the break-even point is 10240 points.
vdb::PlannerInput base() {
  vdb::PlannerInput in;
  in.segment_indexed = true;
  in.has_filter = true;
  in.filter_resolvable = true;
  in.live_points = 1'000'000;
  in.bytes_per_vector = 1024;
  in.full_scan_threshold_bytes = 10ull << 20;
  return in;
}

vdb::Cardinality exact(size_t n) {
  return {n, n, n};
}

}  // namespace

TEST(Planner, ExactRequestedNeverUsesGraph) {
  vdb::PlannerInput in = base();
  in.exact_requested = true;
  in.estimate = exact(900'000);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
  in.filter_resolvable = false;
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Plain);
  in.has_filter = false;
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Plain);
}

TEST(Planner, UnindexedSegmentScans) {
  vdb::PlannerInput in = base();
  in.segment_indexed = false;
  in.estimate = exact(500'000);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
  in.filter_resolvable = false;
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Plain);
}

TEST(Planner, NoFilterUsesGraph) {
  vdb::PlannerInput in = base();
  in.has_filter = false;
  in.estimate = exact(in.live_points);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Graph);
}

TEST(Planner, SmallMatchSetScansCandidates) {
  vdb::PlannerInput in = base();
  in.estimate = exact(100);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
}

TEST(Planner, LargeMatchSetUsesGraph) {
  vdb::PlannerInput in = base();
  in.estimate = exact(200'000);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Graph);
}

TEST(Planner, ThresholdBoundary) {
  vdb::PlannerInput in = base();
  in.estimate = exact(10239);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
  in.estimate = exact(10240);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Graph);
}

// The same 1% selectivity is a scan in a small segment and a graph walk in
// a big one: the planner thinks in bytes, not percentages.
TEST(Planner, SameSelectivityDifferentSegmentSizes) {
  vdb::PlannerInput small = base();
  small.live_points = 100'000;
  small.estimate = exact(1'000);
  EXPECT_EQ(vdb::choose_strategy(small),
            vdb::SearchStrategy::PlainFromCandidates);

  vdb::PlannerInput big = base();
  big.live_points = 10'000'000;
  big.estimate = exact(100'000);
  EXPECT_EQ(vdb::choose_strategy(big), vdb::SearchStrategy::Graph);
}

TEST(Planner, SmallerVectorsRaiseBreakEven) {
  vdb::PlannerInput in = base();
  in.estimate = exact(20'000);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Graph);
  in.bytes_per_vector = 256;  // e.g. SQ8 codes of 256 dims
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
}

TEST(Planner, AmbiguousBandDecidesOnExpected) {
  vdb::PlannerInput in = base();
  in.estimate = {0, 5'000, 500'000};
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
  in.estimate = {0, 50'000, 500'000};
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Graph);
}

TEST(Planner, UnresolvableFilterOnSmallSegmentScansAll) {
  vdb::PlannerInput in = base();
  in.filter_resolvable = false;
  in.live_points = 5'000;
  in.estimate = {0, 2'500, 5'000};
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Plain);
}

TEST(Planner, UnresolvableSelectiveFilterScansPerPoint) {
  vdb::PlannerInput in = base();
  in.filter_resolvable = false;
  in.estimate = {0, 10, 1'000'000};
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Plain);
}

TEST(Planner, UnresolvableBroadFilterUsesGraph) {
  vdb::PlannerInput in = base();
  in.filter_resolvable = false;
  in.estimate = {0, 500'000, 1'000'000};
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::Graph);
}

TEST(Planner, ZeroMatchesScansCandidates) {
  vdb::PlannerInput in = base();
  in.estimate = exact(0);
  EXPECT_EQ(vdb::choose_strategy(in), vdb::SearchStrategy::PlainFromCandidates);
}

TEST(Planner, StrategyNames) {
  EXPECT_EQ(vdb::to_string(vdb::SearchStrategy::Plain), "plain");
  EXPECT_EQ(vdb::to_string(vdb::SearchStrategy::PlainFromCandidates),
            "plain_from_candidates");
  EXPECT_EQ(vdb::to_string(vdb::SearchStrategy::Graph), "graph");
}
