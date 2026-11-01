#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <random>
#include <span>

#include <util/aliases.h>
#include <vdb/types.h>

// Helpers shared by the unit tests. Everything here is deliberately scalar
// and independent of the library so it can serve as a reference oracle.
namespace vdb::test {

inline vec<float> random_vector(s::mt19937_64& rng, uint32_t dim) {
  s::normal_distribution<float> dist(0.0f, 1.0f);
  vec<float> v(dim);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

inline void normalize_ref(s::span<float> v) {
  double n = 0.0;
  for (float x : v) {
    n += double(x) * x;
  }
  n = s::sqrt(n);
  if (n == 0.0) {
    return;
  }
  for (float& x : v) {
    x = float(x / n);
  }
}

inline vec<float> random_unit_vector(s::mt19937_64& rng, uint32_t dim) {
  vec<float> v = random_vector(rng, dim);
  normalize_ref(v);
  return v;
}

// n row-major vectors drawn around `clusters` gaussian centers. Clustered
// data behaves like real embeddings much more than iid noise does.
inline vec<float> clustered_dataset(
    s::mt19937_64& rng,
    size_t n,
    uint32_t dim,
    size_t clusters,
    float spread = 0.3f)
{
  vec<vec<float>> centers;
  for (size_t c = 0; c < clusters; ++c) {
    centers.push_back(random_vector(rng, dim));
  }
  s::uniform_int_distribution<size_t> pick(0, clusters - 1);
  s::normal_distribution<float> noise(0.0f, spread);
  vec<float> rows(n * dim);
  for (size_t i = 0; i < n; ++i) {
    const vec<float>& center = centers[pick(rng)];
    for (uint32_t d = 0; d < dim; ++d) {
      rows[i * dim + d] = center[d] + noise(rng);
    }
  }
  return rows;
}

inline double l2_sqr_ref(s::span<const float> a, s::span<const float> b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = double(a[i]) - b[i];
    sum += d * d;
  }
  return sum;
}

inline double dot_ref(s::span<const float> a, s::span<const float> b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += double(a[i]) * b[i];
  }
  return sum;
}

inline double distance_ref(
    Metric metric,
    s::span<const float> a,
    s::span<const float> b)
{
  switch (metric) {
  case Metric::L2:
    return l2_sqr_ref(a, b);

  break; case Metric::InnerProduct:
    return -dot_ref(a, b);

  break; case Metric::Cosine: {
    double na = s::sqrt(dot_ref(a, a));
    double nb = s::sqrt(dot_ref(b, b));
    if (na == 0.0 or nb == 0.0) {
      return 1.0;
    }
    return 1.0 - dot_ref(a, b) / (na * nb);
  }

  break; default:
    return 0.0;
  }
}

// Exact top-k ids over row-major `rows`, ordered by (distance, id).
inline vec<InternalId> brute_force_ids(
    Metric metric,
    s::span<const float> rows,
    uint32_t dim,
    s::span<const float> query,
    size_t k)
{
  size_t n = rows.size() / dim;
  vec<s::pair<double, InternalId>> all;
  all.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    all.emplace_back(
        distance_ref(metric, rows.subspan(i * dim, dim), query),
        InternalId(i));
  }
  k = s::min(k, n);
  s::partial_sort(all.begin(), all.begin() + k, all.end());
  vec<InternalId> ids;
  for (size_t i = 0; i < k; ++i) {
    ids.push_back(all[i].second);
  }
  return ids;
}

// |found ∩ truth| / |truth|; 1 when truth is empty.
template <typename A, typename B>
double recall(const A& found, const B& truth) {
  if (truth.empty()) {
    return 1.0;
  }
  uset<typename B::value_type> want(truth.begin(), truth.end());
  size_t hit = 0;
  for (const auto& id : found) {
    hit += want.count(id);
  }
  return double(hit) / double(truth.size());
}

// Unique temporary directory removed on destruction.
struct TempDir {
  TempDir() {
    static s::atomic<int> counter{0};
    s::random_device rd;
    mPath = sf::temp_directory_path() /
        ("vdb-test-" + s::to_string(rd()) + "-" + s::to_string(++counter));
    sf::create_directories(mPath);
  }
  ~TempDir() {
    s::error_code ec;
    sf::remove_all(mPath, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const sf::path& path() const { return mPath; }

private:
  sf::path mPath;
};

}  // namespace vdb::test

#endif  // TEST_UTIL_H
