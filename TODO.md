# vdb — design and implementation plan

A personal, local-first vector database for **RAG** and **agent memory**:
an embeddable C++26 library (`libvdb`) plus a `vdb` command line tool that
can also run as a local **MCP server**. Linux and macOS.

This file is both the design document and the progress tracker. Tick the
boxes as components land. Every component lists its header (the contract),
the test that checks it, and what "done" means.

---

## 0. How to work in this repo

```
make                 # configure + build (Release) into ./build
make debug           # Debug build
make test            # build + run every enabled test (ctest)
make test T=hnsw     # run only ./build/test_hnsw  (GTEST_FILTER=... to narrow)
make asan | tsan     # sanitizer builds in build-asan / build-tsan
make mcp             # vdb serve --transport stdio against ./data
```

- Sources: `src/vdb/*.h` (public headers, already written: the design
  contract), `src/vdb/*.cpp` (yours to write), `src/main.cpp` (CLI),
  `src/mcp/*.cpp` (MCP server glue), `src/util/aliases.h`.
- Every `src/vdb/*.cpp` is compiled into `libvdb` automatically.
- **Tests turn on by themselves.** `test/CMakeLists.txt` registers each test
  with the implementation files it needs (`REQUIRES vdb/topk.cpp`). Until
  those files exist the test is skipped at configure time (look for
  `test_xxx: skipped` in the cmake output). Create the `.cpp`, re-run
  `make`, and the test is built and run.
- Headers are the contract the tests are written against. Private data
  members and `Impl` structs in headers are suggestions; change them
  freely. Changing a public signature means changing the test too.
- All executables land directly in `build/` (`build/vdb`, `build/test_*`).

### Dependencies (all pinned, fetched by CMake `FetchContent`)

| Library | Version | Used for | Why this one |
|---|---|---|---|
| [Google Highway](https://github.com/google/highway) | 1.4.0 | SIMD distance kernels, SQ8 kernels | One source, runtime dispatch to SSE4/AVX2/AVX-512 (x86) and NEON/SVE (ARM). Portable across Linux/macOS without `-march=native`. |
| [CRoaring](https://github.com/RoaringBitmap/CRoaring) | 5.4.0 | Filter results, payload index postings, deleted sets | Compressed bitmaps with fast AND/OR/ANDNOT; the standard currency for filtered search. |
| [simdjson](https://github.com/simdjson/simdjson) | 5.1.1 | Payload / filter / NDJSON import / MCP argument parsing | GB/s parsing; bulk imports are JSON-heavy. |
| [CLI11](https://github.com/CLIUtils/CLI11) | 2.7.2 | `vdb` command line | Requested. Header-only. |
| [cxxmcp](https://github.com/caomengxuan666/cxxmcp) | 1.3.0 | `vdb serve` MCP server | See §3.13. MIT, C++17, stdio + Streamable HTTP, 272/272 server conformance. Only the executable links it. |
| [GoogleTest](https://github.com/google/googletest) | 1.18.0 | Unit tests | |
| [Apple MLX](https://github.com/ml-explore/mlx) | system (`brew install mlx`) | Optional GPU batch backend, macOS only | See §3.15. Auto-detected (`VDB_WITH_MLX=AUTO`). |

Later, optional: [llama.cpp](https://github.com/ggml-org/llama.cpp) for an
in-process embedder (M12). Not fetched yet.

### Toolchain notes

- C++26 (`-std=c++26`). Apple clang 21 / libc++ lacks `std::function_ref`
  and `std::atomic<std::shared_ptr>`; the design avoids both (`IdFilter` is
  a `std::function`; segment snapshots are published under a mutex or with
  `std::atomic_load` on `shared_ptr`). `std::expected`, `std::span`,
  `std::flat_map`, `std::print` are available.
- Errors are values: `Result<T> = std::expected<T, Error>`, `Status =
  Result<void>`. No exceptions cross the public API.
- Code style: `.claude/skills/format-cpp` (struct over class, `mMember`,
  snake_case functions, header guards, `not/and/or`, aliases from
  `util/aliases.h`).

---

## 1. Scope contract (the irreversible decisions)

| # | Decision | Choice | Rejected, and why |
|---|---|---|---|
| 1 | Deployment shape | **Embedded library first**; `vdb serve` (MCP) is a thin shell over the same objects | Client-server first: networking/auth for no gain on one machine. Qdrant Edge and Chroma `PersistentClient` show the library-first route. |
| 2 | Storage layer | **Own format**: directory of immutable segment files + manifest + WAL | SQLite catalog (Chroma): one more dependency for a few KB of metadata. RocksDB: Qdrant removed it after compaction stalls. |
| 3 | Memory model | **In RAM, segment files loaded or mmap'd**; quantized codes keep the hot set small | Own buffer pool (caliby): ~10x the work; only needed past RAM. Revisit only if a dataset outgrows memory. |
| 4 | Primary ANN index | **HNSW** per sealed segment, **flat scan** forever as oracle and small-segment path | Vamana/DiskANN: earns its keep when the graph lives on SSD. IVF: lower recall at equal latency. |
| 5 | Mutability | **LSM-style segments**: one appendable (growing, flat) segment + immutable indexed segments, rebuilt by optimizers, never patched | One mutable graph: deletes erode it and locking is hard. |
| 6 | IDs | User **string ids**; dense `uint32` `InternalId` per segment; every point carries an `OpNum` version | UUID-only / int-only: agent and RAG callers think in strings ("doc-17#chunk-3"). |
| 7 | Deletes | **Tombstone** in a per-segment deleted bitmap, **vacuum** by rebuild above 20% deleted | In-place graph repair: complex; rebuild with HNSW reuse later if needed. |
| 8 | Concurrency | **Single writer per collection, lock-free-ish readers over a snapshot** (shared_ptr list of segments); growing segment behind a `shared_mutex` | Fine-grained / OLC graph latches: unnecessary when indexed segments are immutable. |
| 9 | Multi-vector per record | **One vector per record** | Named / late-interaction vectors: changes the ID model; not needed for RAG chunks or memories. |
| 10 | Filter DSL | **Mongo-style JSON** (`$and/$or/$not/$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists`) | SQL `WHERE`: a parser bigger than the index (PardusDB). Django-style keys: awkward for nesting. |
| 11 | Filter strategy | **Bytes-based planner** per segment (Qdrant `full_scan_threshold`) with interval cardinality estimates | Fixed selectivity % threshold: the same 1% is 100 points in one segment and 100k in another. |
| 12 | Metric | L2 / inner product / cosine; **cosine normalized on insert** so it runs as `1 - dot` | |
| 13 | Embeddings | **Bring your own vectors** in the library; `Embedder` interface used by the MCP layer; model id recorded per collection | Embedding inside the core engine: model versioning becomes the engine's problem. |
| 14 | Durability | **WAL with CRC-framed records**, `Durability::{None,Buffered,Sync}` | fsync per page write: slow; no WAL: torn state on crash. |
| 15 | Background work | **None when embedded** (`optimize()` is explicit); a low-priority optimizer thread only when asked (`vdb serve`) | Hidden threads in a library surprise callers (Qdrant Edge rule). |

---

## 2. Architecture

```
           C++ API: Database / Collection          vdb CLI (CLI11)   vdb serve (cxxmcp)
                         │                                   │            │
                         │                               McpTools (JSON in/out, Embedder)
                         ▼                                   └─────┬──────┘
   ┌──────────────── Collection ──────────────────────────────────┘
   │  write path: validate → WAL append (OpNum) → apply to growing segment
   │  read path:  snapshot segments → per-segment plan → search → merge →
   │              hybrid (RRF) / memory re-score → fetch payloads
   │  optimize:   seal growing → build HNSW (+SQ8) → vacuum → merge → swap
   ▼
   Segment (a tiny complete database)
   ├── IdMap            string id ↔ InternalId, per-point OpNum version
   ├── VectorStore      float32 rows (64-byte aligned)        [vector_store.h]
   ├── SQ8 codes        optional, uint8 rows                  [sq8.h]
   ├── HnswIndex        sealed segments only                  [hnsw_index.h]
   ├── PayloadStore     per-point Payload + document text      [payload.h]
   ├── PayloadIndex     declared field indexes (roaring)      [payload_index.h]
   ├── Bm25Index        optional text index                   [bm25.h]
   └── deleted bitmap   tombstones
   ▼
   Storage: <dir>/LOCK, MANIFEST, <collection>/{config.json, wal/, segments/<id>/}
```

Query flow for `search(vector, k, filter)`:

1. Validate (dim, k, text needs text index). Normalize the query for cosine.
2. Take a snapshot: the current list of segments (shared_ptr copy).
3. For each segment: build `PlannerInput` from `PayloadIndex::estimate`
   (+ sampling for unindexed fields) and `choose_strategy` →
   `Plain` / `PlainFromCandidates` / `Graph`.
4. Execute; the allow predicate always also excludes the segment's
   tombstones and any point superseded by a newer version in another
   segment. Graph returning fewer than k → re-run that segment as a scan.
5. Merge per-segment top-k with `TopK`. SQ8: shortlist `k * oversampling`
   on codes, rescore with float32.
6. Hybrid: BM25 top-k' per segment → merge → `rrf()` with the vector list.
   Memory: take `candidate_pool` by similarity, `memory_scores()`, re-sort.
7. Fill `SearchHit` (id, score, distance, document, payload, vector).

---

## 3. Components

Each component: **purpose**, **header**, **key design points**,
**optimization notes**, **test**. Milestones in §4 order the work.

### 3.1 Core types — `types.h` (done: header only)
Shared vocabulary: `InternalId` (dense u32 per segment), `OpNum` (WAL
sequence = point version), `Metric`, `Error`/`ErrorCode`, `Result<T>`,
`Hit {id, distance}` (smaller is closer), `IdFilter`.

### 3.2 Distance kernels — `distance.h`, `distance.cpp`
The only code that knows what "similar" means; everything else exists to
call it less.
- `l2_sqr`, `dot`, `norm`, `normalize`, `distance`, `distance_prenormalized`,
  `distance_batch`.
- No `sqrt` (monotone, doesn't change ranking). Cosine collections
  normalize at insert, so the hot path is `1 - dot`.
- Implement with Highway `HWY_DYNAMIC_DISPATCH` (`foreach_target.h`), 4
  accumulators to hide FMA latency, masked/scalar tail. Accumulate in
  float; the tests tolerate 1e-4 relative error.
- `distance_batch` keeps the query in registers and streams rows; add
  software prefetch of the next row.
- Test: `test_distance` (reference match over 16 dims incl. odd tails,
  symmetry, triangle inequality, normalization, batch == single,
  unaligned input, huge magnitudes).

### 3.3 Top-k heap — `topk.h`, `topk.cpp`
Bounded max-heap of the k best. `accepts()` lets callers skip rescoring.
Deterministic: ties at the boundary are rejected, output ordered by
(distance, id). Test: `test_topk`.

### 3.4 Vector store — `vector_store.h`, `vector_store.cpp`
Row-major float32, 64-byte aligned (`std::aligned_alloc` / custom deleter),
geometric growth. `save`/`load` with magic `VDBV`, version, dim, count,
crc32c. Later: `load` via `mmap` for sealed segments (read-only).
Test: `test_vector_store` (alignment, growth, round trip, bad magic,
future version, flipped byte, every truncation).

### 3.5 Flat search — `flat_search.h`, `flat_search.cpp`
Exact k-NN: the oracle for every approximate path and the strategy for
small or heavily filtered segments. Never delete it.
`flat_search_candidates` iterates a roaring bitmap (pre-filter path).
Optimization: block the scan (`distance_batch` over 64–256 rows) and only
call the filter for rows that `TopK::accepts`. Test: `test_flat_search`.

### 3.6 Payload — `payload.h`, `payload.cpp`
Flat typed map `field → Value` (null, bool, int64, double, string, string
list, int list); nested JSON flattens to dotted keys. simdjson On-Demand
for `parse_payload`; hand-written compact `to_json`; a varint/tag binary
`encode_payload` for segment files (decode must reject truncation with
`Corruption`). Test: `test_payload`.

### 3.7 Filter — `filter.h`, `filter.cpp`
Mongo-style JSON → `Filter` AST, plus `matches()` (the reference
evaluator used for unindexed fields and as the oracle for the index).
Semantics written in the header: missing field, numeric cross-type
compare, list fields match ANY element, Ne/Nin match NO element, 32-level
depth cap (fuzz safety). Test: `test_filter` (operators, list semantics,
missing fields, parse errors, deep nesting, metamorphic laws).
Later: fuzz `parse_filter` with libFuzzer (`-fsanitize=fuzzer`).

### 3.8 Payload index — `payload_index.h`, `payload_index.cpp`
Per-segment secondary indexes over **declared** fields:
- Keyword: `unordered_map<str, Roaring>`.
- Integer / Float: sorted `(value, id)` vector built at seal time (growing
  segments may use `std::map<value, Roaring>`); range → union of runs.
- Bool: two bitmaps. Plus a "has field" bitmap per field for `$exists`,
  `$ne`, `$nin` and `Not`.
- `evaluate()` → exact bitmap or `nullopt` if any leaf is unindexed.
- `estimate()` → `Cardinality {min, exp, max}` with the combination rules
  in the header (exact for indexed leaves).
- `is_tenant` fields: always resolved through the index; Qdrant also
  co-locates tenant points, which we get for free if a later optimizer
  sorts rows by tenant at seal time.
Test: `test_payload_index` (evaluate == scan for 20 filter shapes, after
removals; estimate interval always contains the truth; AND independence).

### 3.9 HNSW index — `hnsw_index.h`, `hnsw_index.cpp`
Hierarchical NSW over a `VectorStore`; stores ids and links only.
- Level `floor(-ln U / ln m)` from a seeded RNG (determinism test).
- Layer 0 degree `2m`, upper `m`; heuristic neighbor selection; prune
  reverse links with the same heuristic.
- Search: greedy descent, then beam `ef` on layer 0. Visited set as an
  epoch-stamped `vector<uint32_t>` per thread (no clearing). Rejected
  points (`allow` false) are still expanded but never enter the result
  heap.
- Filtered search: raise ef roughly by 1/selectivity (capped), and the
  collection falls back to a scan when fewer than k come back.
- `build(threads)`: per-node spinlocks on link lists during parallel
  insert (hnswlib style).
- Links stored as one flat `vector<InternalId>` for layer 0 (fixed stride
  `2m + 1` with a count) and per-node vectors for upper layers.
- Later: co-locate SQ8 code + layer-0 links per node (one cache line fetch
  per hop); reorder ids by graph locality at seal time.
Test: `test_hnsw` (recall@10 ≥ 0.95 L2 and cosine, recall@1 on stored
vectors, **recall monotone in ef**, degree/level invariants, layer-0
reachability, level distribution, determinism, parallel build, identical
vectors, outlier, inline filter, save/load, corruption).

### 3.10 SQ8 quantizer — `sq8.h`, `sq8.cpp`
Per-dimension `[lo, hi] → 0..255`, optional quantile clipping. Asymmetric
distance (float query vs code) with Highway (widen u8 → f32, FMA).
Collections with `Quantization::SQ8` traverse HNSW on codes and rescore
the `k * oversampling` shortlist in float32. Rule: **never return a
quantized distance**. Test: `test_sq8` (half-step reconstruction bound,
clamping, constant dims, quantiles, asymmetric ≈ exact, shortlist+rescore
recall ≥ 0.98).

### 3.11 CRC-32C and WAL — `crc32c.h`, `wal.h`
- `crc32c`: hardware instructions (`_mm_crc32_u64` / `__crc32cd`) with a
  slicing-by-8 table fallback. Test: `test_crc32c` (RFC 3720 vectors,
  chaining, offsets/lengths vs bitwise reference).
- `Wal`: files `wal-<first op>.log`, records `len | crc | op | body`,
  roll at 32 MiB, `sync()` for group commit, `truncate_before(op)` drops
  whole files, torn tail truncated on open, corruption in closed files is
  an error. Test: `test_wal` (ordering, reopen numbering, rolling,
  truncation, every tail truncation recovers, bad checksum, closed-file
  corruption, 5 MiB record).

### 3.12 Planner — `planner.h`, `planner.cpp`
Pure function `choose_strategy(PlannerInput)`. Thresholds in bytes of
vectors scanned (default 10 MiB ≈ the break-even of a scan vs an HNSW
query at memory bandwidth). Three-band estimate: only the ambiguous band
looks at `exp`; unresolvable filters get `exp` from sampling ~1000
payloads. Test: `test_planner`.

### 3.13 Segment, Collection, Database — `collection.h`, `database.h`
The public API. Segments are internal (`src/vdb/segment.h` is yours to
design); suggested shape:

```cpp
struct Segment {               // immutable once sealed
  uint64_t id; bool appendable;
  IdMap ids;                   // umap<str, InternalId> + vec<str> + vec<OpNum>
  VectorStore vectors; opt<SQ8Quantizer> sq8; vec<uint8_t> codes;
  opt<HnswIndex> graph; PayloadIndex index; opt<Bm25Index> text;
  vec<vec<uint8_t>> payloads; vec<str> documents;
  roaring::Roaring deleted;    // copy-on-write for sealed segments
};
struct Snapshot { vec<sptr<const Segment>> sealed; sptr<Segment> growing; };
```

- **Write path**: validate the whole batch → encode an `Upsert`/`Delete`
  WAL record → `append` (+ `sync` under `Durability::Sync`) → apply: new
  version goes to the growing segment; older versions in other segments
  get tombstoned (the id→segment map lives in the collection). Replay
  uses point versions so applying an op twice is a no-op.
- **Read your writes**: apply happens before `upsert` returns, under the
  collection writer mutex; readers copy the snapshot pointer.
- **optimize(force)**: seal growing when over `indexing_threshold_bytes`
  (or `force`), build HNSW (+SQ8, BM25) off to the side, write segment
  files, swap the snapshot, update MANIFEST, truncate WAL. Vacuum
  rebuilds sealed segments above `vacuum_deleted_ratio`. Merge packs small
  sealed segments. Searches during optimize see the old snapshot.
- **Database**: `flock` on `<dir>/LOCK`, MANIFEST written via temp file +
  `fsync` + `rename` + directory `fsync`. Names `[A-Za-z0-9_-]{1,64}`.
  In-memory mode skips all I/O.
- **Hybrid**: RRF (k = 60) of vector top-k' and BM25 top-k'.
- **Memory scoring**: `MemoryScoring` re-ranks a similarity candidate pool
  with `memory_scores()` using payload `ts` / `importance`.
Tests: `test_collection` (API semantics, validation, scores, hybrid,
memory, explain, SQ8 recall, optimize), `test_persistence` (reopen, lock,
drops, durability modes, corrupt manifest, **fork + exit without close**,
**SIGKILL during ingest keeps the acked prefix**), `test_filtered_search`
(**selectivity sweep 0.01%–100% returns exactly min(k, matches) with
recall ≥ 0.95**, indexed and unindexed, anti-correlated filter, tenants,
metamorphic), `test_churn` (**10 delete/insert rounds keep recall ≥ 0.95**,
delete all, re-insert, delete the entry region), `test_concurrency`
(readers + writer + optimize; cross-thread read-your-writes; run under
`make tsan`).

### 3.14 BM25, RRF, memory score — `bm25.h`, `fusion.h`, `memory_score.h`
- `Bm25Index`: postings `term → vec<(id, tf)>`, doc lengths, live count;
  Lucene IDF; ASCII-lowercase tokenizer that keeps UTF-8 bytes. Later:
  stopwords / stemming behind an option. Test: `test_bm25` (formula
  reference, ranking, tie-break, removal, re-add, params).
- `rrf()`: rank-based fusion, no score normalization. Test: `test_fusion`.
- `memory_scores()`: Generative-Agents relevance + recency (0.995/h) +
  importance, min-max normalized. Test: `test_memory_score`.

### 3.15 MLX batch backend — `mlx_backend.h`, `mlx_backend.cpp` (macOS)
Honest framing: per-query latency is best on CPU SIMD (PCIe/launch
overhead dominates small work). MLX wins on **throughput**: many queries ×
many vectors as one GEMM (`-2 Q·Xᵀ + ‖q‖² + ‖x‖²`, `mx::argpartition`
for top-k) on unified memory with no copies. Use it for: ground-truth
generation in benchmarks, batch search API, future k-means / PQ training.
Linux builds never compile it. Test: `test_mlx_backend` (equals
`flat_search` up to near-tie swaps).

### 3.16 MCP tool layer — `mcp_tools.h`, `mcp_tools.cpp`
Transport-free tools (JSON args → JSON result) so they are unit-testable:
`vdb_list_collections`, `vdb_create_collection`, `vdb_upsert`,
`vdb_search`, `vdb_get`, `vdb_delete`, `memory_remember`, `memory_recall`,
`memory_forget`. With an `Embedder`, text replaces vectors; memory tools
use one `agent_memory` collection with `namespace` as the tenant field.
Test: `test_mcp_tools` (schemas, errors, CRUD, text-in/text-out with a
fake embedder, memory isolation and forgetting).

### 3.17 MCP server — `src/mcp/server.cpp` (+ `vdb serve`)
**Framework choice: cxxmcp 1.3.0.** Candidates considered (Nov 2026):
- *cxxmcp* — C++17, MIT, conformance-tested (server 272/272), protocol
  revisions 2024-11-05 … 2026-07-28, stdio + Streamable HTTP via bundled
  header-only cpp-httplib, no Boost, FetchContent-friendly. **Chosen.**
- *gopher-mcp* — feature-rich (many transports, bindings) but much heavier
  and enterprise-oriented; more than a local tool needs.
- *mcp-cpp-sdk (yurirocha15)* — C++20 coroutines on Boost.Asio; a Boost
  dependency and very new.
- Hand-rolled JSON-RPC over stdio — viable (~300 lines) but forfeits
  conformance testing and HTTP.

"Efficient" here means: the transport is never the bottleneck (MCP calls
are a handful per agent turn; the engine call dominates). Keep the glue
thin: cxxmcp `ServerPeer` → `.tool<Json, Json>(name, handler)` → handler
dumps args to text → `McpTools::call` → parse result. If profiling ever
shows JSON overhead, pass `nlohmann::json` straight through instead.

`vdb serve --path DIR [--transport stdio|http] [--port 8848]
[--embedder http://localhost:11434/v1 --model nomic-embed-text]`.
Background optimizer on. Register in Claude Code with
`claude mcp add vdb -- /path/to/build/vdb serve --path ~/.vdb`.

### 3.18 CLI — `src/main.cpp`
CLI11 subcommands (`require_subcommand(1)`; options bound to locals in a
struct per subcommand):

```
vdb create   --path P NAME --dim N [--metric cosine|l2|ip] [--index f:type[:tenant]]... [--text] [--sq8]
vdb list     --path P
vdb drop     --path P NAME
vdb import   --path P NAME FILE.ndjson        # {"id","vector","document","payload"} per line
vdb search   --path P NAME (--vector JSON | --text Q) [-k 10] [--filter JSON] [--exact] [--explain]
vdb get      --path P NAME ID...
vdb delete   --path P NAME (ID... | --filter JSON)
vdb optimize --path P NAME [--force]
vdb stats    --path P [NAME]
vdb serve    --path P [--transport stdio|http] [--port N] [--embedder URL --model M]
vdb bench    [--n 100000] [--dim 128] [--queries 1000] [--fvecs FILE] [--k 10] [--ef 10,20,40,...]
```
Output JSON lines by default (scriptable), `--pretty` for humans.

### 3.19 Embedders — `embedder.h` (+ later `http_embedder.cpp`, `llama_embedder.cpp`)
- `HttpEmbedder`: POST `/v1/embeddings` (OpenAI-compatible: Ollama,
  LM Studio, llama.cpp server) via cpp-httplib (already present through
  cxxmcp). Batches, normalizes, checks dim.
- `LlamaEmbedder` (optional, `VDB_WITH_LLAMA`): llama.cpp in-process,
  GGUF embedding model, Metal on macOS.

---

## 4. Milestones and progress

Rule 1 (oracle): flat search lands first and is never removed; every
approximate path is tested against it. Rule 2 (ratchet): from M3 on,
`vdb bench` numbers are recorded in `BENCH.md`; a change that loses > 0.5
point of recall@10 or > 10% QPS needs a reason.

### M0 — Scaffold (done)
- [x] CMake (C++26), FetchContent deps, Makefile shortcuts, executables in `build/`
- [x] Public headers for every component (design contract)
- [x] Unit tests for every component, auto-enabled per implementation file
- [x] `TODO.md` design + plan

### M1 — Exact vector search (core math)
- [ ] `distance.cpp` — Highway kernels with runtime dispatch → `test_distance`
- [ ] `topk.cpp` → `test_topk`
- [ ] `vector_store.cpp` (aligned storage, save/load) → `test_vector_store`
- [ ] `flat_search.cpp` → `test_flat_search`
- [ ] Record kernel throughput (distances/s, single thread, dim 128/768) in `BENCH.md`

Done when: 100k × 128 exact top-10 matches the reference; all four tests pass.

### M2 — Payloads and filters
- [ ] `payload.cpp` (simdjson parse, JSON + binary encode) → `test_payload`
- [ ] `filter.cpp` (parser + `matches`) → `test_filter`
- [ ] `payload_index.cpp` (keyword/int/float/bool, evaluate, estimate) → `test_payload_index`

Done when: indexed evaluation equals scanning for every filter shape.

### M3 — HNSW
- [ ] `hnsw_index.cpp` insert + search → `test_hnsw` recall tests
- [ ] invariants, determinism, filtered search
- [ ] parallel `build()`
- [ ] save/load with checksum
- [ ] `vdb bench` first version (synthetic + `.fvecs`, recall vs QPS over ef)

Done when: recall@10 ≥ 0.95 at ≥ 10× brute-force QPS on 1M × 128 (SIFT1M
if you have it), recall monotone in ef.

### M4 — Quantization
- [ ] `sq8.cpp` → `test_sq8`
- [ ] HNSW traversal on SQ8 codes + float rescore (collection wiring in M7)

Done when: 4× smaller hot set, recall@10 after rescore within 1 point.

### M5 — Durability primitives
- [ ] `crc32c.cpp` (hardware + table fallback) → `test_crc32c`
- [ ] `wal.cpp` → `test_wal`

### M6 — Planner
- [ ] `planner.cpp` → `test_planner`
- [ ] sampling estimator for unindexed filters (inside the segment)

### M7 — Segments, Collection, Database (the database)
- [ ] `segment.h/.cpp`: growing segment (flat, mutable) and sealed segment (HNSW, immutable), payload + document storage, deleted bitmap, point versions
- [ ] `collection.cpp`: validation, write path, snapshot reads, merge, explain → `test_collection`
- [ ] `database.cpp`: directory layout, LOCK, MANIFEST, config.json, in-memory mode → `test_collection`
- [ ] segment file formats + WAL replay on open → `test_persistence`
- [ ] planner wiring + scan fallback → `test_filtered_search`
- [ ] tombstones, vacuum, merge optimizers → `test_churn`
- [ ] snapshot concurrency → `test_concurrency` (also `make tsan`)
- [ ] SQ8 wiring (`Quantization::SQ8`) → `Collection.Sq8QuantizedRecall`

Done when: the selectivity sweep, churn, and SIGKILL tests pass.

### M8 — Hybrid search and agent memory
- [ ] `bm25.cpp` → `test_bm25`
- [ ] `fusion.cpp` → `test_fusion`
- [ ] `memory_score.cpp` → `test_memory_score`
- [ ] collection wiring: `text`, hybrid RRF, `MemoryScoring` → `Collection.TextAndHybridSearch`, `Collection.MemoryScoringPrefersRecentImportant`

### M9 — MCP
- [ ] `mcp_tools.cpp` → `test_mcp_tools`
- [ ] `HttpEmbedder` (OpenAI-compatible endpoint)
- [ ] `src/mcp/server.cpp` with cxxmcp: stdio + Streamable HTTP
- [ ] background optimizer thread (`DatabaseOptions::background_optimizer`)
- [ ] manual check with Claude Code (`claude mcp add ...`) and the MCP inspector

### M10 — CLI
- [ ] subcommands per §3.18 (create/list/drop/import/search/get/delete/optimize/stats/serve)
- [ ] NDJSON import with simdjson `iterate_many`, batched upserts

### M11 — Performance pass
- [ ] mmap sealed segment files (read-only) instead of reading into RAM
- [ ] co-locate SQ8 codes with layer-0 links; graph-order id relabeling at seal
- [ ] prefetch next candidates in HNSW search
- [ ] batch query API (`Collection::search_batch`) using a thread pool
- [ ] HNSW graph reuse + healing on vacuum (Qdrant `graph_layers_healer` idea)
- [ ] `BENCH.md`: recall/QPS curves, build time, index size, memory (RSS)

### M12 — Optional extensions
- [ ] `mlx_backend.cpp` (macOS) → `test_mlx_backend`; use for `vdb bench` ground truth and `search_batch`
- [ ] `LlamaEmbedder` via llama.cpp (`VDB_WITH_LLAMA`)
- [ ] binary / RaBitQ quantization with asymmetric query + oversampling
- [ ] payload-aware extra HNSW links for tenant / keyword blocks (Qdrant `payload_m`)
- [ ] bi-temporal memory fields (`valid_from/valid_to`, `invalidated_at`) and `memory_update` (invalidate + insert)
- [ ] filter DSL fuzzer (libFuzzer), docs-as-tests for the README quickstart

---

## 5. Testing strategy

| Layer | Where |
|---|---|
| Unit / reference math | `test_distance`, `test_topk`, `test_crc32c`, `test_bm25`, `test_fusion`, `test_memory_score` |
| Format round-trip + corruption + truncation | `test_vector_store`, `test_payload`, `test_wal`, `test_hnsw` (save/load) |
| Oracle / recall vs brute force | `test_hnsw`, `test_sq8`, `test_collection`, `test_filtered_search`, `test_churn` |
| Property / invariants | HNSW degree, levels, reachability, recall monotone in ef; cardinality interval contains truth |
| Metamorphic | filter laws (`test_filter`, `test_filtered_search`) |
| Crash / durability | fork + `_exit`, fork + `SIGKILL` (`test_persistence`) |
| Concurrency | `test_concurrency` under `make tsan` |
| Planner decisions | `test_planner`, `Collection.ExplainReflectsPlanner` |
| Protocol surface | `test_mcp_tools` |

Run `make asan` and `make tsan` before ticking a milestone.

---

## 6. Open questions (decide when you get there)

- Growing-segment size before sealing: 10 MiB of vectors by default
  (Qdrant); for agent memory per-namespace sizes are tiny, so most memory
  collections may never seal — that is fine, the planner scans.
- Whether `get_or_create` should also compare `indexes` (currently only
  dim, metric, embedding model).
- Snapshot publication: mutex-protected `shared_ptr` copy vs
  `std::atomic_load` (deprecated but available) vs a hand-rolled RCU;
  measure with `test_concurrency` + a read-heavy bench.
- Recency source for memory scoring: creation `ts` only, or also update
  a `last_accessed` field on recall (reads become writes; batch them).
