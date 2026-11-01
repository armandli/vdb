# Third-party dependencies. All are pinned to release tags and fetched as
# SYSTEM so their headers do not trigger vdb's warning flags.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# CLI11: command line parsing for the vdb executable.
FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.7.2
  GIT_SHALLOW TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

# simdjson: payload / filter / NDJSON import parsing.
FetchContent_Declare(simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG v5.1.1
  GIT_SHALLOW TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

# Google Highway: portable SIMD with runtime dispatch
# (SSE4/AVX2/AVX-512 on x86, NEON/SVE on ARM) for distance kernels.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.4.0
  GIT_SHALLOW TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

# CRoaring: compressed bitmaps, the currency of filter evaluation.
set(ENABLE_ROARING_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_ROARING_MICROBENCHMARKS OFF CACHE BOOL "" FORCE)
set(ROARING_USE_CPM OFF CACHE BOOL "" FORCE)
FetchContent_Declare(croaring
  GIT_REPOSITORY https://github.com/RoaringBitmap/CRoaring.git
  GIT_TAG v5.4.0
  GIT_SHALLOW TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(cli11 simdjson highway croaring)

if(VDB_WITH_MCP)
  # cxxmcp: MCP server SDK (stdio + Streamable HTTP over bundled
  # cpp-httplib, nlohmann::json, MIT). Only the `vdb serve` code links it.
  set(CXXMCP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CXXMCP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(CXXMCP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(CXXMCP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(CXXMCP_BUILD_CLIENT OFF CACHE BOOL "" FORCE)
  set(CXXMCP_ENABLE_HTTP ON CACHE BOOL "" FORCE)
  set(CXXMCP_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
  set(CXXMCP_ENABLE_AUTH OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(cxxmcp
    GIT_REPOSITORY https://github.com/caomengxuan666/cxxmcp.git
    GIT_TAG v1.3.0
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(cxxmcp)
endif()

if(VDB_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(googletest)
endif()

# Apple MLX: optional GPU batch backend (bulk brute force, k-means training).
# Apple silicon only; Linux builds always run the CPU/Highway path.
set(VDB_HAS_MLX OFF)
if(APPLE AND NOT VDB_WITH_MLX STREQUAL "OFF")
  find_package(MLX CONFIG QUIET HINTS /opt/homebrew /usr/local)
  if(MLX_FOUND)
    set(VDB_HAS_MLX ON)
  elseif(VDB_WITH_MLX STREQUAL "ON")
    message(FATAL_ERROR "VDB_WITH_MLX=ON but MLX was not found "
                        "(brew install mlx)")
  endif()
elseif(VDB_WITH_MLX STREQUAL "ON")
  message(FATAL_ERROR "VDB_WITH_MLX=ON is only supported on macOS")
endif()
