include(GoogleTest)

# vdb_add_test(<name> SOURCES <files...> REQUIRES <src files...>)
#
# Builds ${CMAKE_BINARY_DIR}/test_<name>. REQUIRES lists the implementation
# files (relative to src/) the test links against; until all of them exist
# the test is skipped, so unimplemented milestones never break the build.
function(vdb_add_test name)
  cmake_parse_arguments(arg "" "" "SOURCES;REQUIRES;LIBS" ${ARGN})
  foreach(required IN LISTS arg_REQUIRES)
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/src/${required})
      message(STATUS "test_${name}: skipped (src/${required} not implemented)")
      return()
    endif()
  endforeach()
  add_executable(test_${name} ${arg_SOURCES})
  target_link_libraries(test_${name}
    PRIVATE vdb_lib GTest::gtest_main ${arg_LIBS})
  target_include_directories(test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  gtest_discover_tests(test_${name}
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS ${name})
  set_property(GLOBAL APPEND PROPERTY VDB_TESTS test_${name})
endfunction()

# Ordered by milestone (see TODO.md).
# M1 core math
vdb_add_test(distance SOURCES test_distance.cpp REQUIRES vdb/distance.cpp)
vdb_add_test(topk SOURCES test_topk.cpp REQUIRES vdb/topk.cpp)
vdb_add_test(vector_store
  SOURCES test_vector_store.cpp
  REQUIRES vdb/vector_store.cpp)
vdb_add_test(flat_search
  SOURCES test_flat_search.cpp
  REQUIRES vdb/distance.cpp vdb/topk.cpp vdb/vector_store.cpp
           vdb/flat_search.cpp)

# M2 payload + filter
vdb_add_test(payload SOURCES test_payload.cpp REQUIRES vdb/payload.cpp)
vdb_add_test(filter
  SOURCES test_filter.cpp
  REQUIRES vdb/payload.cpp vdb/filter.cpp)
vdb_add_test(payload_index
  SOURCES test_payload_index.cpp
  REQUIRES vdb/payload.cpp vdb/filter.cpp vdb/payload_index.cpp)

# M3 HNSW
vdb_add_test(hnsw
  SOURCES test_hnsw.cpp
  REQUIRES vdb/distance.cpp vdb/topk.cpp vdb/vector_store.cpp
           vdb/flat_search.cpp vdb/hnsw_index.cpp)

# M4 quantization
vdb_add_test(sq8
  SOURCES test_sq8.cpp
  REQUIRES vdb/distance.cpp vdb/vector_store.cpp vdb/sq8.cpp)

# M5 durability
vdb_add_test(crc32c SOURCES test_crc32c.cpp REQUIRES vdb/crc32c.cpp)
vdb_add_test(wal
  SOURCES test_wal.cpp
  REQUIRES vdb/crc32c.cpp vdb/wal.cpp)

# M6 planner
vdb_add_test(planner SOURCES test_planner.cpp REQUIRES vdb/planner.cpp)

# M7 collection / database (the public API)
vdb_add_test(collection
  SOURCES test_collection.cpp
  REQUIRES vdb/database.cpp vdb/collection.cpp)
vdb_add_test(persistence
  SOURCES test_persistence.cpp
  REQUIRES vdb/database.cpp vdb/collection.cpp)
vdb_add_test(filtered_search
  SOURCES test_filtered_search.cpp
  REQUIRES vdb/database.cpp vdb/collection.cpp)
vdb_add_test(churn
  SOURCES test_churn.cpp
  REQUIRES vdb/database.cpp vdb/collection.cpp)
vdb_add_test(concurrency
  SOURCES test_concurrency.cpp
  REQUIRES vdb/database.cpp vdb/collection.cpp)

# M9 hybrid search + agent memory
vdb_add_test(bm25 SOURCES test_bm25.cpp REQUIRES vdb/bm25.cpp)
vdb_add_test(fusion SOURCES test_fusion.cpp REQUIRES vdb/fusion.cpp)
vdb_add_test(memory_score
  SOURCES test_memory_score.cpp
  REQUIRES vdb/memory_score.cpp)

# M10 MCP tools
vdb_add_test(mcp_tools
  SOURCES test_mcp_tools.cpp
  REQUIRES vdb/database.cpp vdb/collection.cpp vdb/mcp_tools.cpp
  LIBS simdjson)

# M12 MLX batch backend (Apple silicon only)
if(VDB_HAS_MLX)
  vdb_add_test(mlx_backend
    SOURCES test_mlx_backend.cpp
    REQUIRES vdb/flat_search.cpp vdb/mlx_backend.cpp)
endif()
