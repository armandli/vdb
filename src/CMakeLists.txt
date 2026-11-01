# libvdb: the embeddable engine. Every src/vdb/*.cpp is picked up
# automatically, so adding a component never requires editing this file.
file(GLOB VDB_LIB_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/vdb/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp)

# mlx_*.cpp only compiles against MLX, so it is dropped when MLX is absent.
if(NOT VDB_HAS_MLX)
  list(FILTER VDB_LIB_SOURCES EXCLUDE REGEX "/vdb/mlx_[^/]*\\.cpp$")
endif()

configure_file(vdb/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/vdb/version.h @ONLY)

# Until the first component is implemented the library has no sources;
# a generated empty translation unit keeps the target valid.
if(NOT VDB_LIB_SOURCES)
  set(placeholder ${CMAKE_CURRENT_BINARY_DIR}/generated/placeholder.cpp)
  file(WRITE ${placeholder} "")
  set(VDB_LIB_SOURCES ${placeholder})
endif()

add_library(vdb_lib STATIC ${VDB_LIB_SOURCES})
add_library(vdb::vdb ALIAS vdb_lib)
set_target_properties(vdb_lib PROPERTIES OUTPUT_NAME vdb)

target_include_directories(vdb_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_link_libraries(vdb_lib
  PUBLIC vdb_compile_options roaring::roaring
  PRIVATE simdjson hwy)

find_package(Threads REQUIRED)
target_link_libraries(vdb_lib PUBLIC Threads::Threads)

if(VDB_HAS_MLX)
  target_link_libraries(vdb_lib PRIVATE mlx)
  target_compile_definitions(vdb_lib PUBLIC VDB_HAS_MLX=1)
endif()

# vdb: the command line tool, including `vdb serve` (MCP server).
set(VDB_EXE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
if(VDB_WITH_MCP)
  file(GLOB VDB_MCP_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/mcp/*.cpp)
  list(APPEND VDB_EXE_SOURCES ${VDB_MCP_SOURCES})
endif()

add_executable(vdb ${VDB_EXE_SOURCES})
target_link_libraries(vdb PRIVATE vdb_lib CLI11::CLI11)

if(VDB_WITH_MCP)
  target_link_libraries(vdb PRIVATE cxxmcp::server)
  target_compile_definitions(vdb PRIVATE VDB_WITH_MCP=1)
endif()
