cmake_minimum_required(VERSION 3.28)

project(vdb
  VERSION 0.1.0
  DESCRIPTION "Local embedded vector database for RAG and agent memory"
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Every executable (vdb, test_*) lands directly in the build directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} ${CMAKE_BINARY_DIR})
endforeach()

option(VDB_BUILD_TESTS "Build unit tests" ON)
option(VDB_WITH_MCP "Build the MCP server (vdb serve)" ON)
option(VDB_NATIVE "Compile with -march=native (non-portable binary)" OFF)
set(VDB_WITH_MLX AUTO CACHE STRING "Apple MLX batch backend: AUTO, ON, OFF")
set_property(CACHE VDB_WITH_MLX PROPERTY STRINGS AUTO ON OFF)
set(VDB_SANITIZE "" CACHE STRING "Sanitizer: address, thread, undefined, or empty")
set_property(CACHE VDB_SANITIZE PROPERTY STRINGS "" address thread undefined)

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

if(VDB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

message(STATUS "vdb ${PROJECT_VERSION}: build=${CMAKE_BUILD_TYPE} "
               "mcp=${VDB_WITH_MCP} mlx=${VDB_HAS_MLX} "
               "sanitize='${VDB_SANITIZE}' native=${VDB_NATIVE}")
