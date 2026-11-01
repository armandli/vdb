# Warning and optimization flags shared by every vdb target. Third-party
# targets are fetched as SYSTEM, so these warnings never apply to them.
add_library(vdb_compile_options INTERFACE)

target_compile_options(vdb_compile_options INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:RelWithDebInfo>:-O2 -g>
  $<$<CONFIG:Debug>:-O0 -g3>)

if(VDB_NATIVE)
  target_compile_options(vdb_compile_options INTERFACE -march=native)
endif()

if(VDB_SANITIZE)
  if(NOT VDB_SANITIZE MATCHES "^(address|thread|undefined)$")
    message(FATAL_ERROR "VDB_SANITIZE must be address, thread or undefined")
  endif()
  set(sanitize_flags -fsanitize=${VDB_SANITIZE} -fno-omit-frame-pointer)
  if(VDB_SANITIZE STREQUAL "address")
    list(APPEND sanitize_flags -fsanitize=undefined)
  endif()
  target_compile_options(vdb_compile_options INTERFACE ${sanitize_flags})
  target_link_options(vdb_compile_options INTERFACE ${sanitize_flags})
endif()
