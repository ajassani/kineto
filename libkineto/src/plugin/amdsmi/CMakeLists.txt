cmake_minimum_required(VERSION 3.18)
project(kineto_amdsmi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ===== amd-smi =====
# We require both header and library; default search hint is /opt/rocm.
find_path(AMDSMI_INCLUDE_DIR
  NAMES amd_smi/amdsmi.h
  HINTS ${ROCM_PATH} /opt/rocm /opt/rocm/include
  PATH_SUFFIXES include
)
find_library(AMDSMI_LIBRARY
  NAMES amd_smi
  HINTS ${ROCM_PATH} /opt/rocm /opt/rocm/lib /opt/rocm/lib64
  PATH_SUFFIXES lib lib64
)
if(NOT AMDSMI_INCLUDE_DIR OR NOT AMDSMI_LIBRARY)
  message(FATAL_ERROR
    "Could not find amd-smi.  Set ROCM_PATH or install amd-smi-lib-dev.\n"
    "  AMDSMI_INCLUDE_DIR=${AMDSMI_INCLUDE_DIR}\n"
    "  AMDSMI_LIBRARY=${AMDSMI_LIBRARY}")
endif()
message(STATUS "Found amd-smi: ${AMDSMI_LIBRARY}")
message(STATUS "Found amd-smi includes: ${AMDSMI_INCLUDE_DIR}")

# ===== kineto headers (shipped with torch) =====
# We accept either an explicit KINETO_INCLUDE_DIR cache var, or auto-detect
# via a Python helper that locates torch and uses its include dir.  We need
# headers like libkineto.h, IActivityProfiler.h, GenericTraceActivity.h.
if(NOT KINETO_INCLUDE_DIR)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import torch, os; print(os.path.dirname(torch.__file__))"
      OUTPUT_VARIABLE TORCH_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE TORCH_FIND_RC
    )
    if(TORCH_FIND_RC EQUAL 0 AND TORCH_DIR)
      set(KINETO_INCLUDE_DIR "${TORCH_DIR}/include/kineto" CACHE PATH "" FORCE)
      set(KINETO_TORCH_LIB_DIR "${TORCH_DIR}/lib" CACHE PATH "" FORCE)
    endif()
  endif()
endif()
if(NOT KINETO_INCLUDE_DIR OR NOT EXISTS "${KINETO_INCLUDE_DIR}/libkineto.h")
  message(FATAL_ERROR
    "Could not find kineto headers (libkineto.h).  Set KINETO_INCLUDE_DIR "
    "to a directory containing libkineto.h, IActivityProfiler.h, etc.")
endif()
message(STATUS "Found kineto includes: ${KINETO_INCLUDE_DIR}")

# ===== fmt (kineto headers transitively include <fmt/format.h>) =====
# In common ROCm PyTorch containers fmt is installed system-wide via conda at
# /opt/conda/include.  Allow override with FMT_INCLUDE_DIR.
find_path(FMT_INCLUDE_DIR
  NAMES fmt/format.h
  HINTS ${FMT_PATH} /opt/conda/include /usr/include /usr/local/include
)
if(NOT FMT_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not find fmt/format.h.  Set FMT_INCLUDE_DIR to a directory "
    "containing fmt/.  In common ROCm PyTorch containers it is /opt/conda/include.")
endif()
message(STATUS "Found fmt includes: ${FMT_INCLUDE_DIR}")

# Find a fmt library.  libtorch has fmt symbols statically linked but does
# not re-export some (notably fmt::vformat) which we transitively need via
# the kineto headers' addMetadata template that calls fmt::format.
#
# Prefer libfmt.a if torch ships one (this avoids the heap corruption seen
# in environments where two distinct copies of fmt v11 coexist between
# libtorch_cpu's static fmt and a dynamically loaded /opt/conda/lib/libfmt.so).
# Fall back to libfmt.so.
set(FMT_LIBRARY "")
if(KINETO_TORCH_LIB_DIR)
  find_library(FMT_STATIC_LIBRARY
    NAMES libfmt.a fmt
    HINTS ${KINETO_TORCH_LIB_DIR}
    NO_DEFAULT_PATH
  )
  if(FMT_STATIC_LIBRARY)
    message(STATUS "Found static fmt (preferred): ${FMT_STATIC_LIBRARY}")
    set(FMT_LIBRARY "${FMT_STATIC_LIBRARY}")
  endif()
endif()
if(NOT FMT_LIBRARY)
  find_library(FMT_DYNAMIC_LIBRARY
    NAMES fmt
    HINTS ${FMT_PATH} /opt/conda/lib /usr/lib /usr/lib/x86_64-linux-gnu
  )
  if(FMT_DYNAMIC_LIBRARY)
    message(STATUS "Found dynamic fmt: ${FMT_DYNAMIC_LIBRARY}")
    set(FMT_LIBRARY "${FMT_DYNAMIC_LIBRARY}")
  endif()
endif()
if(NOT FMT_LIBRARY)
  message(WARNING
    "Could not find any fmt library; the plugin .so may have undefined "
    "references to fmt::vformat at load time.")
endif()

# ===== libtorch_cpu (kineto symbols are statically linked into it) =====
if(NOT KINETO_TORCH_LIB_DIR)
  message(FATAL_ERROR
    "KINETO_TORCH_LIB_DIR not set.  Need libtorch_cpu.so for kineto symbols.")
endif()
find_library(TORCH_CPU_LIBRARY
  NAMES torch_cpu
  HINTS ${KINETO_TORCH_LIB_DIR}
  NO_DEFAULT_PATH
)
if(NOT TORCH_CPU_LIBRARY)
  message(FATAL_ERROR
    "Could not find libtorch_cpu.so in ${KINETO_TORCH_LIB_DIR}")
endif()
message(STATUS "Found libtorch_cpu: ${TORCH_CPU_LIBRARY}")

# ===== plugin .so =====
add_library(kineto_amdsmi SHARED
  src/AmdSmiDriver.cpp
  src/AmdSmiSampler.cpp
  src/AmdSmiActivityProfilerSession.cpp
  src/AmdSmiActivityProfiler.cpp
  src/PluginRegister.cpp
)

target_include_directories(kineto_amdsmi PRIVATE
  include
  ${AMDSMI_INCLUDE_DIR}
  ${KINETO_INCLUDE_DIR}
  ${FMT_INCLUDE_DIR}
)

target_link_libraries(kineto_amdsmi PRIVATE
  ${AMDSMI_LIBRARY}
  ${TORCH_CPU_LIBRARY}
  pthread
)
# Note: we deliberately do NOT link against any fmt library.  Setting
# FMT_HEADER_ONLY above causes us to instantiate fmt template internals
# (including fmt::vformat) directly into our .so as inline functions.

# Defensive flags.
target_compile_options(kineto_amdsmi PRIVATE -Wall -Wextra -Wno-unused-parameter)
# Compile fmt as header-only so we don't depend on a libfmt.so/.a at link
# or load time.  This matches what kineto upstream does for its own .so
# builds and avoids a class of issues with duplicate fmt static state.
target_compile_definitions(kineto_amdsmi PRIVATE FMT_HEADER_ONLY=1)
# Hide fmt symbols so they don't conflict with libtorch_cpu's own
# statically-linked fmt.  Hidden visibility keeps our copy local to this
# .so without affecting the user's ability to call our exported symbols.
target_compile_options(kineto_amdsmi PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

# ===== standalone driver smoke test =====
add_executable(amdsmi_dump
  src/AmdSmiDriver.cpp
  src/standalone_dump.cpp
)
target_include_directories(amdsmi_dump PRIVATE
  include
  ${AMDSMI_INCLUDE_DIR}
)
target_link_libraries(amdsmi_dump PRIVATE ${AMDSMI_LIBRARY} pthread)

install(TARGETS kineto_amdsmi LIBRARY DESTINATION lib)
install(TARGETS amdsmi_dump RUNTIME DESTINATION bin)
