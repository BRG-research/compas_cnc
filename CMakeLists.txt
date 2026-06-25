cmake_minimum_required(VERSION 3.15...3.27)

project(clipper2_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)              # Clipper2 requires C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build for size (these extensions are small; smaller wheels are nicer).
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if(MSVC)
    set(CMAKE_GENERATOR_PLATFORM x64)
    add_compile_options(/O1 /EHsc)
else()
    add_compile_options(-Os)
endif()

# =====================================================================
# Dependency: Clipper2 (header + 3 source files, no transitive deps).
# Downloaded at configure time and compiled straight into our module so
# we never depend on Clipper2's own CMake / install rules.
# =====================================================================
include(FetchContent)

FetchContent_Declare(
    clipper2
    URL https://github.com/AngusJohnson/Clipper2/archive/refs/tags/Clipper2_1.5.4.zip
)
# The repo root has no CMakeLists.txt (the C++ tree lives under CPP/), so
# MakeAvailable just populates the sources without configuring Clipper2.
FetchContent_MakeAvailable(clipper2)

set(CLIPPER2_LIB_DIR "${clipper2_SOURCE_DIR}/CPP/Clipper2Lib")
message(STATUS "Clipper2 sources: ${CLIPPER2_LIB_DIR}")

# =====================================================================
# nanobind
# =====================================================================
if(NOT SKBUILD)
    message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'. Running
  it directly will almost certainly not produce the desired result. To build
  and install the package use:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  Add -Ceditable.rebuild=true to auto-rebuild on import while developing.")
endif()

find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# =====================================================================
# The extension module: our binding + the three Clipper2 sources.
# =====================================================================
nanobind_add_module(_clipper2 STABLE_ABI NB_STATIC
    src/clipper2.cpp
    "${CLIPPER2_LIB_DIR}/src/clipper.engine.cpp"
    "${CLIPPER2_LIB_DIR}/src/clipper.offset.cpp"
    "${CLIPPER2_LIB_DIR}/src/clipper.rectclip.cpp"
)

target_include_directories(_clipper2 PRIVATE "${CLIPPER2_LIB_DIR}/include")

install(TARGETS _clipper2 LIBRARY DESTINATION compas_cnc)
