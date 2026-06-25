# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

* Added a Clipper2 wrapper as the nanobind extension `compas_cnc._clipper2`, with the friendly API `compas_cnc.offset_polyline` (grow/shrink a polyline) and `compas_cnc.hatch` (fill a closed polyline with parallel lines clipped to its interior).
* Added `examples/example_clipper2_offset_hatch.py` demonstrating concentric offsets and an angled hatch fill.
* Added `compas_cnc.toolpath_helix_drill`, a helical drilling / boring tool-path that spirals a cutting tool down a cylindrical hole axis (tool centre rides a helix of radius `hole_radius - tool_radius`), degenerating to a straight plunge when the tool is wider than the hole. Supports a `length` override (anchor the bottom, extend the top along the axis to an absolute length), a `floor` Z-limit (the drill stops at that depth and goes no lower), and a `safe_z` Z-safety retract (a final point above where the tool finished, at the last point's X/Y, like `toolpath_2d_rectangle`'s lead-out). Wired into `examples/exmple_model_12_fab_column.py` to bore the column's cylindrical cut features with a 3 mm end mill (length 30, floor z=-1, safe_z 60).

### Changed

* Switched the build backend from `setuptools` to `scikit-build-core` + `nanobind` so the package can ship the compiled Clipper2 extension. Clipper2 (v1.5.4) is fetched and compiled at build time via CMake `FetchContent`; building now requires a C++17 compiler.

### Removed

