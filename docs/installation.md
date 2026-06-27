# Installation

## Stable release

```bash
pip install compas_cnc
```

This pulls a pre-built wheel (the C++ Clipper2 extension is already compiled), so no
build tools are needed.

## Development with uv

`compas_cnc` compiles a C++ extension (a Clipper2 wrapper) via `scikit-build-core` +
`nanobind`, so a development install needs a C++17 compiler, CMake (>=3.15) and Git.
The project environment is managed with [uv](https://docs.astral.sh/uv/).

Set up the environment:

```bash
git clone https://github.com/BRG-research/compas_cnc.git
cd compas_cnc
uv venv --python 3.12          # create .venv
```

Install the build backend, then the package itself with build isolation off (so the
extension is rebuilt against this environment):

```bash
uv pip install "nanobind>=2.12" "scikit-build-core>=0.10"
uv pip install --no-build-isolation -ve .
```

Add `-Ceditable.rebuild=true` to the last command to auto-recompile the extension on
import while developing.

## Running

The examples use the COMPAS Viewer for visualisation:

```bash
uv pip install compas_viewer
uv run python docs/examples/toolpaths/drilling.py
```

Run the test suite with:

```bash
uv run pytest
```

## Documentation

To build and serve the documentation locally:

```bash
uv pip install -r requirements-docs.txt
uv run mkdocs serve
```
