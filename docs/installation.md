# Installation

## Development

`compas_cnc` compiles a C++ extension (a Clipper2 wrapper), so install it from a
local clone with build isolation off.

```bash
git clone https://github.com/petrasvestartas/compas_cnc.git
cd compas_cnc
pip install "nanobind>=2.12" "scikit-build-core>=0.10"
pip install --no-build-isolation -ve .
```

The examples use the COMPAS Viewer for visualisation:

```bash
pip install compas_viewer
```

## Documentation

To build and serve the documentation locally:

```bash
pip install -e ".[docs]"
python docs/screenshots.py   # regenerate the example screenshots (needs compas_viewer)
mkdocs serve
```
