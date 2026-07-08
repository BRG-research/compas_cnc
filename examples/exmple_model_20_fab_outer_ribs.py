"""Outer ribs -- flat-machined from stock (data/custom_toolpath_outer_ribs/).

Ø6 mm surfaces the up-facing face (an N-gon -> raster-pocketed) and Ø3.175 mm surfaces
the finer faces, then Ø3.175 mm ramps the contours through the stock with 4 hold-down
tabs. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_20_fab_outer_ribs.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_outer_ribs", "Outer ribs")
    .surface()      # Ø6 + Ø3.175 mm: mill inside the up-facing faces (N-gon via hatch)
    .ramp(tabs=4)   # Ø3.175 mm: ramp contours through, 4 hold-down tabs
    .run()          # write one .nc per tool + open the viewer
)
