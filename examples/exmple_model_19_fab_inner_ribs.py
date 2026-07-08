"""Inner ribs -- flat-machined from stock (data/custom_toolpath_inner_ribs/).

Ø6 mm and Ø3.175 mm surface the up-facing faces, then Ø3.175 mm ramps the contours
through the stock with 4 hold-down tabs. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_19_fab_inner_ribs.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_inner_ribs", "Inner ribs")
    .surface()      # Ø6 + Ø3.175 mm: mill inside the up-facing faces
    .ramp(tabs=4)   # Ø3.175 mm: ramp contours through, 4 hold-down tabs
    .run()          # write one .nc per tool + open the viewer
)
