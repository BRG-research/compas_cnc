"""Inner beams -- flat-machined from stock (data/custom_toolpath_inner_beams/).

Ø6 mm surfaces the up-facing faces, then Ø3.175 mm ramps the contours through the stock
(4 hold-down tabs so the freed parts stay put) and helically drills the holes. Two tools
-> two .nc files, written by ``run()``. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_18_fab_inner_beams.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_inner_beams", "Inner beams")
    .surface()      # Ø6 mm: mill inside every up-facing rectangle
    .ramp(tabs=4)   # Ø3.175 mm: ramp contours through, 4 hold-down tabs
    .drill()        # Ø3.175 mm: helical-drill the holes
    .run()          # write one .nc per tool + open the viewer
)
