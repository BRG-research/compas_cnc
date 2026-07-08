"""Wedges -- flat-machined from stock (data/custom_toolpath_wedges/).

Ø6 mm surfaces the up-facing faces (some tilted -> continuous inclined cutting), starting
each sweep from the top-outer corner of the whole part's bounding box, and Ø6 mm ramps the
contours through the stock with 4 hold-down tabs. Everything is one tool, so ``run()``
writes a single .nc. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_22_fab_wedges.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_wedges", "Wedges")
    .surface(start="outer")  # Ø6 mm: start from the top-outer bbox corner, cut inside
    .ramp(tabs=4)            # Ø6 mm: ramp contours through, 4 hold-down tabs
    .run()                   # one tool -> a single .nc, then open the viewer
)
