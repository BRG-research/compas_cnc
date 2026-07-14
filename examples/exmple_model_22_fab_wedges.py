"""Wedges -- flat-machined from stock (data/custom_toolpath_wedges/).

Ø6 mm surfaces the up-facing faces (some tilted). The tilted faces drop up to ~25 mm, so
they are ROUGHED first in constant-Z terraces (light flat steps down from the stock top)
and only then finished with the continuous inclined sweep -- the tool hogs the wedge out
gently instead of one fully-buried pass. Each sweep starts from the top-outer bbox corner.
Ø6 mm then ramps the contours through the stock with 6 hold-down tabs. Everything is one
tool, so ``run()`` writes a single .nc. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_22_fab_wedges.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_wedges", "Wedges")
    # flip=[3,4]: rotate the sweep 90deg on the two tilted-face finishes behind slider
    # paths 5 and 7 (roughing adds a path per slope, so those paths are surfacing faces 3 & 4).
    .surface(start="outer", rough=True, flip=[3, 4])  # Ø6 mm: terrace-rough tilted faces, then finish
    # ramp ends exactly at the contour Z (overcut 0 -> never crosses Z0); 6 tabs, each a real
    # 10mm-long x 2mm-thick uncut bridge (lift zone auto-widened by the tool diameter).
    .ramp(tabs=6, tab_width=10.0, tab_height=2.0)  # Ø6 mm: ramp contours through, 6 hold-down tabs
    .run()                               # one tool -> a single .nc, then open the viewer
)
