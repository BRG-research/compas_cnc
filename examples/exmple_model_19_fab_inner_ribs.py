"""Inner ribs -- flat-machined from stock (data/custom_toolpath_inner_ribs/).

Ø6 mm and Ø3.175 mm surface the up-facing faces, then Ø3.175 mm cuts the rib walls
given as PAIRED top/bottom polylines: each top loop is matched to its near-identical
bottom loop and the tool descends a helix that interpolates between the two, with 4
hold-down tabs holding each rib to the stock. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_19_fab_inner_ribs.py

(Pass ``separate=True`` to ``paired_ramp`` to instead post the rib-wall cut to its own
``..._3_175mm_paired_ramp.nc`` so it can be re-run on its own.)
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_inner_ribs", "Inner ribs")
    .surface(flat_stepdown=1.5)  # Ø6 + Ø3.175 mm faces; the -6 pockets step down -4.5, -6
    .paired_ramp(tabs=4)         # Ø3.175 mm: interpolate each top/bottom rib pair + 4 tabs
    .run()                       # write one .nc per tool + open the viewer
)
