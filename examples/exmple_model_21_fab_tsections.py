"""T-sections -- flat-machined from stock (data/custom_toolpath_tsections/).

Ø6 mm surfaces the up-facing face, then Ø3.175 mm ramps the INCLINED faces given as
paired top/bottom rail loops: each top loop is matched to its bottom loop, both grown
outward by the tool radius (waste side), and the tool descends a helix morphing top into
bottom. Each of the six T-sections is held to the stock by 4 hold-down tabs so it does
not shift once its profile is cut through. The parts are only ~2.7 mm thick, so the tab
bridge is thinned to 1 mm (the 2 mm default suits the thicker ribs/wedges) -- enough to
hold a flat part, thin enough to snap free. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_21_fab_tsections.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_tsections", "T-sections")
    .surface()                              # Ø6 mm: mill inside the up-facing face
    .paired_ramp(tabs=4, tab_height=1.0)    # Ø3.175 mm: paired rails + 4 tabs (1mm bridge) each
    .run()                                  # write one .nc per tool + open the viewer
)
