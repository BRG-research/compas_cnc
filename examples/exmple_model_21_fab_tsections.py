"""T-sections -- flat-machined from stock (data/custom_toolpath_tsections/).

Ø6 mm surfaces the up-facing face, then Ø3.175 mm ramps the INCLINED faces given as
paired top/bottom rail loops: each top loop is matched to its bottom loop, both grown
outward by the tool radius (waste side), and the tool descends a helix morphing top into
bottom. Run with the project .venv:

    .venv/Scripts/python.exe examples/exmple_model_21_fab_tsections.py
"""

import _custom_toolpath as ct

(
    ct.Job("custom_toolpath_tsections", "T-sections")
    .surface()        # Ø6 mm: mill inside the up-facing face
    .paired_ramp()    # Ø3.175 mm: inclined faces from paired top/bottom rails
    .run()            # write one .nc per tool + open the viewer
)
