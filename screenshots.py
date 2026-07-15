"""Regenerate the example screenshots used in the docs.

Renders every ``docs/examples/**/<name>.py`` offscreen with compas_viewer and writes
``docs/assets/images/example_<name>.jpg``: it patches :meth:`Viewer.show` to zoom-extents
(the "F" key) and grab the framebuffer instead of opening a window.

Usage (needs ``compas_viewer``)::

    python docs/screenshots.py            # render every example
    python docs/screenshots.py <py> <jpg> # render one example (used internally per subprocess)
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "docs" / "examples"
IMAGES = ROOT / "docs" / "assets" / "images"


def _render_one(example: str, out: str) -> None:
    import runpy

    from PySide6 import QtCore

    from compas_viewer import Viewer
    from compas_viewer.commands import zoom_selected

    # The side-dock tool slider is interactive UI; for a static doc image the
    # tool-path is the subject, so drop the (tall) tool so the frame stays clean.
    import compas_cnc.tools as tools

    tools.add_tool_slider = lambda viewer, tool, path, **style: None

    orig_show = Viewer.show

    def show(self):
        self.config.renderer.show_grid = False  # clean background for the doc image (checked per-frame)

        def grab():
            try:
                zoom_selected(self)  # zoom-extents so the geometry fills the frame
                QtCore.QCoreApplication.processEvents()
                self.renderer.grabFramebuffer().save(out, "JPG", 92)
            finally:
                QtCore.QCoreApplication.quit()

        QtCore.QTimer.singleShot(1800, grab)
        orig_show(self)

    Viewer.show = show
    runpy.run_path(example, run_name="__main__")


def _render_all() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    for example in sorted(EXAMPLES.rglob("*.py")):
        out = IMAGES / ("example_" + example.stem + ".jpg")
        proc = subprocess.run([sys.executable, __file__, str(example), str(out)], capture_output=True)
        ok = out.exists() and proc.returncode == 0
        print(("ok   " if ok else "FAIL ") + str(example.relative_to(EXAMPLES)))
        if not ok and proc.stderr:
            print(proc.stderr.decode(errors="replace")[-800:])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        _render_one(sys.argv[1], sys.argv[2])
    else:
        _render_all()
