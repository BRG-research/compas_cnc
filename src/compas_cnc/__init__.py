__author__ = ["Petras Vestartas"]
__copyright__ = "Petras Vestartas"
__license__ = "MIT License"
__email__ = "petrasvestartas@gmail.com"
__version__ = "0.1.0"

from compas_cnc.dxf import load_dxf
from compas_cnc.helix_drill import toolpath_helix_drill
from compas_cnc.ramp_line import toolpath_ramp_line
from compas_cnc.ramp_path import toolpath_ramp_path
from compas_cnc.toolpath import toolpath_2d_rectangle

__all__ = [
    "load_dxf",
    "toolpath_2d_rectangle",
    "toolpath_helix_drill",
    "toolpath_ramp_line",
    "toolpath_ramp_path",
]

# The Clipper2 wrapper lives in a compiled extension (compas_cnc._clipper2).
# Expose its friendly API when the extension has been built, but don't make
# the pure-python parts of the package unimportable if it hasn't been.
try:
    from compas_cnc.clipper2 import clip
    from compas_cnc.clipper2 import hatch
    from compas_cnc.clipper2 import offset_polyline
    from compas_cnc.clipper2 import outline

    __all__ += ["offset_polyline", "outline", "hatch", "clip"]
except ImportError:
    pass
