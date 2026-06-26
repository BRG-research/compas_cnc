__author__ = ["Petras Vestartas"]
__copyright__ = "Petras Vestartas"
__license__ = "MIT License"
__email__ = "petrasvestartas@gmail.com"
__version__ = "0.1.0"

from compas_cnc.dxf import load_dxf
from compas_cnc.postprocessor import Postprocessor
from compas_cnc.toolpath_2d_drill import toolpath_2d_drill
from compas_cnc.toolpath_2d_hatch import toolpath_2d_hatch
from compas_cnc.toolpath_2d_ramp import toolpath_2d_ramp
from compas_cnc.toolpath_2d_surfacing import toolpath_2d_surfacing
from compas_cnc.toolpath_merge import toolpath_merge
from compas_cnc.tools import Tool

__all__ = [
    "load_dxf",
    "Tool",
    "Postprocessor",
    "toolpath_2d_drill",
    "toolpath_2d_hatch",
    "toolpath_2d_ramp",
    "toolpath_2d_surfacing",
    "toolpath_merge",
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
