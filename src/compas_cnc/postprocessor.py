import pathlib
import warnings

from compas.geometry import Point

__all__ = ["Postprocessor", "CARVERA_AIR_TRAVEL", "CARVERA_AIR_WORKAREA"]

# Carvera Air travel envelope (mm), per axis (X, Y, Z). The firmware homes
# ``home_to_max``, so the only true HARD (physical) switch sits at the home/max
# corner (= machine 0); the FAR end of each axis has NO switch, so the firmware
# SOFT endstop is the last line of defence before a mechanical crash. These
# factory soft-endstop spans (MakeraInc/CarveraFirmware src/config2.default,
# header "Carvera_Air settings": x_min -302, y_min -212, z_min -121) are therefore
# the practical hard limit a job must stay within. Actual travel varies a little
# per machine -- Makera advise jogging to verify -- so pass a `margin` for safety.
CARVERA_AIR_TRAVEL = (302.0, 212.0, 121.0)
# Rated cutting area Makera advertises. NOTE Z here (130) is the spec-sheet
# clearance figure -- the firmware-controlled Z travel is only 121 mm
# (CARVERA_AIR_TRAVEL), so that envelope, not this one, governs the Z check.
CARVERA_AIR_WORKAREA = (300.0, 200.0, 130.0)


class Postprocessor:
    """Convert tool-path polylines into Carvera Air G-code and write a ``.nc`` file.

    The Carvera / Carvera Air desktop CNC (Makera) runs a Smoothieware-derived
    controller that reads ordinary RS-274 G-code. This post-processor turns the
    tool-CENTRE polylines produced by the ``toolpath_2d_*`` generators (or any
    merged path from :func:`toolpath_merge`) into the dialect the machine expects:
    a ``%`` start marker, a commented header (units, tool change, coolant/air,
    spindle), a body of ``G1`` linear moves, and a shutdown footer.

    The path already encodes its own safe-Z approach, plunge, cut and retract, so
    every motion is emitted as a ``G1`` linear feed after a single initial ``G0``
    lift to the rapid clearance plane -- exactly as a hand-written Carvera program
    does. Only the axis words that CHANGE between consecutive points are written
    (modal coordinates), so a move that keeps Z prints just ``X.. Y..``.

    Before any G-code is written the path is checked against the machine travel
    envelope (:data:`CARVERA_AIR_TRAVEL`, the factory soft-endstop spans -- the
    practical hard limit, since the far end of each axis has no switch). A job
    whose bounding-box span overruns the reach raises by default (see
    :attr:`on_exceed`), so an out-of-bounds program is caught here instead of as
    a soft-limit halt mid-cut.

    The output reproduces this shape::

        %
        ; 3-Axis
        ; Material: Aluminum
        ; Stock Size: 300(X) * 200(Y) * 20(Z) mm
        ; Tool List
        ; T1-3.175*19mm Flat End
        ; Path List
        ; [T1]2D Contour
        G90 G21              ; Absolute positioning, units in millimeters
        T1 M6                ; Tool change to Tool 1
        M7                   ; Coolant ON (mist)
        S12000 M3            ; Spindle ON clockwise at 12000 RPM
        G0 Z50
        G1 F200
        G1 X-149.8351 Y36.0727
        G1 Z20
        G1 X-149.8351 Y36.0727 Z-18.8
        ...
        M9        ; Turn off coolant
        M05       ; Stop the spindle
        G28       ; Return all axes to the machine home position
        M02       ; End of program

    Parameters
    ----------
    tool : :class:`compas_cnc.tools.Tool`, optional
        The cutting tool. Its name, diameter and height fill the tool-list
        comment (``T{n}-{diameter}*{height}mm {name}``). If omitted a generic
        description is written.
    tool_number : int, optional
        Carvera tool-changer slot, used in the ``T{n} M6`` tool change and the
        header comments. Defaults to ``1``.
    feed : float, optional
        Cutting feed rate (mm/min) set once with ``G1 F{feed}``. Defaults to
        ``200``.
    spindle_speed : int, optional
        Spindle RPM for ``S{rpm} M3``. Defaults to ``12000``.
    coolant : str | None, optional
        ``"mist"`` / ``"air"`` -> ``M7``, ``"flood"`` -> ``M8``, ``None`` (or
        ``False``) turns coolant off entirely (no ``M7``/``M9``). Defaults to
        ``"mist"``.
    rapid_z : float, optional
        World-Z for the opening ``G0`` lift to the travel plane. Defaults to the
        highest Z in the path (the tool-path's own safe traverse height), so the
        first cutting move starts from a known clearance height.
    material : str, optional
        Stock material, written into the header comment. Defaults to ``"Aluminum"``.
    stock_size : tuple(float, float, float) | None, optional
        ``(X, Y, Z)`` stock dimensions in mm for the header comment. ``None``
        omits the line. Defaults to ``(300, 200, 20)``.
    program : str, optional
        Path-list label in the header (``; [T{n}]{program}``). Defaults to
        ``"2D Contour"``.
    precision : int, optional
        Decimal places for coordinates; trailing zeros are stripped. Defaults to
        ``4``.
    travel : tuple(float, float, float) | None, optional
        Per-axis ``(X, Y, Z)`` machine reach in mm the job must fit within. A
        ``None`` entry skips that axis; ``travel=None`` disables the check
        entirely. Defaults to :data:`CARVERA_AIR_TRAVEL` ``(302, 212, 121)``.
        Pass your own jogged values (Makera advise verifying per machine) or
        :data:`CARVERA_AIR_WORKAREA` for the conservative rated area.
    margin : float, optional
        Safety margin in mm subtracted from every ``travel`` axis before checking,
        so the job must fit with this much to spare. Defaults to ``0.0``.
    on_exceed : str, optional
        What to do when the path overruns the envelope: ``"raise"`` (default) a
        :class:`ValueError`, ``"warn"`` a :class:`UserWarning`, or ``"ignore"``.

    Attributes
    ----------
    tool, tool_number, feed, spindle_speed, coolant, rapid_z, material,
    stock_size, program, precision, travel, margin, on_exceed
        The configured options above.
    """

    def __init__(
        self,
        tool=None,
        tool_number=1,
        feed=200.0,
        spindle_speed=12000,
        coolant="mist",
        rapid_z=None,
        material="Aluminum",
        stock_size=(300.0, 200.0, 20.0),
        program="2D Contour",
        precision=4,
        travel=CARVERA_AIR_TRAVEL,
        margin=0.0,
        on_exceed="raise",
    ):
        if on_exceed not in ("raise", "warn", "ignore"):
            raise ValueError("on_exceed must be 'raise', 'warn' or 'ignore'.")
        self.tool = tool
        self.tool_number = int(tool_number)
        self.feed = float(feed)
        self.spindle_speed = int(spindle_speed)
        self.coolant = coolant
        self.rapid_z = None if rapid_z is None else float(rapid_z)
        self.material = material
        self.stock_size = None if stock_size is None else tuple(stock_size)
        self.program = program
        self.precision = int(precision)
        self.travel = None if travel is None else tuple(travel)
        self.margin = float(margin)
        self.on_exceed = on_exceed

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #

    def _fmt(self, value):
        """Coordinate as a clean fixed-point string (trailing zeros stripped, no ``-0``)."""
        value = round(float(value), self.precision)
        if value == 0:
            value = 0.0
        text = f"{value:.{self.precision}f}".rstrip("0").rstrip(".")
        return text or "0"

    def _tool_description(self):
        """Header tool string, e.g. ``T1-3.175*19mm Flat End``."""
        if self.tool is None:
            return f"T{self.tool_number}-End Mill"
        return f"T{self.tool_number}-{self.tool.diameter}*{self.tool.height}mm {self.tool.name}"

    @property
    def _coolant_code(self):
        if not self.coolant:
            return None
        return "M8" if str(self.coolant).lower() == "flood" else "M7"

    # ------------------------------------------------------------------ #
    # G-code generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _points(toolpaths):
        """Flatten tool-path objects / polylines into one list of points, dropping
        consecutive coincident points (matching :func:`toolpath_merge`)."""
        points = []
        for tp in toolpaths:
            path = getattr(tp, "path", tp)
            for p in path:
                point = Point(*p)
                if points and points[-1].distance_to_point(point) < 1e-9:
                    continue
                points.append(point)
        return points

    def header(self, points):
        """Header block (``%`` marker, comments, units, tool change, coolant, spindle)."""
        tool_desc = self._tool_description()
        rapid_z = self.rapid_z if self.rapid_z is not None else max(p.z for p in points)
        first = points[0]

        lines = ["%", "; 3-Axis"]
        if self.material:
            lines.append(f"; Material: {self.material}")
        if self.stock_size:
            sx, sy, sz = self.stock_size
            lines.append(f"; Stock Size: {self._fmt(sx)}(X) * {self._fmt(sy)}(Y) * {self._fmt(sz)}(Z) mm")
        lines.append("; Tool List")
        lines.append(f"; {tool_desc}")
        lines.append("; Path List")
        lines.append(f"; [T{self.tool_number}]{self.program}")
        lines.append("G90 G21              ; Absolute positioning, units in millimeters")
        lines.append(f"; {tool_desc}")
        lines.append(f"T{self.tool_number} M6                ; Tool change to Tool {self.tool_number}")
        coolant = self._coolant_code
        if coolant:
            kind = "flood" if coolant == "M8" else "mist"
            lines.append(f"{coolant}                   ; Coolant ON ({kind})")
        lines.append(f"; G0 X{self._fmt(first.x)} Y{self._fmt(first.y)} ; Rapid move to position (informational)")
        lines.append(f"S{self.spindle_speed} M3            ; Spindle ON clockwise at {self.spindle_speed} RPM")
        lines.append(f"G0 Z{self._fmt(rapid_z)}")
        lines.append(f"G1 F{self._fmt(self.feed)}")
        return lines

    def body(self, points):
        """Linear ``G1`` moves with modal coordinates (only changed axes printed)."""
        rapid_z = self.rapid_z if self.rapid_z is not None else max(p.z for p in points)
        prev = {"X": None, "Y": None, "Z": self._fmt(rapid_z)}  # Z is known after the G0 lift
        lines = []
        for point in points:
            words = []
            for axis, value in (("X", point.x), ("Y", point.y), ("Z", point.z)):
                text = self._fmt(value)
                if prev[axis] != text:
                    words.append(axis + text)
                    prev[axis] = text
            if words:
                lines.append("G1 " + " ".join(words))
        return lines

    def footer(self):
        """Shutdown block (coolant off, spindle stop, home, end of program)."""
        lines = []
        if self._coolant_code:
            lines.append("M9        ; Turn off coolant")
        lines.append("M05       ; Stop the spindle")
        lines.append("G28       ; Return all axes to the machine home position")
        lines.append("M02       ; End of program")
        return lines

    # ------------------------------------------------------------------ #
    # Travel-envelope guard
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bbox(points):
        """``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` of the path."""
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        return ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))

    def check_limits(self, *toolpaths):
        """Travel-envelope violations for ``toolpaths`` -- one ``dict`` per axis
        whose path SPAN exceeds the machine reach.

        Returns a list with an entry ``{"axis", "min", "max", "span", "limit",
        "over"}`` for every axis where the bounding-box span is larger than
        :attr:`travel` (shrunk by :attr:`margin`); an empty list means the job
        fits. The SPAN (not the absolute coordinate) is checked because the work
        origin is set on the machine at setup, so what matters at post time is
        whether the part fits the reachable travel -- the firmware soft-endstop
        span, which on the Carvera Air is the practical hard limit (the far end of
        each axis has no switch, only a crash beyond it).
        """
        return self._violations(self._points(toolpaths))

    def _violations(self, points):
        if not points or self.travel is None:
            return []
        out = []
        for axis, (lo, hi), reach in zip("XYZ", self._bbox(points), self.travel):
            if reach is None:
                continue
            limit = float(reach) - self.margin
            span = hi - lo
            if span > limit + 1e-6:
                out.append({"axis": axis, "min": lo, "max": hi, "span": span, "limit": limit, "over": span - limit})
        return out

    def _enforce_limits(self, points):
        violations = self._violations(points)
        if not violations or self.on_exceed == "ignore":
            return
        detail = "; ".join(
            f"{v['axis']} span {v['span']:.2f} mm > {v['limit']:.2f} mm reach "
            f"(over by {v['over']:.2f} mm, {v['axis']} {v['min']:.2f}..{v['max']:.2f})"
            for v in violations
        )
        message = f"tool-path exceeds the Carvera Air travel envelope: {detail}"
        if self.on_exceed == "warn":
            warnings.warn(message, stacklevel=3)
        else:
            raise ValueError(message)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def to_gcode(self, *toolpaths):
        """Full ``.nc`` program text for one or more tool-paths / polylines.

        Each argument may be a ``toolpath_2d_*`` object (anything with a ``.path``)
        or a raw :class:`compas.geometry.Polyline`; they are concatenated in order.
        The path is checked against the machine travel envelope first (see
        :attr:`on_exceed`).
        """
        points = self._points(toolpaths)
        if not points:
            raise ValueError("no tool-path points to post-process.")
        self._enforce_limits(points)
        lines = self.header(points) + self.body(points) + self.footer()
        return "\n".join(lines) + "\n"

    def write(self, filepath, *toolpaths):
        """Write the ``.nc`` program for ``toolpaths`` to ``filepath`` (``.nc`` added
        if missing). Returns the resolved :class:`pathlib.Path`."""
        path = pathlib.Path(filepath)
        if path.suffix.lower() != ".nc":
            path = path.with_suffix(".nc")
        path.write_text(self.to_gcode(*toolpaths), encoding="utf-8")
        return path

    def __repr__(self):
        return (
            f"Postprocessor(tool_number={self.tool_number}, feed={self.feed}, "
            f"spindle_speed={self.spindle_speed}, coolant={self.coolant!r})"
        )
