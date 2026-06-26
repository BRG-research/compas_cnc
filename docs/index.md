# COMPAS CNC

`compas_cnc` generates 2D CNC milling tool-paths for subtractive fabrication and
post-processes them to G-code, built on the COMPAS framework.

It provides a small set of tool-path generators -- `compas_cnc.toolpath_2d_drill`
(helical drilling), `compas_cnc.toolpath_2d_ramp` (ramp / slot plunging with
dogbone corner relief), `compas_cnc.toolpath_2d_surfacing` (zig-zag face
clearing) and `compas_cnc.toolpath_2d_hatch` (raster pocket fill with islands and
layered roughing) -- that all return one continuous tool-centre path starting and
ending at a safe height.

`compas_cnc.toolpath_merge` concatenates several tool-paths into one job, and
`compas_cnc.Postprocessor` writes the result to **Carvera Air** G-code. Cutting
geometry is offset and hatched through a compiled Clipper2 wrapper, and
`compas_cnc.tools.Tool` draws the cutter for previewing a path in the COMPAS
Viewer.
