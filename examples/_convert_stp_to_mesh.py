"""Offline STEP -> mesh converter for the custom-toolpath examples.

The fabrication examples (``exmple_model_18..22``) run in the project ``.venv``, which
has ``compas_cnc`` + the viewer stack but NO STEP reader. Only ``compas_occ`` (in the
``occ`` conda env) can tessellate the ``.stp`` solids. So this one-time script -- run
with the ``occ`` env python, NOT the ``.venv`` -- reads each folder's ``.stp`` via
``compas_occ`` and writes a plain faced-mesh ``*_geometry.obj`` beside it. The examples
then just ``Mesh.from_obj(...)`` that file for geometry context (mirroring how
``exmple_model_17`` loads a pre-meshed ``.3dm``).

Tessellation uses ``OCCBrep.to_viewmesh`` (OCC's ``BRepMesh`` incremental mesher), which
RESPECTS the face TRIM boundaries. The naive ``to_meshes`` renders each planar face as
its bounding quad (2 triangles), so L-shapes, notches and faces with holes come out
untrimmed/filled -- exactly the "badly exported" look we are avoiding here.

Usage (from the repo root)::

    C:/Users/petrasv/.conda/envs/occ/python.exe examples/_convert_stp_to_mesh.py
"""

import pathlib

from compas_occ.brep import OCCBrep

# Chord/deflection for the OCC mesher (mm). These parts are planar, so this only bounds
# how finely trimmed edges are sampled; 0.1 mm is smooth without exploding the triangle
# count. Lower it for curved parts if edges look faceted.
DEFLECTION = 0.1

FOLDERS = [
    "custom_toolpath_inner_beams",
    "custom_toolpath_inner_ribs",
    "custom_toolpath_outer_ribs",
    "custom_toolpath_tsections",
    "custom_toolpath_wedges",
]

data_dir = pathlib.Path(__file__).parent.parent / "data"


for folder in FOLDERS:
    # The .stp is usually named after the folder, but some folders carry a differently
    # prefixed export (e.g. inner_beams ships custom_toolpath_inner_ribs.stp), so glob.
    candidates = sorted((data_dir / folder).glob("*.stp"))
    if not candidates:
        print(f"[skip] {folder}: no .stp found")
        continue
    stp = candidates[0]
    brep = OCCBrep.from_step(str(stp))
    # to_viewmesh -> (mesh, edges); the mesh is a single TRIMMED triangulation of the
    # whole Brep. We keep just the mesh; edges are the wireframe (unused here).
    mesh, _edges = brep.to_viewmesh(DEFLECTION)
    out = data_dir / folder / f"{folder}_geometry.obj"
    mesh.to_obj(str(out))
    zs = [mesh.vertex_coordinates(v)[2] for v in mesh.vertices()]
    print(f"[ok] {folder}: {mesh.number_of_faces()} trimmed faces, z-max {max(zs):.3f} -> {out.name}")
