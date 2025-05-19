import pyvista as pv
import tetgen
import numpy as np

def tetrahedralize_stl(file_path):
    # Step 1: Load surface mesh from STL
    surface = pv.read(file_path)

    if not surface.is_manifold:
        print("Warning: STL mesh is not manifold. Tetrahedralization may fail.")

    # Step 2: Convert to TetGen-compatible format
    tet = tetgen.TetGen(surface)

    # Step 3: Tetrahedralize (set switches for quality if needed)
    tet_mesh = tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5)

    return tet_mesh

def slice_tetmesh(tet_mesh, origin=None, normal=(0, 0, 1)):
    # Default slice through center
    if origin is None:
        origin = tet_mesh.center

    # Create a slicing plane
    plane = tet_mesh.slice(normal=normal, origin=origin)
    return plane

def visualize_tet_and_slice(tet_mesh, slice_surface):
    p = pv.Plotter()
    p.add_mesh(tet_mesh, show_edges=True, opacity=0.2, color='lightblue')
    p.add_mesh(slice_surface, color='red', line_width=2)
    p.add_axes()
    p.show()

# === Example Usage ===
stl_file = "your_file.stl"  # Replace with your STL path
tet_mesh = tetrahedralize_stl(stl_file)
slice_surface = slice_tetmesh(tet_mesh, normal=(0, 0, 1))
visualize_tet_and_slice(tet_mesh, slice_surface)
