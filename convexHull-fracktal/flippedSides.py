# import trimesh
# import tetgen
# import pyvista as pv
# import numpy as np

# # Load meshes
# bunny = trimesh.load("bunny_original.obj")
# hull = trimesh.load("bunny_hull_scaled.obj")

# # Ensure both meshes are watertight volumes
# if not bunny.is_watertight:
#     print("Warning: Bunny mesh is not watertight!")
# if not hull.is_watertight:
#     print("Warning: Hull mesh is not watertight!")

# # Invert the bunny mesh to make it a hollow cavity inside the hull
# bunny.invert()

# bunnyV = bunny.vertices
# bunnyF = bunny.faces.astype(np.int32)

# bunnytgen = tetgen.TetGen(bunnyV, bunnyF)
# bunnytgen.tetrahedralize(order=1, mindihedral=10, minratio=1.5)

# # Combine meshes: outer hull + inverted inner bunny
# combined = trimesh.util.concatenate([hull, bunny])

# # Prepare vertices and faces for TetGen
# vertices = combined.vertices
# faces = combined.faces.astype(np.int32)

# # Run TetGen tetrahedralization
# tgen = tetgen.TetGen(vertices, faces)
# tgen.tetrahedralize(order=1, mindihedral=10, minratio=1.5)

# tgen = tetgen.TetGen(vertices, faces)
# tgen.tetrahedralize(order=1, mindihedral=10, minratio=1.5)

# # Get tetrahedral mesh
# tetra_mesh = tgen.mesh

# bunny_mesh = bunnytgen.mesh



# print(f"Number of tetrahedra: {tetra_mesh.n_cells}")

# # Visualization with PyVista
# hull_pv = pv.PolyData(hull.vertices, np.hstack([np.full((hull.faces.shape[0], 1), 3), hull.faces]))
# bunny_pv = pv.PolyData(bunny.vertices, np.hstack([np.full((bunny.faces.shape[0], 1), 3), bunny.faces]))

# clipped_tet = tetra_mesh.clip(normal='x', origin=(0, 0, 0))
# clipped_hull = hull_pv.clip(normal='x', origin=(0, 0, 0))
# clipped_bunny = bunny_mesh.clip(normal='x', origin=(0, 0, 0))

# plotter = pv.Plotter()
# plotter.add_mesh(clipped_bunny, color='gray', opacity=0.2, show_edges=True)
# plotter.add_mesh(clipped_hull, color='white', opacity=0.5, show_edges=True)
# plotter.add_mesh(clipped_tet, color='red', opacity=1, show_edges=True)

# plotter.add_axes()
# plotter.show_bounds(grid='front', location='outer', all_edges=True)
# plotter.show()


# import pyvista as pv

# # Load the mesh
# hull_mesh = pv.read('bunny_hull_scaled.obj')
# bunny_mesh = pv.read('bunny_original.obj')

# # Create a plotter and show the mesh
# plotter = pv.Plotter()
# plotter.add_mesh(hull_mesh, color='lightblue', show_edges=True,opacity = 0.3)
# plotter.add_mesh(bunny_mesh, color='red', show_edges=True,opacity = 0.7)
# plotter.add_mesh(hull_mesh, color='lightblue', show_edges=True,opacity = 0.3)
# plotter.show()


import trimesh
import tetgen
import pyvista as pv
import numpy as np

# Load bunny mesh (make sure it is watertight)
bunny = trimesh.load("bunny_original.obj")
if not bunny.is_watertight:
    print("Warning: Bunny mesh is not watertight — results may be incorrect")

# Extract vertices and faces for tetgen
vertices = bunny.vertices
faces = bunny.faces.astype(np.int32)

# Create TetGen object and tetrahedralize
tgen = tetgen.TetGen(vertices, faces)
tgen.tetrahedralize(order=1, mindihedral=10, minratio=1.5)

# Get the tetrahedral mesh (PyVista object)
tetra_mesh = tgen.mesh

print(f"Number of tetrahedra: {tetra_mesh.n_cells}")

# Prepare the bunny surface mesh for visualization
bunny_pv = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3), faces]))
clipped_bunny = tetra_mesh.clip(normal='x', origin=(0, 0, 0))
# Visualize the surface and tetrahedral mesh
plotter = pv.Plotter()
#plotter.add_mesh(bunny_pv, color='white', opacity=0.5, show_edges=True)
plotter.add_mesh(clipped_bunny, color='red', opacity=1, show_edges=True)

plotter.add_axes()
plotter.show_bounds(grid='front', location='outer', all_edges=True)
plotter.show()


