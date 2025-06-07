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
# hull_mesh = pv.read(r'C:\Users\Sumukh\Downloads\hull_fixed.obj')
# bunny_mesh = pv.read(r"convexHull-fracktal\bunny_original.obj")

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

# Load bunny mesh
bunny = trimesh.load(r"C:\Users\Sumukh\Desktop\silicone_mold\convexHull-fracktal\bunny_original.obj")
# Load hull mesh
hull = trimesh.load(r"C:\Users\Sumukh\Desktop\silicone_mold\convexHull-fracktal\bunny_hull_scaled.obj")
#hull = trimesh.load(r"convexHull-fracktal\hull_fixed.obj")
if not bunny.is_watertight:
    print("Warning: Bunny mesh is not watertight — results may be incorrect")


# Extract vertices and faces
bunny_vertices = bunny.vertices
bunny_faces = bunny.faces.astype(np.int32)

# extract vertices and faces from the hull mesh
hull_vertices = hull.vertices
hull_faces = hull.faces.astype(np.int32)
# Invert the bunny mesh to create a hollow cavity inside the hull
bunny.invert()
# Ensure the bunny mesh is watertight after inversion
if not bunny.is_watertight:
    print("Warning: Inverted bunny mesh is still not watertight — results may be incorrect")    

# Ensure the hull mesh is watertight        
if not hull.is_watertight:
    print("Warning: Hull mesh is not watertight — results may be incorrect")



combined_mesh = trimesh.util.concatenate([hull, bunny])

# Extract vertices and faces from the combined mesh
combined_vertices = combined_mesh.vertices
combined_faces = combined_mesh.faces.astype(np.int32)
# Ensure the combined mesh is watertight
if not combined_mesh.is_watertight:
    print("Warning: Combined mesh is not watertight — results may be incorrect")

# create a TetGen object with the combined mesh
combined_tgen = tetgen.TetGen(combined_vertices, combined_faces)
# Tetrahedralize the combined mesh
combined_tgen.tetrahedralize('pq1.2aA')
# Get the tetrahedral mesh as an UnstructuredGrid
combined_ugrid = combined_tgen.grid
# Get tetrahedra and node info (optional print/debug)
combined_tets = combined_ugrid.cells_dict[10]  # VTK_TETRA
combined_nodes = combined_ugrid.points
print(f"Combined mesh generated {len(combined_tets)} tetrahedra with {len(combined_nodes)} nodes.")




# Create TetGen object and tetrahedralize
bunny_tgen = tetgen.TetGen(bunny_vertices, bunny_faces)
bunny_tgen.tetrahedralize('pq1.2aA')

# Create TetGen object for the hull mesh
hull_tgen = tetgen.TetGen(hull_vertices, hull_faces)    
hull_tgen.tetrahedralize('pq1.2aA')

# Get the tetrahedral mesh as an UnstructuredGrid
bunny_ugrid = bunny_tgen.grid

# Get tetrahedra and node info (optional print/debug)
bunny_tets = bunny_ugrid.cells_dict[10]  # VTK_TETRA
bunny_nodes = bunny_ugrid.points
print(f"Generated {len(bunny_tets)} tetrahedra with {len(bunny_nodes)} nodes.")

# Get the hull tetrahedral mesh as an UnstructuredGrid
hull_ugrid = hull_tgen.grid
# Get tetrahedra and node info for the hull mesh
hull_tets = hull_ugrid.cells_dict[10]  # VTK_TETRA
hull_nodes = hull_ugrid.points
print(f"Hull mesh has {len(hull_tets)} tetrahedra with {len(hull_nodes)} nodes.")



# Compute centroid for clipping
centroid = np.mean(bunny_nodes, axis=0)

# Clip the tetrahedral mesh
bunny_clipped_mesh = bunny_ugrid.clip(normal='x', origin=centroid)
# Clip the hull mesh
hull_clipped_mesh = hull_ugrid.clip(normal='x', origin=centroid)
# Clip the combined mesh
combined_clipped_mesh = combined_ugrid.clip(normal='x', origin=centroid)


# Visualization
plotter = pv.Plotter()
#plotter.add_mesh(bunny_clipped_mesh, color='red', opacity=1, show_edges=True)
plotter.add_mesh(combined_clipped_mesh, color='orange', opacity=1, show_edges=True)
#plotter.add_mesh(hull_clipped_mesh, color='lightblue', opacity=1, show_edges=True)
plotter.add_axes()
plotter.show_bounds(grid='front', location='outer', all_edges=True)
plotter.show()
