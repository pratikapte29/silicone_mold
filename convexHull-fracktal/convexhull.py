import trimesh
import tetgen
import pyvista as pv
import numpy as np

# Load meshes
bunny = trimesh.load(r"C:\Users\Sumukh\Desktop\silicone_mold\convexHull-fracktal\bunny_original.obj")
hull = trimesh.load(r"C:\Users\Sumukh\Desktop\silicone_mold\convexHull-fracktal\hull_fixed.obj")
print("Loaded meshes successfully.")
# Check watertightness
if not bunny.is_watertight:
    print("Warning: Bunny mesh is not watertight — results may be incorrect")
if not hull.is_watertight:
    print("Warning: Hull mesh is not watertight — results may be incorrect")


print(f"Bunny vertices: {len(bunny.vertices)}, faces: {len(bunny.faces)}")
print(f"Hull vertices: {len(hull.vertices)}, faces: {len(hull.faces)}")

# Invert bunny for inner cavity
bunny.invert()
print("Bunny mesh inverted to create inner cavity.")

if not bunny.is_watertight:
    print("Warning: Inverted bunny mesh is still not watertight")

# Concatenate the outer hull and inner (inverted) bunny into one surface mesh
combined_mesh = trimesh.util.concatenate([hull, bunny])

print(f"Combined mesh has {len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces.")

# Extract vertices and faces
vertices = combined_mesh.vertices
faces = combined_mesh.faces.astype(np.int32)
print("tetrahedralizing the combined mesh...")
# Tetrahedralize the combined surface mesh
tgen = tetgen.TetGen(vertices, faces)
print("Winding consistent:", combined_mesh.is_winding_consistent)
print("Is volume:", combined_mesh.is_volume)
print("Is watertight:", combined_mesh.is_watertight)
#print("Validation report:", combined_mesh.validate())




combined_mesh.update_faces(combined_mesh.unique_faces())
combined_mesh.update_faces(combined_mesh.nondegenerate_faces())

combined_mesh.remove_unreferenced_vertices()
combined_mesh.merge_vertices()
combined_mesh.rezero()
combined_mesh.process(validate=True)


tgen.tetrahedralize()

print("Tetrahedralization complete.")
# Extract tetrahedral mesh (as PyVista UnstructuredGrid)
ugrid = tgen.grid
tets = ugrid.cells_dict[10]
nodes = ugrid.points
print(f"Generated {len(tets)} tetrahedra with {len(nodes)} nodes.")

print(f"Combined mesh generated {len(tets)} tetrahedra with {len(nodes)} nodes.")

# Clip for visualization
centroid = np.mean(nodes, axis=0)
clipped_mesh = ugrid.clip(normal='x', origin=centroid)

# Visualization
print("Visualizing the clipped mesh...")
plotter = pv.Plotter()
plotter.add_mesh(clipped_mesh, color='orange', opacity=1, show_edges=True)
#plotter.add_mesh(bunny, color='orange', opacity=1, show_edges=True)
plotter.add_axes()
plotter.show_bounds(grid='front', location='outer', all_edges=True)
plotter.show()


