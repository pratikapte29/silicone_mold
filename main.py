import trimesh
import open3d as o3d
import numpy as np

from src.finalize_draw_direction import FinalizeDrawDirection
from src.convex_hull_operations import compute_convex_hull_from_stl
from src.convex_hull_operations import create_mesh, split_convex_hull, display_hull
# from src.split_mesh import extract_unique_vertices_from_faces, closest_distance, face_centroid
from src.split_mesh import split_mesh_faces, display_split_faces, offset_stl, pv_to_trimesh
import time
import sys

"""
STEPS:
0. Load the mesh from a terminal command
1. Compute convex hull of the input mesh 
2. Finalize the draw direction from the candidate directions
3. Split Convex Hull into two parts based on alignment with the draw direction
4. Split mesh faces based on proximity to the convex hull sides
"""
# ! later, try to overload the display function to display everything, that will be much cleaner.

if len(sys.argv) < 3:
    print("Usage: python main.py <mesh_path> <num_vectors>")
    sys.exit(1)

# Set the mesh path and number of candidate vectors
mesh_path = sys.argv[1]
try:
    num_vectors = int(sys.argv[2])
except ValueError:
    print("Error: <num_vectors> must be an integer.")
    sys.exit(1)

start_time = time.time()

""" COMPUTE CONVEX HULL OF THE MESH """

convex_hull, o3dmesh, pcd, convex_hull_path = compute_convex_hull_from_stl(mesh_path)
tri_convex_hull = trimesh.load(convex_hull_path)

# ! Added
""" COMPUTE THE OFFSET SURFACE OF THE MESH """

# Hull bounds will be used for offset surface distance calculation
hull_bounds = tri_convex_hull.bounds
# offset_distance = 0.25 * np.linalg.norm(hull_bounds.extents)
offset_distance = 0.05 * np.max(hull_bounds[1] - hull_bounds[0])
# Compute the offset surface of the input mesh
pv_mesh, offset_mesh = offset_stl(mesh_path, offset_distance)

# Convert offset mesh into trimesh object
offset_mesh = pv_to_trimesh(offset_mesh)
print(len(offset_mesh.faces))
print(len(offset_mesh.vertices))
# offset_mesh.show()

""" FINALIZE THE DRAW DIRECTION """

fd = FinalizeDrawDirection(mesh_path, num_vectors)

candidate_vectors = fd.createCandidateVectors()

draw_direction = fd.computeVisibleAreas(candidate_vectors)
print("Ideal Draw Direction: ", draw_direction)

""" SPLIT CONVEX HULL """

d1_hull_mesh, d2_hull_mesh, d1_aligned_faces, d2_aligned_faces = split_convex_hull(tri_convex_hull, draw_direction)

# ! Added
# """ SPLIT OFFSET SURFACE """
#
# d1_off_mesh, d2_off_mesh, d1_off_faces, d2_off_faces = split_convex_hull(offset_mesh, draw_direction)
# display_hull(d1_off_mesh, d2_off_mesh)

# ! Added
# """ SPLIT MESH FACES BASED ON OFFSET SURFACE """
#
# tri_mesh = trimesh.load(mesh_path)
#
# red_mesh, blue_mesh = split_mesh_faces(tri_mesh, offset_mesh, d1_off_faces, d2_off_faces)

""" SPLIT MESH FACES """

tri_mesh = trimesh.load(mesh_path)

red_mesh, blue_mesh = split_mesh_faces(tri_mesh, tri_convex_hull, d1_aligned_faces, d2_aligned_faces)

end_time = time.time()
print(f"Total time taken is {end_time - start_time:.2f} seconds")

""" DISPLAY THE SPLIT MESHES """

display_split_faces(red_mesh, blue_mesh)
