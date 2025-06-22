import trimesh
import open3d as o3d
import numpy as np

from src.finalize_draw_direction import FinalizeDrawDirection
from src.convex_hull_operations import compute_convex_hull_from_stl
from src.convex_hull_operations import create_mesh, split_convex_hull, display_hull
# from src.split_mesh import extract_unique_vertices_from_faces, closest_distance, face_centroid
from src.split_mesh import split_mesh_faces, display_split_faces, split_mesh_edges, display_split_edges
from src.offset_surface_operations import offset_stl_sdf, mesh_hull_dist, display_offset_surface, split_offset_surface
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
""" GET DISTANCE BETWEEN MESH AND ITS CONVEX HULL """

dist = mesh_hull_dist(mesh_path, convex_hull_path)

# ! Added
""" COMPUTE THE OFFSET SURFACE OF THE MESH USING SDF """

offset_distance = dist * 0.7
# Compute the offset surface of the input mesh
offset_stl_path = offset_stl_sdf(mesh_path, offset_distance)

# Convert offset mesh into trimesh object
offset_mesh = trimesh.load(offset_stl_path)
print(len(offset_mesh.faces))
print(len(offset_mesh.vertices))

# Display the offset surface along with mesh and convex hull
display_offset_surface(offset_stl_path, mesh_path, convex_hull_path)
# offset_mesh.show()

""" FINALIZE THE DRAW DIRECTION """

fd = FinalizeDrawDirection(mesh_path, num_vectors)

candidate_vectors = fd.createCandidateVectors()

draw_direction = fd.computeVisibleAreas(candidate_vectors)
print("Ideal Draw Direction: ", draw_direction)

""" SPLIT CONVEX HULL """

d1_hull_mesh, d2_hull_mesh, d1_aligned_faces, d2_aligned_faces = split_convex_hull(tri_convex_hull, draw_direction)

""" SPLIT MESH BASED ON EDGES """
tri_mesh = trimesh.load(mesh_path)
# edge_list = split_mesh_edges(tri_mesh, tri_convex_hull, d1_aligned_faces, d2_aligned_faces)

""" DISPLAY THE SPLIT EDGES """
# display_split_edges(mesh_path, edge_list)


""" SPLIT MESH FACES """

tri_mesh = trimesh.load(mesh_path)

red_mesh, blue_mesh = split_mesh_faces(tri_mesh, tri_convex_hull, offset_mesh, d1_aligned_faces, d2_aligned_faces)

end_time = time.time()
print(f"Total time taken is {end_time - start_time:.2f} seconds")

""" DISPLAY THE SPLIT MESHES """

display_split_faces(red_mesh, blue_mesh)
