import trimesh
import pyvista as pv
import numpy as np
import open3d as o3d

from src.finalize_draw_direction import FinalizeDrawDirection
from src.convex_hull_operations import compute_convex_hull_from_stl
from src.convex_hull_operations import create_mesh, split_convex_hull, display_hull
# from src.split_mesh import extract_unique_vertices_from_faces, closest_distance, face_centroid
from src.split_mesh import split_mesh_faces, display_split_faces, split_mesh_edges, display_split_edges
from src.offset_surface_operations import offset_stl_sdf, mesh_hull_dist, display_offset_surface, split_offset_surface
from src.merge_isolated_regions import cleanup_isolated_regions
from src.ruledSurface import ruledSurface
from src.generate_metamold import generate_metamold_red
from src.generate_metamold import generate_metamold_blue
from src.generate_metamold import validate_metamold_files
from src.clean_mesh import STLMeshRepair

# For creating the metamold covers:
from src.create_metamold_cover import translate_sfc_boundary, calculate_mesh_height
from src.create_metamold_cover import create_delaunay_surface, create_ruled_surface

from src.topological_membranes import analyze_mold_extractability

import time
import sys
import os

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

offset_distance = dist
# Compute the offset surface of the input mesh
offset_stl_path = offset_stl_sdf(mesh_path, offset_distance)

# Convert offset mesh into trimesh object
offset_mesh = trimesh.load(offset_stl_path)
print(len(offset_mesh.faces))
print(len(offset_mesh.vertices))

# Display the offset surface along with mesh and convex hull
# display_offset_surface(offset_stl_path, mesh_path, convex_hull_path)
# offset_mesh.show()

""" FINALIZE THE DRAW DIRECTION """

fd = FinalizeDrawDirection(mesh_path, num_vectors)

candidate_vectors = fd.createCandidateVectors()

draw_direction = fd.computeVisibleAreas(candidate_vectors)
print("Ideal Draw Direction: ", draw_direction)

""" SPLIT CONVEX HULL """

d1_hull_mesh, d2_hull_mesh, d1_aligned_faces, d2_aligned_faces = split_convex_hull(tri_convex_hull, draw_direction)

""" SPLIT MESH FACES """

tri_mesh = trimesh.load(mesh_path)

red_mesh, blue_mesh = split_mesh_faces(tri_mesh, tri_convex_hull, offset_mesh, d1_aligned_faces, d2_aligned_faces)

display_split_faces(red_mesh, blue_mesh)

""" MERGE ISOLATED REGIONS """

merged_red, merged_blue = red_mesh, blue_mesh# cleanup_isolated_regions(red_mesh, blue_mesh)

# Save the merged meshes

input_file_name = os.path.splitext(os.path.basename(mesh_path))[0]

# Create the results directory path
results_dir = os.path.join("results", input_file_name)
os.makedirs(results_dir, exist_ok=True)

merged_red_path = os.path.join(results_dir, "merged_red.stl")
merged_blue_path = os.path.join(results_dir, "merged_blue.stl")

merged_red.save(merged_red_path)
merged_blue.save(merged_blue_path)

end_time = time.time()
print(f"Total time taken is {end_time - start_time:.2f} seconds")

""" DISPLAY THE SPLIT MESHES """

display_split_faces(merged_red, merged_blue)

""" CREATE RULED SURFACE AND THE COMBINED PARTING SURFACE """
combined_parting_surface = ruledSurface(
        merged_blue_path, merged_red_path, mesh_path
    )

# ! CREATE THE MOLD BOX:

sfc_centroid = combined_parting_surface.center
height = calculate_mesh_height(merged_red, draw_direction)

# Translate the surface boundary points
translated_points, translated_mesh, boundary_mesh, boundary_points = translate_sfc_boundary(
    os.path.join(results_dir, "combined_parting_surface.stl"),
    draw_direction=draw_direction,
    height=height * 1.10  # Upto 10% above the metamold height
)

# Create Delaunay Surface of the translated points
cover_base = create_delaunay_surface(translated_points)
cover_side = create_ruled_surface(boundary_points, translated_points)

# Translate the points to a plane instead of a fixed distance
# Create Delaunay surface of the plane
# Translate the points inward along the combined_parting_surface only
# Create ruled surfaces on the sides
# Create Delaunay surface to close off the thickness

plotter = pv.Plotter()
plotter.add_mesh(translated_mesh, color="lightblue", point_size=10, render_points_as_spheres=True)
plotter.add_mesh(boundary_mesh, color="red", point_size=10, render_points_as_spheres=True)
plotter.add_mesh(merged_red, color="green", opacity=0.5)
plotter.add_mesh(cover_base, color="yellow", opacity=0.5)
plotter.add_mesh(cover_side, color="orange", opacity=0.5)

plotter.show()

""" GENERATE THE METAMOLD HALVES """

combined_mesh_path = os.path.join(results_dir, "combined_parting_surface.stl")

print("Generating metamold halves...")


# ! Note: Interchange the merged_red_path and merged_blue_path if the mold half is incorrectly generated.
# ! To be fixed later.
# Generate red metamold and save to file
metamold_red_path, plane_normal_red = generate_metamold_red(
    combined_mesh_path, merged_blue_path, draw_direction, combined_parting_surface, results_dir
)

# Generate blue metamold and save to file
metamold_blue_path, plane_normal_blue = generate_metamold_blue(
    combined_mesh_path, merged_red_path, draw_direction, combined_parting_surface, results_dir
)

# Validate that the metamold files were created successfully
if not validate_metamold_files(results_dir):
    print("Error: Metamold generation failed. Cannot proceed with secondary membranes.")
    sys.exit(1)

print("Metamold generation completed successfully!")
print(f"Red metamold saved to: {metamold_red_path}")
print(f"Blue metamold saved to: {metamold_blue_path}")

""" REPAIR THE METAMOLD MESHES """

# RED METAMOLD REPAIR
repair_tool = STLMeshRepair(metamold_red_path)

# Analyze the original mesh
repair_tool.print_analysis()

# Repair the mesh using wrap method (similar to Fusion 360)
repair_tool.repair_mesh(method='wrap')

# Compare results
repair_tool.compare_meshes()
repair_tool.export_mesh(
    os.path.join(results_dir, "repaired_red_metamold.stl"),
    mesh_type='repaired'
)

# Visualize results (choose your preferred method)
# Option 1: Trimesh viewer (simple and fast)
# repair_tool.visualize_mesh(use_trimesh_viewer=True)

# BLUE METAMOLD REPAIR

repair_tool = STLMeshRepair(metamold_blue_path)

# Analyze the original mesh
repair_tool.print_analysis()

# Repair the mesh using wrap method (similar to Fusion 360)
repair_tool.repair_mesh(method='wrap')

# Compare results
repair_tool.compare_meshes()
repair_tool.export_mesh(
    os.path.join(results_dir, "repaired_blue_metamold.stl"),
    mesh_type='repaired'
)

# Visualize results (choose your preferred method)
# Option 1: Trimesh viewer (simple and fast)
# repair_tool.visualize_mesh(use_trimesh_viewer=True)


""" FLAG THE FACES NEEDING SECONDARY MEMBRANES """

# BASE NORMALS

def get_base_plane_normal(stl_path):
    """
    Finds the base face (lowest in Z) and returns its *inward* normal.
    """
    mesh = trimesh.load(stl_path)
    face_normals = mesh.face_normals
    face_centroids = mesh.triangles_center

    # Find face with the lowest Z centroid — i.e., the bottom
    base_face_idx = np.argmin(face_centroids[:, 2])

    # Flip the normal to point *into* the STL
    base_normal = -face_normals[base_face_idx]
    return base_normal

blue_base_normal = get_base_plane_normal(os.path.join(results_dir, "repaired_blue_metamold.stl"))
red_base_normal = get_base_plane_normal(os.path.join(results_dir, "repaired_red_metamold.stl"))


# BLUE METAMOLD
analyze_mold_extractability(
    stl_path=os.path.join(results_dir, "repaired_blue_metamold.stl"),
    reference_normal=plane_normal_blue,
    angle_threshold=100,  # Optional: adjust as needed
    silicone_stretch_limit=5.0,  # Optional: adjust based on your silicone type
    save_with_membranes=True,
    output_path=os.path.join(results_dir, "blue_metamold_with_membranes.stl")
)

# RED METAMOLD
analyze_mold_extractability(
    stl_path=os.path.join(results_dir, "repaired_red_metamold.stl"),
    reference_normal=-1 * plane_normal_red,
    angle_threshold=100,  # Optional: adjust as needed
    silicone_stretch_limit=5.0,  # Optional: adjust based on your silicone type
    save_with_membranes=True,
    output_path=os.path.join(results_dir, "red_metamold_with_membranes.stl")
)

