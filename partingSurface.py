import numpy as np
import pyvista as pv
import trimesh

from src.ruledSurface import create_ruled_surface_mesh
from src.finalize_draw_direction import FinalizeDrawDirection
from src.convex_hull_operations import compute_convex_hull_from_stl
from src.offset_surface_operations import mesh_hull_dist

import os


def load_red_mesh(path):
    return pv.read(path)


def compute_centroid(mesh):
    return mesh.center


def compute_max_projection_distance(points, centroid, direction):
    vecs_from_centroid = points - centroid
    projection_lengths = vecs_from_centroid @ direction
    return np.max(projection_lengths)


def create_projection_plane(origin, normal, size=1000):
    return pv.Plane(center=origin, direction=normal, i_size=size, j_size=size)


def project_points_onto_plane(points, plane_origin, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vecs = points - plane_origin
    dists = np.dot(vecs, plane_normal)
    return points - np.outer(dists, plane_normal)


def main():
    # === INPUTS ===
    mesh_path = r"/home/sumukhs-ubuntu/Desktop/silicone_mold/assets/stl/bunny.stl"         # Original STL mesh
    red_mesh_path = r"/home/sumukhs-ubuntu/Desktop/silicone_mold/merged_red.stl"           # Result of red side mesh split
    num_vectors = 100                          # Candidate directions to test

    # === Step 1: Compute convex hull (needed for draw direction logic) ===
    convex_hull, o3dmesh, pcd, convex_hull_path = compute_convex_hull_from_stl(mesh_path)
    print("[✓] Convex hull computed.")

    # === Step 2: Compute draw direction using visible area comparison ===
    fd = FinalizeDrawDirection(mesh_path, num_vectors)
    candidate_vectors = fd.createCandidateVectors()
    draw_direction = fd.computeVisibleAreas(candidate_vectors)
    print(f"[✓] Final draw direction: {draw_direction}")

    # === Step 3: Load red mesh ===
    red_mesh = load_red_mesh(red_mesh_path)
    red_points = red_mesh.points
    red_centroid = compute_centroid(red_mesh)

    # === Step 4: Get max projected length from red centroid ===
    max_proj_dist = compute_max_projection_distance(red_points, red_centroid, draw_direction)
    print(f"[✓] Max projected distance: {max_proj_dist:.3f}")

    # === Step 5: Create projection plane (offset 10% above farthest point) ===
    plane_origin = red_centroid + draw_direction * max_proj_dist * 1.1
    plane = create_projection_plane(plane_origin, draw_direction)
    print(f"[✓] Projection plane created at {plane_origin}")

    # === Step 6: Project red mesh points onto plane ===
    projected_points = project_points_onto_plane(red_points, plane_origin, draw_direction)

    # === Step 7: Create ruled surface between red points and their projections ===
    ruled_surface = create_ruled_surface_mesh(red_points, projected_points)
    ruled_surface.save("ruled_surface.vtk")
    print("[✓] Ruled surface saved as 'ruled_surface.vtk'")

    # === Optional: Visualize everything ===
    plotter = pv.Plotter()
    plotter.add_mesh(red_mesh, color='red', opacity=0.5, label='Red Mesh')
    plotter.add_points(projected_points, color='blue', point_size=6,
                       render_points_as_spheres=True, label='Projected Points')
    plotter.add_mesh(plane, color='green', opacity=0.3, label='Projection Plane')
    plotter.add_mesh(ruled_surface, color='cyan', show_edges=True, label='Ruled Surface')
    plotter.add_legend()
    plotter.show()


if __name__ == "__main__":
    main()
