import numpy as np
import open3d as o3d
import os
import trimesh
import pyvista as pv


def create_mesh(vertices, faces):
    """
    Create Pyvista Polydata from vertices and faces
    :param vertices: trimesh.Trimesh.vertices
    :param faces: trimesh.Trimesh.faces
    :return: pyvista.PolyData
    """
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(int)
    return pv.PolyData(vertices, faces)


def compute_convex_hull_from_stl(input_stl_path, voxel_size=None,
                                 scale_factor=1.05, tessellation_level=3):
    # Load STL mesh
    print(f"Loading mesh from {input_stl_path}")
    mesh = o3d.io.read_triangle_mesh(input_stl_path)

    if not mesh.has_vertices():
        raise ValueError("The loaded mesh has no vertices.")

    print(f"Mesh has {np.asarray(mesh.vertices).shape[0]} vertices and {np.asarray(mesh.triangles).shape[0]} triangles")

    # Convert mesh vertices to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    # Optional downsampling
    if voxel_size is not None:
        print(f"Downsampling point cloud with voxel size = {voxel_size}...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled point cloud has {len(pcd.points)} points")

    # Compute convex hull
    print("Computing convex hull...")
    hull_mesh, _ = pcd.compute_convex_hull()
    hull_mesh.compute_vertex_normals()

    # Tessellate the convex hull by uniformly subdividing it
    print(f"Original hull has {len(hull_mesh.triangles)} triangles")

    if tessellation_level > 0:
        print(f"Tessellating convex hull with level {tessellation_level}...")

        # Method 1: Use Open3D's subdivision function
        tessellated_hull = hull_mesh.subdivide_midpoint(number_of_iterations=tessellation_level)
        print(f"After tessellation: {len(tessellated_hull.triangles)} triangles")
        hull_mesh = tessellated_hull

    hull_mesh.compute_vertex_normals()

    # Scale the hull mesh around its centroid
    print(f"Scaling convex hull by {scale_factor * 100 - 100:.0f}%...")
    vertices = np.asarray(hull_mesh.vertices)
    centroid = vertices.mean(axis=0)
    scaled_vertices = (vertices - centroid) * scale_factor + centroid
    hull_mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)

    base_name = os.path.splitext(os.path.basename(input_stl_path))[0]
    output_dir = os.path.join("results", base_name)
    os.makedirs(output_dir, exist_ok=True)
    output_stl_path = os.path.join(output_dir, base_name + "_hull.stl")

    # Save the scaled convex hull
    print(f"Saving convex hull mesh to {output_stl_path}")
    o3d.io.write_triangle_mesh(output_stl_path, hull_mesh)

    print("Done!")
    return hull_mesh, mesh, pcd, output_stl_path


def split_convex_hull(hull_mesh: trimesh.Trimesh, draw_direction: np.ndarray):
    convex_hull_vertices = hull_mesh.vertices
    convex_hull_faces = hull_mesh.faces
    convex_hull_pv = create_mesh(convex_hull_vertices, convex_hull_faces)

    # Calculate face normals for the convex hull
    convex_hull_pv.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    convex_hull_normals = convex_hull_pv.cell_normals

    d1 = draw_direction
    d2 = -d1

    # Normalize direction vectors
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    d1_aligned_faces = []
    d2_aligned_faces = []

    for i, face in enumerate(convex_hull_faces):
        # Get the face normal
        normal = convex_hull_normals[i]

        # Calculate dot products with both directions
        # We use absolute values since we care about alignment regardless of direction
        dot_d1 = np.dot(normal, d1_norm)
        dot_d2 = np.dot(normal, d2_norm)

        # print(f"Face {i}: Normal {normal}, Dot with d1: {dot_d1}, Dot with d2: {dot_d2}")

        # Assign to direction based on which has the larger dot product with the face normal
        if dot_d1 > dot_d2:
            d1_aligned_faces.append(face)
        else:
            d2_aligned_faces.append(face)

    # Convert the face lists to numpy arrays
    d1_aligned_faces = np.array(d1_aligned_faces) if d1_aligned_faces else np.empty((0, 3), dtype=int)
    d2_aligned_faces = np.array(d2_aligned_faces) if d2_aligned_faces else np.empty((0, 3), dtype=int)

    # Create separate meshes for visualization
    d1_hull_mesh = create_mesh(convex_hull_vertices, d1_aligned_faces) if len(d1_aligned_faces) > 0 else None
    d2_hull_mesh = create_mesh(convex_hull_vertices, d2_aligned_faces) if len(d2_aligned_faces) > 0 else None

    return d1_hull_mesh, d2_hull_mesh, d1_aligned_faces, d2_aligned_faces


def display_hull(d1_hull_mesh: trimesh.Trimesh, d2_hull_mesh: trimesh.Trimesh):
    """
    Display the convex hull mesh using trimesh
    :param hull_mesh: Convex hull mesh
    """
    # Create a plotter for visualization
    hull_plotter = pv.Plotter()

    # Add the meshes and arrows to the plot
    if d1_hull_mesh is not None:
        hull_plotter.add_mesh(d1_hull_mesh, color='red', opacity=1, show_edges=True, line_width=2,
                              label='Faces aligned with d1')
    if d2_hull_mesh is not None:
        hull_plotter.add_mesh(d2_hull_mesh, color='blue', opacity=1, show_edges=True, line_width=2,
                              label='Faces aligned with d2')

    # Add original mesh for reference
    # hull_plotter.add_mesh(mesh, color='lightgray', opacity=0.3, label='Original Mesh')

    # Add legend and display the plot
    hull_plotter.add_legend()
    hull_plotter.add_title("Convex Hull Split by Direction Vectors")
    hull_plotter.show()
