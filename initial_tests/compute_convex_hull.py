import numpy as np
import open3d as o3d


def compute_convex_hull_from_stl(input_stl_path, output_stl_path, voxel_size=None, scale_factor=1.1):
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

    # Scale the hull mesh around its centroid
    print(f"Scaling convex hull by {scale_factor * 100 - 100:.0f}%...")
    vertices = np.asarray(hull_mesh.vertices)
    centroid = vertices.mean(axis=0)
    scaled_vertices = (vertices - centroid) * scale_factor + centroid
    hull_mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)

    # Save the scaled convex hull
    print(f"Saving convex hull mesh to {output_stl_path}")
    o3d.io.write_triangle_mesh(output_stl_path, hull_mesh)

    print("Done!")
    return hull_mesh, mesh, pcd


if __name__ == "__main__":
    input_file = r"G:\Chrome downloads\xyzrgb-dragon-by-renato-tarabella\xyzrgb_dragon_90.stl"
    output_file = r"G:\Chrome downloads\xyzrgb-dragon-by-renato-tarabella\xyzrgb_dragon_90_hull_scaled.stl"

    hull_mesh, original_mesh, pcd = compute_convex_hull_from_stl(
        input_file,
        output_file,
        voxel_size=None,
        scale_factor=1.1  # 10% outward scale
    )

    # Visualize original and scaled hull
    print("Visualizing result...")
    hull_mesh.paint_uniform_color([1, 0, 0])  # Red hull
    original_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray original

    o3d.visualization.draw_geometries([original_mesh, hull_mesh], mesh_show_back_face=True)
