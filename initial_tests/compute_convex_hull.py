import numpy as np
import open3d as o3d


def compute_convex_hull_open3d(input_ply_path, output_ply_path, voxel_size=5.0):
    # Load the PLY file
    print(f"Loading point cloud from {input_ply_path}")
    pcd = o3d.io.read_point_cloud(input_ply_path)

    print(f"Original point cloud has {len(pcd.points)} points")

    # Downsample the point cloud to reduce computation time
    print(f"Downsampling point cloud with voxel size = {voxel_size}...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled point cloud has {len(pcd_down.points)} points")

    # Compute convex hull using Open3D
    print("Computing convex hull using Open3D...")
    hull_mesh, _ = pcd_down.compute_convex_hull()

    # Optionally simplify and compute normals
    hull_mesh.compute_vertex_normals()
    hull_mesh.compute_triangle_normals()

    # Save the convex hull mesh
    print(f"Saving convex hull mesh to {output_ply_path}")
    o3d.io.write_triangle_mesh(output_ply_path, hull_mesh)

    print("Done!")
    return hull_mesh, pcd, pcd_down


if __name__ == "__main__":
    input_file = r"G:\Chrome downloads\xyzrgb_dragon.ply\xyzrgb_dragon.ply"  # Change to your input file
    output_file = r"G:\Chrome downloads\xyzrgb_dragon.ply\xyzrgb_dragon_hull.ply"  # Output file path

    hull_mesh, pcd, pcd_down = compute_convex_hull_open3d(input_file, output_file, voxel_size=5.0)

    # Visualize the result (optional)
    print("Visualizing result...")
    o3d.visualization.draw_geometries([pcd])
