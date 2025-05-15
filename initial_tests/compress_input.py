import open3d as o3d
import numpy as np

def point_cloud_to_stl(input_ply_path, output_stl_path, poisson_depth=9):
    # Step 1: Load point cloud
    print(f"Loading point cloud from: {input_ply_path}")
    pcd = o3d.io.read_point_cloud(input_ply_path)

    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty!")

    # Step 2: Estimate normals (required for Poisson)
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Step 3: Surface reconstruction using Poisson
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    # Optional: Crop low-density vertices to clean up the mesh
    print("Removing low-density vertices...")
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 5)
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

    # Step 4: Save mesh to STL
    print(f"Saving mesh to: {output_stl_path}")
    o3d.io.write_triangle_mesh(output_stl_path, mesh)
    print("Conversion complete.")

    return mesh


if __name__ == "__main__":
    input_ply = r"G:\Chrome downloads\xyzrgb_dragon.ply\xyzrgb_dragon.ply"
    output_stl = r"G:\Chrome downloads\xyzrgb_dragon.ply\xyzrgb_dragon_from_pointcloud.stl"

    mesh = point_cloud_to_stl(input_ply, output_stl, poisson_depth=10)

    # Visualize result
    print("Visualizing result...")
    o3d.visualization.draw_geometries([mesh])
