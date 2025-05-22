import numpy as np
import trimesh
from scipy.spatial.distance import cdist
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_stl_mesh(stl_file_path):
    """
    Load an STL file and return a trimesh object.

    Parameters:
    - stl_file_path: Path to the STL file

    Returns:
    - mesh: Trimesh object
    """
    mesh = trimesh.load(stl_file_path)
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight. Results may be unreliable.")
    return mesh


def compute_signed_distance_field(mesh, grid_resolution=100, padding_factor=0.2, chunk_size=50000):
    """
    Compute the signed distance field for a mesh using chunked processing to avoid memory issues.

    Parameters:
    - mesh: Trimesh object
    - grid_resolution: Number of grid points along each axis
    - padding_factor: Extra space around the mesh as a fraction of mesh size
    - chunk_size: Number of points to process at once (reduces memory usage)

    Returns:
    - sdf_grid: 3D array containing signed distance values
    - grid_coords: Tuple of (x_coords, y_coords, z_coords) for the grid
    - grid_spacing: The spacing between grid points
    """
    # Get mesh bounds
    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    bbox_size = bbox_max - bbox_min

    # Add padding around the mesh
    padding = bbox_size * padding_factor
    bbox_min_padded = bbox_min - padding
    bbox_max_padded = bbox_max + padding

    # Create grid coordinates
    x_coords = np.linspace(bbox_min_padded[0], bbox_max_padded[0], grid_resolution)
    y_coords = np.linspace(bbox_min_padded[1], bbox_max_padded[1], grid_resolution)
    z_coords = np.linspace(bbox_min_padded[2], bbox_max_padded[2], grid_resolution)

    grid_spacing = (bbox_max_padded - bbox_min_padded) / (grid_resolution - 1)

    # Create meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    total_points = len(grid_points)
    print(f"Computing SDF for {total_points} grid points in chunks of {chunk_size}...")

    # Process in chunks to avoid memory issues
    signed_distances = np.zeros(total_points, dtype=np.float32)

    for i in range(0, total_points, chunk_size):
        end_idx = min(i + chunk_size, total_points)
        chunk_points = grid_points[i:end_idx]

        print(f"Processing chunk {i // chunk_size + 1}/{(total_points + chunk_size - 1) // chunk_size} "
              f"({len(chunk_points)} points)")

        try:
            # Compute signed distances for this chunk
            chunk_distances = mesh.nearest.signed_distance(chunk_points)
            signed_distances[i:end_idx] = chunk_distances
        except Exception as e:
            print(f"Error in chunk processing: {e}")
            print("Falling back to manual computation for this chunk...")
            # Fallback to manual computation
            chunk_distances = compute_sdf_chunk_manual(mesh, chunk_points)
            signed_distances[i:end_idx] = chunk_distances

    # Reshape to grid
    sdf_grid = signed_distances.reshape(X.shape)

    return sdf_grid, (x_coords, y_coords, z_coords), grid_spacing


def compute_sdf_chunk_manual(mesh, points):
    """
    Manual SDF computation for a chunk of points as fallback.
    """
    print("Using manual SDF computation (slower but more memory efficient)...")

    # Get mesh data
    vertices = mesh.vertices
    faces = mesh.faces

    # Initialize distances
    min_distances = np.full(len(points), np.inf, dtype=np.float32)

    # Compute distance to each face
    for face_idx, face in enumerate(faces):
        if face_idx % 1000 == 0:
            print(f"Processing face {face_idx}/{len(faces)}")

        tri_vertices = vertices[face]
        distances = point_to_triangle_distance_vectorized(points, tri_vertices)
        min_distances = np.minimum(min_distances, distances)

    # Determine signs using containment test
    try:
        inside_mask = mesh.contains(points)
        signed_distances = np.where(inside_mask, -min_distances, min_distances)
    except:
        # If containment test fails, use ray casting fallback
        print("Containment test failed, using simplified sign determination...")
        signed_distances = min_distances  # All positive (outside)

    return signed_distances.astype(np.float32)


def point_to_triangle_distance_vectorized(points, triangle_vertices):
    """
    Optimized vectorized distance computation to triangle.
    """
    A, B, C = triangle_vertices.astype(np.float32)
    points = points.astype(np.float32)

    # Vector from A to other vertices and points
    AB = B - A
    AC = C - A
    AP = points - A[np.newaxis, :]

    # Dot products
    d1 = np.sum(AB * AP, axis=1)
    d2 = np.sum(AC * AP, axis=1)

    # Check vertex region A
    mask_A = (d1 <= 0) & (d2 <= 0)

    # Check vertex region B
    BP = points - B[np.newaxis, :]
    d3 = np.sum(AB * BP, axis=1)
    d4 = np.sum(AC * BP, axis=1)
    mask_B = (d3 >= 0) & (d4 <= d3) & ~mask_A

    # Check vertex region C
    CP = points - C[np.newaxis, :]
    d5 = np.sum(AB * CP, axis=1)
    d6 = np.sum(AC * CP, axis=1)
    mask_C = (d6 >= 0) & (d5 <= d6) & ~mask_A & ~mask_B

    # Initialize closest points
    closest_points = np.zeros_like(points)
    closest_points[mask_A] = A
    closest_points[mask_B] = B
    closest_points[mask_C] = C

    # Edge AB
    vc = d1 * d4 - d3 * d2
    mask_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~mask_A & ~mask_B & ~mask_C
    if np.any(mask_AB):
        v = d1[mask_AB] / (d1[mask_AB] - d3[mask_AB])
        closest_points[mask_AB] = A + v[:, np.newaxis] * AB

    # Edge AC
    vb = d5 * d2 - d1 * d6
    mask_AC = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~mask_A & ~mask_B & ~mask_C & ~mask_AB
    if np.any(mask_AC):
        w = d2[mask_AC] / (d2[mask_AC] - d6[mask_AC])
        closest_points[mask_AC] = A + w[:, np.newaxis] * AC

    # Edge BC
    va = d3 * d6 - d5 * d4
    mask_BC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~mask_A & ~mask_B & ~mask_C & ~mask_AB & ~mask_AC
    if np.any(mask_BC):
        w = (d4[mask_BC] - d3[mask_BC]) / ((d4[mask_BC] - d3[mask_BC]) + (d5[mask_BC] - d6[mask_BC]))
        closest_points[mask_BC] = B + w[:, np.newaxis] * (C - B)

    # Inside triangle
    mask_inside = ~(mask_A | mask_B | mask_C | mask_AB | mask_AC | mask_BC)
    if np.any(mask_inside):
        denom = va[mask_inside] + vb[mask_inside] + vc[mask_inside]
        # Avoid division by zero
        valid_denom = np.abs(denom) > 1e-10
        if np.any(valid_denom):
            denom_safe = denom[valid_denom]
            v = vb[mask_inside][valid_denom] / denom_safe
            w = vc[mask_inside][valid_denom] / denom_safe
            inside_valid = np.where(mask_inside)[0][valid_denom]
            closest_points[inside_valid] = A + v[:, np.newaxis] * AB + w[:, np.newaxis] * AC

    # Compute distances
    distances = np.linalg.norm(points - closest_points, axis=1)
    return distances.astype(np.float32)


def manual_signed_distance_field(mesh, grid_resolution=100, padding_factor=0.2):
    """
    Alternative manual implementation of signed distance field computation.
    This is slower but more educational and customizable.
    """
    # Get mesh bounds
    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    bbox_size = bbox_max - bbox_min

    # Add padding
    padding = bbox_size * padding_factor
    bbox_min_padded = bbox_min - padding
    bbox_max_padded = bbox_max + padding

    # Create grid
    x_coords = np.linspace(bbox_min_padded[0], bbox_max_padded[0], grid_resolution)
    y_coords = np.linspace(bbox_min_padded[1], bbox_max_padded[1], grid_resolution)
    z_coords = np.linspace(bbox_min_padded[2], bbox_max_padded[2], grid_resolution)

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    print(f"Computing manual SDF for {len(grid_points)} grid points...")

    # Get mesh vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Compute distances to all triangles
    min_distances = np.full(len(grid_points), np.inf)

    for i, face in enumerate(faces):
        if i % 1000 == 0:
            print(f"Processing face {i}/{len(faces)}")

        # Get triangle vertices
        tri_verts = vertices[face]

        # Compute distance from each grid point to this triangle
        distances = point_to_triangle_distance(grid_points, tri_verts)

        # Keep minimum distance
        min_distances = np.minimum(min_distances, distances)

    # Determine sign using ray casting or mesh.contains
    inside_mask = mesh.contains(grid_points)
    signed_distances = np.where(inside_mask, -min_distances, min_distances)

    sdf_grid = signed_distances.reshape(X.shape)
    grid_spacing = (bbox_max_padded - bbox_min_padded) / (grid_resolution - 1)

    return sdf_grid, (x_coords, y_coords, z_coords), grid_spacing


def point_to_triangle_distance(points, triangle_vertices):
    """
    Compute the distance from points to a triangle.

    Parameters:
    - points: Nx3 array of query points
    - triangle_vertices: 3x3 array of triangle vertices

    Returns:
    - distances: N array of distances
    """
    A, B, C = triangle_vertices

    # Vectorized computation of closest point on triangle
    AB = B - A
    AC = C - A
    AP = points - A

    d1 = np.sum(AB * AP, axis=1)
    d2 = np.sum(AC * AP, axis=1)

    # Check if P is in vertex region outside A
    mask1 = (d1 <= 0) & (d2 <= 0)
    closest_points = np.tile(A, (len(points), 1))

    # Check if P is in vertex region outside B
    BP = points - B
    d3 = np.sum(AB * BP, axis=1)
    d4 = np.sum(AC * BP, axis=1)
    mask2 = (d3 >= 0) & (d4 <= d3) & ~mask1
    closest_points[mask2] = B

    # Check if P is in vertex region outside C
    CP = points - C
    d5 = np.sum(AB * CP, axis=1)
    d6 = np.sum(AC * CP, axis=1)
    mask3 = (d6 >= 0) & (d5 <= d6) & ~mask1 & ~mask2
    closest_points[mask3] = C

    # Check if P is in edge region of AB
    vc = d1 * d4 - d3 * d2
    mask4 = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~mask1 & ~mask2 & ~mask3
    v = d1[mask4] / (d1[mask4] - d3[mask4])
    closest_points[mask4] = A + v[:, np.newaxis] * AB

    # Check if P is in edge region of AC
    vb = d5 * d2 - d1 * d6
    mask5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~mask1 & ~mask2 & ~mask3 & ~mask4
    w = d2[mask5] / (d2[mask5] - d6[mask5])
    closest_points[mask5] = A + w[:, np.newaxis] * AC

    # Check if P is in edge region of BC
    va = d3 * d6 - d5 * d4
    mask6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~mask1 & ~mask2 & ~mask3 & ~mask4 & ~mask5
    w = (d4[mask6] - d3[mask6]) / ((d4[mask6] - d3[mask6]) + (d5[mask6] - d6[mask6]))
    closest_points[mask6] = B + w[:, np.newaxis] * (C - B)

    # P is inside the triangle
    mask7 = ~(mask1 | mask2 | mask3 | mask4 | mask5 | mask6)
    denom = 1.0 / (va[mask7] + vb[mask7] + vc[mask7])
    v = vb[mask7] * denom
    w = vc[mask7] * denom
    closest_points[mask7] = A + v[:, np.newaxis] * AB + w[:, np.newaxis] * AC

    # Compute distances
    distances = np.linalg.norm(points - closest_points, axis=1)
    return distances


def create_offset_surface(sdf_grid, grid_coords, offset_distance, grid_spacing):
    """
    Create an offset surface from the signed distance field.

    Parameters:
    - sdf_grid: 3D signed distance field
    - grid_coords: Tuple of coordinate arrays
    - offset_distance: Distance to offset (positive for outward, negative for inward)
    - grid_spacing: Spacing between grid points

    Returns:
    - vertices: Vertices of the offset surface
    - faces: Faces of the offset surface
    """
    print(f"Creating offset surface with distance: {offset_distance}")

    # Create offset SDF by subtracting the offset distance
    offset_sdf = sdf_grid - offset_distance

    # Use marching cubes to extract the zero-level set
    vertices, faces, normals, values = measure.marching_cubes(
        offset_sdf,
        level=0,
        spacing=grid_spacing
    )

    # Adjust vertices to world coordinates
    x_coords, y_coords, z_coords = grid_coords
    vertices[:, 0] += x_coords[0]
    vertices[:, 1] += y_coords[0]
    vertices[:, 2] += z_coords[0]

    return vertices, faces


def save_offset_mesh(vertices, faces, output_path):
    """
    Save the offset mesh to an STL file.

    Parameters:
    - vertices: Mesh vertices
    - faces: Mesh faces
    - output_path: Path to save the STL file
    """
    # Create trimesh object
    offset_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Fix mesh issues
    offset_mesh.remove_degenerate_faces()
    offset_mesh.remove_duplicate_faces()

    # Save to STL
    offset_mesh.export(output_path)
    print(f"Offset mesh saved to: {output_path}")

    return offset_mesh


def visualize_meshes(original_mesh, offset_mesh):
    """
    Visualize the original and offset meshes.
    """
    fig = plt.figure(figsize=(15, 5))

    # Original mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(original_mesh.vertices[:, 0],
                     original_mesh.vertices[:, 1],
                     original_mesh.vertices[:, 2],
                     triangles=original_mesh.faces,
                     alpha=0.7, color='blue')
    ax1.set_title('Original Mesh')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Offset mesh
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(offset_mesh.vertices[:, 0],
                     offset_mesh.vertices[:, 1],
                     offset_mesh.vertices[:, 2],
                     triangles=offset_mesh.faces,
                     alpha=0.7, color='red')
    ax2.set_title('Offset Mesh')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Both meshes together
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_trisurf(original_mesh.vertices[:, 0],
                     original_mesh.vertices[:, 1],
                     original_mesh.vertices[:, 2],
                     triangles=original_mesh.faces,
                     alpha=0.5, color='blue', label='Original')
    ax3.plot_trisurf(offset_mesh.vertices[:, 0],
                     offset_mesh.vertices[:, 1],
                     offset_mesh.vertices[:, 2],
                     triangles=offset_mesh.faces,
                     alpha=0.5, color='red', label='Offset')
    ax3.set_title('Both Meshes')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()


def create_offset_surface_from_stl(stl_file_path, offset_distance,
                                   output_path=None, grid_resolution=80,
                                   chunk_size=25000, visualize=True):
    """
    Complete pipeline to create an offset surface from an STL file with memory optimization.

    Parameters:
    - stl_file_path: Path to input STL file
    - offset_distance: Offset distance (positive=outward, negative=inward)
    - output_path: Path to save output STL (optional)
    - grid_resolution: Resolution of the SDF grid (reduced default for memory efficiency)
    - chunk_size: Number of points to process at once
    - visualize: Whether to show visualization

    Returns:
    - offset_mesh: Trimesh object of the offset surface
    """
    print("Step 1: Loading STL file...")
    original_mesh = load_stl_mesh(stl_file_path)

    print("Step 2: Computing signed distance field...")
    sdf_grid, grid_coords, grid_spacing = compute_signed_distance_field(
        original_mesh, grid_resolution=grid_resolution, chunk_size=chunk_size
    )

    print("Step 3: Creating offset surface...")
    vertices, faces = create_offset_surface(
        sdf_grid, grid_coords, offset_distance, grid_spacing
    )

    print("Step 4: Creating mesh object...")
    offset_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    offset_mesh.remove_degenerate_faces()
    offset_mesh.remove_duplicate_faces()

    if output_path:
        print("Step 5: Saving offset mesh...")
        offset_mesh.export(output_path)
        print(f"Offset mesh saved to: {output_path}")

    if visualize:
        print("Step 6: Visualizing results...")
        visualize_meshes(original_mesh, offset_mesh)

    print("Done!")
    return offset_mesh


# # For large meshes or limited memory systems - use smaller resolution and chunk size
# offset_mesh_conservative = create_offset_surface_from_stl(
#     stl_file_path="large_model.stl",
#     offset_distance=2.0,
#     output_path="output_conservative.stl",
#     grid_resolution=60,      # Lower resolution
#     chunk_size=10000,        # Smaller chunks
#     visualize=False          # Skip visualization to save memory
# )

# For better quality with adequate memory
offset_mesh_quality = create_offset_surface_from_stl(
    stl_file_path=r"..\assets\stl\lucy.stl",
    offset_distance=2.0,
    output_path="output_quality.stl",
    grid_resolution=120,     # Higher resolution
    chunk_size=50000,        # Larger chunks
    visualize=True
)

# # Ultra-conservative for very large models or low memory
# offset_mesh_minimal = create_offset_surface_from_stl(
#     stl_file_path="huge_model.stl",
#     offset_distance=1.0,
#     output_path="output_minimal.stl",
#     grid_resolution=40,      # Very low resolution
#     chunk_size=5000,         # Very small chunks
#     visualize=False
# )
