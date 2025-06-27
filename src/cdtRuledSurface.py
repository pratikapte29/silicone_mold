import numpy as np
from stl import mesh
import pyvista as pv
from scipy.spatial import KDTree
import trimesh
import numpy as np
from trimesh import repair as tm_repair
from trimesh.boolean import difference
import os


# Force attach to log to see detailed errors (optional but helpful)
trimesh.util.attach_to_log()

# Ensure OpenSCAD path is in the environment
openscad_dir = "/usr/bin"
if openscad_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + openscad_dir


def load_obj_file(filename):
    """
    Load OBJ file and return PyVista PolyData object.
    """
    try:
        # Read OBJ file using PyVista
        mesh_data = pv.read(filename)
        print(f"Loaded OBJ file with {mesh_data.n_points} points and {mesh_data.n_cells} faces")
        return mesh_data
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return None


def create_ruled_surface_mesh(inner_points, outer_points):
    """
    Create a ruled surface mesh between two ordered point sets.
    Both arrays must have the same number of points.
    """
    if len(outer_points) != len(inner_points):
        raise ValueError("Both arrays must have the same number of points.")

    # Create an array to store the faces
    faces = []

    # Create triangular faces by connecting points from both lines
    for i in range(len(inner_points) - 1):
        # Define two triangles for each quadrilateral face
        faces.append([3, i, i + len(inner_points), i + 1])
        faces.append([3, i + 1, i + len(inner_points), i + len(inner_points) + 1])

    # Convert the list of faces to a numpy array
    faces = np.array(faces).flatten()

    # Combine the points into a single array
    points = np.vstack([inner_points, outer_points])

    # Create a PolyData object
    ruled_surface = pv.PolyData(points, faces)
    
    return ruled_surface


def expand_points_by_translation(points, centroid, distance):
    """
    Move each point outward from the centroid by a fixed distance.
    """
    vectors = points - centroid
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for points at the centroid
    norms[norms == 0] = 1
    directions = vectors / norms
    expanded_points = points + directions * distance
    return expanded_points


def debug_mesh_info(mesh):
    """
    Print detailed information about the mesh to help debug boundary detection issues.
    """
    print(f"Mesh info:")
    print(f"  Points: {mesh.n_points}")
    print(f"  Cells: {mesh.n_cells}")
    print(f"  Is triangulated: {mesh.is_all_triangles}")
    
    # Check for different types of edges
    boundary_edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, 
                                               feature_edges=False, manifold_edges=False)
    print(f"  Boundary edges: {boundary_edges.n_cells}")
    
    non_manifold_edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=True, 
                                                   feature_edges=False, manifold_edges=False)
    print(f"  Non-manifold edges: {non_manifold_edges.n_cells}")
    
    feature_edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=False, 
                                              feature_edges=True, manifold_edges=False)
    print(f"  Feature edges: {feature_edges.n_cells}")
    
    manifold_edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=False, 
                                               feature_edges=False, manifold_edges=True)
    print(f"  Manifold edges: {manifold_edges.n_cells}")


def find_boundary_points_alternative(cdt_surface):
    """
    Alternative method to find boundary points using edge connectivity analysis.
    """
    # Convert to trimesh for better edge analysis
    faces = cdt_surface.faces.reshape((-1, 4))[:, 1:]  # Remove the first column (face size)
    tm_mesh = trimesh.Trimesh(vertices=cdt_surface.points, faces=faces)
    
    # Get boundary edges (edges that are part of only one face)
    boundary_edges = tm_mesh.edges[trimesh.grouping.group_rows(
        tm_mesh.edges_sorted, require_count=1)]
    
    if len(boundary_edges) == 0:
        print("No boundary edges found using trimesh method either.")
        return np.array([])
    
    print(f"Found {len(boundary_edges)} boundary edges using trimesh.")
    
    # Extract unique boundary vertices
    boundary_vertex_indices = np.unique(boundary_edges.flatten())
    boundary_points = tm_mesh.vertices[boundary_vertex_indices]
    
    # Sort boundary points to form a continuous loop
    if len(boundary_points) > 2:
        boundary_points_sorted = sort_points_by_connectivity(boundary_points, boundary_edges, tm_mesh.vertices)
        return boundary_points_sorted
    else:
        return boundary_points


def sort_points_by_connectivity(boundary_points, boundary_edges, all_vertices):
    """
    Sort boundary points to form a continuous path using edge connectivity.
    """
    # Create a mapping from coordinates to indices
    coord_to_idx = {}
    for i, point in enumerate(all_vertices):
        key = tuple(np.round(point, 6))  # Round to avoid floating point issues
        coord_to_idx[key] = i
    
    # Find indices of boundary points
    boundary_indices = []
    for point in boundary_points:
        key = tuple(np.round(point, 6))
        if key in coord_to_idx:
            boundary_indices.append(coord_to_idx[key])
    
    # Build adjacency list from boundary edges
    adjacency = {idx: [] for idx in boundary_indices}
    for edge in boundary_edges:
        if edge[0] in boundary_indices and edge[1] in boundary_indices:
            adjacency[edge[0]].append(edge[1])
            adjacency[edge[1]].append(edge[0])
    
    # Sort points by following the connectivity
    if not boundary_indices:
        return np.array([])
    
    sorted_indices = [boundary_indices[0]]  # Start with first boundary point
    current = boundary_indices[0]
    visited = {current}
    
    while len(sorted_indices) < len(boundary_indices):
        next_candidates = [idx for idx in adjacency[current] if idx not in visited]
        if not next_candidates:
            break
        next_point = next_candidates[0]  # Take the first available neighbor
        sorted_indices.append(next_point)
        visited.add(next_point)
        current = next_point
    
    # Get the sorted points
    sorted_points = all_vertices[sorted_indices]
    
    # Close the loop by adding the first point at the end
    if len(sorted_points) > 0:
        sorted_points = np.vstack([sorted_points, sorted_points[0]])
    
    return sorted_points


def sort_boundary_points(cdt_surface):
    """
    Sort boundary points of the CDT surface to create an ordered sequence.
    Enhanced version with debugging and fallback methods.
    """
    # First, debug the mesh
    debug_mesh_info(cdt_surface)
    
    # Try original method first
    edges = cdt_surface.extract_feature_edges(
        boundary_edges=True, 
        non_manifold_edges=False, 
        feature_edges=False, 
        manifold_edges=False
    )
    
    if edges.n_cells == 0:
        print("No boundary edges found with original method. Trying alternative approaches...")
        
        # Try with different parameters
        edges = cdt_surface.extract_feature_edges(
            boundary_edges=True, 
            non_manifold_edges=True, 
            feature_edges=True, 
            manifold_edges=True
        )
        
        if edges.n_cells == 0:
            print("Still no edges found. Trying trimesh-based approach...")
            return find_boundary_points_alternative(cdt_surface)
    
    print(f"Found {edges.n_cells} edges for boundary detection.")
    
    # Extract the boundary points from the boundary edges
    boundary_points = edges.points
    boundary_points_array = np.array(boundary_points)
    
    if len(boundary_points_array) == 0:
        print("No boundary points found.")
        return np.array([])
    
    edge_points_list = []
    boundary_points_sorted = []
    
    # Iterate over each edge (cell) in the edges
    for i in range(edges.n_cells):
        edge = edges.get_cell(i)  # Get the i-th edge
        edge_points = edge.points  # Get the points of the edge
        # Append the two points as a list
        edge_points_list.append(edge_points[:2])  # Take the first two points
    
    # Convert to lists for easier processing
    boundary_points_list = boundary_points_array.tolist()
    edge_points_list = [edge.tolist() for edge in edge_points_list]
    
    if len(boundary_points_list) == 0:
        return np.array([])
    
    # Start with the first point
    boundary_points_sorted.append(boundary_points_list[0])
    remaining_edges = edge_points_list.copy()
    
    # Iterate until all nodes are sorted
    while len(boundary_points_sorted) < len(boundary_points_list):
        last_node = boundary_points_sorted[-1]
        found_next = False
        
        for edge in remaining_edges[:]:
            if edge[0] == last_node and edge[1] not in boundary_points_sorted:
                boundary_points_sorted.append(edge[1])
                remaining_edges.remove(edge)
                found_next = True
                break
            elif edge[1] == last_node and edge[0] not in boundary_points_sorted:
                boundary_points_sorted.append(edge[0])
                remaining_edges.remove(edge)
                found_next = True
                break
        
        if not found_next:
            print(f"Could not find next connected point. Processed {len(boundary_points_sorted)} out of {len(boundary_points_list)} points.")
            break
    
    # Close the loop by adding the first point at the end
    if len(boundary_points_sorted) > 0 and len(boundary_points_sorted) > 1:
        boundary_points_sorted.append(boundary_points_sorted[0])
    
    return np.array(boundary_points_sorted)


def create_boundary_from_convex_hull(cdt_surface):
    """
    Fallback method: create boundary points from the convex hull of the mesh.
    This is useful when the mesh doesn't have clear boundary edges.
    """
    print("Attempting to create boundary from convex hull...")
    
    # Get the convex hull
    hull = cdt_surface.convex_hull()
    
    # Project points to a plane (assuming the CDT surface is roughly planar)
    points = cdt_surface.points
    
    # Find the principal plane by computing PCA
    centered_points = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    
    # Project points onto the first two principal components
    projected = centered_points @ vh[:2].T
    
    # Find convex hull in 2D
    from scipy.spatial import ConvexHull
    hull_2d = ConvexHull(projected)
    
    # Get the boundary points in 3D
    boundary_indices = hull_2d.vertices
    boundary_points = points[boundary_indices]
    
    # Close the loop
    boundary_points = np.vstack([boundary_points, boundary_points[0]])
    
    print(f"Created boundary with {len(boundary_points)} points from convex hull.")
    return boundary_points


def visualize_cdt_and_boundary(cdt_surface, boundary_points_sorted):
    """
    Visualize the CDT surface and sorted boundary points.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(cdt_surface, color='cyan', show_edges=True, opacity=1, label='CDT Surface')
    
    if len(boundary_points_sorted) > 0:
        boundary_polydata = pv.PolyData(boundary_points_sorted)
        plotter.add_mesh(boundary_polydata, color='red', point_size=10, 
                        render_points_as_spheres=True, label='Sorted Boundary Points')
    
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('CDT Surface with Sorted Boundary Points')
    plotter.show()


def visualize_final_ruled_surface(cdt_surface, ruled_surface, boundary_points_sorted, expanded_points):
    """
    Visualize the final result with CDT surface and ruled surface.
    """
    plotter = pv.Plotter()
    
    # Add CDT surface
    plotter.add_mesh(cdt_surface, color='cyan', opacity=1, show_edges=True, label='CDT Surface')
    
    # Add ruled surface
    plotter.add_mesh(ruled_surface, color='lightblue', opacity=1, show_edges=True, label='Ruled Surface')
    
    # Add boundary points
    if len(boundary_points_sorted) > 0:
        plotter.add_mesh(pv.PolyData(boundary_points_sorted), color='red', point_size=8, 
                        render_points_as_spheres=True, label='Boundary Points')
    
    # Add expanded points
    if len(expanded_points) > 0:
        plotter.add_mesh(pv.PolyData(expanded_points), color='magenta', point_size=8, 
                        render_points_as_spheres=True, label='Expanded Points')
    
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Complete Ruled Surface with CDT Base')
    plotter.show()


def visualize_combined_surface(combined_surface):
    plotter = pv.Plotter()
    plotter.add_mesh(combined_surface, color='orange', opacity=1, show_edges=True, label='Combined Surface')
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Combined CDT and Ruled Surface')
    plotter.show()


def pv_to_trimesh(pv_mesh):
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]
    return trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)


def trimesh_to_pyvista(tm):
    return pv.PolyData(tm.vertices, np.hstack([np.full((len(tm.faces), 1), 3), tm.faces]))


def main():
    # File paths
    cdt_obj_file = r"/home/sumukhs-ubuntu/triangulation1.obj"  # Your CDT triangulated OBJ file
    file1 = r"/home/sumukhs-ubuntu/Desktop/silicone_mold/merged_blue.stl"  # For centroid calculation
    file3 = r"/home/sumukhs-ubuntu/Desktop/silicone_mold/assets/stl/bunny.stl"  # Bunny mesh for clipping

    # Load the CDT triangulated surface
    print("Loading CDT triangulated surface...")
    cdt_surface = load_obj_file(cdt_obj_file)
    if cdt_surface is None:
        print("Failed to load CDT surface.")
        return

    # Load mesh for centroid calculation
    mesh1 = pv.read(file1)
    centroid = mesh1.center
    print(f"Using centroid: {centroid}")

    # Load bunny mesh for clipping
    bunny_mesh = pv.read(file3)

    print("Sorting boundary points...")
    boundary_pts = sort_boundary_points(cdt_surface)
    
    # If no boundary points found, try convex hull approach
    if len(boundary_pts) == 0:
        print("No boundary points found with edge detection. Trying convex hull approach...")
        boundary_pts = create_boundary_from_convex_hull(cdt_surface)
    
    print(f"Found {len(boundary_pts)} boundary points.")
    if len(boundary_pts) == 0:
        print("No boundary points found after all attempts.")
        return
    
    # Visualize CDT surface and boundary points
    visualize_cdt_and_boundary(cdt_surface, boundary_pts)

    print("Translating sorted points...")
    expanded_pts = expand_points_by_translation(boundary_pts, centroid, 500)

    print("Creating ruled surface...")
    try:
        ruled_surface = create_ruled_surface_mesh(boundary_pts, expanded_pts)
        cdt_surface.save('cdt_surface.vtk')
        ruled_surface.save('ruled_surface.vtk')
        print("Surfaces saved successfully.")
        visualize_final_ruled_surface(cdt_surface,
                                      ruled_surface,
                                      boundary_pts,
                                      expanded_pts)
    except Exception as e:
        print(f"Error creating ruled surface: {e}")
        return

    # Combine the surfaces
    print("Combining CDT and ruled surfaces...")
    combined_surface = cdt_surface + ruled_surface
    combined_surface = combined_surface.triangulate()
    combined_surface.save('combined_parting_surface.vtk')
    print("Combined surface saved successfully.")
    visualize_combined_surface(combined_surface)

    # Clip with bunny mesh
    print("Clipping combined surface with bunny mesh...")
    try:
        # Convert PyVista meshes to Trimesh
        tm_combined = pv_to_trimesh(combined_surface)
        tm_bunny = pv_to_trimesh(bunny_mesh)

        # Perform boolean difference
        result = difference([tm_combined, tm_bunny], engine='blender', check_volume=False)
        if result is None:
            print("Boolean operation failed. Check mesh watertightness.")
            return

        # Convert back to PyVista for visualization/export
        result_pv = trimesh_to_pyvista(result)
        result_pv.save("clipped_with_bunny.vtk")
        print("Clipped surface saved successfully.")
        visualize_combined_surface(result_pv)

    except Exception as e:
        print(f"Clipping with bunny mesh failed: {e}")


if __name__ == "__main__":
    main()