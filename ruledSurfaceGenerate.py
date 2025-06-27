import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import pyvista as pv
from src.ruledSurface import trimesh_to_pyvista, combine_and_triangulate_surfaces


def step1_get_draw_directions(draw_direction):
    """
    Step 1: Get the draw directions from your existing pipeline
    
    Args:
        draw_direction (np.array): The draw direction from your main pipeline
    
    Returns:
        tuple: (red_draw_direction, blue_draw_direction)
    """
    # Use the computed draw direction and its opposite
    red_draw_direction = np.array(draw_direction)
    blue_draw_direction = -np.array(draw_direction)  # Opposite direction
    
    print(f"Red Draw Direction: {red_draw_direction}")
    print(f"Blue Draw Direction: {blue_draw_direction}")
    
    return red_draw_direction, blue_draw_direction


def step2_calculate_max_extension_distance(red_mesh, blue_draw_direction):
    """
    Step 2: For red mesh, calculate directions of all extended points from their centroid 
    along blue draw directions and save the max distance.
    
    Args:
        red_mesh (trimesh.Trimesh): The red mesh
        blue_draw_direction (np.array): Blue draw direction vector
    
    Returns:
        tuple: (max_distance, centroid, boundary_points)
    """
    # Get centroid of red mesh
    centroid = red_mesh.centroid
    
    # Get boundary/edge vertices
    try:
        # Get boundary edges
        boundary_edges = red_mesh.outline()
        if boundary_edges is not None:
            boundary_points = boundary_edges.vertices
        else:
            # Fallback: use convex hull vertices
            hull = red_mesh.convex_hull
            boundary_points = hull.vertices
    except:
        # Final fallback: use all vertices
        boundary_points = red_mesh.vertices
    
    # Calculate vectors from centroid to each boundary vertex
    vectors_to_vertices = boundary_points - centroid
    
    # Normalize blue direction
    blue_direction_normalized = blue_draw_direction / np.linalg.norm(blue_draw_direction)
    
    # Project these vectors onto the blue draw direction
    projections = np.dot(vectors_to_vertices, blue_direction_normalized)
    
    # Find the maximum projection distance (absolute value)
    max_distance = np.max(np.abs(projections))
    
    print(f"Centroid: {centroid}")
    print(f"Max extension distance: {max_distance}")
    print(f"Number of boundary points: {len(boundary_points)}")
    
    return max_distance, centroid, boundary_points

def step3_create_projection_plane(centroid, blue_draw_direction, max_distance, extension_factor=0.1):
    """
    Step 3: Create a plane with its normal aligned to blue draw direction and origin 
    will be the centroid translated to the max dist + some 10%
    
    Args:
        centroid (np.array): Centroid of the red mesh
        blue_draw_direction (np.array): Blue draw direction vector
        max_distance (float): Maximum extension distance from step 2
        extension_factor (float): Additional extension factor (default 10%)
    
    Returns:
        tuple: (plane_origin, plane_normal)
    """
    # Normalize the blue draw direction
    plane_normal = blue_draw_direction / np.linalg.norm(blue_draw_direction)
    
    # Calculate plane origin - centroid translated by max_distance + 10%
    translation_distance = max_distance * (1 + extension_factor)
    plane_origin = centroid + plane_normal * translation_distance
    
    print(f"Plane origin: {plane_origin}")
    print(f"Plane normal: {plane_normal}")
    print(f"Translation distance: {translation_distance}")
    
    return plane_origin, plane_normal


def step4_project_points_on_plane(boundary_points, plane_origin, plane_normal):
    """
    Step 4: Project all the boundary points on the plane
    
    Args:
        boundary_points (np.array): Boundary points from step 2
        plane_origin (np.array): Plane origin from step 3
        plane_normal (np.array): Plane normal from step 3
    
    Returns:
        np.array: Projected points on the plane
    """
    # Vector from plane origin to each boundary point
    vectors_to_points = boundary_points - plane_origin
    
    # Calculate the distance from each point to the plane
    distances_to_plane = np.dot(vectors_to_points, plane_normal)
    
    # Project points onto the plane by subtracting the normal component
    projected_points = boundary_points - np.outer(distances_to_plane, plane_normal)
    
    print(f"Number of projected points: {len(projected_points)}")
    if len(distances_to_plane) > 0:
        print(f"Average distance to plane: {np.mean(np.abs(distances_to_plane)):.6f}")
    
    return projected_points


def step5_create_ruled_surface(boundary_points, projected_points):
    """
    Step 5: Create ruled surface between boundary points and their projections
    
    Args:
        boundary_points (np.array): Original boundary points
        projected_points (np.array): Projected points on the plane
    
    Returns:
        pv.PolyData: Ruled surface as PyVista mesh
    """
    if len(boundary_points) != len(projected_points):
        raise ValueError("Boundary points and projected points must have the same length")
    
    if len(boundary_points) < 3:
        print("Not enough points to create surface")
        return None
    
    try:
        # Get convex hull of boundary points to order them properly
        hull_2d = ConvexHull(boundary_points[:, :2])  # Use 2D projection for ordering
        ordered_indices = hull_2d.vertices
        
        # Reorder points based on convex hull
        ordered_boundary = boundary_points[ordered_indices]
        ordered_projected = projected_points[ordered_indices]
        
    except:
        # If convex hull fails, use original order
        ordered_boundary = boundary_points
        ordered_projected = projected_points
    
    # Combine points
    n_points = len(ordered_boundary)
    combined_points = np.vstack([ordered_boundary, ordered_projected])
    
    # Create faces for ruled surface
    faces = []
    
    for i in range(n_points):
        j = (i + 1) % n_points  # Next point (wrap around)
        
        # Create two triangles for each "strip" between corresponding points
        # Triangle 1: boundary[i], projected[i], boundary[j]
        faces.extend([3, i, i + n_points, j])
        
        # Triangle 2: boundary[j], projected[i], projected[j]
        faces.extend([3, j, i + n_points, j + n_points])
    
    # Create PyVista mesh
    ruled_surface = pv.PolyData(combined_points, faces)
    
    print(f"Ruled surface created with {ruled_surface.n_points} points and {ruled_surface.n_faces} faces")
    
    return ruled_surface

def visualize_ruled_surface_process(boundary_points, projected_points, ruled_surface, 
                                   plane_origin, plane_normal, centroid,red_mesh):
    """
    Visualization function to see the entire process
    
    Args:
        boundary_points (np.array): Original boundary points
        projected_points (np.array): Projected points
        ruled_surface (pv.PolyData): The created ruled surface
        plane_origin (np.array): Plane origin
        plane_normal (np.array): Plane normal
        centroid (np.array): Original mesh centroid
    """
    plotter = pv.Plotter()
    
    # Add the ruled surface
    if ruled_surface is not None:
        plotter.add_mesh(ruled_surface, color='lightblue', opacity=1, 
                        show_edges=True, label='Ruled Surface')
    if red_mesh is not None:
        plotter.add_mesh(red_mesh, color='lightblue', opacity=1, 
                        show_edges=True, label='Ruled Surface')
    
    # Add boundary points
    if len(boundary_points) > 0:
        plotter.add_mesh(pv.PolyData(boundary_points), color='lightblue', point_size=8,opacity=1,
                        render_points_as_spheres=True, label='Boundary Points')
    
    # Add projected points
    if len(projected_points) > 0:
        plotter.add_mesh(pv.PolyData(projected_points), color='lightblue', point_size=8,opacity=1,
                        render_points_as_spheres=True, label='Projected Points')
    
    # Add centroid
    plotter.add_mesh(pv.PolyData(centroid.reshape(1, -1)), color='lightblue', point_size=12,opacity=1,
                    render_points_as_spheres=True, label='Centroid')
    
    # Create plane for visualization
    try:
        if len(boundary_points) > 0:
            plane_size = np.linalg.norm(boundary_points - centroid, axis=1).max() * 2
            plane_mesh = pv.Plane(center=plane_origin, direction=plane_normal, 
                                 i_size=plane_size, j_size=plane_size)
            plotter.add_mesh(plane_mesh, color='yellow', opacity=0.2, label='Projection Plane')
    except Exception as e:
        print(f"Could not create plane visualization: {e}")
    
    plotter.show_axes()
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_title('Ruled Surface Creation Process')
    plotter.show()


def main(mesh_path, draw_direction):
    """
    Main function to create and visualize a ruled surface from a mesh and draw direction.
    
    Args:
        mesh_path (str): Path to the input mesh file
        draw_direction (np.array): The draw direction vector [x, y, z]
    """
    # Load the mesh
    try:
        red_mesh = trimesh.load(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return
    
    # Step 1: Get draw directions
    red_draw_direction, blue_draw_direction = step1_get_draw_directions(draw_direction)
    
    # Step 2: Calculate max extension distance and get boundary points
    max_distance, centroid, boundary_points = step2_calculate_max_extension_distance(
        red_mesh, blue_draw_direction)
    
    # Step 3: Create projection plane
    plane_origin, plane_normal = step3_create_projection_plane(
        centroid, blue_draw_direction, max_distance)
    
    # Step 4: Project boundary points onto plane
    projected_points = step4_project_points_on_plane(
        boundary_points, plane_origin, plane_normal)
    
    # Step 5: Create ruled surface
    ruled_surface = step5_create_ruled_surface(boundary_points, projected_points)
    
    # Visualize the process
    visualize_ruled_surface_process(
        boundary_points, projected_points, ruled_surface, 
        plane_origin, plane_normal, centroid,red_mesh)

# Example usage
if __name__ == "__main__":
    # Example mesh path and draw direction
    example_mesh_path = r"/home/sumukhs-ubuntu/Desktop/silicone_mold/combined_parting_surface.stl"  # Replace with actual mesh path
    example_draw_direction = np.array([0.0, 0.0, 1.0])  # Example: along z-axis
    
    main(example_mesh_path, example_draw_direction)