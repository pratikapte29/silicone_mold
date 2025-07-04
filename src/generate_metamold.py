import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import pyvista as pv
from src.ruledSurface import trimesh_to_pyvista, combine_and_triangulate_surfaces

from scipy.spatial import Delaunay


def step1_get_draw_directions(draw_direction, merged_red_mesh):
    """
    Step 1: Get the draw directions based on merged_red mesh normal alignment

    Args:
        draw_direction (np.array): The original draw direction from pipeline
        merged_red_mesh (trimesh.Trimesh): The merged red mesh

    Returns:
        tuple: (red_draw_direction, blue_draw_direction)
    """
    # Calculate average face normal of merged_red mesh
    red_face_normals = merged_red_mesh.face_normals
    red_avg_normal = np.mean(red_face_normals, axis=0)
    red_avg_normal = red_avg_normal / np.linalg.norm(red_avg_normal)

    # Normalize original draw direction
    draw_direction_normalized = draw_direction / np.linalg.norm(draw_direction)

    # Check alignment using dot product
    alignment = np.dot(red_avg_normal, draw_direction_normalized)

    # If alignment is positive, use original direction for red
    # If alignment is negative, use opposite direction for red
    if alignment > 0:
        red_draw_direction = draw_direction_normalized
        blue_draw_direction = -draw_direction_normalized
    else:
        red_draw_direction = -draw_direction_normalized
        blue_draw_direction = draw_direction_normalized

    print(f"Red mesh normal alignment: {alignment:.3f}")
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
    pyvista_mesh = trimesh_to_pyvista(red_mesh)

    # Get boundary/edge vertices
    try:
        # Get boundary edges
        boundary_edges = pyvista_mesh.extract_feature_edges(boundary_edges=True,
                                                            non_manifold_edges=False,
                                                            feature_edges=False,
                                                            manifold_edges=False)
        if boundary_edges is not None:
            boundary_points = boundary_edges.points
            boundary_points_array = np.array(boundary_points)
        else:
            # Fallback: use convex hull vertices
            hull = red_mesh.convex_hull
            boundary_points = hull.vertices
            boundary_points_array = np.array(boundary_points)
            print("No boundary edges found, using convex hull vertices.")
    except:
        # Final fallback: use all vertices
        boundary_points = red_mesh.vertices
    edge_points_list = []
    boundary_points_sorted = []

    for i in range(boundary_edges.n_cells):
        edge = boundary_edges.get_cell(i)  # Get the i-th edge
        edge_points = edge.points  # Get the points of the edge

        # Append the two points as a list
        edge_points_list.append(edge_points[:2])

    edge_points_array = np.array(edge_points_list)
    boundary_points_array = boundary_points_array.tolist()
    edge_points_array = edge_points_array.tolist()

    boundary_points_sorted.append(boundary_points_array[0])
    remaining_edges = edge_points_array.copy()

    while len(boundary_points_sorted) < len(boundary_points):
        last_node = boundary_points_sorted[-1]
        for edge in remaining_edges[:]:
            if edge[0] == last_node and edge[1] not in boundary_points_sorted:
                boundary_points_sorted.append(edge[1])
                remaining_edges.remove(edge)
                break
            elif edge[1] == last_node and edge[0] not in boundary_points_sorted:
                boundary_points_sorted.append(edge[0])
                remaining_edges.remove(edge)
                break
    boundary_points_sorted.append(boundary_points_sorted[0])
    # Calculate vectors from centroid to each boundary vertex
    vectors_to_vertices = boundary_points_sorted - centroid

    # Normalize blue direction
    blue_direction_normalized = blue_draw_direction / np.linalg.norm(blue_draw_direction)

    # Project these vectors onto the blue draw direction
    projections = np.dot(vectors_to_vertices, blue_direction_normalized)

    # Find the maximum projection distance (absolute value)
    # maximum projection length of any boundary point from the centroid
    max_distance = np.max(np.abs(projections))

    print(f"Centroid: {centroid}")
    print(f"Max extension distance: {max_distance}")
    print(f"Number of boundary points: {len(boundary_points)}")

    return max_distance, centroid, boundary_points_sorted


def step3_create_projection_plane_red(centroid, mesh_faces, mesh_vertices, max_distance, extension_factor=0.15):
    """
    Step 3: Create a plane with its normal aligned to the average face normal and origin
    will be the centroid translated to the max dist + some 10%

    Args:
        centroid (np.array): Centroid of the red mesh
        mesh_faces (np.array): Array of face indices (Nx3)
        mesh_vertices (np.array): Array of vertex coordinates (Mx3)
        max_distance (float): Maximum extension distance from step 2
        extension_factor (float): Additional extension factor (default 10%)

    Returns:
        tuple: (plane_origin, plane_normal)
    """
    # Initialize sum of normals
    normal_sum = np.zeros(3)
    mesh_faces[:, [1, 2]] = mesh_faces[:, [2, 1]]

    # Iterate over all faces and sum their normals
    for face in mesh_faces:
        # Get the three vertices of the face
        v0 = mesh_vertices[face[0]]
        v1 = mesh_vertices[face[1]]
        v2 = mesh_vertices[face[2]]

        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)

        # Add to sum (we'll normalize later)
        normal_sum += face_normal

    # Find the unit normal (average normal direction)
    plane_normal = normal_sum / np.linalg.norm(normal_sum)

    # Calculate plane origin - centroid translated by max_distance + extension factor
    translation_distance = max_distance * (1.5 + extension_factor)
    plane_origin = centroid + plane_normal * translation_distance

    print(f"Plane origin: {plane_origin}")
    print(f"Plane normal: {plane_normal}")
    print(f"Translation distance: {translation_distance}")
    print(f"Number of faces processed: {len(mesh_faces)}")

    return plane_origin, plane_normal

def step3_create_projection_plane_blue(centroid, mesh_faces, mesh_vertices, max_distance, extension_factor=0.15):
    """
    Step 3: Create a plane with its normal aligned to the average face normal and origin
    will be the centroid translated to the max dist + some 10%

    Args:
        centroid (np.array): Centroid of the red mesh
        mesh_faces (np.array): Array of face indices (Nx3)
        mesh_vertices (np.array): Array of vertex coordinates (Mx3)
        max_distance (float): Maximum extension distance from step 2
        extension_factor (float): Additional extension factor (default 10%)

    Returns:
        tuple: (plane_origin, plane_normal)
    """
    # Initialize sum of normals
    normal_sum = np.zeros(3)
    mesh_faces[:, [1, 2]] = mesh_faces[:, [2, 1]]

    # Iterate over all faces and sum their normals
    for face in mesh_faces:
        # Get the three vertices of the face
        v0 = mesh_vertices[face[0]]
        v1 = mesh_vertices[face[1]]
        v2 = mesh_vertices[face[2]]

        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)

        # Add to sum (we'll normalize later)
        normal_sum += face_normal

    # Find the unit normal (average normal direction)
    plane_normal = normal_sum / np.linalg.norm(normal_sum)

    # Calculate plane origin - centroid translated by max_distance + extension factor
    translation_distance = - max_distance * (1.5 + extension_factor)
    plane_origin = centroid + plane_normal * translation_distance

    print(f"Plane origin: {plane_origin}")
    print(f"Plane normal: {plane_normal}")
    print(f"Translation distance: {translation_distance}")
    print(f"Number of faces processed: {len(mesh_faces)}")

    return plane_origin, plane_normal



def step4_project_points_on_plane(boundary_points, plane_origin, plane_normal, scale=1.0):

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
    projected_points = boundary_points - np.outer(scale * distances_to_plane, plane_normal)


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


def bottom_surface(projected_points):
    points_2d = projected_points[:, :2]  # Assuming projection is in XY plane

    tri = Delaunay(points_2d)
    faces = np.hstack([np.full((len(tri.simplices), 1), 3), tri.simplices]).flatten()

    # Create a surface mesh
    mesh = pv.PolyData(projected_points, faces)
    return mesh


def visualize_ruled_surface_process(
    boundary_points, projected_points, ruled_surface,
    plane_origin, plane_normal, centroid,
    red_mesh, merged_red, projected_mesh,
    retracted_points=None, retracted_projected=None, inner_wall=None,
    opp_projected_points=None, opp_plane_origin=None, opp_plane_normal=None, opp_mesh=None,
    retracted_mesh=None, retracted_projected_mesh=None  # ← Add these!
):

    plotter = pv.Plotter()

    # Add ruled surface
    if ruled_surface is not None:
        plotter.add_mesh(ruled_surface, color='lightblue', opacity=1,
                         show_edges=True, label='Ruled Surface')
    if red_mesh is not None:
        plotter.add_mesh(red_mesh, color='lightblue', opacity=1,
                         show_edges=True, label='Base Mesh')
    if merged_red is not None:
        plotter.add_mesh(merged_red, color='red', opacity=1,
                         show_edges=True, label='Merged Red')
    if projected_mesh is not None:
        plotter.add_mesh(projected_mesh, color='red', opacity=1,
                         show_edges=True, label='Bottom Mesh')

    # Add boundary points
    if len(boundary_points) > 0:
        plotter.add_mesh(pv.PolyData(boundary_points), color='black', point_size=12, opacity=1,
                         render_points_as_spheres=True, label='Boundary Points')

    # Add downward projected points
    if len(projected_points) > 0:
        plotter.add_mesh(pv.PolyData(projected_points), color='red', point_size=8, opacity=1,
                         render_points_as_spheres=True, label='Projected Downward')

    # Add upward projected points
    if opp_projected_points is not None and len(opp_projected_points) > 0:
        plotter.add_mesh(pv.PolyData(opp_projected_points), color='blue', point_size=8, opacity=1,
                         render_points_as_spheres=True, label='Projected Upward')

        # Add lines from each boundary point to its upward projected point
        if len(boundary_points) == len(opp_projected_points):
            lines = []
            pts = []
            for i in range(len(boundary_points)):
                start = boundary_points[i]
                end = opp_projected_points[i]
                idx = len(pts)
                pts.append(start)
                pts.append(end)
                lines.append([2, idx, idx + 1])  # VTK line format

            line_mesh = pv.PolyData()
            line_mesh.points = np.array(pts)
            line_mesh.lines = np.array(lines)
            plotter.add_mesh(line_mesh, color='blue', line_width=2, label='Projection Lines Upward')

    # Add centroid
    plotter.add_mesh(pv.PolyData(centroid.reshape(1, -1)), color='lightblue', point_size=12, opacity=1,
                     render_points_as_spheres=True, label='Centroid')

    # Add bottom projection plane
    try:
        if len(boundary_points) > 0:
            plane_size = np.linalg.norm(boundary_points - centroid, axis=1).max() * 2
            plane_mesh = pv.Plane(center=plane_origin, direction=plane_normal,
                                  i_size=plane_size, j_size=plane_size)
            plotter.add_mesh(plane_mesh, color='yellow', opacity=1, label='Bottom Plane')
    except Exception as e:
        print(f"Could not create bottom plane visualization: {e}")

    # Add upper projection plane
    try:
        if opp_plane_origin is not None and opp_plane_normal is not None:
            plane_size = np.linalg.norm(boundary_points - centroid, axis=1).max() * 2
            opp_plane_mesh = pv.Plane(center=opp_plane_origin, direction=opp_plane_normal,
                                      i_size=plane_size, j_size=plane_size)
            plotter.add_mesh(opp_plane_mesh, color='cyan', opacity=1, label='Top Plane')
    except Exception as e:
        print(f"Could not create top plane visualization: {e}")

    # Add upper mesh (optional)
    if opp_mesh is not None:
        plotter.add_mesh(opp_mesh, color='blue', opacity=1, show_edges=True, label='Upper Surface')

    # Add retracted points
    if retracted_points is not None:
        plotter.add_mesh(pv.PolyData(retracted_points), color='green', point_size=10,opacity=1,
                         render_points_as_spheres=True, label='Retracted Points')

    # Add retracted projections
    if retracted_projected is not None:
        plotter.add_mesh(pv.PolyData(retracted_projected), color='blue', point_size=10,
                         render_points_as_spheres=True, label='Retracted Projected')

    # Add inner wall surface
    if inner_wall is not None:
        plotter.add_mesh(inner_wall, color='orange', opacity=1, show_edges=True,
                         label='Inner Wall Surface')
    # Add retracted base mesh
    if retracted_mesh is not None:
        plotter.add_mesh(retracted_mesh, color='green', opacity=1, show_edges=True, label='Retracted Base')

# Add retracted projected mesh
    if retracted_projected_mesh is not None:
        plotter.add_mesh(retracted_projected_mesh, color='blue', opacity=1, show_edges=True, label='Retracted Top')

    # Final plot setup
    plotter.show_axes()
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_title('Projection Visualization')
    plotter.show()

# You can place this anywhere in your script, e.g., after your imports

def get_largest_dimension(mesh):
    """
    Returns the largest dimension (max axis length) of the mesh's bounding box.

    Args:
        mesh (trimesh.Trimesh or pv.PolyData): The mesh object.

    Returns:
        float: Largest dimension (max axis length).
    """
    if hasattr(mesh, "bounds"):  # Works for both trimesh and pyvista
        bounds = mesh.bounds  # shape (2, 3): [[minx, miny, minz], [maxx, maxy, maxz]]
        min_corner, max_corner = bounds
        axis_lengths = max_corner - min_corner
        largest_dim = np.max(axis_lengths)
        print(f"Largest dimension: {largest_dim:.4f}")
        return largest_dim
    else:
        raise ValueError("Mesh object does not have 'bounds' attribute.")

def retract_boundary_by_absolute_distance(boundary_points, mesh, retract_ratio=0.05):
    """
    Retract boundary points inward toward their centroid by a fixed absolute distance,
    computed as a percentage of the largest mesh dimension.

    Args:
        boundary_points (np.array): Nx3 array of boundary point coordinates.
        mesh (trimesh.Trimesh or pv.PolyData): Mesh to compute bounding box from.
        retract_ratio (float): Fraction of largest dimension to retract by (e.g., 0.05 = 5%).

    Returns:
        np.array: Retracted boundary points.
    """
    # Step 1: Get largest dimension of the mesh
    largest_dim = get_largest_dimension(mesh)
    retract_distance = retract_ratio * largest_dim

    # Step 2: Compute 2D centroid in XY (or whatever plane you want)
    centroid_2d = np.mean(boundary_points[:, :2], axis=0)

    # Step 3: Move each point toward the centroid by that fixed distance
    retracted_points = []
    for pt in boundary_points:
        direction = centroid_2d - pt[:2]
        norm = np.linalg.norm(direction)
        if norm == 0:
            new_xy = pt[:2]  # Leave point unchanged if it's at the centroid
        else:
            unit_dir = direction / norm
            new_xy = pt[:2] + unit_dir * min(retract_distance, norm)

        # Keep original Z
        retracted_points.append([new_xy[0], new_xy[1], pt[2]])

    retracted_points = np.array(retracted_points)
    print(f"Retracted {len(boundary_points)} points by {retract_distance:.4f} units toward centroid.")
    return retracted_points



def trimesh_to_pvpoly(tri_mesh):
    return pv.PolyData(tri_mesh.vertices,
                       np.hstack((np.full((len(tri_mesh.faces), 1), 3), tri_mesh.faces)))

def project_retracted_opposite(retracted_points, plane_normal, offset):
    """
    Projects retracted points in the opposite direction of the plane normal.

    Args:
        retracted_points (np.array): Nx3 array of retracted points
        plane_normal (np.array): Normal of the projection plane (should be unit length)
        offset (float): Distance to project the retracted points along the *opposite* direction

    Returns:
        np.array: Projected retracted points
    """
    return retracted_points + offset * plane_normal


def generate_metamold_red(mesh_path, mold_half_path, draw_direction):
    """
    Main function to create and visualize a ruled surface from a mesh and draw direction.

    Args:
        mesh_path (str): combined surface mesh
        mold_half_path (str): Path to the mold half mesh (merged_red / merged_blue)
        draw_direction (np.array): The draw direction vector [x, y, z]
    """
    # Load the mesh
    try:
        red_mesh = trimesh.load(mesh_path)
        merged_red = trimesh.load(mold_half_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Step 1: Get draw directions
    red_draw_direction, blue_draw_direction = step1_get_draw_directions(draw_direction, merged_red)

    # Step 2: Calculate max extension distance and get boundary points
    max_distance, centroid, boundary_points = step2_calculate_max_extension_distance(
        red_mesh, blue_draw_direction)

    # Step 3: Create projection plane
    plane_origin, plane_normal = step3_create_projection_plane_red(
        centroid=centroid,
        mesh_faces=red_mesh.faces,
        mesh_vertices=red_mesh.vertices,
        max_distance=max_distance,
        extension_factor=0.01  # optional, defaults to 0.1
    )
    retracted_points = retract_boundary_by_absolute_distance(
        np.array(boundary_points), red_mesh, retract_ratio=0.05)
    # Step 4: Project boundary points onto plane
    projected_points = step4_project_points_on_plane(
        boundary_points, plane_origin, plane_normal)

    projected_mesh = bottom_surface(projected_points)
    # Step 5: Create ruled surface
    ruled_surface = step5_create_ruled_surface(boundary_points, projected_points)

    retracted_projected_up = project_retracted_opposite(
        retracted_points, -plane_normal, offset=30)  # or any small positive distance
    inner_wall = step5_create_ruled_surface(retracted_points, retracted_projected_up)

    # Visualize the process
    visualize_ruled_surface_process(
        retracted_points, projected_points, ruled_surface,
        plane_origin, plane_normal, centroid,
        red_mesh, merged_red, projected_mesh,
        retracted_points=retracted_points,
        retracted_projected=retracted_projected_up,
        inner_wall=inner_wall
    )

def sort_boundary_points(pv_mesh):
    """
    Extracts and sorts boundary points from a PyVista mesh so they are ordered one after the other.

    Args:
        pv_mesh (pyvista.PolyData): The PyVista mesh (typically a surface).

    Returns:
        list: Sorted list of boundary points (as Nx3 list of coordinates).
    """
    # Extract boundary edges
    edges = pv_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )

    boundary_points = edges.points
    boundary_points_array = np.array(boundary_points).tolist()

    edge_points_list = []

    # Build edge list (pairs of connected points)
    for i in range(edges.n_cells):
        edge = edges.get_cell(i)
        edge_points = edge.points[:2]
        edge_points_list.append(edge_points)

    edge_points_array = np.array(edge_points_list).tolist()

    # Start with first point
    boundary_points_sorted = [boundary_points_array[0]]
    remaining_edges = edge_points_array.copy()

    # Walk through edges to build ordered list
    while len(boundary_points_sorted) < len(boundary_points):
        last_node = boundary_points_sorted[-1]
        for edge in remaining_edges[:]:
            if np.allclose(edge[0], last_node) and not any(np.allclose(edge[1], p) for p in boundary_points_sorted):
                boundary_points_sorted.append(edge[1])
                remaining_edges.remove(edge)
                break
            elif np.allclose(edge[1], last_node) and not any(np.allclose(edge[0], p) for p in boundary_points_sorted):
                boundary_points_sorted.append(edge[0])
                remaining_edges.remove(edge)
                break

    boundary_points_sorted.append(boundary_points_sorted[0])  # close the loop
    print("Sorted boundary length:", len(boundary_points_sorted))
    return boundary_points_sorted

    
def generate_metamold_blue(mesh_path, mold_half_path, draw_direction):
    """
    Main function to create and visualize a ruled surface from a mesh and draw direction.

    Args:
        mesh_path (str): combined surface mesh
        mold_half_path (str): Path to the mold half mesh (merged_red / merged_blue)
        draw_direction (np.array): The draw direction vector [x, y, z]
    """
    # Load the mesh
    try:
        red_mesh = trimesh.load(mesh_path)
        merged_red = trimesh.load(mold_half_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Step 1: Get draw directions
    red_draw_direction, blue_draw_direction = step1_get_draw_directions(draw_direction, merged_red)

    # Step 2: Calculate max extension distance and get boundary points
    max_distance, centroid, boundary_points = step2_calculate_max_extension_distance(
        red_mesh, red_draw_direction)
    
    max_distance_opp, centroid_opp, boundary_points_opp = step2_calculate_max_extension_distance(
        red_mesh, blue_draw_direction)

    # Step 3: Create projection plane
    plane_origin, plane_normal = step3_create_projection_plane_blue(
        centroid=centroid,
        mesh_faces=red_mesh.faces,
        mesh_vertices=red_mesh.vertices,
        max_distance=max_distance,
        extension_factor=0.1  # optional, defaults to 0.1
    )
    plane_origin_opp, plane_normal_opp = step3_create_projection_plane_blue(
        centroid=centroid_opp,
        mesh_faces=red_mesh.faces,
        mesh_vertices=red_mesh.vertices,
        max_distance=max_distance_opp,
        extension_factor=0.1  # optional, defaults to 0.1
    )

    # Step 4: Project boundary points onto plane
    projected_points = step4_project_points_on_plane(
        boundary_points, plane_origin, plane_normal)
    projected_points_opp = step4_project_points_on_plane(
        boundary_points_opp, plane_origin_opp, plane_normal_opp)
    
    retracted_points = retract_boundary_by_absolute_distance(
        np.array(boundary_points), red_mesh, retract_ratio=0.05)

    
    retracted_points_opp = step4_project_points_on_plane(
        retracted_points, plane_origin_opp, plane_normal_opp)
    retracted_mesh = bottom_surface(retracted_points)
    retracted_mesh_opp = bottom_surface(retracted_points_opp)

    projected_mesh = bottom_surface(projected_points)
    projected_mesh_opp = bottom_surface(projected_points_opp)
    # Step 5: Create ruled surface
    ruled_surface = step5_create_ruled_surface(boundary_points, projected_points)
    

    # Visualize the process
    inner_wall = step5_create_ruled_surface(retracted_points, retracted_points_opp)

    visualize_ruled_surface_process(
    boundary_points, projected_points, ruled_surface,
    plane_origin, plane_normal, centroid,
    red_mesh, merged_red, projected_mesh,
    retracted_points=retracted_points,
    retracted_projected=retracted_points_opp,
    inner_wall=inner_wall,
    opp_projected_points=projected_points_opp,
    opp_plane_origin=plane_origin_opp,
    opp_plane_normal=plane_normal_opp,
    opp_mesh=projected_mesh_opp,
    retracted_mesh=retracted_mesh,
    retracted_projected_mesh=retracted_mesh_opp
    )

def visualize_projection_process(
    red_mesh, merged_red,
    boundary_points, projected_points_opp,
    retracted_points, retracted_points_opp,
    centroid, red_draw_direction,
    plane_origin_opp, plane_normal_opp,
    retracted_mesh, retracted_mesh_opp, inner_wall
):
    plotter = pv.Plotter()

    # Base meshes
    plotter.add_mesh(pv.wrap(red_mesh), color='lightblue', opacity=1, show_edges=True, label='Combined Surface')
    plotter.add_mesh(pv.wrap(merged_red), color='red', opacity=1, show_edges=True, label='Mold Half (Bunny)')

    # Points
    if boundary_points is not None and len(boundary_points) > 0:
        plotter.add_mesh(pv.PolyData(np.array(boundary_points)), color='black', point_size=12,
                         render_points_as_spheres=True, label='Boundary Points')

    if projected_points_opp is not None and len(projected_points_opp) > 0:
        plotter.add_mesh(pv.PolyData(np.array(projected_points_opp)), color='blue', point_size=8,
                         render_points_as_spheres=True, label='Projected Points (Opp)')

    if retracted_points is not None:
        plotter.add_mesh(pv.PolyData(np.array(retracted_points)), color='green', point_size=10,
                         render_points_as_spheres=True, label='Retracted Points')

    if retracted_points_opp is not None:
        plotter.add_mesh(pv.PolyData(np.array(retracted_points_opp)), color='blue', point_size=10,
                         render_points_as_spheres=True, label='Retracted Projected')

    # Connecting lines
    if boundary_points is not None and projected_points_opp is not None and len(boundary_points) == len(projected_points_opp):
        lines = []
        pts = []
        for i in range(len(boundary_points)):
            pts.extend([boundary_points[i], projected_points_opp[i]])
            lines.append([2, 2*i, 2*i+1])
        line_mesh = pv.PolyData()
        line_mesh.points = np.array(pts)
        line_mesh.lines = np.array(lines)
        plotter.add_mesh(line_mesh, color='blue', line_width=2, label='Projection Lines Upward')

    # Centroid
    plotter.add_mesh(pv.PolyData(centroid.reshape(1, -1)), color='lightblue', point_size=12,
                     render_points_as_spheres=True, label='Centroid')

    # Planes
    try:
        plane_size = np.linalg.norm(np.array(boundary_points) - centroid, axis=1).max() * 2
        bottom_plane = pv.Plane(center=centroid, direction=red_draw_direction, i_size=plane_size, j_size=plane_size)
        plotter.add_mesh(bottom_plane, color='yellow', opacity=0.5, label='Bottom Plane')
    except Exception as e:
        print(f"Could not create bottom plane: {e}")

    try:
        plane_size = np.linalg.norm(np.array(boundary_points) - centroid, axis=1).max() * 2
        top_plane = pv.Plane(center=plane_origin_opp, direction=plane_normal_opp, i_size=plane_size, j_size=plane_size)
        plotter.add_mesh(top_plane, color='cyan', opacity=0.5, label='Top Plane')
    except Exception as e:
        print(f"Could not create top plane: {e}")

    # Surfaces
    if retracted_mesh is not None:
        plotter.add_mesh(retracted_mesh, color='green', opacity=0.3, show_edges=True, label='Retracted Base')
    if retracted_mesh_opp is not None:
        plotter.add_mesh(retracted_mesh_opp, color='blue', opacity=1, show_edges=True, label='Retracted Top')
    if inner_wall is not None:
        plotter.add_mesh(inner_wall, color='orange', opacity=1, show_edges=True, label='Inner Wall Surface')

    # Plotting
    plotter.show_axes()
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_title('Projection Visualization')
    plotter.show()


def retracted_projected_points(mesh_path, mold_half_path, draw_direction):
    try:
        red_mesh = trimesh.load(mesh_path)
        merged_red = trimesh.load(mold_half_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    red_draw_direction, blue_draw_direction = step1_get_draw_directions(draw_direction,merged_red)

    max_distance, centroid, boundary_points = step2_calculate_max_extension_distance(
        red_mesh, red_draw_direction)

    max_distance_opp, centroid_opp, boundary_points_opp = step2_calculate_max_extension_distance(
        red_mesh, blue_draw_direction)

    plane_origin_opp, plane_normal_opp = step3_create_projection_plane_blue(
        centroid=centroid_opp,
        mesh_faces=red_mesh.faces,
        mesh_vertices=red_mesh.vertices,
        max_distance=max_distance_opp,
        extension_factor=0.1
    )

    projected_points_opp = step4_project_points_on_plane(
        boundary_points_opp, plane_origin_opp, plane_normal_opp)

    retracted_points = retract_boundary_by_absolute_distance(
        np.array(boundary_points), red_mesh, retract_ratio=0.05)

    retracted_points_opp = step4_project_points_on_plane(
        retracted_points, plane_origin_opp, plane_normal_opp)

    retracted_mesh = bottom_surface(retracted_points)
    retracted_mesh_opp = bottom_surface(retracted_points_opp)
    inner_wall = step5_create_ruled_surface(retracted_points, retracted_points_opp)

    visualize_projection_process(
        red_mesh, merged_red,
        boundary_points, projected_points_opp,
        retracted_points, retracted_points_opp,
        centroid, red_draw_direction,
        plane_origin_opp, plane_normal_opp,
        retracted_mesh, retracted_mesh_opp, inner_wall
    )
