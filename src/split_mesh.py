import numpy as np
import trimesh
from scipy.spatial import KDTree
import pyvista as pv
from src.convex_hull_operations import create_mesh
import trimesh


def extract_unique_vertices_from_faces(vertices, faces):
    # Get unique vertex indices from the faces
    unique_indices = np.unique(faces.flatten())
    # Return the corresponding vertices
    return vertices[unique_indices]


def closest_distance(point, kdtree):
    if kdtree is None:
        return float('inf')
    distance, _ = kdtree.query(point)
    return distance


def face_centroid(mesh, face_index):
    face_vertices = mesh.vertices[mesh.faces[face_index]]
    centroid = np.mean(face_vertices, axis=0)
    return centroid


def pv_to_trimesh(pv_mesh):
    """
    Convert a PyVista PolyData mesh to a Trimesh object.
    :param pv_mesh: pyvista.PolyData
    :return: trimesh.Trimesh
    """
    # Extract vertices and faces from the PyVista mesh
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove the face size (always 3 for triangles)

    # Create a Trimesh object
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return tri_mesh


def split_mesh_faces(mesh: trimesh.Trimesh, convex_hull: trimesh.Trimesh,
                     offset_mesh: trimesh.Trimesh, hull_faces_1, hull_faces_2):
    import trimesh

    bounding_box = mesh.bounds
    max_dimension = np.max(bounding_box[1] - bounding_box[0])

    red_hull_vertices = extract_unique_vertices_from_faces(convex_hull.vertices, hull_faces_1) if len(hull_faces_1) > 0 else np.empty((0, 3))
    blue_hull_vertices = extract_unique_vertices_from_faces(convex_hull.vertices, hull_faces_2) if len(hull_faces_2) > 0 else np.empty((0, 3))

    boundary_threshold = 0.15 * max_dimension

    def identify_boundary_hull_vertices(red_vertices, blue_vertices, threshold):
        if len(red_vertices) == 0 or len(blue_vertices) == 0:
            return np.ones(len(red_vertices), dtype=bool), np.ones(len(blue_vertices), dtype=bool)
        red_tree = KDTree(red_vertices)
        blue_tree = KDTree(blue_vertices)
        distances_red_to_blue, _ = blue_tree.query(red_vertices, k=1)
        distances_blue_to_red, _ = red_tree.query(blue_vertices, k=1)
        red_far_from_boundary = distances_red_to_blue > threshold
        blue_far_from_boundary = distances_blue_to_red > threshold
        return red_far_from_boundary, blue_far_from_boundary

    red_far_from_boundary, blue_far_from_boundary = identify_boundary_hull_vertices(
        red_hull_vertices, blue_hull_vertices, boundary_threshold)

    red_hull_vertices_filtered = red_hull_vertices[red_far_from_boundary]
    blue_hull_vertices_filtered = blue_hull_vertices[blue_far_from_boundary]

    red_kdtree = KDTree(red_hull_vertices_filtered) if len(red_hull_vertices_filtered) > 0 else None
    blue_kdtree = KDTree(blue_hull_vertices_filtered) if len(blue_hull_vertices_filtered) > 0 else None

    # Prepare ray intersector for offset mesh
    try:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(offset_mesh)
    except Exception:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(offset_mesh)

    def calculate_total_distance(face_center, face_normal, hull_kdtree, hull_vertices_filtered):
        if hull_kdtree is None:
            return float('inf')
        projection_distance = 0.1 * max_dimension
        projected_point = face_center + face_normal * projection_distance
        hull_distance, hull_idx = hull_kdtree.query(projected_point, k=1)
        closest_hull_point = hull_vertices_filtered[hull_idx]

        # Ray: origin = closest_hull_point, direction = face_normal
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins=[closest_hull_point],
            ray_directions=[face_normal],
            multiple_hits=False
        )
        if len(locations) > 0:
            offset_distance = np.linalg.norm(locations[0] - closest_hull_point)
        else:
            offset_distance = float('inf')
        return hull_distance + offset_distance

    red_proximal_faces = []
    blue_proximal_faces = []

    for face_idx, face in enumerate(mesh.faces):
        face_center = face_centroid(mesh, face_idx)
        face_normal = mesh.face_normals[face_idx]
        red_total_distance = calculate_total_distance(face_center, face_normal, red_kdtree, red_hull_vertices_filtered)
        blue_total_distance = calculate_total_distance(face_center, face_normal, blue_kdtree, blue_hull_vertices_filtered)
        if red_total_distance <= blue_total_distance:
            red_proximal_faces.append(face)
        else:
            blue_proximal_faces.append(face)

    red_proximal_faces = np.array(red_proximal_faces) if red_proximal_faces else np.empty((0, 3), dtype=int)
    blue_proximal_faces = np.array(blue_proximal_faces) if blue_proximal_faces else np.empty((0, 3), dtype=int)

    red_mesh = create_mesh(mesh.vertices, red_proximal_faces) if len(red_proximal_faces) > 0 else None
    blue_mesh = create_mesh(mesh.vertices, blue_proximal_faces) if len(blue_proximal_faces) > 0 else None

    return red_mesh, blue_mesh

def split_mesh_edges(mesh: trimesh.Trimesh, convex_hull: trimesh.Trimesh, hull_faces_1, hull_faces_2) -> list:
    """
    Split the mesh on the edges whose vertices have proximity to different sections of the hull
    :param mesh: Input mesh body of type trimesh.Trimesh
    :param convex_hull: Convex hull of the mesh of type trimesh.Trimesh
    :param hull_faces_1: Faces of the first hull section
    :param hull_faces_2: Faces of the second hull section
    :return: edge_list: List of edges that are split
    """

    bounding_box = mesh.bounds
    max_dimension = np.max(bounding_box[1] - bounding_box[0])

    # Extract unique vertices from red and blue hull sections
    red_hull_vertices = extract_unique_vertices_from_faces(convex_hull.vertices, hull_faces_1) if len(
        hull_faces_1) > 0 else np.empty((0, 3))
    blue_hull_vertices = extract_unique_vertices_from_faces(convex_hull.vertices, hull_faces_2) if len(
        hull_faces_2) > 0 else np.empty((0, 3))

    # Define boundary threshold distance
    boundary_threshold = 0.15 * max_dimension  # d is your threshold distance

    # Identify hull vertices that are close to the boundary between the two regions
    def identify_boundary_hull_vertices(red_vertices, blue_vertices, threshold):
        """
        Identify hull vertices that are close to the boundary between red and blue regions.
        Returns a boolean mask for each set indicating whether each vertex is far enough from the boundary.
        """
        if len(red_vertices) == 0 or len(blue_vertices) == 0:
            return np.ones(len(red_vertices), dtype=bool), np.ones(len(blue_vertices), dtype=bool)

        # Create KD-Trees for efficient nearest neighbor search
        red_tree = KDTree(red_vertices)
        blue_tree = KDTree(blue_vertices)

        # Find the distance from each red vertex to the closest blue vertex
        distances_red_to_blue, _ = blue_tree.query(red_vertices, k=1)

        # Find the distance from each blue vertex to the closest red vertex
        distances_blue_to_red, _ = red_tree.query(blue_vertices, k=1)

        # Create masks for vertices that are far enough from the boundary
        red_far_from_boundary = distances_red_to_blue > threshold
        blue_far_from_boundary = distances_blue_to_red > threshold

        return red_far_from_boundary, blue_far_from_boundary

    # Apply the filter to identify hull vertices away from the boundary
    red_far_from_boundary, blue_far_from_boundary = identify_boundary_hull_vertices(
        red_hull_vertices, blue_hull_vertices, boundary_threshold)

    # Filter hull vertices for KD-Trees - exclude those close to the boundary
    red_hull_vertices_filtered = red_hull_vertices[red_far_from_boundary]
    blue_hull_vertices_filtered = blue_hull_vertices[blue_far_from_boundary]

    # Create KD-Trees using only hull vertices far from the boundary
    red_kdtree = KDTree(red_hull_vertices_filtered) if len(red_hull_vertices_filtered) > 0 else None
    blue_kdtree = KDTree(blue_hull_vertices_filtered) if len(blue_hull_vertices_filtered) > 0 else None

    edge_list = []

    # Get all edges from the mesh
    mesh_edges = mesh.edges

    # Process all mesh edges
    for edge in mesh_edges:
        # Get the two vertices of the edge
        v1_idx, v2_idx = edge
        v1 = mesh.vertices[v1_idx]
        v2 = mesh.vertices[v2_idx]

        # Find the closest distance from each vertex to red and blue hull sections
        # using only hull vertices that are far from the boundary
        v1_red_distance = closest_distance(v1, red_kdtree)
        v1_blue_distance = closest_distance(v1, blue_kdtree)

        v2_red_distance = closest_distance(v2, red_kdtree)
        v2_blue_distance = closest_distance(v2, blue_kdtree)

        # Determine which hull section each vertex is closest to
        v1_closest_to_red = v1_red_distance <= v1_blue_distance
        v2_closest_to_red = v2_red_distance <= v2_blue_distance

        # If the two vertices are closest to different hull sections, add the edge to the list
        if v1_closest_to_red != v2_closest_to_red:
            edge_list.append(edge)

    return edge_list


def display_split_edges(mesh_path: str, edge_list: list,
                                    mesh_color='lightgray', edge_color='red',
                                    edge_width=5, show_wireframe=True,
                                    wireframe_opacity=0.3):
    """
    Visualize a mesh with highlighted split edges.

    :param mesh_path: Input mesh path
    :param edge_list: List of edges to highlight (from split_mesh_edges function)
    :param mesh_color: Color of the base mesh surface
    :param edge_color: Color of the highlighted edges
    :param edge_width: Width of the highlighted edges
    :param show_wireframe: Whether to show the mesh wireframe
    :param wireframe_opacity: Opacity of the wireframe (if shown)
    """

    # Convert trimesh to PyVista mesh
    pv_mesh = pv.read(mesh_path)
    mesh = trimesh.load(mesh_path)

    # Create plotter
    plotter = pv.Plotter()

    # Add the main mesh
    plotter.add_mesh(pv_mesh, color=mesh_color, opacity=0.8,
                     show_edges=show_wireframe, edge_opacity=wireframe_opacity)

    # Create lines for the split edges
    if len(edge_list) > 0:
        # Convert edge list to line segments for PyVista
        lines = []
        for edge in edge_list:
            v1_idx, v2_idx = edge
            v1 = mesh.vertices[v1_idx]
            v2 = mesh.vertices[v2_idx]

            # Add line segment (format: [2, point1_idx, point2_idx])
            lines.extend([2, len(lines) // 3, len(lines) // 3 + 1])

        # Create points array for the line vertices
        points = []
        for edge in edge_list:
            v1_idx, v2_idx = edge
            points.append(mesh.vertices[v1_idx])
            points.append(mesh.vertices[v2_idx])

        if points:
            points = np.array(points)

            # Reshape lines for PyVista format
            lines_array = []
            for i in range(0, len(points), 2):
                lines_array.extend([2, i, i + 1])

            # Create PyVista lines object
            lines_mesh = pv.PolyData(points)
            lines_mesh.lines = np.array(lines_array)

            # Add highlighted edges to the plot
            plotter.add_mesh(lines_mesh, color=edge_color, line_width=edge_width)

    # Set up the view
    plotter.show_axes()
    plotter.add_text(f"Mesh with {len(edge_list)} split edges highlighted",
                     position='upper_left')

    # Show the plot
    plotter.show()


def display_split_faces(red_mesh: trimesh.Trimesh, blue_mesh: trimesh.Trimesh):
    """
    Display the split meshes using PyVista
    :param red_mesh: Red mesh
    :param blue_mesh: Blue mesh
    """
    proximity_plotter = pv.Plotter()

    # Add the split original mesh
    if red_mesh is not None:
        proximity_plotter.add_mesh(red_mesh, color='red', opacity=1, show_edges=False, label='Closest to Red Hull')
    if blue_mesh is not None:
        proximity_plotter.add_mesh(blue_mesh, color='blue', opacity=1, show_edges=False, label='Closest to Blue Hull')

    proximity_plotter.add_legend()
    proximity_plotter.add_title("Original Mesh Split by Proximity to Hull Sections")
    proximity_plotter.show()
