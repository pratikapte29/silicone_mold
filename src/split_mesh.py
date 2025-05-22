import numpy as np
import trimesh
from scipy.spatial import KDTree
import pyvista as pv
from src.convex_hull_operations import create_mesh


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


def offset_stl(file_path, offset_distance):
    # Load STL
    mesh = pv.read(file_path)

    # Compute point normals
    mesh = mesh.compute_normals(auto_orient_normals=True, point_normals=True, inplace=False)

    # Offset the points along their normals
    offset_points = mesh.points + offset_distance * mesh.point_normals

    # Create new mesh with offset points and same faces
    offset_mesh = pv.PolyData(offset_points, mesh.faces)

    return mesh, offset_mesh


def split_mesh_faces(mesh: trimesh.Trimesh, convex_hull: trimesh.Trimesh, hull_faces_1, hull_faces_2):
    """
    Split the mesh faces based on proximity to the convex hull sections.
    However, skip the parts of the hull which are very close to the boundary surface.

    :param mesh: input mesh body
    :param convex_hull: convex hull of the mesh
    :param hull_faces_1:
    :param hull_faces_2:
    :return: red_mesh, blue_mesh (PyVista meshes)
    """

    bounding_box = convex_hull.bounds
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

    red_proximal_faces = []
    blue_proximal_faces = []

    # Process all mesh faces based on proximity to filtered hull vertices
    for face_idx, face in enumerate(mesh.faces):
        # Compute the centroid of this face
        face_center = face_centroid(mesh, face_idx)

        # Find the closest distance to red and blue hull sections
        # using only hull vertices that are far from the boundary
        red_distance = closest_distance(face_center, red_kdtree)
        blue_distance = closest_distance(face_center, blue_kdtree)

        # Assign the face to the closer hull section
        if red_distance <= blue_distance:
            red_proximal_faces.append(face)
        else:
            blue_proximal_faces.append(face)

    # Convert to numpy arrays
    red_proximal_faces = np.array(red_proximal_faces) if red_proximal_faces else np.empty((0, 3), dtype=int)
    blue_proximal_faces = np.array(blue_proximal_faces) if blue_proximal_faces else np.empty((0, 3), dtype=int)

    # Create PyVista meshes for visualization
    red_mesh = create_mesh(mesh.vertices, red_proximal_faces) if len(red_proximal_faces) > 0 else None
    blue_mesh = create_mesh(mesh.vertices, blue_proximal_faces) if len(blue_proximal_faces) > 0 else None

    return red_mesh, blue_mesh


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
