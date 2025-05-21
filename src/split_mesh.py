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


def split_mesh_faces(mesh: trimesh.Trimesh, convex_hull: trimesh.Trimesh, hull_faces_1, hull_faces_2):
    """
    :param mesh: input mesh body
    :param convex_hull: convex hull of the mesh
    :param hull_faces_1:
    :param hull_faces_2:
    :return: red_mesh, blue_mesh (PyVista meshes)
    """

    # Extract unique vertices from red and blue hull sections
    red_hull_vertices = extract_unique_vertices_from_faces(convex_hull.vertices, hull_faces_1) if len(
        hull_faces_1) > 0 else np.empty((0, 3))
    blue_hull_vertices = extract_unique_vertices_from_faces(convex_hull.vertices, hull_faces_2) if len(
        hull_faces_2) > 0 else np.empty((0, 3))

    # Create KD-Trees for efficient nearest point lookup
    red_kdtree = KDTree(red_hull_vertices) if len(red_hull_vertices) > 0 else None
    blue_kdtree = KDTree(blue_hull_vertices) if len(blue_hull_vertices) > 0 else None

    red_proximal_faces = []
    blue_proximal_faces = []

    # For each face in the original mesh, compute its centroid and find the closest hull section
    for face_idx, face in enumerate(mesh.faces):
        # Compute the centroid of this face
        face_center = face_centroid(mesh, face_idx)

        # Find the closest distance to red and blue hull sections
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
