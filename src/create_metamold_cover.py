"""
Input: combined_parting_surface variable in the main.py file
Steps:
1. Get the boundary of the combined parting surface
2. Offset the boundary inward by a small distance to give the cover thickness
3. Move the inner and outer set of points towards the correct draw direction to give the cover height
   (Inner ones will be slightly lower)
4.
"""

import os
import pyvista as pv
import trimesh
import numpy as np


def translate_sfc_boundary(stl_file_path, translation_vector, distance):
    """
    Translates boundary points of an STL file along a given vector by a specified distance.

    Parameters:
    -----------
    stl_file_path : str
        Path to the STL file (e.g., 'combined_parting_surface.stl')
    translation_vector : numpy.ndarray
        3D vector indicating the direction of translation (will be normalized)
    distance : float
        Distance to translate the boundary points

    Returns:
    --------
    translated_points : numpy.ndarray
    """

    # Load the STL file
    mesh = pv.read(stl_file_path)

    # Get boundary edges
    boundary_edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False
    )

    # Extract boundary points
    boundary_points = boundary_edges.points

    # Normalize the translation vector
    translation_vector = np.array(translation_vector, dtype=float)
    unit_vector = translation_vector / np.linalg.norm(translation_vector)

    # Calculate translation offset
    translation_offset = unit_vector * distance

    # Translate boundary points
    translated_points = boundary_points + translation_offset

    # Create meshes for visualization (optional)
    boundary_mesh = pv.PolyData(boundary_points)
    translated_mesh = pv.PolyData(translated_points)

    return translated_points, translated_mesh, boundary_mesh


def calculate_mesh_height(mesh, direction_vector):
    """
    Alternative method using bounding box approach (works with both PyVista and Trimesh).
    This method first aligns the mesh with the direction vector.

    Parameters:
    -----------
    mesh : pyvista.PolyData or trimesh.Trimesh
        Input mesh
    direction_vector : numpy.ndarray
        3D vector indicating the direction to measure height

    Returns:
    --------
    float
        Height along the specified direction
    """

    # Get points based on mesh type
    if hasattr(mesh, 'points'):  # PyVista
        points = mesh.points
    else:  # Trimesh
        points = mesh.vertices

    # Normalize direction vector
    direction_vector = np.array(direction_vector, dtype=float)
    unit_vector = direction_vector / np.linalg.norm(direction_vector)

    # Create rotation matrix to align direction vector with Z-axis
    z_axis = np.array([0, 0, 1])

    # If vectors are already aligned, no rotation needed
    if np.allclose(unit_vector, z_axis):
        rotation_matrix = np.eye(3)
    elif np.allclose(unit_vector, -z_axis):
        rotation_matrix = np.diag([1, 1, -1])
    else:
        # Rodrigues' rotation formula
        v = np.cross(unit_vector, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(unit_vector, z_axis)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    # Rotate points
    rotated_points = np.dot(points, rotation_matrix.T)

    # Height is the difference in Z coordinates after rotation
    height = np.max(rotated_points[:, 2]) - np.min(rotated_points[:, 2])

    return height
