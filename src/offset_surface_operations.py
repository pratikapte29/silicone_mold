import os

import trimesh
from skimage import measure
import numpy as np
import pyvista as pv
from meshlib import mrmeshpy
from scipy.spatial import KDTree


def mesh_hull_dist(mesh_path: str, convex_hull_path: str) -> float:
    """
    Return the distance between the surfaces of mesh and convex hull
    This will be used to calculate the offset surface distance

    :param mesh_path: Input mesh stl file path
    :param convex_hull_path: Path of stl file of convex hull
    :return: Mean distance between the two surfaces
    """
    mesh = pv.read(mesh_path)
    convex_hull = pv.read(convex_hull_path)

    tree = KDTree(convex_hull.points)
    d_kdtree, idx = tree.query(mesh.points)
    mesh['distances'] = d_kdtree

    return np.max(d_kdtree)


def offset_stl(file_path, offset_distance):
    """
    Naively offset the mesh
    :param file_path: str
    :param offset_distance: float
    :return: pv.PolyData, pv.PolyData
    """
    # Load STL
    mesh = pv.read(file_path)

    # Compute point normals
    mesh = mesh.compute_normals(auto_orient_normals=True, point_normals=True, inplace=False)

    # Offset the points along their normals
    offset_points = mesh.points + offset_distance * mesh.point_normals

    # Create new mesh with offset points and same faces
    offset_mesh = pv.PolyData(offset_points, mesh.faces)

    return mesh, offset_mesh


def offset_stl_sdf(mesh_path: str, offset_distance: float, voxel_size=10):
    """
    Compute the offset surface of a mesh using Signed Distance Function (SDF)
    :param mesh_path: str
    :param offset_distance: float
    :param voxel_size: default - 1
    :return: offset_mesh_path: str
    """
    # Load mesh using Trimesh
    closedMesh = mrmeshpy.loadMesh(mesh_path)

    # setup offset parameters
    params = mrmeshpy.OffsetParameters()
    params.voxelSize = voxel_size

    # create positive offset mesh
    posOffset = mrmeshpy.offsetMesh(closedMesh, offset_distance, params)

    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    output_dir = os.path.join("results", base_name)
    os.makedirs(output_dir, exist_ok=True)
    output_stl_path = os.path.join(output_dir, base_name + "_offset.stl")

    # save results
    mrmeshpy.saveMesh(posOffset, output_stl_path)

    return output_stl_path


def split_offset_surface(offset_mesh_path: str, draw_direction: np.ndarray):
    """
    Split the offset surface into two parts based on the draw direction probably using region growing
    :param offset_mesh_path: str
    :param draw_direction: np.ndarray
    :return:
    """

    # Load the offset mesh
    offset_mesh = trimesh.load(offset_mesh_path)
    offset_mesh.edges_unique


def display_offset_surface(offset_mesh_path: str, mesh_path: str, convex_hull_path: str):
    """
    Display the offset surface mesh
    :param offset_mesh_path: str
    :param mesh_path: str
    :param convex_hull_path: str
    :return:
    """

    # Load the offset mesh
    offset_mesh = pv.read(offset_mesh_path)
    mesh = pv.read(mesh_path)
    convex_hull = pv.read(convex_hull_path)

    # Create a plotter object
    plotter = pv.Plotter()

    # Add the offset mesh to the plotter
    plotter.add_mesh(offset_mesh, color='lightblue', opacity=0.5, show_edges=False, label='Offset Surface')
    plotter.add_mesh(mesh, color='red', opacity=1, show_edges=False, label='Original Mesh')
    plotter.add_mesh(convex_hull, color='gray', opacity=0.5, show_edges=False, label='Convex Hull')

    # Set the camera position and orientation
    plotter.camera_position = 'xy'
    plotter.show_grid()

    # Show the plot
    plotter.show()


def split_offset_surface(offset_surface_path: str, d1_aligned_faces: list, d2_aligned_faces: list):
    """
    Split the offset surface into two parts based on the proximity of faces with convex hull
    :param offset_surface_path: str
    :param d1_aligned_faces: list of face indices aligned with direction 1
    :param d2_aligned_faces: list of face indices aligned with direction 2
    :return: tuple (d1_mesh, d2_mesh)
    """

    # Offset surface splitting won't be needed at the moment.
    # Only the distance will be used such that it can reduce the impact of proximity to convex hull
    pass
