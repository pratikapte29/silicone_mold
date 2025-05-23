import os

import trimesh
from skimage import measure
import numpy as np
import pyvista as pv
from meshlib import mrmeshpy


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


def offset_stl_sdf(mesh_path: str, offset_distance: float, voxel_size=1):
    """
    Compute the offset surface of a mesh using Signed Distance Function (SDF)
    :param mesh_path: str
    :param offset_distance: float
    :param voxel_size: default - 0.01
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

