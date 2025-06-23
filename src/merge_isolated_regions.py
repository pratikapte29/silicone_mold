import pyvista as pv
import numpy as np


def cleanup_isolated_regions(red_mesh, blue_mesh):
    """
    Transfer all isolated regions to the opposite mesh.
    Only the largest connected component stays in each mesh.

    Parameters:
    - red_mesh: pyvista mesh
    - blue_mesh: pyvista mesh

    Returns:
    - cleaned_red_mesh: pyvista mesh
    - cleaned_blue_mesh: pyvista mesh
    """

    # Extract the largest connected component from red mesh
    red_all = red_mesh.connectivity('all')
    region_ids_red = np.unique(red_all['RegionId'])
    red_new = red_all.connectivity('largest')

    # Remove the noisy regions from red half
    red_noise_ids = region_ids_red[1::]
    red_noise = red_all.connectivity('specified', red_noise_ids)

    # Extract the largest connected component from blue mesh
    blue_all = blue_mesh.connectivity('all')
    region_ids_blue = np.unique(blue_all['RegionId'])
    blue_new = blue_all.connectivity('largest')

    # Remove the noisy regions from blue half
    blue_noise_ids = region_ids_blue[1::]
    blue_noise = blue_all.connectivity('specified', blue_noise_ids)

    # Merge the noisy regions from red into blue
    if red_noise is not None:
        cleaned_blue_mesh = pv.merge([blue_new, red_noise])
    else:
        cleaned_blue_mesh = blue_new

    # Merge the noisy regions from blue into red
    if blue_noise is not None:
        cleaned_red_mesh = pv.merge([red_new, blue_noise])
    else:
        cleaned_red_mesh = red_new

    return cleaned_red_mesh, cleaned_blue_mesh

