import pyvista as pv
import numpy as np


def highlight_intersections(stl_path, direction):
    # Load the STL mesh
    mesh = pv.read(stl_path)

    # Get the centroid of the mesh
    origin = mesh.center

    # Ensure direction is a unit vector
    direction = direction / np.linalg.norm(direction)

    # Create rays in both directions
    ray1_end = origin + 1000 * direction
    ray2_end = origin - 1000 * direction

    # Intersect the mesh with the rays
    points1, inds1 = mesh.ray_trace(origin, ray1_end)
    points2, inds2 = mesh.ray_trace(origin, ray2_end)

    # Merge hit face indices and remove duplicates
    all_inds = np.unique(np.concatenate([inds1, inds2]))

    # Extract the intersected faces
    intersected = mesh.extract_cells(all_inds)

    # Plot
    p = pv.Plotter()
    p.add_mesh(mesh, color='lightgray', opacity=0.3, label='Original Mesh')
    p.add_mesh(intersected, color='red', label='Intersected Faces')
    p.add_mesh(pv.Line(origin, ray1_end), color='blue', line_width=3, label='Ray +d')
    p.add_mesh(pv.Line(origin, ray2_end), color='green', line_width=3, label='Ray -d')
    p.add_legend()
    p.show()


# Example usage:
if __name__ == "__main__":
    stl_file = r"..\assets\stl\xyzrgb_dragon.stl"
    d = np.array([0.2165375, 0.82532754, 0.52148438])  # replace with your input direction
    highlight_intersections(stl_file, d)
