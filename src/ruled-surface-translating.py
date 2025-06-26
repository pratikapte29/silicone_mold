import numpy as np
from stl import mesh
import pyvista as pv
from scipy.spatial import KDTree
import trimesh
import pyvista as pv
import numpy as np

def visualize_ruled_surface(common_points, expanded_points):
    """
    Visualize a ruled surface between two ordered point sets using PyVista.
    """
    if len(common_points) == 0 or len(expanded_points) == 0:
        print("No points to create a ruled surface.")
        return

    # Stack the two curves into a grid: shape (2, N, 3)
    points = np.stack([common_points, expanded_points], axis=0)  # shape (2, N, 3)
    # Reshape for StructuredGrid: (nx, ny, 3)
    grid = pv.StructuredGrid()
    grid.points = points.reshape(-1, 3)
    grid.dimensions = [2, len(common_points), 1]

    plotter = pv.Plotter()
    plotter.add_mesh(grid, color='cyan', opacity=0.7, show_edges=True, label='Ruled Surface')
    plotter.add_mesh(pv.PolyData(common_points), color='yellow', point_size=12, render_points_as_spheres=True, label='Common Points')
    plotter.add_mesh(pv.PolyData(expanded_points), color='magenta', point_size=12, render_points_as_spheres=True, label='Expanded Points')
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Ruled Surface between Common and Expanded Points')
    plotter.show()

# Example usage in your main:
# visualize_ruled_surface(common_points, expanded_points)
def load_stl_vertices(filename):
    """Load STL file and extract unique vertices"""
    stl_mesh = mesh.Mesh.from_file(filename)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    vertices_rounded = np.round(vertices, decimals=6)
    unique_vertices = np.unique(vertices_rounded, axis=0)
    return unique_vertices
def expand_points_by_translation(points, centroid, distance):
    """
    Move each point outward from the centroid by a fixed distance.
    """
    vectors = points - centroid
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for points at the centroid
    norms[norms == 0] = 1
    directions = vectors / norms
    expanded_points = points + directions * distance
    return expanded_points

def visualize_translated(file1, file2, vertices1_minus, vertices2_minus, common_points_minus, expanded_points):
    plotter = pv.Plotter()
    # Show translated red and blue meshes as point clouds
    plotter.add_mesh(pv.PolyData(vertices1_minus), color='red', point_size=5, render_points_as_spheres=True, label='Red STL -distance')
    plotter.add_mesh(pv.PolyData(vertices2_minus), color='blue', point_size=5, render_points_as_spheres=True, label='Blue STL -distance')
    # Show translated common points
    plotter.add_mesh(pv.PolyData(common_points_minus), color='yellow', point_size=10, render_points_as_spheres=True, label='Common Points -distance')
    # Show expanded points (already at +distance)
    plotter.add_mesh(pv.PolyData(expanded_points), color='magenta', point_size=10, render_points_as_spheres=True, label='Expanded Points +distance')
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Translated Meshes and Points')
    plotter.show()


def visualize_with_pyvista(file1, file2, common_points):
    """Visualize the STL files and common points using PyVista"""
    plotter = pv.Plotter()
    mesh1 = pv.read(file1)
    mesh2 = pv.read(file2)
    plotter.add_mesh(mesh1, color='red', opacity=1, label='Red STL')
    plotter.add_mesh(mesh2, color='blue', opacity=1, label='Blue STL')
    if len(common_points) > 0:
        point_cloud = pv.PolyData(common_points)
        plotter.add_mesh(point_cloud, color='yellow', point_size=10, 
                        render_points_as_spheres=True, 
                        label=f'Common Points ({len(common_points)})')
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('STL Files with Common Points')
    plotter.show()


def find_common_points(vertices1, vertices2, tolerance=1e-6):
    """Find common points between two sets of vertices"""
    common_points = []
    for v1 in vertices1:
        distances = np.linalg.norm(vertices2 - v1, axis=1)
        if np.min(distances) < tolerance:
            common_points.append(v1)
    return np.array(common_points) if common_points else np.array([]).reshape(0, 3)



def main():
    file1 = "merged_red.stl"
    file2 = "merged_blue.stl"

    # Load meshes and compute centroid from the first mesh
    mesh1 = pv.read(file1)
    centroid = mesh1.center

    # Load vertices
    vertices1 = load_stl_vertices(file1)
    vertices2 = load_stl_vertices(file2)

    # Find common points
    common_points = find_common_points(vertices1, vertices2)
    print(f"Found {len(common_points)} common points.")

    # Expand common points outward from centroid by a fixed distance
    distance = 500  # Set your desired translation distance here
    if len(common_points) > 0:
        expanded_points = expand_points_by_translation(common_points, centroid, distance)
    else:
        expanded_points = np.array([]).reshape(0, 3)

    # Visualize ruled surface between common and expanded points
    print("Visualizing ruled surface between common and expanded points...")
    visualize_ruled_surface(common_points, expanded_points)

if __name__ == "__main__":
    main()