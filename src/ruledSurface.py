import numpy as np
from stl import mesh
import pyvista as pv
from scipy.spatial import KDTree
import trimesh
import pyvista as pv
import numpy as np
from trimesh import repair as tm_repair
from trimesh.boolean import difference
import os


# Force attach to log to see detailed errors (optional but helpful)
trimesh.util.attach_to_log()

# Ensure OpenSCAD path is in the environment
openscad_dir = "/usr/bin"
if openscad_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + openscad_dir


def create_ruled_surface_mesh(inner_points, outer_points):
    """
    Create a ruled surface mesh between two ordered point sets.
    Both arrays must have the same number of points.
    """
    if len(outer_points) != len(inner_points):
        raise ValueError("Both arrays must have the same number of points.")

    # Create an array to store the faces
    faces = []

    # Create triangular faces by connecting points from both lines
    for i in range(len(inner_points) - 1):
        # Define two triangles for each quadrilateral face
        faces.append([3, i, i + len(inner_points), i + 1])
        faces.append([3, i + 1, i + len(inner_points), i + len(inner_points) + 1])

    # Convert the list of faces to a numpy array
    faces = np.array(faces).flatten()

    # Combine the points into a single array
    points = np.vstack([inner_points, outer_points])

    # Create a PolyData object
    ruled_surface = pv.PolyData(points, faces)
    
    return ruled_surface

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

def apply_delaunay_triangulation(common_points):
    """
    Apply Delaunay triangulation to common points and return the surface.
    """
    if len(common_points) < 3:
        print("Not enough points for Delaunay triangulation.")
        return None
    
    # Create PyVista PolyData from common points
    surface_points = pv.PolyData(common_points)
    
    # Apply Delaunay 2D triangulation
    delaunay_surf = surface_points.delaunay_2d()
    
    # Smooth the surface
    #smooth_surf = delaunay_surf.smooth(n_iter=50, relaxation_factor=0.5)
    
    return delaunay_surf

def sort_boundary_points(delaunay_surface):
    """
    Sort boundary points of the Delaunay surface to create an ordered sequence.
    Based on the reference code approach.
    """
    # Extract the boundary edges of the surface
    edges = delaunay_surface.extract_feature_edges(
        boundary_edges=True, 
        non_manifold_edges=False, 
        feature_edges=False, 
        manifold_edges=False
    )
    
    # Extract the boundary points from the boundary edges
    boundary_points = edges.points
    boundary_points_array = np.array(boundary_points)
    
    if len(boundary_points_array) == 0:
        print("No boundary points found.")
        return np.array([])
    
    edge_points_list = []
    boundary_points_sorted = []
    
    # Iterate over each edge (cell) in the edges
    for i in range(edges.n_cells):
        edge = edges.get_cell(i)  # Get the i-th edge
        edge_points = edge.points  # Get the points of the edge
        # Append the two points as a list
        edge_points_list.append(edge_points[:2])  # Take the first two points
    
    # Convert to lists for easier processing
    boundary_points_list = boundary_points_array.tolist()
    edge_points_list = [edge.tolist() for edge in edge_points_list]
    
    if len(boundary_points_list) == 0:
        return np.array([])
    
    # Start with the first point
    boundary_points_sorted.append(boundary_points_list[0])
    remaining_edges = edge_points_list.copy()
    
    # Iterate until all nodes are sorted
    while len(boundary_points_sorted) < len(boundary_points_list):
        last_node = boundary_points_sorted[-1]
        found_next = False
        
        for edge in remaining_edges[:]:
            if edge[0] == last_node and edge[1] not in boundary_points_sorted:
                boundary_points_sorted.append(edge[1])
                remaining_edges.remove(edge)
                found_next = True
                break
            elif edge[1] == last_node and edge[0] not in boundary_points_sorted:
                boundary_points_sorted.append(edge[0])
                remaining_edges.remove(edge)
                found_next = True
                break
        
        if not found_next:
            break
    
    # Close the loop by adding the first point at the end
    if len(boundary_points_sorted) > 0:
        boundary_points_sorted.append(boundary_points_sorted[0])
    
    return np.array(boundary_points_sorted)

def visualize_delaunay_and_boundary(delaunay_surface, boundary_points_sorted):
    """
    Visualize the Delaunay surface and sorted boundary points.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(delaunay_surface, color='cyan', show_edges=True, opacity=1, label='Delaunay Surface')
    
    if len(boundary_points_sorted) > 0:
        boundary_polydata = pv.PolyData(boundary_points_sorted)
        plotter.add_mesh(boundary_polydata, color='red', point_size=10, 
                        render_points_as_spheres=True, label='Sorted Boundary Points')
    
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Delaunay Surface with Sorted Boundary Points')
    plotter.show()

def visualize_final_ruled_surface(delaunay_surface, ruled_surface, boundary_points_sorted, expanded_points):
    """
    Visualize the final result with Delaunay surface and ruled surface.
    """
    plotter = pv.Plotter()
    
    # Add Delaunay surface
    plotter.add_mesh(delaunay_surface, color='cyan', opacity=1, show_edges=True, label='Delaunay Surface')
    
    # Add ruled surface
    plotter.add_mesh(ruled_surface, color='lightblue', opacity=1, show_edges=True, label='Ruled Surface')
    
    # Add boundary points
    if len(boundary_points_sorted) > 0:
        plotter.add_mesh(pv.PolyData(boundary_points_sorted), color='red', point_size=8, 
                        render_points_as_spheres=True, label='Boundary Points')
    
    # Add expanded points
    if len(expanded_points) > 0:
        plotter.add_mesh(pv.PolyData(expanded_points), color='magenta', point_size=8, 
                        render_points_as_spheres=True, label='Expanded Points')
    
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Complete Ruled Surface with Delaunay Base')
    plotter.show()

def find_common_points(vertices1, vertices2, tolerance=1e-6):
    """Find common points between two sets of vertices"""
    common_points = []
    for v1 in vertices1:
        distances = np.linalg.norm(vertices2 - v1, axis=1)
        if np.min(distances) < tolerance:
            common_points.append(v1)
    return np.array(common_points) if common_points else np.array([]).reshape(0, 3)

def visualize_combined_surface(combined_surface):
    plotter = pv.Plotter()
    plotter.add_mesh(combined_surface, color='orange', opacity=1, show_edges=True, label='Combined Surface')
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Combined Delaunay and Ruled Surface')
    plotter.show()


def pv_to_trimesh(pv_mesh):
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]
    return trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)


def trimesh_to_pyvista(tm):
    return pv.PolyData(tm.vertices, np.hstack([np.full((len(tm.faces), 1), 3), tm.faces]))

def combine_and_triangulate_surfaces(surface1, surface2):
    combined = surface1 + surface2
    triangulated = combined.triangulate()
    return triangulated


def main():
    file1 = r"merged_blue.stl"
    file2 = "merged_red.stl"
    file3 = r"/home/sumukhs-ubuntu/Desktop/silicone_mold/assets/stl/bunny.stl"

    mesh1 = pv.read(file1)
    centroid = mesh1.center
    bunny_mesh = pv.read(file3)

    v1 = load_stl_vertices(file1)
    v2 = load_stl_vertices(file2)
    common = find_common_points(v1, v2)
    print(f"Found {len(common)} common points.")
    if len(common) < 3:
        print("Not enough common points for processing.")
        return

    print("Applying Delaunay triangulation...")
    delaunay_surface = apply_delaunay_triangulation(common)
    if delaunay_surface is None:
        print("Failed to create Delaunay surface.")
        return

    print("Sorting boundary points...")
    boundary_pts = sort_boundary_points(delaunay_surface)
    print(f"Sorted {len(boundary_pts)} boundary points.")
    if len(boundary_pts) == 0:
        print("No boundary points found after sorting.")
        return
    visualize_delaunay_and_boundary(delaunay_surface, boundary_pts)

    print("Translating sorted points...")
    expanded_pts = expand_points_by_translation(boundary_pts, centroid, 500)

    print("Creating ruled surface...")
    try:
        ruled_surface = create_ruled_surface_mesh(boundary_pts, expanded_pts)
        delaunay_surface.save('delaunay_surface.vtk')
        ruled_surface.save('ruled_surface.vtk')
        print("Surfaces saved successfully.")
        visualize_final_ruled_surface(delaunay_surface,
                                      ruled_surface,
                                      boundary_pts,
                                      expanded_pts)
    except Exception as e:
        print(f"Error creating ruled surface: {e}")
        return

    combined_surface = delaunay_surface + ruled_surface
    combined_surface = combined_surface.triangulate()
    combined_surface.save('combined_parting_surface.stl')
    print("Combined surface saved successfully.")
    visualize_combined_surface(combined_surface)

    combined_surface = pv.PolyData(combined_surface)
    bunny_mesh = pv.PolyData(bunny_mesh)
    clipped = combined_surface.clip_surface(bunny_mesh, invert=False)
    visualize_combined_surface(clipped)

    print("Clipping combined surface with bunny mesh...")
    try:
        # Convert PyVista meshes to Trimesh
        tm_combined = pv_to_trimesh(combined_surface)
        tm_bunny = pv_to_trimesh(bunny_mesh)

        # Perform boolean difference
        result = difference([tm_bunny, tm_combined], engine='blender', check_volume=False)
        if result is None:
            print("Boolean operation failed. Check mesh watertightness.")
            return

        # Convert back to PyVista for visualization/export
        result_pv = trimesh_to_pyvista(result)
        result_pv.save("clipped_with_bunny.vtk")
        print("Clipped surface saved successfully.")
        visualize_combined_surface(result_pv)

    except Exception as e:
        print(f"Clipping with bunny mesh failed: {e}")

if __name__ == "__main__":
    main()
