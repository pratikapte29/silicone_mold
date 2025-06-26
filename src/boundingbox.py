import numpy as np
from stl import mesh
import pyvista as pv
from scipy.spatial import KDTree
import trimesh
def load_stl_vertices(filename):
    """Load STL file and extract unique vertices"""
    stl_mesh = mesh.Mesh.from_file(filename)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    vertices_rounded = np.round(vertices, decimals=6)
    unique_vertices = np.unique(vertices_rounded, axis=0)
    return unique_vertices



def pv_to_trimesh(pv_mesh):
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # PyVista faces: [3, v0, v1, v2]
    return trimesh.Trimesh(vertices=pv_mesh.points, faces=faces, process=False)

def find_common_points(vertices1, vertices2, tolerance=1e-6):
    """Find common points between two sets of vertices"""
    common_points = []
    for v1 in vertices1:
        distances = np.linalg.norm(vertices2 - v1, axis=1)
        if np.min(distances) < tolerance:
            common_points.append(v1)
    return np.array(common_points) if common_points else np.array([]).reshape(0, 3)

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

def scale_points_outside_mesh(points, centroid, mesh_vertices, margin=1.05):
    """
    Scale each point outward from the centroid so that all are outside the mesh.
    margin: how much farther than the max mesh radius (e.g., 1.05 for 5% farther)
    """
    # Compute max distance from centroid to any mesh vertex
    all_distances = np.linalg.norm(mesh_vertices - centroid, axis=1)
    max_distance = np.max(all_distances)
    target_distance = max_distance * margin

    vectors = points - centroid
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    scales = target_distance / norms
    expanded_points = centroid + vectors * scales
    return expanded_points

# def shortest_distance_to_mesh(points, mesh):
#     tree = KDTree(mesh.points)
#     distances, index = tree.query(points, k=1)
#     nearest_point = mesh.points[index]
#     print(f"Nearest point on mesh to given points: {nearest_point}")
#     print(f"Distance to nearest point: {distances}")
#     return distances,nearest_point
def boutline_on_offset_surface(common_points, offset_pv_mesh):
    tmesh = pv_to_trimesh(offset_pv_mesh)
    closest_points, distances, triangle_id = trimesh.proximity.closest_point(tmesh, common_points)
    return distances, closest_points

def ray_intersections_with_offset(common_points, centroid, offset_pv_mesh, tol=1e-2):
    tmesh = pv_to_trimesh(offset_pv_mesh)
    origins = common_points
    directions = common_points - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1
    directions = directions / norms

    intersection_points = np.full_like(common_points, np.nan)
    used_locations = []

    for i, (origin, direction) in enumerate(zip(origins, directions)):
        # Cast ray in original direction
        locs, idx_ray, idx_tri = tmesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction],
            multiple_hits=False
        )
        if len(locs) > 0:
            # Check if this intersection is close to any previous one
            duplicate = any(np.linalg.norm(locs[0] - ul) < tol for ul in used_locations)
            if duplicate:
                # Flip direction and try again
                flipped_dir = -direction
                locs_flip, _, _ = tmesh.ray.intersects_location(
                    ray_origins=[origin],
                    ray_directions=[flipped_dir],
                    multiple_hits=False
                )
                if len(locs_flip) > 0:
                    intersection_points[i] = locs_flip[0]
                    used_locations.append(locs_flip[0])
                else:
                    intersection_points[i] = locs[0]
                    used_locations.append(locs[0])
            else:
                intersection_points[i] = locs[0]
                used_locations.append(locs[0])
        else:
            # Try flipping direction if no intersection
            flipped_dir = -direction
            locs_flip, _, _ = tmesh.ray.intersects_location(
                ray_origins=[origin],
                ray_directions=[flipped_dir],
                multiple_hits=False
            )
            if len(locs_flip) > 0:
                intersection_points[i] = locs_flip[0]
                used_locations.append(locs_flip[0])
    return intersection_points


def visualize_translated_points_and_meshes(vertices1, vertices2, common_points, x=100):
    """
    Visualize:
    - Red mesh (vertices1) translated by -x
    - Blue mesh (vertices2) translated by -x
    - Common points translated by -x
    - Common points translated by +x
    """
    import pyvista as pv
    translation_vector = np.array([x, 0, 0])

    # Translate
    vertices1_minus_x = vertices1 - translation_vector
    vertices2_minus_x = vertices2 - translation_vector
    common_points_minus_x = common_points - translation_vector
    common_points_plus_x = common_points + translation_vector

    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(vertices1_minus_x), color='red', point_size=5, render_points_as_spheres=True, label='Red STL -x')
    plotter.add_mesh(pv.PolyData(vertices2_minus_x), color='blue', point_size=5, render_points_as_spheres=True, label='Blue STL -x')
    plotter.add_mesh(pv.PolyData(common_points_minus_x), color='yellow', point_size=10, render_points_as_spheres=True, label='Common Points -x')
    plotter.add_mesh(pv.PolyData(common_points_plus_x), color='magenta', point_size=10, render_points_as_spheres=True, label='Common Points +x')
    plotter.add_legend()
    plotter.show()


def create_offset_surface(mesh, offset_distance):
    """
    Create an offset surface by moving each vertex along its normal.
    mesh: a PyVista PolyData mesh
    offset_distance: the distance to offset (positive = outward, negative = inward)
    Returns a new PyVista PolyData mesh.
    """
    mesh = mesh.copy()
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    normals = mesh.point_normals
    new_points = mesh.points + normals * offset_distance
    offset_mesh = mesh.copy()
    offset_mesh.points = new_points
    return offset_mesh

def main():
    # File paths
    file1 = "merged_red.stl"
    file2 = "merged_blue.stl"

    mesh1 = pv.read(r"/home/sumukhs-ubuntu/Desktop/silicone_mold/assets/stl/bunny.stl")
    centroid = mesh1.center

    offset_distance = 100  # or any value you want
    offset_mesh = create_offset_surface(mesh1, offset_distance)

    print(f"Centroid of the mesh: {centroid}")
    try:
        print("Loading STL files...")
        vertices1 = load_stl_vertices(file1)
        vertices2 = load_stl_vertices(file2)
        
        print(f"Red STL has {len(vertices1)} unique vertices")
        print(f"Blue STL has {len(vertices2)} unique vertices")
        
        print("Finding common points...")
        common_points = find_common_points(vertices1, vertices2)
        
        print(f"Found {len(common_points)} common points")
        
        if len(common_points) > 0:
            print("\nFirst few common points:")
            for i, point in enumerate(common_points[:5]):
                print(f"  {i+1}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
            if len(common_points) > 5:
                print(f"  ... and {len(common_points) - 5} more")
        
            # Scale points so all are outside the mesh
            # Scale points so all are outside the mesh (if you still want to use expanded_points for something else)
            expanded_points = scale_points_outside_mesh(common_points, centroid, vertices1, margin=1.05)
            print(f"Scaled {len(expanded_points)} points so they are outside the mesh.")

            # Find distances from original common points to offset mesh
            #distances, nearest_points = outline_on_offset_surface(common_points, offset_mesh)
            print("Shortest distances from common points to offset mesh:")
            #print(distances)
            # intersection_points = ray_intersections_with_offset(common_points, centroid, offset_mesh)
            # print("Intersection points (NaN if no intersection):")
            # print(intersection_points)

            #Get the index of smallest distance
            # min_idx = np.argmin(distances)
            # print(f"Index of nearest point: {min_idx}")
            # print(f"Nearest point on offset mesh: {nearest_points[min_idx]}")
            # print(f"Distance to nearest point: {distances[min_idx]}")

            # Visualize using PyVista (show both original and expanded points if you want)
            print("\nGenerating visualization...")
            visualize_with_pyvista(file1, file2, common_points)
            # Visualize using PyVista (show both original and expanded points)
            print("\nGenerating visualization...")
            visualize_with_pyvista(file1, file2, expanded_points)
        else:
            print("No common points found, skipping expansion and visualization.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find STL file. Make sure both files exist in the current directory.")
        print(f"Looking for: {file1} and {file2}")
    except Exception as e:
        print(f"Error processing STL files: {e}")
    # Visualize original and offset surfaces
    # Visualize bunny, offset surface, common points, and nearest points
    # Visualize bunny, offset surface, common points, and nearest points
    plotter = pv.Plotter()
    plotter.add_mesh(mesh1, color='red', opacity=0.5, label='Original Bunny')
    plotter.add_mesh(offset_mesh, color='green', opacity=0.4, label='Offset Surface')

    if len(common_points) > 0:
        plotter.add_mesh(pv.PolyData(common_points), color='yellow', point_size=10, render_points_as_spheres=True, label='Common Points')

    # if len(nearest_points) > 0:
    #     plotter.add_mesh(pv.PolyData(nearest_points), color='magenta', point_size=12, render_points_as_spheres=True, label='Nearest Points')

    plotter.add_legend()
    plotter.show()

if __name__ == "__main__":
    main()