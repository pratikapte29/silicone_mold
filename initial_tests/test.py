import trimesh
import numpy as np
import pyvista as pv
import time
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn import datasets
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from IPython.display import display, clear_output
from sklearn.decomposition import PCA

'''input is stl file and its convex hull, output is the delaunay surface - parting_surface.vtk'''

starttime = time.time()

'''PART 1: CREATE CANDIDATE DIRECTION VECTORS'''
# Load the STL file
mesh = trimesh.load(r"G:\Chrome downloads\xyzrgb-dragon-by-renato-tarabella\xyzrgb_dragon_90.stl")

# Specify the margin by which the bounding box should be expanded
margin = 5.0  # You can adjust this value as needed

# Compute the bounding box of the mesh
bounding_box = mesh.bounds
min_bound = bounding_box[0] - margin
max_bound = bounding_box[1] + margin

# Create the vertices of the expanded box
box_vertices = [
    [min_bound[0], min_bound[1], min_bound[2]],
    [min_bound[0], min_bound[1], max_bound[2]],
    [min_bound[0], max_bound[1], min_bound[2]],
    [min_bound[0], max_bound[1], max_bound[2]],
    [max_bound[0], min_bound[1], min_bound[2]],
    [max_bound[0], min_bound[1], max_bound[2]],
    [max_bound[0], max_bound[1], min_bound[2]],
    [max_bound[0], max_bound[1], max_bound[2]],
]

box_faces = [
    [0, 1, 3], [0, 3, 2],  # Bottom face
    [4, 5, 7], [4, 7, 6],  # Top face
    [0, 1, 5], [0, 5, 4],  # Front face
    [2, 3, 7], [2, 7, 6],  # Back face
    [0, 2, 6], [0, 6, 4],  # Left face
    [1, 3, 7], [1, 7, 5],  # Right face
]

mesh2 = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
mesh2.export("expanded_bounding_box.stl")

vertices = mesh.vertices
normals = mesh.face_normals
center = mesh.centroid
print(center)

# Number of vectors
num_vectors = 64
# Generate evenly distributed points on a sphere using the Fibonacci lattice
indices = np.arange(0, num_vectors, dtype=float) + 0.5
phi = np.arccos(1 - 2*indices/num_vectors)
theta = np.pi * (1 + 5**0.5) * indices

# Convert spherical coordinates to Cartesian coordinates
x = np.cos(theta) * np.sin(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(phi)

# Stack the coordinates into a 2D array
vectors = np.column_stack((x, y, z))

# Normalize the vectors to get unit vectors (not strictly necessary as they are already unit vectors)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

direction_vector_data = {}


def project_to_plane(points, vector):
    normal = vector
    return points - np.outer(np.dot(points, normal), normal)

'''PART 2: SELECT DIRECTION VECTORS AND SPLIT STL INTO 2 PARTS'''

# Function to compute 2D projection and convex hull area
def compute_projection_area(vertices, vector):
    # Project vertices onto the plane defined by the vector
    projected_points = project_to_plane(vertices, vector)
    # Take only x, y for 2D projection
    projected_points_2d = projected_points[:, :2]

    # Check if points are not degenerate
    if np.unique(projected_points_2d, axis=0).shape[0] > 2:
        # Compute the convex hull of the projected 2D points
        hull = ConvexHull(projected_points_2d)
        hull_points = projected_points_2d[hull.vertices]
        return hull_points, hull
    else:
        return np.array([]), None

def face_centroid(mesh, face_index):
    face_vertices = mesh.vertices[mesh.faces[face_index]]
    centroid = np.mean(face_vertices, axis=0)
    return centroid


for vector in vectors:
    # Compute the 2D projection area for each vector
    _, hull = compute_projection_area(mesh.vertices, vector)
    if hull is not None:
        area = hull.volume  # area for 2D convex hull
    else:
        area = 0
    direction_vector_data[tuple(vector)] = area

# Get the direction vectors
d1 = max(direction_vector_data, key=direction_vector_data.get)  # direction vector 1
max_value = direction_vector_data[d1]
print(d1)
d2 = -np.array(d1)  # direction vector 2
print(d2)

# Scale the vectors for better visibility
bounding_box = mesh.bounds
max_dimension = np.max(bounding_box[1] - bounding_box[0])

# Set the scale for the arrows to be 1/4th the max dimension
scale = max_dimension * 1.4

d1_scaled = np.array(d1) * scale
d2_scaled = np.array(d2) * scale

# Split faces based on dot product with direction vectors
dir1_faces = []
dir2_faces = []

# FIND CLOSEST 2 FACES ALIGNING WITH EACH DIRECTION VECTOR

d1 = d1 / np.linalg.norm(d1)
d2 = d2 / np.linalg.norm(d2)

# Find the face that is closest aligned to d1
dot_products1 = np.dot(mesh.face_normals, d1)
index1 = np.argmax(dot_products1)
closest_face_d1 = mesh.faces[index1]

# Find the face that is closest aligned to d2
dot_products2 = np.dot(mesh.face_normals, d2)
index2 = np.argmax(dot_products2)
closest_face_d2 = mesh.faces[index2]

# GROUP THE FACES ALONG TWO VECTORS BASED ON INCLINATION OF NORMAL

dir1_vertices = []
dir2_vertices = []
faces = mesh.faces
face_normals = mesh.face_normals
dot_products1 = np.dot(face_normals, d1)
dot_products2 = np.dot(face_normals, d2)
vertices = mesh.vertices

for index, face in enumerate(faces):
    # CHANGE HERE! baher ach calculate the dot product as an array
    if dot_products1[index] > 0:
        dir1_faces.append(face)
        dir1_vertices.extend(mesh.vertices[face])
    if dot_products2[index] > 0:
        dir2_faces.append(face)
        dir2_vertices.extend(mesh.vertices[face])

# Get the boundary vertices
vertices_set_1 = {tuple(row) for row in dir1_vertices}
vertices_set_2 = {tuple(row) for row in dir2_vertices}

common_set = vertices_set_1.intersection(vertices_set_2)

boundary_vertices = np.array(list(common_set))
# print(boundary_vertices)

# Convert faces to PyVista PolyData
def create_mesh(vertices, faces):
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(int)
    return pv.PolyData(vertices, faces)

dir1_mesh = create_mesh(vertices, np.array(dir1_faces))
dir2_mesh = create_mesh(vertices, np.array(dir2_faces))


########################################################################################################################
'''PART 3: AUTO-CLUSTER THE BOUNDARY VERTICES DBSCAN'''
# Standardize features for DBSCAN
scaler = StandardScaler()
boundary_vertices_scaled = scaler.fit_transform(boundary_vertices)

# DBSCAN clustering
eps = 0.4  # Maximum distance between two samples for them to be considered as in the same neighborhood
min_samples = 5  # The number of samples in a neighborhood for a point to be considered a core point

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(boundary_vertices_scaled)

# Assign points to clusters
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
print(f"Estimated number of clusters: {n_clusters}")

# Create arrays for each cluster
clusters = [boundary_vertices[labels == i] for i in range(n_clusters)]

# plotterDB = pv.Plotter()

colors = plt.cm.get_cmap('tab10', len(unique_labels))
for label in unique_labels:
    if label == -1:
        color = 'k'  # Black for noise
    else:
        color = colors(label)[:3]  # Convert RGBA to RGB
    points = boundary_vertices[labels == label]
    cloud = pv.PolyData(points)
    # plotterDB.add_mesh(cloud, color=color, point_size=10, render_points_as_spheres=True, label=f'Cluster {label}')

# plotterDB.add_legend()
# plotterDB.show()

'''PART 4: GET THE LARGEST CLUSTER AND ASSIGN THAT AS BOUNDARY VERTICES OF STL'''

plotter2 = pv.Plotter()

longest_cluster_index = max(range(n_clusters), key=lambda i: len(clusters[i]))
boundary_vertices = clusters[longest_cluster_index]

boundary_vertices = np.array(boundary_vertices)
# boundary_vertices = np.vstack([boundary_vertices, boundary_vertices[0]])

# Compute the convex hull
hull_scipy = ConvexHull(boundary_vertices)

# Extract the vertices and faces from the hull
hull_faces = hull_scipy.simplices

# Create a PyVista PolyData object from the convex hull faces
hull_polydata = pv.PolyData(boundary_vertices)

spline = pv.Spline(boundary_vertices, 100)

'''PART 5: COMPUTE DELAUNAY SURFACE OF INNER STL FILE (mesh)'''
surface_points = pv.PolyData(boundary_vertices)
surf = surface_points.delaunay_2d()

finalPlot = pv.Plotter()
finalPlot.add_mesh(surf, show_edges=True, color='cyan', edge_color='black', label='Delaunay Surface')
finalPlot.add_points(boundary_vertices, color='red', point_size=5, label='Boundary Vertices')
finalPlot.add_points(vertices, color='green', point_size=3, label='Vertices')

# Add legend
finalPlot.add_legend()

# Show the plot
finalPlot.show()

# Smoothen the delaunay surface

smooth_surf = surf.smooth(n_iter=50, relaxation_factor=0.5)

'''PART 6: ORDER THE BOUNDARY_POINTS ONE AFTER THE OTHER'''

# Extract the boundary edges of the surface
edges = smooth_surf.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)

# Extract the boundary points from the boundary edges
# boundary points are the parting surface ones, and boundary vertices are the mesh ones
boundary_points = edges.points
boundary_points_array = np.array(boundary_points)
# print(boundary_points_array)

edge_points_list = []
boundary_points_sorted = []  # sorted means points arranged one after other along the delaunay surface edge

# Iterate over each edge (cell) in the edges
for i in range(edges.n_cells):
    edge = edges.get_cell(i)  # Get the i-th edge
    edge_points = edge.points  # Get the points of the edge

    # Append the two points as a list
    edge_points_list.append(edge_points[:2])  # Take the first two points

# Convert the list to a NumPy array
edge_points_array = np.array(edge_points_list)
# print(edge_points_array)

boundary_points_array = boundary_points_array.tolist()
edge_points_array = edge_points_array.tolist()

boundary_points_sorted.append(boundary_points_array[0])  # add first point

remaining_edges = edge_points_array.copy()

# Iterate until all nodes are sorted
while len(boundary_points_sorted) < len(boundary_points):
    last_node = boundary_points_sorted[-1]
    for edge in remaining_edges[:]:
        if edge[0] == last_node and edge[1] not in boundary_points_sorted:
            boundary_points_sorted.append(edge[1])
            remaining_edges.remove(edge)
            break
        elif edge[1] == last_node and edge[0] not in boundary_points_sorted:
            boundary_points_sorted.append(edge[0])
            remaining_edges.remove(edge)
            break

print("length of sorted boundary points: ", len(boundary_points_sorted))
boundary_points_sorted.append(boundary_points_sorted[0])

# Create a PolyData object for the boundary points
boundary_points_polydata = pv.PolyData(boundary_points)

surface_plotter = pv.Plotter()
surface_plotter.add_mesh(smooth_surf, color='cyan', show_edges=True, opacity=0.5, label='Delaunay Surface')
surface_plotter.add_mesh(boundary_points_polydata, color='red', point_size=10, render_points_as_spheres=True, label='Boundary Points')
surface_plotter.add_legend()
surface_plotter.show()

########################################################################################################################
'''PART 7: PROJECT BOUNDARY POINTS OF DELAUNAY SURFACE ON THE EXTERNAL CONVEX HULL!'''

# Compute the centroid of the mesh
centroid = mesh.centroid
# print(centroid)

origins = np.tile(centroid, (len(boundary_points_sorted), 1))
# print(origins)

# since the imported mesh2 is in trimesh, create the mesh2 again in pyvista mesh format
# from the vertices of trimesh mesh2
outer_vertices = mesh2.vertices
outer_faces = mesh2.faces
mesh2 = create_mesh(outer_vertices, outer_faces)
# print("no. of boundary points of delaunay surface (unextended)", len(boundary_points))
vectors = boundary_points_sorted - origins
normalized_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

extended_points, ind_ray, ind_tri = mesh2.multi_ray_trace(origins, vectors, first_point=True)
rays = [pv.Line(o, v) for o, v in zip(origins, vectors)]

scaled_vectors = normalized_vectors * max_dimension * 0.25
extended_points = extended_points + scaled_vectors

intersections = pv.PolyData(extended_points)

'''PART 8: EXTEND THE PARTING SURFACE TILL THE OUTWARD MOLD BODY'''

combined_points = np.vstack((boundary_vertices, extended_points))
delaunay_points = pv.PolyData(combined_points)
finalSurf = delaunay_points.delaunay_2d()

smoothFinalSurf = finalSurf.smooth(n_iter=50, relaxation_factor=0.5)
smoothFinalSurf.save('delaunay_surface.vtk')

# Plotting to visualize
plotters = pv.Plotter()
plotters.add_mesh(smoothFinalSurf, show_edges=True, edge_color='black', color='cyan', opacity=1)
plotters.add_mesh(dir1_mesh, color='red', opacity=0.4, show_edges=True)
plotters.add_mesh(dir2_mesh, color='green', opacity=0.4, show_edges=True)
plotters.add_mesh(mesh2, color='yellow', opacity=0.4, show_edges=False)

plotters.add_mesh(intersections, color="maroon",
           point_size=10, label="Intersection Points")

# plotters.add_points(vertices, color='green', point_size=3, label='Vertices')
plotters.show()

'''PART 9: CREATE RULED SURFACE'''
outer_points = np.array(extended_points)
inner_points = np.array(boundary_points_sorted)

if len(outer_points) != len(inner_points):
    print(len(outer_points))  # extended_points 328
    print(len(inner_points))  # boundary_points 43
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

plotterFinal = pv.Plotter()
plotterFinal.add_mesh(extended_points, color="maroon", point_size=10, label="Intersection Points")
# plotterFinal.add_mesh(boundary_points_sorted, color="green", point_size=10, label="Inner Points")
plotterFinal.add_mesh(ruled_surface, color='lightblue', show_edges=True)
plotterFinal.show()

print(boundary_points_sorted)

boundary_points_sorted = np.array(boundary_points_sorted)

partingSurface = ruled_surface + smooth_surf

smooth_surf.save('unruled_parting_surface.vtk')
partingSurface.save('parting_surface.vtk')
ruled_surface.save('ruled_surface.vtk')
partingSurface.save('parting_surface.stl')

partingSurfacePlotter = pv.Plotter()
partingSurfacePlotter.add_mesh(partingSurface, color='lightblue', show_edges=True)
partingSurfacePlotter.add_mesh(dir1_mesh, color='red', opacity=1, show_edges=True)
partingSurfacePlotter.add_mesh(dir2_mesh, color='green', opacity=1, show_edges=True)
partingSurfacePlotter.add_mesh(mesh2, color='yellow', opacity=0.4, show_edges=False)
partingSurfacePlotter.show()

# plt.ion()  # Turn on interactive mode
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Set the axes limits to ensure all points are visible
# ax.set_xlim(np.min(boundary_points_sorted[:, 0]) - 10, np.max(boundary_points_sorted[:, 0]) + 10)
# ax.set_ylim(np.min(boundary_points_sorted[:, 1]) - 10, np.max(boundary_points_sorted[:, 1]) + 10)
# ax.set_zlim(np.min(boundary_points_sorted[:, 2]) - 10, np.max(boundary_points_sorted[:, 2]) + 10)
#
# # Plot points one by one
# for point in boundary_points_sorted:
#     ax.scatter(point[0], point[1], point[2], color='r', s=50)  # Plot each point
#     display(fig)  # Display the updated plot
#     plt.pause(0.5)  # Pause for half a second (adjust as needed)
#
# # Keep the plot displayed after the loop finishes
# plt.ioff()  # Turn off interactive mode
# plt.show()

# ! TESTING THE BELOW PART:

'''PART 10: SPLIT CONVEX HULL FACES BASED ON ALIGNMENT WITH DIRECTION VECTORS'''

print("\n" + "="*80)
print("SPLITTING CONVEX HULL BASED ON DIRECTION VECTORS")
print("="*80)

mesh2 = trimesh.load(r"G:\Chrome downloads\xyzrgb-dragon-by-renato-tarabella\xyzrgb_dragon_90_hull_scaled_tessellated.stl")

# Convert the trimesh convex hull (mesh2) to get face normals
# Note: mesh2 is already defined in the code as the expanded bounding box
convex_hull_vertices = mesh2.vertices
convex_hull_faces = mesh2.faces

# Create a pyvista mesh for better visualization
convex_hull_pv = create_mesh(convex_hull_vertices, convex_hull_faces)

# Calculate face normals for the convex hull
convex_hull_pv.compute_normals(cell_normals=True, point_normals=False, inplace=True)
convex_hull_normals = convex_hull_pv.cell_normals

# Normalize direction vectors again to be sure
d1_norm = d1 / np.linalg.norm(d1)
d2_norm = d2 / np.linalg.norm(d2)

print(f"Direction 1: {d1_norm}")
print(f"Direction 2: {d2_norm}")

# Initialize arrays to store faces belonging to each direction
d1_aligned_faces = []
d2_aligned_faces = []

# For each face, check if it's more aligned with d1 or d2
# We do this by comparing the dot product with each direction
for i, face in enumerate(convex_hull_faces):
    # Get the face normal
    normal = convex_hull_normals[i]

    # Calculate dot products with both directions
    # We use absolute values since we care about alignment regardless of direction
    dot_d1 = np.dot(normal, d1_norm)
    dot_d2 = np.dot(normal, d2_norm)

    print(f"Face {i}: Normal {normal}, Dot with d1: {dot_d1}, Dot with d2: {dot_d2}")

    # Assign to direction based on which has the larger dot product with the face normal
    if dot_d1 > dot_d2:
        d1_aligned_faces.append(face)
    else:
        d2_aligned_faces.append(face)

# Convert the face lists to numpy arrays
d1_aligned_faces = np.array(d1_aligned_faces) if d1_aligned_faces else np.empty((0, 3), dtype=int)
d2_aligned_faces = np.array(d2_aligned_faces) if d2_aligned_faces else np.empty((0, 3), dtype=int)

# Create separate meshes for visualization
d1_hull_mesh = create_mesh(convex_hull_vertices, d1_aligned_faces) if len(d1_aligned_faces) > 0 else None
d2_hull_mesh = create_mesh(convex_hull_vertices, d2_aligned_faces) if len(d2_aligned_faces) > 0 else None

# Print information about the split
print(f"Convex hull has {len(convex_hull_faces)} total faces")
print(f"  - {len(d1_aligned_faces)} faces aligned with direction 1 {d1_norm}")
print(f"  - {len(d2_aligned_faces)} faces aligned with direction 2 {d2_norm}")

# Create a plotter for visualization
hull_plotter = pv.Plotter()

# Add arrows to represent the direction vectors
startpoint = center
endpoint1 = center + d1_scaled
endpoint2 = center + d2_scaled
arrow1 = pv.Arrow(startpoint, endpoint1, tip_length=0.2, tip_radius=0.1, shaft_radius=0.05)
arrow2 = pv.Arrow(startpoint, endpoint2, tip_length=0.2, tip_radius=0.1, shaft_radius=0.05)

# Add the meshes and arrows to the plot
if d1_hull_mesh is not None:
    hull_plotter.add_mesh(d1_hull_mesh, color='red', opacity=1, show_edges=True, line_width=2,
                          label='Faces aligned with d1')
if d2_hull_mesh is not None:
    hull_plotter.add_mesh(d2_hull_mesh, color='blue', opacity=1, show_edges=True, line_width=2,
                          label='Faces aligned with d2')

hull_plotter.add_mesh(arrow1, color='darkred', label='Direction 1')
hull_plotter.add_mesh(arrow2, color='darkblue', label='Direction 2')

# Add original mesh for reference
# hull_plotter.add_mesh(mesh, color='lightgray', opacity=0.3, label='Original Mesh')

# Add legend and display the plot
hull_plotter.add_legend()
hull_plotter.add_title("Convex Hull Split by Direction Vectors")
hull_plotter.show()


