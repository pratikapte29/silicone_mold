import trimesh
import numpy as np
import random
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ortools.linear_solver import pywraplp
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Step 1: Load mesh
mesh = trimesh.load_mesh(r"../assets/stl/cow.stl")
target_faces = 500
reduction_fraction = 1 - (target_faces / len(mesh.faces))
mesh = mesh.simplify_quadric_decimation(reduction_fraction)
print(f"Mesh simplified to {mesh.faces.shape[0]} faces")

print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
n = len(mesh.faces)
face_normals = mesh.face_normals
print(f"Computed face normals for {n} faces.")


def compute_tubeness(mesh, max_radius=10):
    # Placeholder: use curvature as a proxy for tubeness
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, max_radius)
    t = 1 / np.log(1 + np.abs(curvature) + 1e-8)
    return t


def compute_visible_faces(mesh, direction, angle_threshold_deg=7):
    normals = mesh.face_normals
    direction = direction / np.linalg.norm(direction)
    visible = np.dot(normals, direction) > np.cos(np.radians(angle_threshold_deg))
    return visible


def shape_aware_geodesic(mesh, visible_faces, tubeness, direction):
    G = nx.Graph()
    for i, neighbors in enumerate(mesh.vertex_neighbors):
        for j in neighbors:
            vi, vj = mesh.vertices[i], mesh.vertices[j]
            l = np.linalg.norm(vi - vj)
            w = (tubeness[i] + tubeness[j]) / 2
            G.add_edge(i, j, weight=l * (1 + w))
    visible_vertices = np.unique(mesh.faces[visible_faces].flatten())
    dist = np.full(len(mesh.vertices), np.inf)
    for v in visible_vertices:
        lengths = nx.single_source_dijkstra_path_length(G, v)
        for k, d in lengths.items():
            if d < dist[k]:
                dist[k] = d
    return dist


def compute_moldability_cost(mesh, directions):
    tubeness = compute_tubeness(mesh)
    mij = np.zeros((len(mesh.faces), len(directions)))
    MAX_DIST = 20
    for j, dj in enumerate(directions):
        visible = compute_visible_faces(mesh, dj)
        dist = shape_aware_geodesic(mesh, visible, tubeness, dj)
        for i, face in enumerate(mesh.faces):
            if visible[i]:
                mij[i, j] = 0
            else:
                safe_dist = np.clip(dist[face], 0, MAX_DIST)
                mij[i, j] = np.mean(np.exp(safe_dist) - 1)
    mij[np.isinf(mij)] = 1e6
    mij = np.nan_to_num(mij, nan=1e6, posinf=1e6, neginf=0)
    return mij


def fibonacci_sphere(samples):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)


def anisotropic_clustering(mesh, num_clusters):
    """
    Perform anisotropic clustering of mesh faces based on principal curvatures
    """
    print(f"Computing anisotropic clustering with {num_clusters} clusters...")

    # Compute principal curvatures for each vertex
    vertex_curvatures = []
    for i, vertex in enumerate(mesh.vertices):
        try:
            # Get neighboring faces for this vertex
            vertex_faces = []
            for f_idx, face in enumerate(mesh.faces):
                if i in face:
                    vertex_faces.append(f_idx)

            if len(vertex_faces) > 0:
                # Use face normals to estimate curvature direction
                face_normals_at_vertex = mesh.face_normals[vertex_faces]
                if len(face_normals_at_vertex) > 1:
                    # Compute covariance of normals to get principal directions
                    cov = np.cov(face_normals_at_vertex.T)
                    eigenvals, eigenvecs = np.linalg.eig(cov)
                    # Sort by eigenvalue magnitude
                    idx = np.argsort(np.abs(eigenvals))[::-1]
                    principal_dir = eigenvecs[:, idx[0]]
                else:
                    principal_dir = face_normals_at_vertex[0]
            else:
                principal_dir = np.array([1, 0, 0])  # Default direction

            vertex_curvatures.append(principal_dir)
        except:
            vertex_curvatures.append(np.array([1, 0, 0]))  # Default direction

    vertex_curvatures = np.array(vertex_curvatures)

    # Deform mesh vertices according to principal curvature directions
    deformed_vertices = mesh.vertices.copy()
    scale_factor = 0.1  # Adjust deformation strength
    for i, (vertex, curvature_dir) in enumerate(zip(mesh.vertices, vertex_curvatures)):
        # Move vertex along principal curvature direction
        deformed_vertices[i] = vertex + scale_factor * curvature_dir

    # Compute face centroids in deformed space
    face_centroids_deformed = []
    for face in mesh.faces:
        centroid = np.mean(deformed_vertices[face], axis=0)
        face_centroids_deformed.append(centroid)
    face_centroids_deformed = np.array(face_centroids_deformed)

    # Cluster face centroids using K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(face_centroids_deformed)

    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
    return cluster_labels


def compute_cluster_moldability(face_moldability, cluster_labels, num_clusters):
    """
    Compute moldability for each cluster by summing face moldabilities
    """
    cluster_moldability = np.zeros((num_clusters, face_moldability.shape[1]))

    for cluster_id in range(num_clusters):
        cluster_faces = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_faces) > 0:
            cluster_moldability[cluster_id] = np.sum(face_moldability[cluster_faces], axis=0)

    return cluster_moldability


def segment_clusters_ILP(num_clusters, k, cluster_moldability, lam=0.1, mu=1.0, T=None):
    """
    Segment clusters using ILP (simplified version without smoothing term)
    """
    if T is None:
        T = k

    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("CBC solver unavailable")

    # Variables: b[i][j] = 1 if cluster i is assigned to direction j
    bij = [[solver.IntVar(0, 1, f'b_{i}_{j}') for j in range(k)] for i in range(num_clusters)]
    gj = [solver.IntVar(0, 1, f'g_{j}') for j in range(k)]

    # Objective: minimize sum of moldability costs
    objective = solver.Objective()
    for i in range(num_clusters):
        for j in range(k):
            objective.SetCoefficient(bij[i][j], cluster_moldability[i, j])

    # Label cost
    for j in range(k):
        objective.SetCoefficient(gj[j], mu)

    objective.SetMinimization()

    # Constraints: each cluster assigned to exactly one direction
    for i in range(num_clusters):
        solver.Add(solver.Sum(bij[i]) == 1)

    # Link direction usage with cluster assignments
    for j in range(k):
        sum_bij_j = solver.Sum([bij[i][j] for i in range(num_clusters)])
        solver.Add(num_clusters * gj[j] >= sum_bij_j)
        solver.Add(sum_bij_j >= gj[j])

    # Use at least 2 directions, at most T directions
    solver.Add(solver.Sum(gj) >= 2)
    solver.Add(solver.Sum(gj) <= T)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        cluster_labels = [np.argmax([bij[i][j].solution_value() for j in range(k)]) for i in range(num_clusters)]
        used_directions = [j for j in range(k) if gj[j].solution_value() > 0.5]
        return cluster_labels, used_directions
    else:
        raise RuntimeError("No optimal solution found for cluster segmentation.")


def segment_mesh_ILP(n, k, mij, I, Suv, lam, mu, T, allowed_directions=None):
    """
    Modified ILP to only consider specific allowed directions
    """
    if allowed_directions is None:
        allowed_directions = list(range(k))

    k_allowed = len(allowed_directions)

    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("CBC solver unavailable")

    bij = [[solver.IntVar(0, 1, f'b_{i}_{j}') for j in range(k_allowed)] for i in range(n)]
    gj = [solver.IntVar(0, 1, f'g_{j}') for j in range(k_allowed)]

    objective = solver.Objective()
    for i in range(n):
        for j_idx, j in enumerate(allowed_directions):
            objective.SetCoefficient(bij[i][j_idx], mij[i, j])

    for (u, v) in I:
        for j_idx, j in enumerate(allowed_directions):
            a, b = sorted((u, v))
            x = solver.IntVar(0, 1, f'xor_{a}_{b}_{j_idx}')
            solver.Add(x >= bij[u][j_idx] - bij[v][j_idx])
            solver.Add(x >= bij[v][j_idx] - bij[u][j_idx])
            solver.Add(x <= bij[u][j_idx] + bij[v][j_idx])
            solver.Add(x <= 2 - bij[u][j_idx] - bij[v][j_idx])
            if (u, v, j) in Suv:
                objective.SetCoefficient(x, lam * Suv[(u, v, j)])

    for j_idx in range(k_allowed):
        objective.SetCoefficient(gj[j_idx], mu)

    objective.SetMinimization()

    for i in range(n):
        solver.Add(solver.Sum(bij[i]) == 1)

    for j_idx in range(k_allowed):
        sum_bij_j = solver.Sum([bij[i][j_idx] for i in range(n)])
        solver.Add(n * gj[j_idx] >= sum_bij_j)
        solver.Add(sum_bij_j >= gj[j_idx])

    solver.Add(solver.Sum(gj) >= 2)
    solver.Add(solver.Sum(gj) <= min(T, k_allowed))

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        result = []
        for i in range(n):
            local_label = np.argmax([bij[i][j_idx].solution_value() for j_idx in range(k_allowed)])
            global_label = allowed_directions[local_label]
            result.append(global_label)
        return result
    else:
        raise RuntimeError("No optimal solution found.")


# Main execution starts here
k = 200
directions = fibonacci_sphere(k)
print(f"Sampled {k} parting directions on the sphere.")

# Step 2: Two-stage optimization
print("\n=== STAGE 1: Clustering and Coarse Optimization ===")

# Anisotropic clustering
num_clusters = min(50, n // 10)  # Use fewer clusters than faces
cluster_labels = anisotropic_clustering(mesh, num_clusters)

# Compute moldability for original faces
print("Computing moldability costs for all faces...")
mij = compute_moldability_cost(mesh, directions)

# Compute cluster moldability
print("Computing cluster moldability...")
cluster_moldability = compute_cluster_moldability(mij, cluster_labels, num_clusters)

# Segment clusters to find optimal directions
print("Segmenting clusters...")
cluster_segmentation, optimal_directions = segment_clusters_ILP(
    num_clusters, k, cluster_moldability, lam=0.1, mu=1.0, T=k
)

print(f"Stage 1 complete. Using {len(optimal_directions)} optimal directions: {optimal_directions}")

print("\n=== STAGE 2: Fine Optimization on Original Mesh ===")

# Get adjacency list of faces
adj = mesh.face_adjacency
I = [(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in adj]
I = list(set(I))
print(f"Computed adjacency list with {len(I)} edges.")

# Compute smoothing weights for optimal directions only
face_areas = mesh.area_faces
max_sq_diff = np.zeros(k)
for j in optimal_directions:
    diffs = []
    for (u, v) in I:
        diffs.append((mij[u, j] - mij[v, j]) ** 2)
    max_sq_diff[j] = max(diffs) if diffs else 1.0

Suv = {}
for (u, v) in I:
    Auv = face_areas[u] + face_areas[v]
    for j in optimal_directions:
        if max_sq_diff[j] > 0:
            Nuv = 1 - ((mij[u, j] - mij[v, j]) ** 2) / max_sq_diff[j]
        else:
            Nuv = 1
        Suv[(u, v, j)] = Auv * Nuv

# Compute label cost
mu_j = []
for j in optimal_directions:
    j_opp_idx = np.argmax([np.dot(directions[j], -directions[other]) for other in optimal_directions if other != j])
    j_opp = optimal_directions[j_opp_idx] if len(optimal_directions) > 1 else j
    mu_j_val = np.sum(np.maximum(mij[:, j] - mij[:, j_opp], 0))
    mu_j.append(mu_j_val)
mu = 1 / (2 * min(mu_j)) if min(mu_j) > 0 else 1.0

# Final segmentation on original mesh with optimal directions
print("Running final segmentation...")
labels = segment_mesh_ILP(n, k, mij, I, Suv, lam=0.1, mu=mu, T=len(optimal_directions),
                          allowed_directions=optimal_directions)

print(f"Final segmentation complete. Used directions: {set(labels)}")

# Post-process to create binary mold split like image (c)
print("\n=== CREATING BINARY MOLD SPLIT ===")


def create_binary_mold_split(labels, directions, optimal_directions):
    """
    Convert multi-part segmentation to binary mold split
    """
    unique_labels = list(set(labels))

    if len(unique_labels) == 2:
        # Already binary
        return labels
    elif len(unique_labels) > 2:
        # Find the two most opposite directions
        direction_pairs = []
        for i, dir1_idx in enumerate(unique_labels):
            for j, dir2_idx in enumerate(unique_labels):
                if i < j:
                    dir1 = directions[dir1_idx]
                    dir2 = directions[dir2_idx]
                    # Compute how opposite they are (closer to -1 is more opposite)
                    dot_product = np.dot(dir1, dir2)
                    direction_pairs.append((dot_product, dir1_idx, dir2_idx))

        # Sort by most opposite (most negative dot product)
        direction_pairs.sort(key=lambda x: x[0])
        most_opposite = direction_pairs[0]

        primary_dir = most_opposite[1]
        secondary_dir = most_opposite[2]

        print(f"Converting to binary split using directions {primary_dir} and {secondary_dir}")
        print(f"Dot product (opposites): {most_opposite[0]:.3f}")

        # Reassign all labels to either primary or secondary
        binary_labels = []
        for label in labels:
            if label == primary_dir:
                binary_labels.append(0)  # First mold half
            elif label == secondary_dir:
                binary_labels.append(1)  # Second mold half
            else:
                # Assign to the direction with lower moldability cost
                if mij[len(binary_labels), primary_dir] <= mij[len(binary_labels), secondary_dir]:
                    binary_labels.append(0)
                else:
                    binary_labels.append(1)

        return binary_labels, [primary_dir, secondary_dir]
    else:
        # Only one direction found, split based on geometry
        print("Only one direction found, splitting based on mesh geometry...")
        face_centroids = np.array([np.mean(mesh.vertices[face], axis=0) for face in mesh.faces])

        # Find the axis with maximum variation
        centroid_mean = np.mean(face_centroids, axis=0)
        variations = np.var(face_centroids, axis=0)
        split_axis = np.argmax(variations)

        # Split along this axis
        binary_labels = []
        for centroid in face_centroids:
            if centroid[split_axis] > centroid_mean[split_axis]:
                binary_labels.append(0)
            else:
                binary_labels.append(1)

        return binary_labels, [unique_labels[0], unique_labels[0]]


# Create binary split
if len(set(labels)) > 1:
    binary_labels, used_directions = create_binary_mold_split(labels, directions, optimal_directions)
else:
    # Fallback: split based on principal axis
    print("Single direction detected, creating geometric split...")
    face_centroids = np.array([np.mean(mesh.vertices[face], axis=0) for face in mesh.faces])
    centroid_mean = np.mean(face_centroids, axis=0)

    # Split along Z-axis (or axis with max variation)
    variations = np.var(face_centroids, axis=0)
    split_axis = np.argmax(variations)

    binary_labels = []
    for centroid in face_centroids:
        if centroid[split_axis] > centroid_mean[split_axis]:
            binary_labels.append(0)
        else:
            binary_labels.append(1)

    used_directions = [0, 1]

labels = binary_labels
print(f"Binary mold split complete. Two parts: {set(labels)}")

# Step 3: Visualize segmentation
print("\n=== VISUALIZATION ===")

# Create binary color scheme like image (c)
# Blue for first half, yellow/orange for second half
mold_colors = {
    0: np.array([0.3, 0.7, 1.0, 1.0]),  # Light blue (like image c)
    1: np.array([1.0, 0.7, 0.2, 1.0])  # Yellow/orange (like image c)
}

face_colors = np.array([mold_colors[label] for label in labels])
print(f"Created binary mold visualization with {len(set(labels))} parts.")

mesh.visual.face_colors = (face_colors[:, :3] * 255).astype(np.uint8)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1.0 / mesh.scale)

# Create scene with mesh and direction arrows
scene = mesh.scene()
centroid = mesh.centroid
arrow_length = mesh.scale * 0.5

# Only show arrows for the two main parting directions
primary_directions = []
if len(set(labels)) == 2:
    # Show arrows for the two mold halves
    if isinstance(used_directions, list) and len(used_directions) >= 2:
        primary_directions = used_directions[:2]
    else:
        # Create opposing directions for visualization
        primary_directions = [0, 1]

for i, label in enumerate([0, 1]):
    if i < len(primary_directions):
        j = primary_directions[i]
        if j < len(directions):
            dj = directions[j]
        else:
            # Default directions
            dj = np.array([0, 0, 1]) if i == 0 else np.array([0, 0, -1])
    else:
        dj = np.array([0, 0, 1]) if i == 0 else np.array([0, 0, -1])

    start = centroid
    end = centroid + dj * arrow_length

    # Create arrow geometry
    arrow_radius = arrow_length * 0.02
    arrow_head_length = arrow_length * 0.2

    # Arrow shaft
    shaft = trimesh.creation.cylinder(radius=arrow_radius, height=arrow_length)
    shaft.apply_translation([0, 0, arrow_length / 2])

    # Arrow head
    head = trimesh.creation.cone(radius=arrow_radius * 2, height=arrow_head_length)
    head.apply_translation([0, 0, arrow_length + arrow_head_length / 2])

    # Combine shaft and head
    arrow = shaft + head

    # Rotate arrow to point in direction dj
    if not np.allclose(dj, [0, 0, 1]):
        rotation_axis = np.cross([0, 0, 1], dj)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot([0, 0, 1], dj), -1, 1))
            rotation = R.from_rotvec(angle * rotation_axis)
            arrow.apply_transform(trimesh.transformations.rotation_matrix(
                angle=np.linalg.norm(rotation.as_rotvec()),
                direction=rotation.as_rotvec() / np.linalg.norm(rotation.as_rotvec()),
                point=[0, 0, 0]
            ))

    # Position arrow at centroid
    arrow.apply_translation(centroid)

    # Color arrow to match mold part
    arrow_color = mold_colors[label]
    arrow.visual.face_colors = (np.array(arrow_color[:3]) * 255).astype(np.uint8)

    scene.add_geometry(arrow)

print(f"Final result: Binary mold split into 2 parts (like image c)")
print(f"Blue part: {np.sum(np.array(labels) == 0)} faces")
print(f"Yellow part: {np.sum(np.array(labels) == 1)} faces")

# Show the scene
scene.show()