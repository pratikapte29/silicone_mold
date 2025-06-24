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


# Step 1: Load mesh
mesh = trimesh.load_mesh(r"../assets/stl/bunny.stl")
target_faces = 500
reduction_fraction = 1 - (target_faces / len(mesh.faces))
mesh = mesh.simplify_quadric_decimation(reduction_fraction)
print(mesh.faces.shape[0])

print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
n = len(mesh.faces)
face_normals = mesh.face_normals
print(f"Computed face normals for {n} faces.")


def compute_tubeness(mesh, max_radius=10):
    """
    Compute tubeness t(vi) for each vertex vi in the mesh.
    This is a simplified placeholder; real implementation would require
    topological analysis as described in the paper.
    """
    # Placeholder: use curvature as a proxy for tubeness
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, max_radius)
    t = 1 / np.log(1 + np.abs(curvature) + 1e-8)
    return t

def compute_visible_faces(mesh, direction, angle_threshold_deg=7):
    """
    Compute visible faces from a given direction using dot product as a proxy.
    For real applications, use GPU-accelerated rendering.
    """
    normals = mesh.face_normals
    direction = direction / np.linalg.norm(direction)
    visible = np.dot(normals, direction) > np.cos(np.radians(angle_threshold_deg))
    return visible

def shape_aware_geodesic(mesh, visible_faces, tubeness, direction):
    """
    Compute shape-aware geodesic distances from visible region boundary.
    """
    # Build a graph with edge weights using tubeness
    G = nx.Graph()
    for i, neighbors in enumerate(mesh.vertex_neighbors):
        for j in neighbors:
            vi, vj = mesh.vertices[i], mesh.vertices[j]
            l = np.linalg.norm(vi - vj)
            w = (tubeness[i] + tubeness[j]) / 2
            G.add_edge(i, j, weight=l * (1 + w))

    # Find visible vertices (vertices of visible faces)
    visible_vertices = np.unique(mesh.faces[visible_faces].flatten())
    # Compute geodesic distances from visible vertices to all others
    dist = np.full(len(mesh.vertices), np.inf)
    for v in visible_vertices:
        lengths = nx.single_source_dijkstra_path_length(G, v)
        for k, d in lengths.items():
            if d < dist[k]:
                dist[k] = d
    return dist

def compute_moldability_cost(mesh, directions):
    """
    Compute moldability cost matrix mij for all faces and directions.
    """
    tubeness = compute_tubeness(mesh)
    mij = np.zeros((len(mesh.faces), len(directions)))
    MAX_DIST = 20  # Cap for geodesic distance to avoid overflow
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

# Step 2: Sample k parting directions on sphere
def sample_sphere(k):
    vecs = []
    for _ in range(k):
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vecs.append([x, y, z])
    return np.array(vecs)

k = 8
directions = sample_sphere(k)
print(f"Sampled {k} parting directions on the sphere.")

# Step 3: Compute data cost mij (dot product based)
mij = compute_moldability_cost(mesh, directions)

# Step 4: Get adjacency list of faces
adj = mesh.face_adjacency
I = [(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in adj]
I = list(set(I))  # remove duplicates
print(f"Computed adjacency list with {len(I)} edges.")

# Step 5: Smoothing weights (use 1.0 or based on angle)
Suv = {}
for u, v in I:
    angle = np.dot(face_normals[u], face_normals[v])
    Suv[(u, v)] = 1 - angle  # encourage similar normals to have same label

# Step 6: Solve ILP (reuse from earlier)

def segment_mesh_ILP(n, k, mij, I, Suv, lam, mu, T):
    solver = pywraplp.Solver.CreateSolver("CBC")
    bij = [[solver.IntVar(0, 1, f'b_{i}_{j}') for j in range(k)] for i in range(n)]
    gj = [solver.IntVar(0, 1, f'g_{j}') for j in range(k)]

    objective = solver.Objective()
    for i in range(n):
        for j in range(k):
            objective.SetCoefficient(bij[i][j], mij[i][j])
    for (u, v) in I:
        for j in range(k):
            a, b = sorted((u, v))
            x = solver.IntVar(0, 1, f'xor_{a}_{b}_{j}')
            solver.Add(x >= bij[u][j] - bij[v][j])
            solver.Add(x >= bij[v][j] - bij[u][j])
            solver.Add(x <= bij[u][j] + bij[v][j])
            solver.Add(x <= 2 - bij[u][j] - bij[v][j])
            objective.SetCoefficient(x, lam * Suv[(u, v)])
    for j in range(k):
        objective.SetCoefficient(gj[j], mu)

    objective.SetMinimization()

    for i in range(n):
        solver.Add(solver.Sum(bij[i]) == 1)

    for j in range(k):
        sum_bij_j = solver.Sum([bij[i][j] for i in range(n)])
        solver.Add(n * gj[j] >= sum_bij_j)
        solver.Add(sum_bij_j >= gj[j])

    # solver.Add(solver.Sum(gj) >= 2)
    # solver.Add(solver.Sum(gj) <= T)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        result = [np.argmax([bij[i][j].solution_value() for j in range(k)]) for i in range(n)]
        return result
    else:
        raise RuntimeError("No optimal solution found.")

# Step 7: Run the segmentation

print("mij min:", np.min(mij), "max:", np.max(mij))
print("Any inf in mij?", np.any(np.isinf(mij)))
print("Any nan in mij?", np.any(np.isnan(mij)))
print("mij sample:", mij[:5, :5])
labels = segment_mesh_ILP(n, k, mij, I, Suv, lam=1.0, mu=0.0, T=k*2)
print(f"Segmented mesh into {k} parts with labels: {set(labels)}")

# Step 8: Visualize segmentation
colors = plt.cm.get_cmap('tab10', k)
face_colors = np.array([colors(label) for label in labels])
print(f"Assigned colors to {len(face_colors)} faces.")
mesh.visual.face_colors = (face_colors[:, :3] * 255).astype(np.uint8)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1.0 / mesh.scale)
mesh.show()