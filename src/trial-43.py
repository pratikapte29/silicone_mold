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
mesh = trimesh.load_mesh(r"/home/sumukhs-ubuntu/Desktop/silicone_mold/assets/stl/bunny.stl")
target_faces = 500
reduction_fraction = 1 - (target_faces / len(mesh.faces))
mesh = mesh.simplify_quadric_decimation(reduction_fraction)
print(mesh.faces.shape[0])

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

# Step 3: Compute data cost mij (flow-based)
mij = compute_moldability_cost(mesh, directions)

# Step 4: Get adjacency list of faces
adj = mesh.face_adjacency
I = [(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in adj]
I = list(set(I))
print(f"Computed adjacency list with {len(I)} edges.")

# Step 5: Smoothing weights and label cost (from paper)
face_areas = mesh.area_faces
max_sq_diff = np.zeros(k)
for j in range(k):
    diffs = []
    for (u, v) in I:
        diffs.append((mij[u, j] - mij[v, j]) ** 2)
    max_sq_diff[j] = max(diffs) if diffs else 1.0

Suv = {}
for (u, v) in I:
    Auv = face_areas[u] + face_areas[v]
    for j in range(k):
        if max_sq_diff[j] > 0:
            Nuv = 1 - ((mij[u, j] - mij[v, j]) ** 2) / max_sq_diff[j]
        else:
            Nuv = 1
        Suv[(u, v, j)] = Auv * Nuv

mu_j = []
for j in range(k):
    j_opp = (j + k // 2) % k  # crude opposite, works if directions are paired
    mu_j_val = np.sum(np.maximum(mij[:, j] - mij[:, j_opp], 0))
    mu_j.append(mu_j_val)
mu = 1 / (2 * min(mu_j)) if min(mu_j) > 0 else 1.0

def segment_mesh_ILP(n, k, mij, I, Suv, lam, mu, T):
    solver = pywraplp.Solver.CreateSolver("CBC")
    bij = [[solver.IntVar(0, 1, f'b_{i}_{j}') for j in range(k)] for i in range(n)]
    gj = [solver.IntVar(0, 1, f'g_{j}') for j in range(k)]

    objective = solver.Objective()
    for i in range(n):
        for j in range(k):
            objective.SetCoefficient(bij[i][j], mij[i, j])
    for (u, v) in I:
        for j in range(k):
            a, b = sorted((u, v))
            x = solver.IntVar(0, 1, f'xor_{a}_{b}_{j}')
            solver.Add(x >= bij[u][j] - bij[v][j])
            solver.Add(x >= bij[v][j] - bij[u][j])
            solver.Add(x <= bij[u][j] + bij[v][j])
            solver.Add(x <= 2 - bij[u][j] - bij[v][j])
            objective.SetCoefficient(x, lam * Suv[(u, v, j)])
    for j in range(k):
        objective.SetCoefficient(gj[j], mu)

    objective.SetMinimization()

    for i in range(n):
        solver.Add(solver.Sum(bij[i]) == 1)

    for j in range(k):
        sum_bij_j = solver.Sum([bij[i][j] for i in range(n)])
        solver.Add(n * gj[j] >= sum_bij_j)
        solver.Add(sum_bij_j >= gj[j])

    solver.Add(solver.Sum(gj) >= 2)
    solver.Add(solver.Sum(gj) <= T)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        result = [np.argmax([bij[i][j].solution_value() for j in range(k)]) for i in range(n)]
        return result
    else:
        raise RuntimeError("No optimal solution found.")

print("mij min:", np.min(mij), "max:", np.max(mij))
print("Any inf in mij?", np.any(np.isinf(mij)))
print("Any nan in mij?", np.any(np.isnan(mij)))
print("mij sample:", mij[:5, :5])
labels = segment_mesh_ILP(n, k, mij, I, Suv, lam=0.1, mu=mu, T=k*2)
print(f"Segmented mesh into {k} parts with labels: {set(labels)}")

# Step 8: Visualize segmentation
colors = plt.cm.get_cmap('tab10', k)
face_colors = np.array([colors(label) for label in labels])
print(f"Assigned colors to {len(face_colors)} faces.")
mesh.visual.face_colors = (face_colors[:, :3] * 255).astype(np.uint8)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1.0 / mesh.scale)

# Visualize candidate vectors as arrows
scene = mesh.scene()
centroid = mesh.centroid
arrow_length = mesh.scale * 0.5  # Adjust as needed

for dj in directions:
    start = centroid
    end = centroid + dj * arrow_length
    path = trimesh.load_path(np.array([start, end]))
    scene.add_geometry(path)

scene.show()