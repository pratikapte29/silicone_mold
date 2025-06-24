import trimesh
import numpy as np
import random
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ortools.linear_solver import pywraplp

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
mij = [[1 - np.dot(face_normals[i], directions[j]) for j in range(k)] for i in range(n)]

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

    solver.Add(solver.Sum(gj) >= 2)
    solver.Add(solver.Sum(gj) <= T)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        result = [np.argmax([bij[i][j].solution_value() for j in range(k)]) for i in range(n)]
        return result
    else:
        raise RuntimeError("No optimal solution found.")

# Step 7: Run the segmentation
labels = segment_mesh_ILP(n, k, mij, I, Suv, lam=1.0, mu=2.0, T=5)
print(f"Segmented mesh into {k} parts with labels: {set(labels)}")

# Step 8: Visualize segmentation
colors = plt.cm.get_cmap('tab10', k)
face_colors = np.array([colors(label) for label in labels])
print(f"Assigned colors to {len(face_colors)} faces.")
mesh.visual.face_colors = (face_colors[:, :3] * 255).astype(np.uint8)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1.0 / mesh.scale)
mesh.show()