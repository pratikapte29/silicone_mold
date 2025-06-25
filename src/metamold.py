"""
This script contains the implementation from the Metamolds:
Computational Design of Silicone Molds research paper.
"""

import trimesh
import numpy as np
import pyomo.environ as pyo
from ortools.linear_solver import pywraplp

mesh = trimesh.load("../assets/stl/cow.stl")
n = len(mesh.faces)


def fibonacci_sphere(k):
    indices = np.arange(0, k, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/k)          # latitude
    theta = np.pi * (1 + 5**0.5) * indices    # longitude (golden angle)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack((x, y, z), axis=1)


k = 32
directions = fibonacci_sphere(k)  # shape: (512, 3)
directions /= np.linalg.norm(directions, axis=1)[:, None]  # Normalize directions

face_normals = mesh.face_normals  # (n, 3)
m = np.zeros((n, k))
for i in range(n):
    for j in range(k):
        angle = np.dot(face_normals[i], directions[j])
        m[i, j] = 1.0 - abs(angle)  # Or any moldability-based penalty

adj = mesh.face_adjacency
I = [(int(u), int(v)) for u, v in adj]  # Pairs of adjacent face indices

model = pyo.ConcreteModel()
model.FACES = range(n)
model.DIRS = range(k)

model.b = pyo.Var(model.FACES, model.DIRS, within=pyo.Binary)
model.g = pyo.Var(model.DIRS, within=pyo.Binary)
λ = 0.2
μ = 0.5
T = 20  # Max directions allowed

# Objective
def objective_rule(model):
    data_term = sum(m[i][j] * model.b[i, j] for i in model.FACES for j in model.DIRS)
    smoothness_term = λ * sum((model.b[u, j] - model.b[v, j])**2
                              for (u, v) in I for j in model.DIRS)
    label_cost = μ * sum(model.g[j] for j in model.DIRS)
    return data_term + smoothness_term + label_cost

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


# Each face gets exactly one direction
def assign_one_label_per_face(model, i):
    return sum(model.b[i, j] for j in model.DIRS) == 1

model.assignment = pyo.Constraint(model.FACES, rule=assign_one_label_per_face)

# Link b[i,j] and g[j]
def link_b_and_g(model, i, j):
    return model.b[i, j] <= model.g[j]

model.label_usage = pyo.Constraint(model.FACES, model.DIRS, rule=link_b_and_g)

# g[j] = 1 if used, else 0
def label_existence_lower(model, j):
    return sum(model.b[i, j] for i in model.FACES) >= model.g[j]

def label_existence_upper(model, j):
    return sum(model.b[i, j] for i in model.FACES) <= n * model.g[j]

model.label_lower = pyo.Constraint(model.DIRS, rule=label_existence_lower)
model.label_upper = pyo.Constraint(model.DIRS, rule=label_existence_upper)

# Limit number of directions used
model.direction_limit = pyo.Constraint(expr=sum(model.g[j] for j in model.DIRS) <= T)
model.direction_min = pyo.Constraint(expr=sum(model.g[j] for j in model.DIRS) >= 2)


solver = pywraplp.Solver.CreateSolver('CBC')  # or glpk, gurobi, etc.
solver.Solve(model, tee=True)


face_labels = np.zeros(n, dtype=int)
for i in range(n):
    for j in range(k):
        if pyo.value(model.b[i, j]) > 0.5:
            face_labels[i] = j
            break

# Map face_labels to directions[j]
used_directions = [j for j in range(k) if pyo.value(model.g[j]) > 0.5]
print(f"{len(used_directions)} directions used.")


