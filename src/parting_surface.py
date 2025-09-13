import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import ConvexHull
import trimesh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx


class InteractivePartingSurface:
    def __init__(self, mesh_path):
        """Initialize with STL file"""
        self.mesh = pv.read(mesh_path)
        self.original_mesh = self.mesh.copy()
        self.parting_plane_normal = None
        self.parting_plane_center = None
        self.boundary_points = []
        self.parting_surface = None
        self.plotter = None
        self.interactive_mode = False

    def find_optimal_parting_direction(self):
        """Find optimal parting direction using PCA and surface analysis"""
        points = self.mesh.points

        # Method 1: PCA-based approach
        pca = PCA(n_components=3)
        pca.fit(points)
        principal_directions = pca.components_

        # Method 2: Surface normal analysis
        normals = self.mesh.point_normals
        # Cluster normals to find dominant directions
        kmeans = KMeans(n_clusters=6, random_state=42)
        normal_clusters = kmeans.fit(normals)
        dominant_normals = normal_clusters.cluster_centers_

        # Method 3: Bounding box analysis
        bounds = np.array(self.mesh.bounds)
        dimensions = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])

        # Choose direction that minimizes cross-sectional complexity
        candidates = np.vstack([principal_directions, dominant_normals, np.eye(3)])

        best_direction = None
        min_complexity = float('inf')

        for direction in candidates:
            direction = direction / np.linalg.norm(direction)
            complexity = self._evaluate_parting_complexity(direction)
            if complexity < min_complexity:
                min_complexity = complexity
                best_direction = direction

        self.parting_plane_normal = best_direction
        self.parting_plane_center = np.mean(points, axis=0)

        return best_direction

    def _evaluate_parting_complexity(self, normal):
        """Evaluate complexity of parting in given direction"""
        points = self.mesh.points
        center = np.mean(points, axis=0)

        # Project points onto the plane perpendicular to normal
        distances = np.dot(points - center, normal)
        projected_points = points - np.outer(distances, normal)

        # Use 2D convex hull perimeter as complexity measure
        if len(projected_points) < 4:
            return float('inf')

        # Project to 2D
        u = np.cross(normal, [1, 0, 0])
        if np.allclose(u, 0):
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        points_2d = np.column_stack([
            np.dot(projected_points, u),
            np.dot(projected_points, v)
        ])

        try:
            hull = ConvexHull(points_2d)
            return hull.area  # Perimeter in 2D
        except:
            return float('inf')

    def create_fast_parting_surface(self):
        """Create initial parting surface using fast plane intersection"""
        if self.parting_plane_normal is None:
            self.find_optimal_parting_direction()

        # Create a large plane for intersection
        bounds = np.array(self.mesh.bounds)
        plane_size = np.linalg.norm(self.mesh.bounds[1::2] - self.mesh.bounds[::2]) * 2
        plane = pv.Plane(
            center=self.parting_plane_center,
            direction=self.parting_plane_normal,
            i_size=plane_size,
            j_size=plane_size
        )

        # Find intersection curve with mesh
        intersection = self.mesh.slice(normal=self.parting_plane_normal, origin=self.parting_plane_center)

        if intersection.n_points > 0:
            # Extract boundary points from intersection
            boundary_points = intersection.points
            self.boundary_points = self._extract_boundary_loop(boundary_points)

            # Create initial parting surface
            self.parting_surface = self._create_surface_from_boundary(self.boundary_points)

        return self.parting_surface

    def _extract_boundary_loop(self, points):
        """Extract main boundary loop from intersection points"""
        if len(points) < 3:
            return points

        # Build adjacency graph
        distances = cdist(points, points)
        # Connect each point to nearest neighbors
        k = min(4, len(points) - 1)

        graph = np.zeros_like(distances)
        for i in range(len(points)):
            # Find k nearest neighbors
            nearest_idx = np.argsort(distances[i])[1:k + 1]
            for j in nearest_idx:
                if distances[i, j] < np.mean(distances) * 0.5:  # Threshold
                    graph[i, j] = distances[i, j]
                    graph[j, i] = distances[i, j]

        # Find longest path (approximate boundary)
        G = nx.from_numpy_array(graph)
        if G.number_of_nodes() == 0:
            return points

        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)

        # Try to find a cycle or longest path
        if len(largest_cc) > 2:
            try:
                cycle = nx.find_cycle(subgraph)
                cycle_points = [points[edge[0]] for edge in cycle]
                return np.array(cycle_points)
            except:
                # Find longest path
                if subgraph.number_of_edges() > 0:
                    # Use traveling salesman approximation
                    nodes = list(largest_cc)
                    if len(nodes) > 2:
                        # Simple greedy approach for boundary ordering
                        ordered_nodes = self._order_boundary_points(points[nodes])
                        return points[np.array(nodes)[ordered_nodes]]

        return points

    def _order_boundary_points(self, points):
        """Order boundary points to form a closed loop"""
        if len(points) < 3:
            return list(range(len(points)))

        # Start with first point
        ordered = [0]
        remaining = set(range(1, len(points)))

        current = 0
        while remaining:
            # Find nearest unvisited point
            distances = [np.linalg.norm(points[current] - points[i]) for i in remaining]
            nearest_idx = min(remaining, key=lambda i: np.linalg.norm(points[current] - points[i]))
            ordered.append(nearest_idx)
            remaining.remove(nearest_idx)
            current = nearest_idx

        return ordered

    def _create_surface_from_boundary(self, boundary_points):
        """Create a surface from boundary points"""
        if len(boundary_points) < 3:
            return None

        # Create a simple triangulated surface
        # Use Delaunay triangulation or simple fan triangulation
        center = np.mean(boundary_points, axis=0)

        faces = []
        n_points = len(boundary_points)

        # Fan triangulation from center
        points = np.vstack([center, boundary_points])

        for i in range(n_points):
            next_i = (i + 1) % n_points
            faces.append([0, i + 1, next_i + 1])

        # Create mesh
        faces = np.array(faces)
        faces_with_size = np.column_stack([np.full(len(faces), 3), faces])

        surface = pv.PolyData(points, faces_with_size)
        return surface

    def refine_boundary_with_geodesics(self, selected_points=None):
        """Refine boundary using geodesic paths"""
        if selected_points is None or len(selected_points) < 2:
            return self.boundary_points

        # Create mesh graph for geodesic computation
        mesh_trimesh = trimesh.Trimesh(
            vertices=self.mesh.points,
            faces=self.mesh.faces.reshape(-1, 4)[:, 1:]  # Remove size column
        )

        # Find geodesic paths between selected points
        refined_boundary = []

        for i in range(len(selected_points)):
            start_point = selected_points[i]
            end_point = selected_points[(i + 1) % len(selected_points)]

            # Find closest vertices on mesh
            start_vertex = np.argmin(cdist([start_point], self.mesh.points)[0])
            end_vertex = np.argmin(cdist([end_point], self.mesh.points)[0])

            # Compute geodesic path
            path = self._compute_geodesic_path(mesh_trimesh, start_vertex, end_vertex)
            refined_boundary.extend(self.mesh.points[path[:-1]])  # Avoid duplicating end point

        return np.array(refined_boundary)

    def _compute_geodesic_path(self, mesh_trimesh, start_vertex, end_vertex):
        """Compute geodesic path between two vertices"""
        try:
            # Use trimesh's graph functionality
            vertex_adjacency = mesh_trimesh.vertex_adjacency_graph
            path = nx.shortest_path(vertex_adjacency, start_vertex, end_vertex, weight='weight')
            return path
        except:
            # Fallback: direct connection
            return [start_vertex, end_vertex]

    def start_interactive_editing(self):
        """Start interactive editing session"""
        self.plotter = pv.Plotter()

        # Add original mesh
        self.plotter.add_mesh(self.mesh, opacity=0.7, color='lightblue')

        # Add parting surface if it exists
        if self.parting_surface is not None:
            self.plotter.add_mesh(self.parting_surface, color='red', opacity=0.8)

        # Add boundary points if they exist
        if len(self.boundary_points) > 0:
            boundary_mesh = pv.PolyData(self.boundary_points)
            self.plotter.add_mesh(boundary_mesh, color='yellow', point_size=10, render_points_as_spheres=True)

        # Enable point picking
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            show_message="Click to select boundary points. Press 'r' to refine with geodesics.",
            color='red',
            point_size=15
        )

        # Add keyboard callbacks
        self.plotter.add_key_event('r', self._refine_callback)
        self.plotter.add_key_event('s', self._save_callback)
        self.plotter.add_key_event('p', self._recompute_parting_plane)

        self.selected_points = []
        self.interactive_mode = True

        print("Interactive Mode Controls:")
        print("- Click points to select boundary")
        print("- Press 'r' to refine boundary with geodesics")
        print("- Press 's' to save current parting surface")
        print("- Press 'p' to recompute parting plane")

        self.plotter.show()

    def _on_point_picked(self, point):
        """Handle point picking events"""
        self.selected_points.append(point)
        print(f"Selected point {len(self.selected_points)}: {point}")

        # Add visual marker
        marker = pv.Sphere(radius=0.01, center=point)
        self.plotter.add_mesh(marker, color='green')

    def _refine_callback(self):
        """Callback for refining boundary with geodesics"""
        if len(self.selected_points) >= 2:
            print("Refining boundary with geodesics...")
            refined_boundary = self.refine_boundary_with_geodesics(self.selected_points)

            # Update boundary points
            self.boundary_points = refined_boundary

            # Recreate parting surface
            self.parting_surface = self._create_surface_from_boundary(self.boundary_points)

            # Update visualization
            self.plotter.clear()
            self.plotter.add_mesh(self.mesh, opacity=0.7, color='lightblue')
            if self.parting_surface is not None:
                self.plotter.add_mesh(self.parting_surface, color='red', opacity=0.8)

            # Add refined boundary
            boundary_mesh = pv.PolyData(self.boundary_points)
            self.plotter.add_mesh(boundary_mesh, color='yellow', point_size=10, render_points_as_spheres=True)

            print("Boundary refined!")

    def _save_callback(self):
        """Callback for saving parting surface"""
        if self.parting_surface is not None:
            self.parting_surface.save('parting_surface.stl')
            print("Parting surface saved as 'parting_surface.stl'")

    def _recompute_parting_plane(self):
        """Callback for recomputing parting plane"""
        print("Recomputing parting plane...")
        self.find_optimal_parting_direction()
        self.create_fast_parting_surface()

        # Update visualization
        self.plotter.clear()
        self.plotter.add_mesh(self.mesh, opacity=0.7, color='lightblue')
        if self.parting_surface is not None:
            self.plotter.add_mesh(self.parting_surface, color='red', opacity=0.8)

        print("Parting plane recomputed!")

    def get_parting_line_2d_view(self):
        """Get 2D view of parting line for easier editing"""
        if self.parting_plane_normal is None:
            self.find_optimal_parting_direction()

        # Project mesh points onto parting plane
        points = self.mesh.points
        center = self.parting_plane_center
        normal = self.parting_plane_normal

        # Create 2D coordinate system on the plane
        u = np.cross(normal, [1, 0, 0])
        if np.allclose(u, 0):
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # Project all points
        relative_points = points - center
        u_coords = np.dot(relative_points, u)
        v_coords = np.dot(relative_points, v)

        return np.column_stack([u_coords, v_coords]), (u, v, center, normal)


# Usage example
def main():
    # Example usage
    stl_file = r"assets/stl/cow_fixed.stl"  # Replace with your STL file path

    # Create parting surface tool
    parting_tool = InteractivePartingSurface(stl_file)

    # Find optimal parting direction
    direction = parting_tool.find_optimal_parting_direction()
    print(f"Optimal parting direction: {direction}")

    # Create initial parting surface
    surface = parting_tool.create_fast_parting_surface()

    if surface is not None:
        print(f"Initial parting surface created with {surface.n_points} points")

        # Start interactive editing
        parting_tool.start_interactive_editing()
    else:
        print("Could not create initial parting surface")


if __name__ == "__main__":
    main()