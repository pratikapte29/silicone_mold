import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from finalize_draw_direction import FinalizeDrawDirection


class BoundaryBasedSegmentation:
    def __init__(self, stl_file_path):
        """
        Simple boundary-based mesh segmentation with PyVista visualization.

        Args:
            stl_file_path: Path to STL file
        """
        if isinstance(stl_file_path, str):
            self.mesh = trimesh.load(stl_file_path)
            print(f"Loaded mesh from file with {len(self.mesh.faces)} faces, {len(self.mesh.vertices)} vertices")
        else:
            # Handle case where a mesh object is passed directly
            self.mesh = stl_file_path
            print(f"Using provided mesh with {len(self.mesh.faces)} faces, {len(self.mesh.vertices)} vertices")

        # Convert to PyVista mesh for visualization
        self.pv_mesh = pv.PolyData(self.mesh.vertices, np.hstack([[3] + list(face) for face in self.mesh.faces]))

    def find_optimal_draw_direction(self, mesh_path, n_candidates=100):
        """
        Find optimal draw direction using visibility analysis.
        Returns the direction vector that minimizes undercuts.
        """
        print("Finding optimal draw direction...")

        fd = FinalizeDrawDirection(mesh_path, n_candidates)

        candidate_vectors = fd.createCandidateVectors()

        draw_direction = fd.computeVisibleAreas(candidate_vectors)
        return draw_direction

    def project_mesh_2d(self, draw_direction):
        """
        Project mesh vertices to 2D plane perpendicular to draw direction.
        """
        # Create orthonormal basis
        z_axis = draw_direction / np.linalg.norm(draw_direction)

        # Find a perpendicular vector
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Project vertices
        projected_2d = np.column_stack([
            np.dot(self.mesh.vertices, x_axis),
            np.dot(self.mesh.vertices, y_axis)
        ])

        return projected_2d, (x_axis, y_axis, z_axis)

    def find_visible_boundary_points(self, draw_direction, n_boundary_points=25):
        """
        Find boundary points visible from the draw direction.
        """
        print("Finding visible boundary points...")

        # Project to 2D
        projected_2d, (x_axis, y_axis, z_axis) = self.project_mesh_2d(draw_direction)

        # Find vertices that are visible from the draw direction
        vertex_visibility = np.zeros(len(self.mesh.vertices), dtype=bool)

        # For each vertex, check if any adjacent face is visible
        for face_idx, face in enumerate(self.mesh.faces):
            face_normal = self.mesh.face_normals[face_idx]
            if np.dot(face_normal, draw_direction) > 0:  # Face is visible
                vertex_visibility[face] = True

        # Get visible vertices
        visible_vertices = np.where(vertex_visibility)[0]
        visible_2d = projected_2d[visible_vertices]

        # Find convex hull (boundary) of projected points
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(visible_2d)
            boundary_indices = hull.vertices
            boundary_vertex_indices = visible_vertices[boundary_indices]
            boundary_2d = visible_2d[boundary_indices]
        except:
            print("Convex hull failed, using all visible vertices")
            boundary_vertex_indices = visible_vertices
            boundary_2d = visible_2d

        # Sample equally spaced points along boundary
        if len(boundary_2d) > n_boundary_points:
            # Compute cumulative distance along boundary
            boundary_closed = np.vstack([boundary_2d, boundary_2d[0]])
            distances = np.sqrt(np.sum(np.diff(boundary_closed, axis=0) ** 2, axis=1))
            cumulative_dist = np.cumsum(distances)
            total_length = cumulative_dist[-1]

            # Sample points at equal intervals
            target_distances = np.linspace(0, total_length, n_boundary_points, endpoint=False)
            sampled_indices = []

            for target_dist in target_distances:
                # Find closest point
                idx = np.argmin(np.abs(cumulative_dist - target_dist))
                if idx < len(boundary_vertex_indices):
                    sampled_indices.append(boundary_vertex_indices[idx])

            boundary_vertex_indices = np.array(sampled_indices)

        print(f"Found {len(boundary_vertex_indices)} boundary points")
        return boundary_vertex_indices, projected_2d, (x_axis, y_axis, z_axis)

    def visualize_from_draw_direction(self, draw_direction, boundary_points=None):
        """
        Visualize the mesh from the draw direction with highlighted boundary points using PyVista.
        """
        print("Creating PyVista visualization...")

        # Create plotter
        plotter = pv.Plotter(shape=(1, 3), title="Mesh Analysis from Draw Direction")

        # === Subplot 1: 3D view with draw direction ===
        plotter.subplot(0, 0)

        # Add main mesh
        plotter.add_mesh(self.pv_mesh, color='lightblue', opacity=0.8, show_edges=False)

        # Add draw direction as arrow
        center = self.mesh.centroid
        arrow_length = np.max(self.mesh.extents) * 0.4
        arrow_end = center + draw_direction * arrow_length

        # Create arrow
        arrow = pv.Arrow(start=center, direction=draw_direction, scale=arrow_length)
        plotter.add_mesh(arrow, color='red', label='Draw Direction')

        # Highlight boundary points if provided
        if boundary_points is not None:
            boundary_coords = self.mesh.vertices[boundary_points]
            boundary_polydata = pv.PolyData(boundary_coords)
            plotter.add_mesh(boundary_polydata, color='red', point_size=10,
                             render_points_as_spheres=True, label='Boundary Points')

        plotter.add_text("3D View", font_size=12)
        plotter.show_axes()

        # === Subplot 2: View from draw direction (2D projection) ===
        plotter.subplot(0, 1)

        # Project mesh to 2D plane
        projected_2d, (x_axis, y_axis, z_axis) = self.project_mesh_2d(draw_direction)

        # Create 2D points in 3D space (z=0 plane)
        projected_3d = np.column_stack([projected_2d, np.zeros(len(projected_2d))])
        projected_polydata = pv.PolyData(projected_3d)

        plotter.add_mesh(projected_polydata, color='lightblue', point_size=2,
                         render_points_as_spheres=True, opacity=0.6, label='Projected vertices')

        # Add boundary points in 2D
        if boundary_points is not None:
            boundary_2d = projected_2d[boundary_points]
            boundary_3d = np.column_stack([boundary_2d, np.zeros(len(boundary_2d))])
            boundary_2d_polydata = pv.PolyData(boundary_3d)
            plotter.add_mesh(boundary_2d_polydata, color='red', point_size=8,
                             render_points_as_spheres=True, label='Boundary points')

            # Connect boundary points to show outline
            if len(boundary_2d) > 2:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(boundary_2d)

                    # Create lines for hull edges
                    for simplex in hull.simplices:
                        p1 = np.append(boundary_2d[simplex[0]], 0)
                        p2 = np.append(boundary_2d[simplex[1]], 0)
                        line = pv.Line(p1, p2)
                        plotter.add_mesh(line, color='red', line_width=3, opacity=0.8)
                except:
                    pass

        plotter.add_text("2D Projection", font_size=12)
        plotter.view_xy()  # Look down at XY plane

        # === Subplot 3: Detailed boundary view ===
        plotter.subplot(0, 2)

        if boundary_points is not None:
            # Show only boundary region in detail
            boundary_coords = self.mesh.vertices[boundary_points]
            boundary_mesh = pv.PolyData(boundary_coords)

            plotter.add_mesh(boundary_mesh, color='red', point_size=12,
                             render_points_as_spheres=True, label='Boundary points')

            # Add point labels
            for i, point in enumerate(boundary_coords):
                plotter.add_point_labels([point], [str(i)], font_size=20, text_color='black')

            # Add original mesh with transparency
            plotter.add_mesh(self.pv_mesh, color='lightblue', opacity=0.3, show_edges=False)

        else:
            plotter.add_mesh(self.pv_mesh, color='lightblue', opacity=0.8)

        plotter.add_text("Boundary Points Detail", font_size=12)

        # Show the plot
        plotter.show()

    def create_draw_direction_view(self, draw_direction, boundary_points=None):
        """
        Create a single view looking along the draw direction.
        """
        plotter = pv.Plotter(title="View Along Draw Direction")

        # Project all vertices to view plane
        projected_2d, (x_axis, y_axis, z_axis) = self.project_mesh_2d(draw_direction)

        # Create the view by positioning camera along draw direction
        # Set up camera to look along draw direction
        center = self.mesh.centroid
        camera_pos = center - draw_direction * np.max(self.mesh.extents) * 2

        # Add main mesh
        plotter.add_mesh(self.pv_mesh, color='lightblue', opacity=0.7, show_edges=True)

        # Add boundary points
        if boundary_points is not None:
            boundary_coords = self.mesh.vertices[boundary_points]
            boundary_mesh = pv.PolyData(boundary_coords)
            plotter.add_mesh(boundary_mesh, color='red', point_size=15,
                             render_points_as_spheres=True, label='Boundary Points')

            # Add labels
            for i, point in enumerate(boundary_coords):
                plotter.add_point_labels([point], [str(i)], font_size=16, text_color='white')

        # Set camera position to look along draw direction
        plotter.camera_position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.view_up = [0, 0, 1] if abs(draw_direction[2]) < 0.9 else [1, 0, 0]

        plotter.add_text(f"View along draw direction: {draw_direction}", position='upper_left', font_size=12)
        plotter.show()

    def analyze_from_draw_direction(self, mesh_path, draw_direction=None, n_boundary_points=25):
        """
        Complete analysis: find draw direction, identify boundary points, and visualize.
        """
        if draw_direction is None:
            draw_direction = self.find_optimal_draw_direction(mesh_path, n_candidates=100)

        # Find boundary points
        boundary_points, projected_2d, axes = self.find_visible_boundary_points(
            draw_direction, n_boundary_points
        )

        # Create comprehensive visualization
        self.visualize_from_draw_direction(draw_direction, boundary_points)

        # Also create dedicated draw direction view
        self.create_draw_direction_view(draw_direction, boundary_points)

        return draw_direction, boundary_points


# Example usage
def main():
    # Load your STL file
    stl_file = r"assets/stl/cow_fixed.stl"  # Replace with your file path

    try:
        segmenter = BoundaryBasedSegmentation(stl_file)

        # Option 1: Let it find the optimal draw direction
        draw_direction, boundary_points = segmenter.analyze_from_draw_direction(mesh_path=stl_file, n_boundary_points=100)

        # Option 2: Use your own draw direction if you already have it
        # custom_direction = np.array([0, 0, 1])  # Example: straight up
        # draw_direction, boundary_points = segmenter.analyze_from_draw_direction(
        #     draw_direction=custom_direction, n_boundary_points=25
        # )

        print(f"Draw direction: {draw_direction}")
        print(f"Boundary points (vertex indices): {boundary_points}")

        # The boundary points are now ready for geodesic connection
        return segmenter, draw_direction, boundary_points

    except Exception as e:
        print(f"Error loading file: {e}")
        print("Make sure to update the stl_file path to your actual file")

        # Create a simple test mesh for demonstration
        print("Creating a test mesh for demonstration...")
        mesh = trimesh.creation.box(extents=[2, 1, 0.5])
        segmenter = BoundaryBasedSegmentation(mesh)
        draw_direction, boundary_points = segmenter.analyze_from_draw_direction(n_boundary_points=25)
        return segmenter, draw_direction, boundary_points


if __name__ == "__main__":
    segmenter, draw_direction, boundary_points = main()