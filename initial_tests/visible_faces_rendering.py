import numpy as np
import trimesh
import pyrender
import pyvista as pv
from scipy.spatial import ConvexHull, KDTree
import matplotlib.pyplot as plt
import os
from PIL import Image
import time


class PartingDirectionFinder:
    def __init__(self, mesh_path, num_candidates=64, render_resolution=512):
        """
        Initialize the parting direction finder using GPU-accelerated rendering.

        Parameters:
        -----------
        mesh_path : str
            Path to the mesh file (.stl, .obj, etc.)
        num_candidates : int
            Number of candidate directions to evaluate
        render_resolution : int
            Resolution of the rendered images (higher = more accurate but slower)
        """
        self.mesh = trimesh.load(mesh_path)
        self.num_candidates = num_candidates
        self.render_resolution = render_resolution
        self.center = self.mesh.centroid

        # Calculate the scale of the mesh for proper camera positioning
        self.scale = np.max(np.linalg.norm(self.mesh.vertices - self.center, axis=1))

        # Generate candidate directions
        self.directions = self._generate_fibonacci_sphere_directions()

        # Store results
        self.visibility_results = {}
        self.invisible_area_fractions = {}

    def _generate_fibonacci_sphere_directions(self):
        """Generate evenly distributed points on a sphere using the Fibonacci lattice."""
        indices = np.arange(0, self.num_candidates, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.num_candidates)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        # Convert spherical coordinates to Cartesian coordinates
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        # Stack the coordinates into a 2D array
        vectors = np.column_stack((x, y, z))

        # Normalize the vectors
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _compute_face_areas(self):
        """Compute the area of each face in the mesh."""
        areas = np.zeros(len(self.mesh.faces))

        for i, face in enumerate(self.mesh.faces):
            # Get the vertices of this face
            vertices = self.mesh.vertices[face]

            # Compute two edges of the triangle
            v0 = vertices[1] - vertices[0]
            v1 = vertices[2] - vertices[0]

            # Compute area using cross product
            area = 0.5 * np.linalg.norm(np.cross(v0, v1))
            areas[i] = area

        return areas

    def _compute_face_visibility(self, direction, depth_buffer):
        """
        Compute which faces are visible from a particular direction.

        Parameters:
        -----------
        direction : ndarray
            Direction vector
        depth_buffer : ndarray
            Depth buffer from the rendered image

        Returns:
        --------
        ndarray
            Boolean array marking which faces are visible
        """
        # Create a PyMesh renderer for better triangle visibility
        renderer = pyrender.OffscreenRenderer(self.render_resolution, self.render_resolution)

        # Create a scene
        scene = pyrender.Scene()

        # Add the mesh with unique colors for each face
        mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        scene.add(mesh)

        # Create a camera that looks from the direction toward the center
        camera_pos = self.center + direction * (self.scale * 2.5)

        # Set up camera looking toward center
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        # Create view matrix - camera looks at center from direction
        z_axis = -direction / np.linalg.norm(direction)  # Camera looks toward center
        world_up = np.array([0, 0, 1])  # World up direction

        # If direction is aligned with world up, use a different reference
        if np.abs(np.dot(z_axis, world_up)) > 0.999:
            world_up = np.array([0, 1, 0])

        # Camera axes
        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Create camera transformation matrix
        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = camera_pos

        # Add camera to scene
        scene.add(camera, pose=pose)

        # Render for visibility
        color, depth = renderer.render(scene)

        # Add a new mesh to the scene with face IDs encoded in colors
        color_mesh = pyrender.Mesh.from_trimesh(
            self.mesh,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.5
            ),
            wireframe=False
        )

        # Remove previous mesh and add the ID-encoded mesh
        scene.clear()
        scene.add(color_mesh)
        scene.add(camera, pose=pose)

        # Render the ID-encoded scene
        # Instead of standard rendering, we'll use a special vertex shader that bakes face IDs into colors
        # This technique is common in GPU picking implementations

        # For our implementation, we'll use a simpler approach:
        # We'll consider a face visible if any of its vertices are visible

        # Project all vertices to screen space
        vertices = self.mesh.vertices

        # Create 4x4 perspective projection matrix
        # Simplified version for this example
        near = 0.1
        far = 100.0
        aspect = 1.0  # Square viewport
        fov = np.pi / 3.0
        f = 1.0 / np.tan(fov / 2)

        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0]
        ])

        # View-projection matrix
        view_proj = projection @ np.linalg.inv(pose)

        # Transform vertices to clip space
        vertices_homogeneous = np.hstack((vertices, np.ones((len(vertices), 1))))
        clip_coords = vertices_homogeneous @ view_proj.T

        # Perspective division to get NDC coordinates
        ndc_coords = clip_coords[:, :3] / clip_coords[:, 3:4]

        # Convert to screen coordinates
        screen_x = (ndc_coords[:, 0] + 1) * self.render_resolution / 2
        screen_y = (1 - ndc_coords[:, 1]) * self.render_resolution / 2

        # Visibility check
        visible_faces = np.zeros(len(self.mesh.faces), dtype=bool)

        for i, face in enumerate(self.mesh.faces):
            # Get the face normal
            normal = self.mesh.face_normals[i]

            # Back-face culling: check if face is facing away from camera
            if np.dot(normal, direction) >= 0:
                # Face is facing away from camera, mark as invisible
                visible_faces[i] = False
                continue

            # Check if any vertex of the face is visible
            for vertex_idx in face:
                x, y = int(screen_x[vertex_idx]), int(screen_y[vertex_idx])

                # Check if point is within screen bounds
                if (0 <= x < self.render_resolution and 0 <= y < self.render_resolution):
                    # Check if point is not occluded
                    # This is a simplified occlusion check
                    # A proper implementation would use depth testing
                    if depth[y, x] > 0:
                        visible_faces[i] = True
                        break

        # Clean up
        renderer.delete()

        return visible_faces

    def compute_visibility_for_all_directions(self):
        """Compute visibility for all candidate directions."""
        print("Computing face areas...")
        face_areas = self._compute_face_areas()
        total_area = np.sum(face_areas)

        print(f"Computing visibility for {self.num_candidates} directions...")
        start_time = time.time()

        # Create renderer
        r = pyrender.OffscreenRenderer(self.render_resolution, self.render_resolution)

        for i, direction in enumerate(self.directions):
            # Create a scene
            scene = pyrender.Scene()

            # Add the mesh
            mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
            scene.add(mesh)

            # Set up camera
            camera_pos = self.center + direction * (self.scale * 2.5)
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

            # Create view matrix
            z_axis = -direction / np.linalg.norm(direction)
            world_up = np.array([0, 0, 1])

            if np.abs(np.dot(z_axis, world_up)) > 0.999:
                world_up = np.array([0, 1, 0])

            x_axis = np.cross(world_up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            pose = np.eye(4)
            pose[:3, 0] = x_axis
            pose[:3, 1] = y_axis
            pose[:3, 2] = z_axis
            pose[:3, 3] = camera_pos

            # Add camera to scene
            scene.add(camera, pose=pose)

            # Render depth image
            _, depth = r.render(scene)

            # Compute which faces are visible
            visible = self._compute_face_visibility(direction, depth)

            # Calculate visible and invisible areas
            visible_area = np.sum(face_areas[visible])
            invisible_area = total_area - visible_area
            invisible_fraction = invisible_area / total_area

            # Store results
            self.visibility_results[tuple(direction)] = {
                'visible_faces': visible,
                'visible_area': visible_area,
                'invisible_area': invisible_area,
                'total_area': total_area,
                'invisible_fraction': invisible_fraction
            }

            self.invisible_area_fractions[tuple(direction)] = invisible_fraction

            # Print progress
            if (i + 1) % 10 == 0 or i == len(self.directions) - 1:
                elapsed = time.time() - start_time
                print(f"Direction {i + 1}/{self.num_candidates} - "
                      f"Visible: {visible_area:.2f}, "
                      f"Invisible: {invisible_area:.2f} ({invisible_fraction:.2%}), "
                      f"Elapsed: {elapsed:.1f}s")

        # Clean up
        r.delete()

        return self.invisible_area_fractions

    def get_best_parting_directions(self):
        """
        Find the two directions that minimize non-visible surface area.
        According to the paper description: "we simply select as the
        parting directions the two directions d1, d2 which minimize
        the non-visible surface area."

        Returns:
        --------
        tuple
            (d1, d2) - The best parting directions
        """
        if not self.invisible_area_fractions:
            self.compute_visibility_for_all_directions()

        # Sort directions by invisible area fraction (ascending)
        sorted_directions = sorted(self.invisible_area_fractions.items(),
                                   key=lambda x: x[1])

        # Select the two directions with minimum invisible area
        d1 = np.array(sorted_directions[0][0])
        d2 = np.array(sorted_directions[1][0])

        invisible_d1 = sorted_directions[0][1]
        invisible_d2 = sorted_directions[1][1]

        print(f"Selected direction 1: {d1} with {invisible_d1:.2%} invisible area")
        print(f"Selected direction 2: {d2} with {invisible_d2:.2%} invisible area")

        return d1, d2

    def visualize_directions(self, d1, d2):
        """Visualize the selected directions."""
        # Create a plotter
        plotter = pv.Plotter()

        # Convert trimesh to pyvista
        vertices = self.mesh.vertices
        faces = np.hstack((np.ones((len(self.mesh.faces), 1), dtype=np.int64) * 3,
                           self.mesh.faces)).flatten()
        pv_mesh = pv.PolyData(vertices, faces)

        # Add mesh to plotter
        plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.7)

        # Add direction arrows
        start_point = self.center
        arrow_length = self.scale * 1.5

        # Direction 1
        end_point1 = start_point + d1 * arrow_length
        arrow1 = pv.Arrow(start_point, end_point1, tip_length=0.2,
                          tip_radius=0.1, shaft_radius=0.05)
        plotter.add_mesh(arrow1, color='red', label='Direction 1')

        # Direction 2
        end_point2 = start_point + d2 * arrow_length
        arrow2 = pv.Arrow(start_point, end_point2, tip_length=0.2,
                          tip_radius=0.1, shaft_radius=0.05)
        plotter.add_mesh(arrow2, color='blue', label='Direction 2')

        # Add legend
        plotter.add_legend()
        plotter.add_title("Selected Parting Directions")

        # Show the plot
        plotter.show()

    def visualize_visible_areas(self, direction):
        """
        Visualize the visible and non-visible areas from a specific direction.

        Parameters:
        -----------
        direction : ndarray
            Direction to visualize
        """
        if tuple(direction) not in self.visibility_results:
            print("Direction not found in results. Computing visibility...")
            self.compute_visibility_for_all_directions()

        # Get visibility information
        result = self.visibility_results[tuple(direction)]
        visible_faces = result['visible_faces']

        # Create separate meshes for visible and invisible parts
        visible_indices = np.where(visible_faces)[0]
        invisible_indices = np.where(~visible_faces)[0]

        # Create face arrays
        visible_faces_array = self.mesh.faces[visible_indices]
        invisible_faces_array = self.mesh.faces[invisible_indices]

        # Create PyVista meshes
        pv_visible = pv.PolyData(
            self.mesh.vertices,
            np.hstack((np.ones((len(visible_faces_array), 1), dtype=np.int64) * 3,
                       visible_faces_array)).flatten()
        ) if len(visible_indices) > 0 else None

        pv_invisible = pv.PolyData(
            self.mesh.vertices,
            np.hstack((np.ones((len(invisible_faces_array), 1), dtype=np.int64) * 3,
                       invisible_faces_array)).flatten()
        ) if len(invisible_indices) > 0 else None

        # Create plotter
        plotter = pv.Plotter()

        # Add meshes
        if pv_visible is not None:
            plotter.add_mesh(pv_visible, color='green', opacity=1.0,
                             label=f'Visible ({result["visible_area"]:.1f} area units)')

        if pv_invisible is not None:
            plotter.add_mesh(pv_invisible, color='red', opacity=0.5,
                             label=f'Non-visible ({result["invisible_area"]:.1f} area units)')

        # Add direction arrow
        start_point = self.center
        arrow_length = self.scale * 1.5
        end_point = start_point + direction * arrow_length
        arrow = pv.Arrow(start_point, end_point, tip_length=0.2,
                         tip_radius=0.1, shaft_radius=0.05)
        plotter.add_mesh(arrow, color='blue', label='View Direction')

        # Add camera at viewing position
        camera_pos = self.center + direction * (self.scale * 2.5)
        plotter.camera_position = [tuple(camera_pos), tuple(self.center), (0, 0, 1)]

        # Add legend and title
        plotter.add_legend()
        invisible_percent = result['invisible_fraction'] * 100
        plotter.add_title(f"Visibility from Direction {direction}\n"
                          f"{invisible_percent:.1f}% Non-visible Area")

        # Show the plot
        plotter.show()

    def split_mesh_by_directions(self, d1, d2):
        """
        Split the mesh into two parts based on the selected parting directions.

        Parameters:
        -----------
        d1, d2 : ndarray
            The selected parting directions

        Returns:
        --------
        tuple
            (part1, part2) - The split mesh parts as trimesh objects
        """
        # First create a convex hull
        hull = self.mesh.convex_hull

        # Convert to PyVista for better normal computation
        hull_pv = pv.PolyData(hull.vertices,
                              np.hstack((np.ones((len(hull.faces), 1), dtype=np.int64) * 3,
                                         hull.faces)).flatten())

        # Compute face normals for the hull
        hull_pv.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        hull_normals = hull_pv.cell_normals

        # Normalize direction vectors
        d1_norm = d1 / np.linalg.norm(d1)
        d2_norm = d2 / np.linalg.norm(d2)

        # Initialize arrays to store faces belonging to each direction
        d1_aligned_faces = []
        d2_aligned_faces = []

        # For each face, check if it's more aligned with d1 or d2
        for i, face in enumerate(hull.faces):
            # Get the face normal
            normal = hull_normals[i]

            # Calculate dot products with both directions
            dot_d1 = abs(np.dot(normal, d1_norm))
            dot_d2 = abs(np.dot(normal, d2_norm))

            # Assign to direction based on which has the larger dot product
            if dot_d1 > dot_d2:
                d1_aligned_faces.append(face)
            else:
                d2_aligned_faces.append(face)

        # Convert the face lists to numpy arrays
        d1_aligned_faces = np.array(d1_aligned_faces) if d1_aligned_faces else np.empty((0, 3), dtype=int)
        d2_aligned_faces = np.array(d2_aligned_faces) if d2_aligned_faces else np.empty((0, 3), dtype=int)

        # Extract unique vertices from each hull section
        def extract_unique_vertices_from_faces(vertices, faces):
            unique_indices = np.unique(faces.flatten())
            return vertices[unique_indices]

        red_hull_vertices = extract_unique_vertices_from_faces(hull.vertices, d1_aligned_faces) if len(
            d1_aligned_faces) > 0 else np.empty((0, 3))
        blue_hull_vertices = extract_unique_vertices_from_faces(hull.vertices, d2_aligned_faces) if len(
            d2_aligned_faces) > 0 else np.empty((0, 3))

        # Create KD-Trees for efficient nearest point lookup
        red_kdtree = KDTree(red_hull_vertices) if len(red_hull_vertices) > 0 else None
        blue_kdtree = KDTree(blue_hull_vertices) if len(blue_hull_vertices) > 0 else None

        # Function to find the closest distance to a point set
        def closest_distance(point, kdtree):
            if kdtree is None:
                return float('inf')
            distance, _ = kdtree.query(point)
            return distance

        # Function to calculate face centroid
        def face_centroid(mesh, face_idx):
            face_vertices = mesh.vertices[mesh.faces[face_idx]]
            centroid = np.mean(face_vertices, axis=0)
            return centroid

        # Initialize arrays to store faces of the original mesh based on proximity
        red_proximal_faces = []
        blue_proximal_faces = []

        # For each face in the original mesh, find the closest hull section
        for face_idx, face in enumerate(self.mesh.faces):
            # Compute the centroid of this face
            face_center = face_centroid(self.mesh, face_idx)

            # Find the closest distance to red and blue hull sections
            red_distance = closest_distance(face_center, red_kdtree)
            blue_distance = closest_distance(face_center, blue_kdtree)

            # Assign the face to the closer hull section
            if red_distance <= blue_distance:
                red_proximal_faces.append(face)
            else:
                blue_proximal_faces.append(face)

        # Convert to numpy arrays
        red_proximal_faces = np.array(red_proximal_faces) if red_proximal_faces else np.empty((0, 3), dtype=int)
        blue_proximal_faces = np.array(blue_proximal_faces) if blue_proximal_faces else np.empty((0, 3), dtype=int)

        # Create separate trimesh objects for each part
        part1 = trimesh.Trimesh(vertices=self.mesh.vertices, faces=red_proximal_faces) if len(
            red_proximal_faces) > 0 else None
        part2 = trimesh.Trimesh(vertices=self.mesh.vertices, faces=blue_proximal_faces) if len(
            blue_proximal_faces) > 0 else None

        return part1, part2

    def visualize_split_mesh(self, part1, part2, d1, d2):
        """Visualize the split mesh."""
        # Create a plotter
        plotter = pv.Plotter()

        # Convert trimesh objects to pyvista meshes
        if part1 is not None:
            part1_pv = pv.PolyData(part1.vertices,
                                   np.hstack((np.ones((len(part1.faces), 1), dtype=np.int64) * 3,
                                              part1.faces)).flatten())
            plotter.add_mesh(part1_pv, color='red', opacity=1, label='Part 1')

        if part2 is not None:
            part2_pv = pv.PolyData(part2.vertices,
                                   np.hstack((np.ones((len(part2.faces), 1), dtype=np.int64) * 3,
                                              part2.faces)).flatten())
            plotter.add_mesh(part2_pv, color='blue', opacity=1, label='Part 2')

        # Add direction vectors as arrows
        scale_factor = self.scale * 1.5
        start_point = self.center

        # Direction 1 arrow
        end_point1 = start_point + d1 * scale_factor
        arrow1 = pv.Arrow(start_point, end_point1,
                          tip_length=0.15, tip_radius=0.08, shaft_radius=0.03)
        plotter.add_mesh(arrow1, color='darkred')

        # Direction 2 arrow
        end_point2 = start_point + d2 * scale_factor
        arrow2 = pv.Arrow(start_point, end_point2,
                          tip_length=0.15, tip_radius=0.08, shaft_radius=0.03)
        plotter.add_mesh(arrow2, color='darkblue')

        # Add legend
        plotter.add_legend()
        plotter.add_title("Mesh Split by Optimal Parting Directions")

        # Show the plot
        plotter.show()

    def save_split_meshes(self, part1, part2, prefix=""):
        """Save the split meshes to files."""
        if part1 is not None:
            part1.export(f"{prefix}part1.stl")
            print(f"Saved part 1 to {prefix}part1.stl")

        if part2 is not None:
            part2.export(f"{prefix}part2.stl")
            print(f"Saved part 2 to {prefix}part2.stl")

    def process(self, visualize=True, save_output=True, output_prefix=""):
        """
        Complete pipeline to find parting directions and split the mesh.

        Parameters:
        -----------
        visualize : bool
            Whether to show visualization plots
        save_output : bool
            Whether to save the output meshes
        output_prefix : str
            Prefix for output file names

        Returns:
        --------
        tuple
            (d1, d2, part1, part2) - The selected directions and split meshes
        """
        # Compute visibility for all directions
        self.compute_visibility_for_all_directions()

        # Get the best parting directions
        d1, d2 = self.get_best_parting_directions()
        print(f"Selected Parting Direction 1: {d1}")
        print(f"Selected Parting Direction 2: {d2}")

        # Visualize the directions
        if visualize:
            self.visualize_directions(d1, d2)
            self.visualize_visible_areas(d1)
            self.visualize_visible_areas(d2)

        # Split the mesh
        part1, part2 = self.split_mesh_by_directions(d1, d2)

        # Report split results
        print(f"Original mesh has {len(self.mesh.faces)} total faces")
        print(f"  - Part 1 has {len(part1.faces) if part1 else 0} faces")
        print(f"  - Part 2 has {len(part2.faces) if part2 else 0} faces")

        # Visualize the split mesh
        if visualize:
            self.visualize_split_mesh(part1, part2, d1, d2)

        # Save the results
        if save_output:
            self.save_split_meshes(part1, part2, output_prefix)

        return d1, d2, part1, part2


# Example usage
if __name__ == "__main__":
    mesh_path = r"E:\Unify\Automatic Mold Designer\Mewtwo.stl"
    finder = PartingDirectionFinder(mesh_path, num_candidates=64, render_resolution=512)
    d1, d2, part1, part2 = finder.process(visualize=True, save_output=True)