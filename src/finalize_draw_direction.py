import trimesh
import pyvista as pv
import numpy as np
import pyrender
from pyrender import PerspectiveCamera, DirectionalLight, OffscreenRenderer
from scipy.spatial import ConvexHull


class FinalizeDrawDirection:
    def __init__(self, mesh_path: str, num_vectors: int, resolution=(256, 256)):
        self.mesh_path = mesh_path
        self.num_vectors = num_vectors
        self.resolution = resolution

        self.mesh = trimesh.load(self.mesh_path)
        self.vertices = self.mesh.vertices
        self.normals = self.mesh.face_normals
        self.center = self.mesh.centroid

    def createCandidateVectors(self):
        """
        Create uniform candidate vectors for determining draw directions using Fibonacci Lattice
        :return: array of candidate vectors
        """

        # Generate evenly distributed points on a sphere using the Fibonacci lattice
        indices = np.arange(0, self.num_vectors, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.num_vectors)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        # Convert spherical coordinates to Cartesian coordinates
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        # Stack the coordinates into a 2D array
        vectors = np.column_stack((x, y, z))

        # Normalize the vectors to get unit vectors (not strictly necessary as they are already unit vectors)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        return vectors

    def renderVisibleArea(self, candidate_direction):
        """
        Render the visible area of the mesh from a point along the candidate draw directions
        :return: None
        """

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.05, 0.05, 0.05])
        pyr_mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        scene.add(pyr_mesh)

        # Calculate the bounding box and its diagonal length
        bounding_box = self.mesh.bounds
        max_dimension = np.max(bounding_box[1] - bounding_box[0])

        # Set cam_distance to be 2x the diagonal length
        cam_distance = 2 * max_dimension
        cam_pos = self.center - candidate_direction * cam_distance
        up = np.array([0, 1, 0]) if not np.allclose(candidate_direction, [0, 1, 0]) else np.array([1, 0, 0])

        # Compute camera transform matrix
        z_axis = candidate_direction / np.linalg.norm(candidate_direction)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        camera_pose[:3, 3] = cam_pos

        # Add camera and light to the scene
        camera = PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)

        light = DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=camera_pose)

        # Render
        renderer = OffscreenRenderer(*self.resolution)
        _, depth = renderer.render(scene)
        renderer.delete()

        # # Visible pixels: valid depth values below a threshold
        # visible_pixels = np.count_nonzero((depth < 10.0) & (depth > 0))
        # return visible_pixels / (self.resolution[0] * self.resolution[1])
        # Get 3D positions of visible points (reverse projection, if intrinsics known)
        mask = (depth < 10.0) & (depth > 0)
        ys, xs = np.nonzero(mask)
        pixels = np.column_stack((xs, ys))

        # Compute convex hull area of 2D projection
        if len(pixels) >= 3:
            hull = ConvexHull(pixels)
            area = hull.volume  # 2D area
        else:
            area = 0

        return area

    def projected_area(self, direction: np.ndarray) -> float:
        """
        Projects the mesh onto a plane perpendicular to 'direction' and computes the convex hull area.
        """

        direction = direction / np.linalg.norm(direction)

        # Get all vertices of the mesh
        points = self.vertices

        # Project points onto a plane perpendicular to direction
        # Plane origin is mesh centroid
        projected_2d = trimesh.points.project_to_plane(
            points,
            plane_normal=direction,
            plane_origin=self.center,
            return_planar=True  # return as (n, 2)
        )

        # Compute the convex hull of the 2D projection
        try:
            hull = ConvexHull(projected_2d)
            area = hull.area  # This is 2D perimeter, not area
            area = hull.volume  # In 2D, volume is actually area
        except Exception:
            area = 0.0  # In case the hull can't be constructed (e.g., degenerate)

        return area

        # # Normalize direction
        # direction = direction / np.linalg.norm(direction)
        #
        # # Projection matrix to drop component along 'direction'
        # projection_matrix = np.eye(3) - np.outer(direction, direction)
        #
        # # Project triangle centroids (to avoid degenerate triangle issues)
        # triangle_centroids = self.mesh.triangles_center @ projection_matrix.T
        # triangle_centroids_2d = triangle_centroids[:, :2]  # Project to 2D
        #
        # if len(triangle_centroids_2d) < 3:
        #     return 0.0
        #
        # try:
        #     hull = ConvexHull(triangle_centroids_2d)
        #     return hull.volume  # 2D area
        # except:
        #     return 0.0

    def computeVisibleAreas(self, vectors: list):
        """
        Compute the visible area score from each direction.

        Returns:
            The direction (unit vector) with the maximum visible area.
        """
        best_dir = None
        max_score = -1
        for dir_vec in vectors:
            score = self.projected_area(dir_vec)
            if score > max_score:
                max_score = score
                best_dir = dir_vec

        return best_dir


# TODO: Add this module to the above class

# def projected_area(mesh: trimesh.Trimesh, direction: np.ndarray) -> float:
#     """
#     Projects the mesh onto a plane perpendicular to 'direction' and computes the silhouette area.
#     """
#     # Build projection matrix (drop the component along direction)
#     direction = direction / np.linalg.norm(direction)
#     projection_matrix = np.eye(3) - np.outer(direction, direction)
#
#     # Project all vertices
#     projected_verts = mesh.vertices @ projection_matrix.T
#
#     # Create 2D polygon from projection
#     projected_2d = projected_verts[:, :2]  # take first 2 dimensions
#     projected_mesh = trimesh.Trimesh(vertices=projected_2d, faces=mesh.faces)
#
#     # Compute convex hull or union of triangles
#     return projected_mesh.convex_hull.area  # or projected_mesh.area
