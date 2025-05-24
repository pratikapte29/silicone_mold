"""
Region Growing Algorithm for Mesh Segmentation
Divides mesh into two regions based on normal alignment while avoiding concavities
"""

import pyvista as pv
import numpy as np
import networkx as nx


class MeshRegionGrowing:
    def __init__(self, mesh, direction, normal_threshold=0.3, concavity_threshold=0.1):
        """
        Initialize the region growing algorithm

        Args:
            mesh: PyVista mesh
            direction: Direction vector for region classification
            normal_threshold: Threshold for normal alignment (0-1)
            concavity_threshold: Threshold for detecting concave areas
        """
        self.mesh = mesh
        self.direction = direction / np.linalg.norm(direction)
        self.normal_threshold = normal_threshold
        self.concavity_threshold = concavity_threshold

        # Compute face properties
        self.face_centers = mesh.cell_centers().points
        self.face_normals = mesh.compute_normals(cell_normals=True, point_normals=False)['Normals']
        self.face_areas = self._compute_face_areas()

        # Build adjacency graph
        self.adjacency_graph = self._build_face_adjacency_graph()

        # Initialize regions
        self.region_labels = np.zeros(mesh.n_cells, dtype=int)  # 0=unassigned, 1=region1, -1=region2
        self.face_distances = np.full(mesh.n_cells, np.inf)  # Distance from nearest seed point

    def _compute_face_areas(self):
        """Compute area of each face"""
        areas = np.zeros(self.mesh.n_cells)
        for i in range(self.mesh.n_cells):
            cell = self.mesh.get_cell(i)
            if cell.GetNumberOfPoints() == 3:  # Triangle
                pts = np.array([cell.GetPoints().GetPoint(j) for j in range(3)])
                v1, v2 = pts[1] - pts[0], pts[2] - pts[0]
                areas[i] = 0.5 * np.linalg.norm(np.cross(v1, v2))
        return areas

    def _build_face_adjacency_graph(self):
        """Build adjacency graph between faces"""
        G = nx.Graph()

        # Add all faces as nodes
        for i in range(self.mesh.n_cells):
            G.add_node(i)

        # Find adjacent faces by shared edges
        edge_to_faces = {}

        for face_id in range(self.mesh.n_cells):
            cell = self.mesh.get_cell(face_id)
            n_points = cell.GetNumberOfPoints()

            # Get edges of this face
            for i in range(n_points):
                p1 = cell.GetPointId(i)
                p2 = cell.GetPointId((i + 1) % n_points)
                edge = tuple(sorted([p1, p2]))

                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_id)

        # Add edges between adjacent faces
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                G.add_edge(faces[0], faces[1])

        return G

    def _compute_normal_alignment(self, face_id):
        """Compute alignment of face normal with direction vector"""
        return np.dot(self.face_normals[face_id], self.direction)

    def _detect_concave_regions(self):
        """Detect concave regions using curvature analysis"""
        concave_faces = set()

        for face_id in range(self.mesh.n_cells):
            neighbors = list(self.adjacency_graph.neighbors(face_id))
            if len(neighbors) < 2:
                continue

            face_normal = self.face_normals[face_id]
            face_center = self.face_centers[face_id]

            # Check curvature with neighbors
            curvature_sum = 0
            valid_neighbors = 0

            for neighbor_id in neighbors:
                neighbor_normal = self.face_normals[neighbor_id]
                neighbor_center = self.face_centers[neighbor_id]

                # Vector from face to neighbor
                direction_vec = neighbor_center - face_center
                if np.linalg.norm(direction_vec) > 1e-6:
                    direction_vec /= np.linalg.norm(direction_vec)

                    # Compute curvature indicator
                    normal_diff = neighbor_normal - face_normal
                    curvature = np.dot(normal_diff, direction_vec)
                    curvature_sum += curvature
                    valid_neighbors += 1

            if valid_neighbors > 0:
                avg_curvature = curvature_sum / valid_neighbors
                if avg_curvature < -self.concavity_threshold:
                    concave_faces.add(face_id)

        return concave_faces

    def _seed_initial_regions(self):
        """Seed initial regions based on strong normal alignment"""
        alignments = np.array([self._compute_normal_alignment(i) for i in range(self.mesh.n_cells)])

        # Find faces with strong positive alignment
        strong_positive = np.where(alignments > (1 - self.normal_threshold))[0]
        # Find faces with strong negative alignment
        strong_negative = np.where(alignments < -(1 - self.normal_threshold))[0]

        # Set initial seeds
        for face_id in strong_positive:
            self.region_labels[face_id] = 1

        for face_id in strong_negative:
            self.region_labels[face_id] = -1

        print(f"Seeded {len(strong_positive)} faces in region +1")
        print(f"Seeded {len(strong_negative)} faces in region -1")

    def _region_growing_step(self, concave_faces):
        """Perform one step of region growing"""
        changed = False
        candidates = []

        # Find candidate faces for growing
        for face_id in range(self.mesh.n_cells):
            if self.region_labels[face_id] == 0:  # Unassigned
                neighbors = list(self.adjacency_graph.neighbors(face_id))
                region_votes = {1: 0, -1: 0}

                # Count votes from assigned neighbors
                for neighbor_id in neighbors:
                    if self.region_labels[neighbor_id] != 0:
                        region_votes[self.region_labels[neighbor_id]] += 1

                # Only consider faces that have at least one assigned neighbor
                if region_votes[1] > 0 or region_votes[-1] > 0:
                    alignment = self._compute_normal_alignment(face_id)

                    # Determine preferred region based on normal alignment
                    preferred_region = 1 if alignment > 0 else -1

                    # Check if face is in concave region - if so, penalize it
                    is_concave = face_id in concave_faces

                    # Calculate confidence based on neighbor votes and alignment
                    total_votes = region_votes[1] + region_votes[-1]
                    neighbor_confidence = max(region_votes[1], region_votes[-1]) / total_votes
                    alignment_confidence = abs(alignment)

                    # Strong penalty for concave faces
                    if is_concave:
                        alignment_confidence *= 0.2  # Heavily reduce confidence for concave faces

                    # Combined confidence: 70% from alignment, 30% from neighbors
                    combined_confidence = 0.7 * alignment_confidence + 0.3 * neighbor_confidence

                    # Additional check: if neighbors strongly disagree with alignment, reduce confidence
                    neighbor_preferred_region = 1 if region_votes[1] > region_votes[-1] else -1
                    if neighbor_preferred_region != preferred_region and neighbor_confidence > 0.6:
                        combined_confidence *= 0.5

                    candidates.append((face_id, preferred_region, combined_confidence))

        # Sort candidates by confidence and assign regions
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Only assign faces with reasonable confidence
        confidence_threshold = 0.3
        for face_id, region, confidence in candidates:
            if confidence > confidence_threshold:
                self.region_labels[face_id] = region
                changed = True

        return changed, len(candidates)

    def segment_mesh(self):
        """Main method to segment the mesh into two regions"""
        print("Detecting concave regions...")
        concave_faces = self._detect_concave_regions()
        print(f"Found {len(concave_faces)} concave faces")

        print("Seeding initial regions...")
        self._seed_initial_regions()

        print("Growing regions...")
        iteration = 0
        max_iterations = 100

        while iteration < max_iterations:
            changed, num_candidates = self._region_growing_step(concave_faces)

            if not changed:
                print(f"Region growing converged after {iteration} iterations")
                break

            iteration += 1

            if iteration % 10 == 0:
                assigned = np.sum(self.region_labels != 0)
                total = len(self.region_labels)
                print(f"Iteration {iteration}: {assigned}/{total} faces assigned ({assigned/total*100:.1f}%)")

        # Handle remaining unassigned faces - assign based on normal alignment only
        unassigned = np.where(self.region_labels == 0)[0]
        print(f"Assigning {len(unassigned)} remaining faces based on normal alignment")

        for face_id in unassigned:
            alignment = self._compute_normal_alignment(face_id)
            # For unassigned faces, use a more conservative assignment
            if abs(alignment) > 0.1:  # Only assign if there's some directional preference
                self.region_labels[face_id] = 1 if alignment > 0 else -1
            else:
                # For truly ambiguous faces, check neighbors
                neighbors = list(self.adjacency_graph.neighbors(face_id))
                region_votes = {1: 0, -1: 0}
                for neighbor_id in neighbors:
                    if self.region_labels[neighbor_id] != 0:
                        region_votes[self.region_labels[neighbor_id]] += 1

                if region_votes[1] > region_votes[-1]:
                    self.region_labels[face_id] = 1
                elif region_votes[-1] > region_votes[1]:
                    self.region_labels[face_id] = -1
                else:
                    # Last resort: use alignment even if weak
                    self.region_labels[face_id] = 1 if alignment >= 0 else -1

        return self.region_labels, concave_faces

    def visualize_results(self, concave_faces):
        """Visualize the segmentation results"""
        # Create region meshes
        region1_faces = np.where(self.region_labels == 1)[0]
        region2_faces = np.where(self.region_labels == -1)[0]
        concave_faces_list = list(concave_faces)

        region1_mesh = self.mesh.extract_cells(region1_faces) if len(region1_faces) > 0 else None
        region2_mesh = self.mesh.extract_cells(region2_faces) if len(region2_faces) > 0 else None
        concave_mesh = self.mesh.extract_cells(concave_faces_list) if len(concave_faces_list) > 0 else None

        # Plot results
        p = pv.Plotter(shape=(1, 2))

        # Original mesh with direction and concave regions
        p.subplot(0, 0)
        p.add_mesh(self.mesh, color='lightgray', opacity=0.5, label='Original Mesh')

        if concave_mesh:
            p.add_mesh(concave_mesh, color='purple', opacity=1.0, label='Concave Regions')

        # Add direction arrow
        origin = self.mesh.center
        arrow_end = origin + 100 * self.direction
        p.add_mesh(pv.Line(origin, arrow_end), color='black', line_width=5, label='Direction')
        p.add_title("Original Mesh with Concave Regions")
        p.add_legend()

        # Segmented mesh
        p.subplot(0, 1)
        if region1_mesh:
            p.add_mesh(region1_mesh, color='red', opacity=0.8, label='Region +1')
        if region2_mesh:
            p.add_mesh(region2_mesh, color='blue', opacity=0.8, label='Region -1')

        p.add_title("Segmented Regions")
        p.add_legend()

        p.show()


def segment_mesh_with_region_growing(stl_path, direction, **kwargs):
    """
    Main function to segment a mesh using region growing

    Args:
        stl_path: Path to STL file
        direction: Direction vector for segmentation
        **kwargs: Additional parameters for MeshRegionGrowing
    """
    # Load mesh
    mesh = pv.read(stl_path)

    # Ensure mesh has normals
    mesh = mesh.compute_normals(cell_normals=True, point_normals=True)

    # Create region growing instance
    rg = MeshRegionGrowing(mesh, direction, **kwargs)

    # Perform segmentation
    labels, concave_faces = rg.segment_mesh()

    # Print statistics
    region1_count = np.sum(labels == 1)
    region2_count = np.sum(labels == -1)
    concave_count = len(concave_faces)

    print(f"\nSegmentation Results:")
    print(f"Region +1: {region1_count} faces ({region1_count/len(labels)*100:.1f}%)")
    print(f"Region -1: {region2_count} faces ({region2_count/len(labels)*100:.1f}%)")
    print(f"Concave faces detected: {concave_count} ({concave_count/len(labels)*100:.1f}%)")

    # Visualize results
    rg.visualize_results(concave_faces)

    return rg, labels, concave_faces


# Example usage
if __name__ == "__main__":
    stl_file = r"..\assets\stl\xyzrgb_dragon.stl"
    direction = np.array([0.2165375, 0.82532754, 0.52148438])

    # Run segmentation with custom parameters
    rg, labels, concave_faces = segment_mesh_with_region_growing(
        stl_file,
        direction,
        normal_threshold=0.2,    # How strict the initial seeding is
        concavity_threshold=0.15  # How sensitive concavity detection is
    )