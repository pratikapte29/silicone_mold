import numpy as np
import trimesh
import matplotlib.pyplot as plt

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    GPU_AVAILABLE = True
    print(f"GPU detected: {cp.cuda.get_device_name()}")
    print(f"GPU memory: {cp.cuda.Device().mem_info[1] / 1024 ** 3:.1f} GB total")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy-cuda11x")

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class GPUMeshSegmentation:
    def __init__(self, stl_file_path, k_directions=200, max_segments=6, use_gpu=True):
        """
        GPU-Accelerated mesh segmentation for mold design.

        Args:
            stl_file_path: Path to STL file
            k_directions: Number of candidate parting directions
            max_segments: Maximum number of allowed segments
            use_gpu: Whether to use GPU acceleration
        """
        self.mesh = trimesh.load(stl_file_path)
        self.k = k_directions
        self.max_segments = max_segments
        self.use_gpu = use_gpu and GPU_AVAILABLE

        print(f"Loaded mesh with {len(self.mesh.faces)} faces")
        print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")

        # Choose computation backend
        self.xp = cp if self.use_gpu else np

        # Generate candidate directions
        self.directions = self._generate_uniform_directions()

        # Compute face adjacency
        self.adjacency_pairs = self._compute_face_adjacency()

        # Transfer mesh data to GPU if using GPU
        if self.use_gpu:
            self._transfer_to_gpu()

        self.moldability_costs = None
        self.smoothness_weights = None

    def _transfer_to_gpu(self):
        """Transfer mesh data to GPU memory."""
        print("Transferring mesh data to GPU...")
        self.gpu_face_normals = cp.asarray(self.mesh.face_normals)
        self.gpu_face_centers = cp.asarray(self.mesh.triangles_center)
        self.gpu_directions = cp.asarray(self.directions)
        self.gpu_centroid = cp.asarray(self.mesh.centroid)

        # Compute bounding box extent on GPU
        bounds = cp.asarray(self.mesh.bounds)
        self.gpu_bbox_extent = cp.linalg.norm(bounds[1] - bounds[0])

        print(f"GPU memory usage: {cp.cuda.Device().mem_info[0] / 1024 ** 2:.1f} MB")

    def _generate_uniform_directions(self):
        """Generate k uniformly distributed directions on unit sphere."""
        print(f"Generating {self.k} uniform directions...")

        # Fibonacci sphere for uniform distribution
        indices = np.arange(0, self.k, dtype=float) + 0.5
        theta = np.arccos(1 - 2 * indices / self.k)
        phi = np.pi * (1 + 5 ** 0.5) * indices

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.column_stack([x, y, z])

    def _compute_face_adjacency(self):
        """Compute adjacent face pairs."""
        return self.mesh.face_adjacency

    def _compute_moldability_costs_gpu(self):
        """
        GPU-accelerated moldability cost computation.

        This demonstrates several GPU computing concepts:
        1. Parallel matrix operations
        2. Memory-efficient batch processing
        3. Kernel fusion for multiple operations
        """
        print("Computing moldability costs on GPU...")

        n_faces = len(self.mesh.faces)

        # Allocate result array on GPU
        costs = cp.zeros((n_faces, self.k), dtype=cp.float32)

        # Process in batches to manage GPU memory
        batch_size = min(50, self.k)  # Process 50 directions at a time

        for batch_start in range(0, self.k, batch_size):
            batch_end = min(batch_start + batch_size, self.k)
            batch_size_actual = batch_end - batch_start

            # Get batch of directions
            batch_directions = self.gpu_directions[batch_start:batch_end]  # [batch_size, 3]

            # Compute visibility for all faces and all directions in batch
            # This is a matrix multiplication: [n_faces, 3] @ [3, batch_size] = [n_faces, batch_size]
            visibility = cp.dot(self.gpu_face_normals, batch_directions.T)

            # Angle costs (absolute value of negative dot products)
            angle_costs = cp.abs(visibility)

            # Depth estimation (vectorized for all faces)
            center_distances = cp.linalg.norm(
                self.gpu_face_centers - self.gpu_centroid, axis=1
            )  # [n_faces,]

            # Broadcast depth costs to match batch dimensions
            depth_costs = (center_distances[:, None] / self.gpu_bbox_extent) * 0.5
            depth_costs = cp.broadcast_to(depth_costs, (n_faces, batch_size_actual))

            # Combine costs
            batch_costs = angle_costs + depth_costs

            # Set visible faces (positive visibility) to zero cost
            visible_mask = visibility > 0
            batch_costs[visible_mask] = 0.0

            # Store results
            costs[:, batch_start:batch_end] = batch_costs

            if (batch_end) % 100 == 0 or batch_end == self.k:
                print(f"  Processed {batch_end}/{self.k} directions")

        # Transfer results back to CPU
        if self.use_gpu:
            self.moldability_costs = cp.asnumpy(costs)
        else:
            self.moldability_costs = costs

        print("Moldability costs computed!")

    def _compute_moldability_costs_cpu(self):
        """CPU fallback for moldability cost computation."""
        print("Computing moldability costs on CPU...")

        n_faces = len(self.mesh.faces)
        self.moldability_costs = np.zeros((n_faces, self.k))

        face_normals = self.mesh.face_normals
        face_centers = self.mesh.triangles_center
        bbox_extent = np.linalg.norm(self.mesh.bounds[1] - self.mesh.bounds[0])

        for j in range(self.k):
            direction = self.directions[j]

            # Vectorized visibility computation
            visibility = np.dot(face_normals, direction)
            visible_mask = visibility > 0

            # Costs for non-visible faces
            angle_costs = np.abs(visibility)
            center_distances = np.linalg.norm(face_centers - self.mesh.centroid, axis=1)
            depth_costs = (center_distances / bbox_extent) * 0.5

            costs = angle_costs + depth_costs
            costs[visible_mask] = 0.0

            self.moldability_costs[:, j] = costs

            if (j + 1) % 50 == 0:
                print(f"  Processed {j + 1}/{self.k} directions")

    def _compute_smoothness_weights_gpu(self):
        """GPU-accelerated smoothness weights computation."""
        print("Computing smoothness weights on GPU...")

        if not self.use_gpu:
            return self._compute_smoothness_weights_cpu()

        n_edges = len(self.adjacency_pairs)

        # Get face indices for adjacent pairs
        face_u_indices = cp.asarray(self.adjacency_pairs[:, 0])
        face_v_indices = cp.asarray(self.adjacency_pairs[:, 1])

        # Get normals for adjacent faces
        normals_u = self.gpu_face_normals[face_u_indices]  # [n_edges, 3]
        normals_v = self.gpu_face_normals[face_v_indices]  # [n_edges, 3]

        # Compute dot products (element-wise for each edge)
        cos_angles = cp.sum(normals_u * normals_v, axis=1)  # [n_edges,]
        cos_angles = cp.clip(cos_angles, -1.0, 1.0)

        # Compute angles and weights
        angles = cp.arccos(cos_angles)
        weights = 1.0 - (angles / cp.pi)

        # Transfer back to CPU
        self.smoothness_weights = cp.asnumpy(weights)
        print(f"Computed {len(self.smoothness_weights)} edge weights")

    def _compute_smoothness_weights_cpu(self):
        """CPU fallback for smoothness weights."""
        print("Computing smoothness weights on CPU...")

        weights = np.zeros(len(self.adjacency_pairs))
        face_normals = self.mesh.face_normals

        for i, (face_u, face_v) in enumerate(self.adjacency_pairs):
            normal_u = face_normals[face_u]
            normal_v = face_normals[face_v]

            cos_angle = np.clip(np.dot(normal_u, normal_v), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            weights[i] = 1.0 - (angle / np.pi)

        self.smoothness_weights = weights
        print(f"Computed {len(weights)} edge weights")

    def compute_costs(self):
        """Compute all required costs using GPU acceleration."""
        if self.use_gpu:
            self._compute_moldability_costs_gpu()
            self._compute_smoothness_weights_gpu()
        else:
            self._compute_moldability_costs_cpu()
            self._compute_smoothness_weights_cpu()

    def solve_greedy_approximation(self, lambda_smooth=1.0, mu_label=1.0):
        """
        Fast greedy approximation with GPU acceleration where possible.
        """
        print("Solving with greedy approximation...")
        n_faces = len(self.mesh.faces)

        # Initialize with best individual directions
        face_labels = np.argmin(self.moldability_costs, axis=1)

        print(f"Initial solution uses {len(np.unique(face_labels))} directions")

        # Iterative improvement
        max_iterations = 20  # Reduced for speed
        for iteration in range(max_iterations):
            improved = False
            old_labels = face_labels.copy()

            # Try to improve each face's label
            for i in range(0, n_faces, 100):  # Process in small batches
                batch_end = min(i + 100, n_faces)

                for face_idx in range(i, batch_end):
                    current_label = face_labels[face_idx]
                    best_cost = float('inf')
                    best_label = current_label

                    # Try a subset of directions for speed
                    candidate_directions = np.random.choice(self.k, size=min(20, self.k), replace=False)

                    for j in candidate_directions:
                        cost = self.moldability_costs[face_idx, j]

                        # Add smoothness penalty from neighbors
                        neighbor_penalty = 0
                        neighbor_count = 0

                        for edge_idx, (u, v) in enumerate(self.adjacency_pairs):
                            if u == face_idx:
                                neighbor_label = face_labels[v]
                                neighbor_count += 1
                            elif v == face_idx:
                                neighbor_label = face_labels[u]
                                neighbor_count += 1
                            else:
                                continue

                            if neighbor_label != j:
                                neighbor_penalty += lambda_smooth * self.smoothness_weights[edge_idx]

                        total_cost = cost + neighbor_penalty

                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_label = j

                    if best_label != current_label:
                        face_labels[face_idx] = best_label
                        improved = True

            if not improved:
                print(f"Converged after {iteration + 1} iterations")
                break

            # Add label cost penalty to encourage fewer segments
            unique_labels = np.unique(face_labels)
            if len(unique_labels) > self.max_segments:
                print(f"Too many segments ({len(unique_labels)}), consolidating...")
                # Simple consolidation: merge least used labels
                label_counts = [(label, np.sum(face_labels == label)) for label in unique_labels]
                label_counts.sort(key=lambda x: x[1])  # Sort by count

                # Keep top max_segments labels
                keep_labels = [lc[0] for lc in label_counts[-self.max_segments:]]

                # Reassign faces with removed labels to nearest kept label
                for i, label in enumerate(face_labels):
                    if label not in keep_labels:
                        # Find closest direction among kept labels
                        face_costs = [self.moldability_costs[i, kept_label] for kept_label in keep_labels]
                        best_kept = keep_labels[np.argmin(face_costs)]
                        face_labels[i] = best_kept

            print(f"Iteration {iteration + 1}: {len(np.unique(face_labels))} segments")

        used_directions = list(np.unique(face_labels))
        print(f"Final solution: {len(used_directions)} segments")

        return face_labels, used_directions, None

    def segment_mesh(self, lambda_smooth=1.0, mu_label=1.0):
        """
        Main segmentation function with GPU acceleration.
        """
        print(f"Segmenting mesh with {len(self.mesh.faces)} faces using {'GPU' if self.use_gpu else 'CPU'}")

        # Compute costs (GPU accelerated)
        import time
        start_time = time.time()

        self.compute_costs()

        cost_time = time.time() - start_time
        print(f"Cost computation took {cost_time:.2f} seconds")

        # Solve optimization
        start_time = time.time()
        face_labels, used_directions, obj_value = self.solve_greedy_approximation(lambda_smooth, mu_label)
        solve_time = time.time() - start_time
        print(f"Optimization took {solve_time:.2f} seconds")

        return face_labels, used_directions, obj_value

    def visualize_segmentation(self, face_labels, used_directions):
        """Visualize segmentation results."""
        n_segments = len(used_directions)
        colors = plt.cm.Set3(np.linspace(0, 1, n_segments))

        # Map face labels to colors
        face_colors = np.zeros((len(self.mesh.faces), 3))
        for i, label in enumerate(face_labels):
            segment_idx = used_directions.index(label)
            face_colors[i] = colors[segment_idx][:3]

        # Create colored mesh
        colored_mesh = self.mesh.copy()
        colored_mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

        # Show mesh
        colored_mesh.show()

        # Print results
        print(f"\n=== Segmentation Results ===")
        print(f"Number of segments: {n_segments}")
        print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")

        for i, dir_idx in enumerate(used_directions):
            direction = self.directions[dir_idx]
            faces_in_segment = np.sum(face_labels == dir_idx)
            print(f"Segment {i + 1}: Direction ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})")
            print(f"  Faces: {faces_in_segment} ({faces_in_segment / len(face_labels) * 100:.1f}%)")

    def benchmark_gpu_vs_cpu(self):
        """Benchmark GPU vs CPU performance."""
        if not GPU_AVAILABLE:
            print("GPU not available for benchmarking")
            return

        print("\n=== GPU vs CPU Benchmark ===")
        import time

        # Test moldability cost computation
        print("Benchmarking moldability cost computation...")

        # CPU timing
        self.use_gpu = False
        self.xp = np
        start = time.time()
        self._compute_moldability_costs_cpu()
        cpu_time = time.time() - start

        # GPU timing
        self.use_gpu = True
        self.xp = cp
        self._transfer_to_gpu()
        start = time.time()
        self._compute_moldability_costs_gpu()
        gpu_time = time.time() - start

        print(f"CPU time: {cpu_time:.2f} seconds")
        print(f"GPU time: {gpu_time:.2f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

        # Test smoothness weights
        print("\nBenchmarking smoothness weights computation...")

        # CPU timing
        self.use_gpu = False
        start = time.time()
        self._compute_smoothness_weights_cpu()
        cpu_time = time.time() - start

        # GPU timing
        self.use_gpu = True
        start = time.time()
        self._compute_smoothness_weights_gpu()
        gpu_time = time.time() - start

        print(f"CPU time: {cpu_time:.2f} seconds")
        print(f"GPU time: {gpu_time:.2f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")


def install_cupy():
    """Helper function to install CuPy for CUDA support."""
    print("To enable GPU acceleration, install CuPy:")
    print("For CUDA 11.x: pip install cupy-cuda11x")
    print("For CUDA 12.x: pip install cupy-cuda12x")
    print("Check your CUDA version with: nvidia-smi")


def main():
    """Example usage with GPU acceleration."""
    stl_file = r"assets/stl/cow_fixed.stl"

    if not GPU_AVAILABLE:
        install_cupy()
        print("Falling back to CPU computation...")

    try:
        # Create GPU-accelerated segmenter
        segmenter = GPUMeshSegmentation(
            stl_file,
            k_directions=200,  # More directions possible with GPU
            max_segments=4,
            use_gpu=True  # Enable GPU acceleration
        )

        # Run benchmark if GPU is available
        if GPU_AVAILABLE:
            segmenter.benchmark_gpu_vs_cpu()

        # Perform segmentation
        face_labels, used_directions, obj_value = segmenter.segment_mesh(
            lambda_smooth=1.0,
            mu_label=3.0
        )

        if face_labels is not None:
            segmenter.visualize_segmentation(face_labels, used_directions)

            if len(used_directions) == 2:
                print("✓ Achieved optimal 2-piece mold!")
            else:
                print(f"Complex geometry required {len(used_directions)} pieces")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install CuPy: pip install cupy-cuda11x")
        print("2. Install trimesh: pip install trimesh")
        print("3. Check CUDA installation: nvidia-smi")


if __name__ == "__main__":
    main()