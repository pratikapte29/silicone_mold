import numpy as np
import trimesh
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import pyvista as pv

warnings.filterwarnings('ignore')

class MetaMoldSplitter:
    def __init__(self, stl_file_path, k_directions=650, lambda_smooth=0.1, mu_label=0.05):
        """
        Initialize the MetaMold splitter based on the paper's methodology.
        
        Args:
            stl_file_path: Path to input STL file
            k_directions: Number of candidate parting directions (default 20)
            lambda_smooth: Smoothing regularization parameter (default 0.1)
            mu_label: Label cost regularization parameter (default 0.05)
        """
        self.mesh = trimesh.load(stl_file_path)
        self.k = k_directions
        self.lambda_smooth = lambda_smooth
        self.mu_label = mu_label
        
        # Generate candidate parting directions uniformly on unit sphere
        self.directions = self._generate_sphere_directions()
        
        # Initialize face properties
        self.n_faces = len(self.mesh.faces)
        self.face_normals = self.mesh.face_normals
        self.face_centers = self.mesh.triangles_center
        
        # Results storage
        self.moldability_costs = None
        self.optimal_labels = None
        self.parting_plane = None
        
    def _generate_sphere_directions(self):
        """Generate k uniformly distributed directions on unit sphere."""
        # Using Fibonacci sphere method for uniform distribution
        indices = np.arange(0, self.k, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/self.k)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        return np.column_stack([x, y, z])
    
    def compute_moldability_matrix(self):
        """
        Compute moldability cost matrix m_ij for each face and direction.
        Based on Section 4.2 of the paper.
        """
        print("Computing moldability costs...")
        self.moldability_costs = np.zeros((self.n_faces, self.k))
        
        for j, direction in enumerate(self.directions):
            for i in range(self.n_faces):
                self.moldability_costs[i, j] = self._compute_face_moldability(i, direction)
                
        return self.moldability_costs
    
    def _compute_face_moldability(self, face_idx, direction):
        """
        Compute moldability cost for a single face and direction.
        Based on visibility and geometric feasibility.
        """
        face_normal = self.face_normals[face_idx]
        face_center = self.face_centers[face_idx]
        
        # Dot product indicates if face is visible from direction
        visibility = np.dot(face_normal, direction)
        
        # If face is back-facing (not visible), high cost
        if visibility <= 0:
            cost = 10.0  # High penalty for non-visible faces
        else:
            # Cost based on how well-aligned the face is with extraction direction
            # Well-aligned faces (parallel to direction) have lower cost
            alignment = abs(np.dot(face_normal, direction))
            cost = 1.0 - alignment
            
            # Add geometric complexity penalty
            cost *= self._compute_geometric_complexity(face_idx, direction)
        
        return cost
    
    def _compute_geometric_complexity(self, face_idx, direction):
        """Compute geometric complexity factor for moldability."""
        # Simple implementation: consider face area and curvature
        face_area = self.mesh.area_faces[face_idx]
        max_area = np.max(self.mesh.area_faces)
        
        # Normalize area factor (larger faces are easier to mold)
        area_factor = 1.0 + (max_area - face_area) / max_area
        
        return area_factor
    
    def compute_smoothing_costs(self):
        """
        Compute smoothing costs S_uv for adjacent faces.
        Based on Equation (2) in the paper.
        """
        print("Computing smoothing costs...")
        # Get face adjacency information
        face_adjacency = self.mesh.face_adjacency
        
        smoothing_costs = {}
        
        for edge_idx in range(len(face_adjacency)):
            face_u, face_v = face_adjacency[edge_idx]
            
            # Compute area-weighted difference
            area_u = self.mesh.area_faces[face_u] 
            area_v = self.mesh.area_faces[face_v]
            
            smoothing_costs[(face_u, face_v)] = area_u + area_v
            
        return smoothing_costs
    
    def optimize_segmentation(self):
        """
        Solve the integer programming problem from Equation (1).
        Uses relaxation and iterative optimization.
        """
        print("Optimizing face segmentation...")
        
        if self.moldability_costs is None:
            self.compute_moldability_matrix()
            
        smoothing_costs = self.compute_smoothing_costs()
        
        # Initialize binary variables b_ij (face i uses direction j)
        b = np.zeros((self.n_faces, self.k))
        
        # Initialize auxiliary variables g_j (direction j is used)
        g = np.zeros(self.k)
        
        # Iterative optimization (simplified approach)
        for iteration in range(10):
            # Step 1: Optimize face assignments given active directions
            for i in range(self.n_faces):
                active_dirs = np.where(g > 0.5)[0]
                if len(active_dirs) > 0:
                    costs = self.moldability_costs[i, active_dirs]
                    best_dir_idx = active_dirs[np.argmin(costs)]
                    b[i, :] = 0
                    b[i, best_dir_idx] = 1
                else:
                    # If no directions active, choose best direction for this face
                    best_dir_idx = np.argmin(self.moldability_costs[i, :])
                    b[i, :] = 0
                    b[i, best_dir_idx] = 1
            
            # Step 2: Update active directions
            for j in range(self.k):
                if np.sum(b[:, j]) > 0:
                    g[j] = 1
                else:
                    g[j] = 0
        
        self.optimal_labels = np.argmax(b, axis=1)
        
        # Compute final parting directions
        unique_labels = np.unique(self.optimal_labels)
        print(f"Optimization complete. Using {len(unique_labels)} parting directions.")
        
        return self.optimal_labels
    
    def generate_parting_plane(self):
        """Generate the parting plane for mold splitting."""
        if self.optimal_labels is None:
            self.optimize_segmentation()
        
        # Find the dominant parting direction
        label_counts = np.bincount(self.optimal_labels)
        dominant_direction_idx = np.argmax(label_counts)
        parting_direction = self.directions[dominant_direction_idx]
        
        # Compute parting plane position (through mesh centroid)
        mesh_centroid = np.mean(self.mesh.vertices, axis=0)
        
        self.parting_plane = {
            'normal': parting_direction,
            'point': mesh_centroid
        }
        
        return self.parting_plane
    
    def split_mesh(self):
        """Split the mesh into two mold pieces."""
        if self.parting_plane is None:
            self.generate_parting_plane()
        
        plane_normal = self.parting_plane['normal']
        plane_point = self.parting_plane['point']
        
        # Classify vertices relative to parting plane
        vertices = self.mesh.vertices
        distances = np.dot(vertices - plane_point, plane_normal)
        
        # Split mesh using the parting plane
        try:
            # Method 1: Use trimesh slice functionality
            mold_part_1 = self.mesh.slice_plane(plane_point, plane_normal)
            mold_part_2 = self.mesh.slice_plane(plane_point, -plane_normal)
            
        except:
            # Method 2: Manual splitting based on face classification
            print("Using manual mesh splitting...")
            face_sides = []
            
            for face in self.mesh.faces:
                face_verts = vertices[face]
                face_distances = distances[face]
                
                # Classify face based on vertex positions
                if np.all(face_distances >= 0):
                    face_sides.append(1)  # Positive side
                elif np.all(face_distances <= 0):
                    face_sides.append(-1)  # Negative side
                else:
                    face_sides.append(0)  # Intersecting face
            
            face_sides = np.array(face_sides)
            
            # Create submeshes
            faces_part_1 = self.mesh.faces[face_sides >= 0]
            faces_part_2 = self.mesh.faces[face_sides <= 0]
            
            mold_part_1 = trimesh.Trimesh(vertices=vertices, faces=faces_part_1)
            mold_part_2 = trimesh.Trimesh(vertices=vertices, faces=faces_part_2)
        
        return mold_part_1, mold_part_2
    
    def visualize_segmentation(self):
        """Visualize the mesh segmentation and parting directions."""
        if self.optimal_labels is None:
            self.optimize_segmentation()
        
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Original mesh
        ax1 = fig.add_subplot(131, projection='3d')
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        triangles=faces, alpha=0.7, color='lightblue')
        ax1.set_title('Original Mesh')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Face segmentation
        ax2 = fig.add_subplot(132, projection='3d')
        
        # Color faces by their assigned direction
        face_colors = plt.cm.tab10(self.optimal_labels / max(1, np.max(self.optimal_labels)))
        
        for i, face in enumerate(faces):
            face_verts = vertices[face]
            color = face_colors[i]
            ax2.plot_trisurf(face_verts[:, 0], face_verts[:, 1], face_verts[:, 2], 
                           color=color, alpha=0.7)
        
        ax2.set_title('Face Segmentation')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Plot 3: Parting directions
        ax3 = fig.add_subplot(133, projection='3d')
        
        # Plot mesh
        ax3.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        triangles=faces, alpha=0.3, color='lightgray')
        
        # Plot parting plane
        if self.parting_plane:
            normal = self.parting_plane['normal']
            point = self.parting_plane['point']
            
            # Create plane visualization
            d = -np.dot(normal, point)
            xx, yy = np.meshgrid(np.linspace(vertices[:, 0].min(), vertices[:, 0].max(), 10),
                               np.linspace(vertices[:, 1].min(), vertices[:, 1].max(), 10))
            zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
            
            ax3.plot_surface(xx, yy, zz, alpha=0.3, color='red')
            
            # Plot normal vector
            ax3.quiver(point[0], point[1], point[2], 
                      normal[0], normal[1], normal[2], 
                      length=0.5*np.ptp(vertices), color='red', arrow_length_ratio=0.1)
        
        ax3.set_title('Parting Plane')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()
    
    def save_mold_parts(self, output_prefix="mold_part"):
        """Save the two mold parts as STL files."""
        mold_part_1, mold_part_2 = self.split_mesh()
        
        # Save parts
        part1_file = f"{output_prefix}_1.stl"
        part2_file = f"{output_prefix}_2.stl"
        
        mold_part_1.export(part1_file)
        mold_part_2.export(part2_file)
        
        print(f"Mold parts saved as {part1_file} and {part2_file}")
        
        return part1_file, part2_file
    
    def print_optimization_summary(self):
        """Print summary of the optimization results."""
        if self.optimal_labels is None:
            print("Optimization not yet performed.")
            return
        
        unique_labels = np.unique(self.optimal_labels)
        print("\n=== MetaMold Optimization Summary ===")
        print(f"Input mesh: {self.n_faces} faces")
        print(f"Candidate directions: {self.k}")
        print(f"Selected parting directions: {len(unique_labels)}")
        
        # Print direction usage
        for label in unique_labels:
            count = np.sum(self.optimal_labels == label)
            direction = self.directions[label]
            print(f"Direction {label}: {count} faces, vector: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
        
        # Compute total cost
        if self.moldability_costs is not None:
            total_moldability_cost = 0
            for i in range(self.n_faces):
                total_moldability_cost += self.moldability_costs[i, self.optimal_labels[i]]
            
            print(f"Total moldability cost: {total_moldability_cost:.3f}")
        
        print("=====================================\n")


# Example usage and testing function
def process_stl_file(stl_file_path, k_directions=650, visualize=True):
    """
    Process an STL file and split it into mold parts.
    
    Args:
        stl_file_path: Path to input STL file
        k_directions: Number of candidate parting directions
        visualize: Whether to show visualization plots
    """
    print(f"Processing STL file: {stl_file_path}")
    
    # Initialize the splitter
    splitter = MetaMoldSplitter(stl_file_path, k_directions=k_directions)
    
    # Perform optimization
    labels = splitter.optimize_segmentation()
    
    # Generate parting plane
    parting_plane = splitter.generate_parting_plane()
    
    # Print summary
    splitter.print_optimization_summary()
    
    # Visualize results
    if visualize:
        try:
            splitter.visualize_segmentation()
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Save mold parts
    part_files = splitter.save_mold_parts()
    
    return splitter, part_files


# Example with a simple test case (creating a simple mesh if no STL provided)
def create_test_mesh():
    """Create a simple test mesh for demonstration."""
    # Create a simple box mesh
    mesh = trimesh.creation.box(extents=[2, 1, 1])
    mesh.export('test_object.stl')
    return 'test_object.stl'


if __name__ == "__main__":
    # Example usage
    print("MetaMold: Computational Design of Silicone Molds")
    print("Based on the research paper methodology")
    print("=" * 50)
    
    # Create a test mesh if needed
    test_file = create_test_mesh()
    print(f"Created test mesh: {test_file}")
    
    # Process the mesh
    try:
        splitter, mold_files = process_stl_file("../assets/stl/beast.stl", k_directions=650, visualize=False)
        print(f"Successfully created mold parts: {mold_files}")
    except Exception as e:
        print(f"Error processing mesh: {e}")
        
    print("\nTo use with your own STL file:")
    print("splitter = MetaMoldSplitter('your_file.stl')")
    print("splitter.optimize_segmentation()")
    print("splitter.save_mold_parts('output_mold')")

    # Load STL files
    mesh1 = pv.read("mold_part_1.stl")
    mesh2 = pv.read("mold_part_2.stl")

    # Create a plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh1, color="red", opacity=1.0, label="Mesh 1")
    plotter.add_mesh(mesh2, color="blue", opacity=1.0, label="Mesh 2")

    # Optional: Add legend and show axes
    plotter.add_legend()
    plotter.show_axes()

    # Display the scene
    plotter.show()
