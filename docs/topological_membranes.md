# Topological Membranes for Positive-Genus Surfaces

## Overview

This document describes the implementation of topological membranes for handling positive-genus surfaces in silicone mold generation. The functionality addresses the problem of objects with genus g > 0, where segmentation may result in mold pieces that pass through tunnel holes, making them physically impossible to extract.

## Background

### The Problem

For objects with positive genus (e.g., donuts, torus shapes), the standard mold segmentation approach can create mold pieces that pass through tunnel holes. These pieces cannot be physically extracted from the mold, making the segmentation invalid for manufacturing.

### The Solution

The solution involves introducing **topological membranes** - thin membranes inserted into the surface mesh at tunnel locations. These membranes:

1. Reduce the genus of the input model
2. Define cuts in the silicone mold volume
3. Make mold extraction physically possible

## Implementation

### Core Functions

#### 1. Genus Computation
```python
def compute_mesh_genus(mesh):
    """
    Compute the genus of a mesh using Euler characteristic.
    χ = V - E + F = 2 - 2g, where g is the genus
    """
```

#### 2. Tunnel Loop Detection
```python
def find_tunnel_loops(mesh, num_generators=None):
    """
    Find tunnel loops using homology computation.
    Uses geodesic distance approach to identify potential tunnel locations.
    """
```

#### 3. Membrane Creation
```python
def create_topological_membrane(mesh, tunnel_loop, membrane_thickness=0.01):
    """
    Create a topological membrane for a given tunnel loop.
    Projects loop vertices to a plane and creates a triangulated surface.
    """
```

#### 4. Membrane Integration
```python
def integrate_membranes_with_mesh(mesh, membranes):
    """
    Integrate topological membranes with the original mesh.
    Ensures manifoldness and proper connectivity.
    """
```

### Algorithm Steps

1. **Genus Detection**: Compute the genus of the input mesh using Euler characteristic
2. **Tunnel Loop Identification**: Find homology generators that hug tunnel loops
3. **Membrane Creation**: Create thin membranes at tunnel locations using surface reconstruction
4. **Mesh Integration**: Integrate membranes with the original mesh while preserving manifoldness
5. **Segmentation**: Run the standard mold segmentation on the modified mesh
6. **Membrane Analysis**: Analyze which membranes create required cuts vs. unnecessary ones
7. **Membrane Removal**: Remove membranes that don't correspond to required cuts

## Usage

### Basic Usage

```python
from src.generate_metamold import process_positive_genus_surface

# Process a mesh with positive genus
processed_mesh, membranes_info = process_positive_genus_surface('input_mesh.stl')

# The processed mesh can now be used in the standard metamold generation pipeline
```

### Integration with Existing Pipeline

The topological membrane functionality is automatically integrated into the `generate_metamold_red` and `generate_metamold_blue` functions:

```python
# The functions now automatically handle positive-genus surfaces
generate_metamold_red(mesh_path, mold_half_path, draw_direction)
generate_metamold_blue(mesh_path, mold_half_path, draw_direction)
```

### Testing

Use the provided test script to verify functionality:

```bash
# Run basic tests
python test_topological_membranes.py

# Test with a specific mesh
python test_topological_membranes.py path/to/your/mesh.stl
```

## Technical Details

### Homology Computation

The implementation uses a simplified approach to homology computation:

1. **Geodesic Distance Analysis**: Uses distance from mesh centroid to identify potential tunnel locations
2. **Loop Creation**: Creates loops around identified tunnel points by following mesh edges
3. **Loop Optimization**: Ensures loops are properly closed and follow tunnel geometry

### Membrane Geometry

Membranes are created using:

1. **Vertex Projection**: Project tunnel loop vertices to a plane perpendicular to the average normal
2. **Triangulation**: Use Delaunay triangulation to create a surface from projected vertices
3. **Extrusion**: Extrude the surface to give it thickness
4. **Integration**: Merge with the original mesh while preserving topology

### Segmentation Analysis

After segmentation, membranes are analyzed to determine which ones to keep:

- **Membranes with same label on both sides**: Create cuts in the mold, keep them
- **Membranes with different labels**: Don't create cuts, remove them
- **Membranes traversed by boundaries**: Don't create cuts, remove them

## Limitations and Future Work

### Current Limitations

1. **Simplified Homology**: The current implementation uses a simplified approach to homology computation
2. **Basic Loop Detection**: Tunnel loop detection is based on geodesic distance rather than full homology analysis
3. **Limited Membrane Removal**: The membrane removal process is simplified and may not handle all cases

### Future Improvements

1. **Advanced Homology**: Implement full homology computation using persistent homology or Reeb graphs
2. **Shortest Basis**: Use the shortest basis algorithm from [Dey et al. 2013] for optimal tunnel loop detection
3. **Screened Poisson**: Implement screened Poisson surface reconstruction for better membrane geometry
4. **Robust Removal**: Improve membrane removal with proper hole filling and topology preservation

## Dependencies

The implementation requires the following Python packages:

- `trimesh`: Mesh processing and manipulation
- `open3d`: Advanced mesh operations and topology analysis
- `pyvista`: Visualization and mesh display
- `numpy`: Numerical computations
- `scipy`: Scientific computing (Delaunay triangulation, sparse matrices)
- `networkx`: Graph operations for topology analysis
- `sklearn`: Nearest neighbor search for membrane analysis

## References

1. Dey, T. K., et al. "Computing a shortest basis for the homology group." *Computational Geometry*, 2013.
2. Biasotti, S., et al. "Reeb graphs for shape analysis and applications." *Theoretical Computer Science*, 2008.
3. Kazhdan, M., and Hoppe, H. "Screened Poisson surface reconstruction." *ACM Transactions on Graphics*, 2013.

## Examples

### Example 1: Torus Processing

```python
# Create a torus (genus 1)
torus = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.2)

# Process with membranes
processed_torus, membranes = process_positive_genus_surface('torus.stl')

# Visualize result
visualize_membranes(processed_torus, membranes)
```

### Example 2: Integration with Main Pipeline

```python
# The main pipeline now automatically handles positive-genus surfaces
# No changes needed to existing code
python main.py input_mesh.stl 10
```

## Troubleshooting

### Common Issues

1. **No membranes created**: Check if the mesh actually has positive genus
2. **Poor membrane quality**: Adjust membrane thickness or loop detection parameters
3. **Visualization errors**: Ensure PyVista is properly installed and configured

### Debugging

Enable debug output by modifying the functions to include more detailed logging:

```python
# Add debug prints to see what's happening
print(f"Mesh genus: {genus}")
print(f"Found {len(tunnel_loops)} tunnel loops")
print(f"Created {len(membranes)} membranes")
``` 