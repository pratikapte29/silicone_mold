#!/usr/bin/env python3
"""
Example script demonstrating topological membrane functionality.
This script shows how to process a positive-genus surface and integrate
it with the existing metamold generation pipeline.
"""

import sys
import os
import numpy as np
import trimesh

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generate_metamold import (
    process_positive_genus_surface,
    visualize_membranes,
    generate_metamold_red,
    generate_metamold_blue
)


def example_with_torus():
    """Example using a torus (genus 1)."""
    print("=== Example: Processing Torus ===")
    
    # Create a torus
    torus = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.2)
    
    # Save the torus
    torus_path = "example_torus.stl"
    torus.export(torus_path)
    print(f"Created torus: {torus_path}")
    
    # Process with topological membranes
    processed_mesh, membranes_info = process_positive_genus_surface(torus_path)
    
    if membranes_info:
        print(f"Added {len(membranes_info)} topological membranes")
        
        # Save processed mesh
        processed_path = "example_torus_with_membranes.stl"
        processed_mesh.export(processed_path)
        print(f"Saved processed mesh: {processed_path}")
        
        # Visualize the result
        print("Opening visualization...")
        visualize_membranes(processed_mesh, membranes_info)
    else:
        print("No membranes were created")
    
    print()


def example_with_existing_mesh(mesh_path):
    """Example using an existing mesh file."""
    print(f"=== Example: Processing {mesh_path} ===")
    
    if not os.path.exists(mesh_path):
        print(f"Mesh file not found: {mesh_path}")
        return
    
    # Process with topological membranes
    processed_mesh, membranes_info = process_positive_genus_surface(mesh_path)
    
    if membranes_info:
        print(f"Added {len(membranes_info)} topological membranes")
        
        # Save processed mesh
        base_name = os.path.splitext(mesh_path)[0]
        processed_path = f"{base_name}_with_membranes.stl"
        processed_mesh.export(processed_path)
        print(f"Saved processed mesh: {processed_path}")
        
        # Visualize the result
        print("Opening visualization...")
        visualize_membranes(processed_mesh, membranes_info)
        
        # Example: Use in metamold generation (if you have the required files)
        print("\nNote: To use this in metamold generation, you would call:")
        print(f"generate_metamold_red('{processed_path}', 'mold_half.stl', draw_direction)")
        print(f"generate_metamold_blue('{processed_path}', 'mold_half.stl', draw_direction)")
    else:
        print("No membranes were created")
    
    print()


def example_integration_with_pipeline():
    """Example showing integration with the main pipeline."""
    print("=== Example: Integration with Main Pipeline ===")
    
    # Create a test mesh with positive genus
    torus = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.2)
    torus_path = "pipeline_test_torus.stl"
    torus.export(torus_path)
    
    # Create dummy mold halves for demonstration
    # In practice, these would come from the main pipeline
    red_half = trimesh.creation.box(extents=[1, 1, 0.5])
    blue_half = trimesh.creation.box(extents=[1, 1, 0.5])
    
    red_path = "pipeline_test_red.stl"
    blue_path = "pipeline_test_blue.stl"
    
    red_half.export(red_path)
    blue_half.export(blue_path)
    
    # Example draw direction
    draw_direction = np.array([0, 0, 1])
    
    print("The metamold generation functions now automatically handle positive-genus surfaces:")
    print("1. They detect if the mesh has positive genus")
    print("2. They add topological membranes if needed")
    print("3. They proceed with the standard metamold generation")
    print()
    
    print("To test this, you would run:")
    print(f"python main.py {torus_path} 10")
    print()
    
    print("The main.py script will automatically:")
    print("- Detect the positive genus")
    print("- Add topological membranes")
    print("- Generate the metamold halves")
    print("- Save results to the results/ directory")
    
    print()


def main():
    """Main example function."""
    print("Topological Membrane Examples")
    print("=" * 40)
    
    # Example 1: Torus
    example_with_torus()
    
    # Example 2: Integration with pipeline
    example_integration_with_pipeline()
    
    # Example 3: With existing mesh if provided
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
        example_with_existing_mesh(mesh_path)
    else:
        print("To test with a specific mesh, provide the mesh path:")
        print("python example_topological_membranes.py path/to/your/mesh.stl")
        print()
    
    print("Examples completed!")
    print()
    print("Next steps:")
    print("1. Run the test suite: python test_topological_membranes.py")
    print("2. Try with your own mesh: python main.py your_mesh.stl 10")
    print("3. Check the documentation: docs/topological_membranes.md")


if __name__ == "__main__":
    main() 