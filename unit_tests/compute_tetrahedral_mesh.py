import gmsh
import meshio
import pyvista as pv

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.merge(r"assets/stl/cow.stl")

# Create a surface loop and volume
s = gmsh.model.getEntities(2)  # get all surfaces
gmsh.model.addPhysicalGroup(2, [s[0][1]], tag=1)

# Create surface loop and volume
sl = gmsh.model.geo.addSurfaceLoop([s[0][1]])
vol = gmsh.model.geo.addVolume([sl])
gmsh.model.addPhysicalGroup(3, [vol], tag=2)

gmsh.model.geo.synchronize()

# Now generate 3D tetrahedral mesh
gmsh.model.mesh.generate(3)
gmsh.write("mesh.msh")
gmsh.finalize()

# --- Convert to VTK using meshio ---
mesh = meshio.read("mesh.msh")
meshio.write("output.vtu", mesh)  # Use .vtu for unstructured tetrahedral mesh

# --- Load in PyVista ---
grid = pv.read("output.vtu")

# --- Clip at centroid ---
centroid = grid.center
clipped = grid.clip(normal='z', origin=centroid, invert=False)

# --- Visualize ---
plotter = pv.Plotter()
plotter.add_mesh(clipped, show_edges=True, color='lightblue')
plotter.add_mesh(grid.outline(), color='black')
plotter.show()
