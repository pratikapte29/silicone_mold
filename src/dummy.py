import pyvista as pv

# Load the OBJ file
mesh = pv.read(r"/home/sumukhs-ubuntu/triangulation1.obj")

# Create a plotter and add the mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, color="lightgrey")
plotter.show()
