from src.split_mesh import offset_stl
import numpy as np
import pyvista as pv

file_path = r"..\assets\stl\lucy.stl"

pv_mesh, offset_mesh = offset_stl(file_path, 100)

plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color="lightgray", label="Original")
plotter.add_mesh(offset_mesh, color="red", opacity=1, label="Offset")
plotter.add_legend()
plotter.show()
