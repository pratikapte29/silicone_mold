from src.split_mesh import offset_stl
from src.split_mesh import offset_stl_sdf
import numpy as np
import pyvista as pv

file_path = r"..\assets\stl\lucy.stl"

pv_mesh, offset_mesh = offset_stl(file_path, 150)
offset_mesh_sdf = offset_stl_sdf(file_path, 150)
print(len(offset_mesh_sdf.faces))

plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color="lightgray", label="Original")
plotter.add_mesh(offset_mesh_sdf, color="red", opacity=0.5, label="Offset")
plotter.add_legend()
plotter.show()
