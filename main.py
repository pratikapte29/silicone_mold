from src.finalize_draw_direction import FinalizeDrawDirection
from src.convex_hull_operations import compute_convex_hull_from_stl
import time

start_time = time.time()
mesh_path = "assets/stl/xyzrgb_dragon.stl"
num_vectors = 1000

fd = FinalizeDrawDirection(mesh_path, num_vectors)

candidate_vectors = fd.createCandidateVectors()

draw_direction = fd.computeVisibleAreas(candidate_vectors)
end_time = time.time()
print(draw_direction)
print(f"Time required for {num_vectors} candidates is {end_time - start_time:.2f} seconds")
