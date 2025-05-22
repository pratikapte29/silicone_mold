from meshlib import mrmeshpy

# load closed mesh
closedMesh = mrmeshpy.loadMesh(r"..\assets\stl\lucy.stl")

# setup offset parameters
params = mrmeshpy.OffsetParameters()
params.voxelSize = 0.04
# params.type = mrmeshpy.OffsetParametersType.Offset  # requires closed mesh

# create positive offset mesh
posOffset = mrmeshpy.offsetMesh(closedMesh, 0.2, params)

# save results
mrmeshpy.saveMesh(posOffset, "posOffset.stl")
