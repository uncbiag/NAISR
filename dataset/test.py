import numpy as np

import pyvista as pv


values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
values = np.ones(values.shape)
values[5:10, 3:4, 4:6] = 0
# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(values.shape)

# Edit the spatial reference
grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.point_data["values"] = values.flatten(order="F")  # Flatten the array!

# Now plot the grid!
grid.plot(show_edges=True, opacity=0.4, )