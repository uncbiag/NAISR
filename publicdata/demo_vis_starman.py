import numpy as np
import matplotlib.pyplot as plt

from skimage import measure


# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp(np.sin(x)**3 + np.cos(y)**2))

# Find contours at a constant value of 0.8
contours = measure.find_contours(r, 0.5)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

datapath = "/playpen-raid/jyn/NAISR/NAISR/examples/starman/train/2dsdf/0000_0_off.npy"
np_sdf = np.load(datapath)


plt.scatter(x=np_sdf[:, 0], y=np_sdf[:, 1], cmap='coolwarm', c=np_sdf[:, 2], s=1)
plt.colorbar()
plt.show()

