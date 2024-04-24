import numpy as np
import matplotlib.pyplot as plt

# Define the size of the grid
x = np.linspace(-10, 10, 11)
y = np.linspace(-10, 10, 11)

# Create a mesh grid for the x and y values
X, Y = np.meshgrid(x, y)

# Calculate the square of the distance to (0,0) for each point in the mesh grid
R2 = X**2 + Y**2

# As direction is upward, U (x component) will be 0 and V (y component) will be the magnitude
U = np.zeros_like(X)
V = 1/np.power(R2+1, 1/3)

# Normalize the vectors to make the plot more readable
magnitude = np.sqrt(U**2 + V**2)
# U = U / magnitude
# V = V / magnitude

# Plot the vector field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, scale=8, color='black', width=0.01)
#plt.title("Vector Field with Magnitude based on Distance^2 to (0,0)")
# plt.xlabel('X')
# plt.ylabel('Y')
plt.axis('off')
plt.grid('off')
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
plt.savefig('../data_generation/vf_y.png', transparent=True)




import numpy as np
import matplotlib.pyplot as plt


x,y = np.meshgrid(np.linspace(0,3,11),np.linspace(0,3,11))

u = 1 / (np.abs(x)**2.+0.1) * np.sign(x)  #x/np.sqrt(x**2) * np.sign(x)
v = 0#y/np.sqrt(x**2)

plt.quiver(x,y,u,v)
plt.axis('off')
plt.savefig('../data_generation/vf_x.png')

plt.close()

x,y = np.meshgrid(np.linspace(-1,1,11),np.linspace(-1,1,11))

v = 1 / (np.abs(x)**2.+0.5) * np.sign(x)  #x/np.sqrt(x**2) * np.sign(x)
u = 0#y/np.sqrt(x**2)

plt.quiver(np.exp(),y,u,v)
plt.axis('off')
plt.savefig('../data_generation/vf_y.png')