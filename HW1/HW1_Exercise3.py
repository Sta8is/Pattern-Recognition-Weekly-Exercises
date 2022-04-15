import numpy as np
import matplotlib.pyplot as plt

# Ερώτημα α
num_of_throws = 1000
rng = np.random.default_rng(seed=123)
z1 = rng.integers(low=1, high=6, endpoint=True, size=num_of_throws)
z2 = rng.integers(low=1, high=6, endpoint=True, size=num_of_throws)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, x_edges, y_edges = np.histogram2d(z1, z2, bins=[i+0.5 for i in range(0, 7)], density=True)
x_pos, y_pos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25, indexing="ij")
x_pos = x_pos.ravel()
y_pos = y_pos.ravel()
z_pos = 0
dx = dy = 0.5 * np.ones_like(z_pos)
dz = hist.ravel()
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, zsort='average')
plt.show()

# Ερώτημα β
z_temp = 1/6 * np.ones(6)
con = np.convolve(z_temp, z_temp, "full")
print(con)
plt.plot(con, "-ok")
plt.show()

y = z1+z2
plt.hist(y, bins=[i for i in range(2, 14)], align="left", rwidth=0.75)
plt.xticks([i for i in range(1, 14)])
plt.title("Y=z1+z2")
plt.show()



