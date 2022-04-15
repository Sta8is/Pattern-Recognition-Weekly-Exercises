import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# ----- Exercise 3 -----
# 3D plot
# Our 2-dimensional distribution will be over variables X and Y
N = 200
X = np.linspace(-7, 7, N)
Y = np.linspace(-7, 7, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu_1 = np.array([1, 2])
mu_2 = np.array([-2, -1])
Sigma = np.array([[4, 1], [1,  9]])
# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
# The distribution on the variables X, Y packed into pos.
Z_1 = 0.25*multivariate_normal(mu_1, Sigma).pdf(pos)
Z_2 = 0.75*multivariate_normal(mu_2, Sigma).pdf(pos)
# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z_1, antialiased=True, rstride=5, cstride=5, linewidth=1, cmap="viridis")
# ax.plot_surface(X, Y, Z_2, antialiased=True, rstride=5, cstride=5, linewidth=1, cmap="viridis")
ax.plot_wireframe(X, Y, Z_1, antialiased=True, rstride=5, cstride=5, linewidth=1)
ax.plot_wireframe(X, Y, Z_2, antialiased=True, rstride=5, cstride=5, linewidth=1)
plt.contourf(X, Y, Z_1, zdir='z', offset=-0.01, alpha=0.7, cmap="viridis")
plt.contourf(X, Y, Z_2, zdir='z', offset=-0.01, alpha=0.5, cmap="viridis")
# Adjust the limits, ticks and view angle
ax.set_zlim(-0.01, 0.03)
ax.set_zticks(np.linspace(0, 0.03, 5))
ax.view_init(1, 1)
x, y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
z = (-(24/35)*x - (9/35)*y + 0.884326574382395)
# plot the plane
ax.plot_surface(x, y, z, alpha=0.2, color="r")
plt.show()
# 2D Plot
x_axis = np.arange(-10, 10, 0.0001)
plt.plot(x_axis, (1/4)*norm.pdf(x_axis, 1.2, np.sqrt(99/35)))
plt.plot(x_axis, (3/4)*norm.pdf(x_axis, -57/35, np.sqrt(99/35)))
plt.axvline(x=(31/35), color='r', linestyle='-')
plt.ylim((0, 0.3))
plt.legend(["N(1.2,2.8286)", "N(-1.6286,2.8286)", "Decision Boundary"], prop={'size': 15})
plt.show()


# ----- Exercise 4 -----
x_axis = np.arange(-10, 10, 0.001)
# Mean_1=2, Std_1=sqrt(0.5)
plt.plot(x_axis, (1/3)*norm.pdf(x_axis, 2, np.sqrt(0.5)), label="N(2,0.5)")
# Mean_1=1.5, Std_1=sqrt(0.2)
plt.plot(x_axis, (2/3)*norm.pdf(x_axis, 1.5, np.sqrt(0.2)), label="N(1.5,0.2)")
f = 1.5*np.power(x_axis, 2) - 3.5*x_axis + 1.166854634062923
plt.plot(x_axis, f, label="Decision Boundary")
plt.legend(loc="upper right")
plt.ylim((-1, 1))
plt.xlim((-1.5, 4.5))
plt.show()
