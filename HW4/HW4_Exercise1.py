import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
rng = np.random.default_rng(seed=1234)


def parzen_gaussian_kernel(v, h):
    return np.mean(norm.pdf(v, loc=0, scale=1), axis=0)/h


x_theoretical = np.linspace(-1, 3, 4001)
p_theoretical = np.where((0 < x_theoretical) & (x_theoretical < 2), 1/2, 0)
plt.figure()
plt.plot(x_theoretical, p_theoretical, label="Theoretical PDF p(x)")
plt.legend(prop={'size': 12})
plt.show()

# Parzen Windows
N_values = [32,256,5000]
h_values = [0.05, 0.2]
for j in range(len(N_values)):
    X = rng.uniform(0, 2, N_values[j]).reshape((N_values[j], 1))  # N samples from Uniform Distribution in range (0,2)
    plt.figure()
    plt.plot(x_theoretical, p_theoretical, label="Theoretical PDF p(x)")
    for i in range(len(h_values)):
        x_points = np.linspace(-1, 3, 401).reshape((1, 401))  # Points in x axis where we try to estimate pdf
        u = np.subtract(x_points, X)/h_values[i]
        temp = (1/(np.sqrt(2*np.pi)))*np.exp((-1/2)*(u**2))  # To plot Gaussian Distributions  at N points
        # for p in range(temp.shape[0]):  # To plot Gaussian Distributions  at N points
        #     plt.plot(x_points.reshape(401, 1), temp[p], label="Gaussian "+str(p+1))  # To plot Gaussian Distributions  at N points
        #     plt.plot(X, np.zeros(32), 'ro')  # To plot Gaussian Distributions  at N points
        ph = np.mean((1/((np.sqrt(2*np.pi))*h_values[i]))*np.exp((-1/2)*(u**2)), axis=0)
        # ph = parzen_gaussian_kernel(u, h_values[i])
        plt.plot(x_points.reshape(401, 1), ph, label="N="+str(N_values[j])+", h= "+str(h_values[i]))
    plt.legend()
    plt.show()


# K Nearest Neighbors
X = rng.uniform(0, 2, 5000).reshape((5000, 1))
K_values = [32, 64, 256]
plt.figure()
plt.plot(x_theoretical, p_theoretical, label="Theoretical PDF p(x)")
for j in range(len(K_values)):
    x_points = np.linspace(0, 2, 201).reshape((1, 201))
    distances = np.abs(np.subtract(x_points, X))  # Manhattan Distance (L1) = Euclidian Distance (L2) because dim = 1
    sorted_dist_ind = np.argsort(distances, axis=0)
    Nearest_Neighbors = X[sorted_dist_ind[0:K_values[j]]]
    V = np.max(Nearest_Neighbors, axis=0)-np.min(Nearest_Neighbors, axis=0)
    ph = K_values[j]/(5000*V)
    plt.plot(x_points.reshape(201, 1), ph, label="K=" + str(K_values[j]), linewidth=j/2+1)
plt.legend(prop={'size': 12})
plt.show()

