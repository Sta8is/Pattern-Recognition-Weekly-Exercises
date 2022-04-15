import sympy

import HW3_Exercise1 as Ex1
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sympy as sym


# Read data from file
my_data = np.load("my_data.npy")

# Assign data for each class
w1_data = my_data[:, 0:3]
w2_data = my_data[:, 3:6]
w3_data = my_data[:, 6:]

# Estimating parameters/statistics of Gaussian's
w1_means = np.mean(w1_data, axis=0)
w2_means = np.mean(w2_data, axis=0)
w3_means = np.mean(w3_data, axis=0)
w1_covariance = np.cov(w1_data, rowvar=False)
w2_covariance = np.cov(w2_data, rowvar=False)
w3_covariance = np.cov(w3_data, rowvar=False)

# ---------- Exercise 2.1 ----------
# Mean and covariance estimation using only feature x1 (First column or column at index 0)
one_dim_w1_mean = w1_means[0]  # mean of class 1 for feature x1
one_dim_w2_mean = w2_means[0]  # mean of class 2 for feature x1
one_dim_w1_covariance = w1_covariance[0, 0]  # covariance of class 1 for feature x1
one_dim_w2_covariance = w2_covariance[0, 0]  # covariance of class 1 for feature x1

# Solving equation with discrete steps
tol = 10 ** (-5)  # tolerance
x = np.arange(-10, 10, tol*10)  # Defining the space where we seek for the solution
g = Ex1.discriminant_function(x, one_dim_w1_mean, one_dim_w1_covariance, 0.5, 1) - Ex1.discriminant_function(x, one_dim_w2_mean, one_dim_w2_covariance, 0.5, 1)
index = list(zip(*np.where(abs(g) < tol)))  # Finding the solutions with tolerance
for i in range(len(list(index))):
    print("g1(x)-g2(x)=0 for x: ", x[index[i]], "", "with error tolerance: ", g[index[i]])
# Solving equation using symbolic variables
z1 = sym.Symbol("z1")
eq = Ex1.discriminant_function(z1, one_dim_w1_mean, one_dim_w1_covariance, 0.5, 1) - Ex1.discriminant_function(z1, one_dim_w2_mean, one_dim_w2_covariance, 0.5, 1)
sol = sympy.solve(eq)
print("Solutions using sympy: ", sol[0], " and ", sol[1])

# Calculating classification error
g_a = Ex1.discriminant_function(w1_data[:, 0], one_dim_w1_mean, one_dim_w1_covariance, 0.5, 1) - Ex1.discriminant_function(w1_data[:, 0], one_dim_w2_mean, one_dim_w2_covariance, 0.5, 1)
g_b = Ex1.discriminant_function(w2_data[:, 0], one_dim_w1_mean, one_dim_w1_covariance, 0.5, 1) - Ex1.discriminant_function(w2_data[:, 0], one_dim_w2_mean, one_dim_w2_covariance, 0.5, 1)
wrong = (g_a < 0).sum() + (g_b > 0).sum()
print(wrong, "/", 20, " points misclassified using only feature x1. Error is : ", wrong/20 * 100, "%")


# ---------- Exercise 2.3 ----------
# Mean and covariance estimation using features x1,x2 (First and second column or column at index 0 and 1)
two_dim_w1_mean = w1_means[0:2]  # means of class 1 for features x1,x2
two_dim_w2_mean = w2_means[0:2]  # means of class 2 for features x1,x2
two_dim_w1_covariance = w1_covariance[0:2, 0:2]  # covariance matrix of class 1 for feature x1,x2
two_dim_w2_covariance = w2_covariance[0:2, 0:2]  # covariance matrix of class 2 for feature x1,x2

wrong = 0
for i in range(10):
    g_a = Ex1.discriminant_function(w1_data[i, 0:2], two_dim_w1_mean, two_dim_w1_covariance, 0.5, 2) - Ex1.discriminant_function(w1_data[i, 0:2], two_dim_w2_mean, two_dim_w2_covariance, 0.5, 2)
    g_b = Ex1.discriminant_function(w2_data[i, 0:2], two_dim_w1_mean, two_dim_w1_covariance, 0.5, 2) - Ex1.discriminant_function(w2_data[i, 0:2], two_dim_w2_mean, two_dim_w2_covariance, 0.5, 2)
    if g_a < 0 and g_b > 0:
        wrong += 2
    elif g_a < 0 or g_b > 0:
        wrong += 1
print(wrong, "/", 20, " points misclassified using features x1 and x2. Error is : ", wrong/20 * 100, "%")


# ---------- Exercise 2.4 ----------
wrong = 0
for i in range(10):
    g_a = Ex1.discriminant_function(w1_data[i, :], w1_means, w1_covariance, 0.5, 3) - Ex1.discriminant_function(w1_data[i, :], w2_means, w2_covariance, 0.5, 3)
    g_b = Ex1.discriminant_function(w2_data[i, :], w1_means, w1_covariance, 0.5, 3) - Ex1.discriminant_function(w2_data[i, :], w2_means, w2_covariance, 0.5, 3)
    if g_a < 0 and g_b > 0:
        wrong += 2
    elif g_a < 0 or g_b > 0:
        wrong += 1
print(wrong, "/", 20, " points misclassified using all features x1, x2, x3. Error is : ", wrong/20 * 100, "%")


# ---------- Exercise 2.6 ----------
print("\n Symbolic Equations for g1,g2,g3: \n")
x1, x2, x3 = sym.symbols("x1 x2 x3")
matrix = sym.Matrix([x1, x2, x3])
# Original shape was (3,)
g_1 = sym.simplify(Ex1.discriminant_function(matrix, w1_means.reshape((3, 1)), w1_covariance, 0.8, 3))
g_2 = sym.simplify(Ex1.discriminant_function(matrix, w2_means.reshape((3, 1)), w2_covariance, 0.1, 3))
g_3 = sym.simplify(Ex1.discriminant_function(matrix, w3_means.reshape((3, 1)), w3_covariance, 0.1, 3))
print(" g1(x) = ", g_1, "\n", "g2(x) = ", g_2, "\n", "g3(x) = ", g_3, "\n")


# Plot results for Exercise 2.1
fig = plt.figure()
x_axis = np.linspace(-10, 10, 10000)
plt.plot(x_axis, (1/2)*norm.pdf(x_axis, one_dim_w1_mean, np.sqrt(one_dim_w1_covariance)), "b")
plt.plot(x_axis, (1/2)*norm.pdf(x_axis, one_dim_w2_mean, np.sqrt(one_dim_w2_covariance)), "y")
plt.plot(w1_data[:, 0], (1/2)*norm.pdf(w1_data[:, 0], one_dim_w1_mean, np.sqrt(one_dim_w1_covariance)), "bo")
plt.plot(w2_data[:, 0], (1/2)*norm.pdf(w2_data[:, 0], one_dim_w2_mean, np.sqrt(one_dim_w2_covariance)), "yo")
res = np.array(sol, dtype=np.float64)
plt.plot(res, (1/2)*norm.pdf(res,  one_dim_w1_mean, np.sqrt(one_dim_w1_covariance)), 'ro')
plt.vlines(res, 0, 0.1, colors="g")
plt.legend(["Gaussian 1", "Gaussian 2", "Gaussian 1 Data", "Gaussian 2 Data", "Intersection points", "Decision Boundaries"], prop={'size': 15})
plt.show()

