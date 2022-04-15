import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def mse_pseudoinverse(data, bin_labels):
    """
    :param data:  Matrix of all data, Size=(Num_of_features x Number of samples) -> Here Nx150
    :param bin_labels: Vector of labels of samples, Size=(1 x Number of samples) -> Here 1x150 == (150,)
    :return: weights(final_weights), errors(number of errors at each iteration)
    """
    pseudo_input = np.ones([1, data.shape[1]])
    data_aug = np.vstack([pseudo_input, data])
    pseudoinverse = np.linalg.pinv(data_aug.transpose())
    # e = 0
    # pseudoinverse_mine = np.dot(np.linalg.inv(np.dot(data_aug, data_aug.transpose())+e*np.eye(data_aug.shape[0])), data_aug)
    weights = np.dot(pseudoinverse, bin_labels).reshape(-1, 1)
    disc_fun = np.multiply(np.dot(weights.transpose(), data_aug), bin_labels)
    errors = np.sum(disc_fun < 0)
    return weights, errors


def hyperplane(x_1, x_2, weights):
    x3 = (-weights[0] - weights[1]*x_1 - weights[2]*x_2)/weights[3]
    return x3


iris_dataset = load_iris()
iris_data = iris_dataset.data.transpose()
iris_labels = iris_dataset.target
iris_data_f_123 = iris_data[0:3]
iris_data_f_234 = iris_data[1:]
unique = np.unique(iris_labels)

print("Results using Features 1,2,3: Sepal length, Sepal width, Petal length")
weights_pinv123 = []
errors_pinv123 = []
binary_labels123 = []
for i in range(len(unique)):
    binary_labels123.append(np.where(iris_labels == i, 1, -1))
    w_pinv, e_pinv = mse_pseudoinverse(iris_data_f_123, binary_labels123[i])
    weights_pinv123.append(w_pinv)
    errors_pinv123.append(e_pinv)
    print("Classifying Class ", i, " from  other 2")
    print("Weights using MSE pseudoinverse method are: \n", w_pinv)
    print(e_pinv, " out of", iris_data.shape[1], " points misclassified ")
    print("Classification error is: ", e_pinv / iris_data.shape[1])
    print("--------------------------------------------------")
print("Total errors: ", sum(errors_pinv123), ". Error percentage :", sum(errors_pinv123)/(3*iris_data.shape[1]))

print("+---"*20+"+")
print("Results using Features 2,3,4: Sepal width, Petal length, Petal width")
weights_pinv234 = []
errors_pinv234 = []
binary_labels234 = []
for i in range(len(unique)):
    binary_labels234.append(np.where(iris_labels == i, 1, -1))
    w_pinv, e_pinv = mse_pseudoinverse(iris_data_f_234, binary_labels234[i])
    weights_pinv234.append(w_pinv)
    errors_pinv234.append(e_pinv)
    print("Classifying Class ", i, " from  other 2")
    print("Weights using MSE pseudoinverse method are: \n", w_pinv)
    print(e_pinv, " out of", iris_data.shape[1], " points misclassified ")
    print("Classification error is: ", e_pinv / iris_data.shape[1])
    print("--------------------------------------------------")
print("Total errors: ", sum(errors_pinv234), ". Error percentage :", sum(errors_pinv234)/(3*iris_data.shape[1]))

points = np.linspace(0, 8, 10)
x1, x2 = np.meshgrid(points, points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Decision Boundaries for each class using features 1,2,3")
ax.scatter(iris_data_f_123[0][:50], iris_data_f_123[1][:50], iris_data_f_123[2][:50], c='r', marker='o', label='Iris Setosa')
ax.scatter(iris_data_f_123[0][50:100], iris_data_f_123[1][50:100], iris_data_f_123[2][50:100], c='g', marker='o', label='Iris Versicolor')
ax.scatter(iris_data_f_123[0][100:], iris_data_f_123[1][100:], iris_data_f_123[2][100:], c='b', marker='o', label='Iris Virginica')
ax.plot_surface(x1, x2, hyperplane(x1, x2, weights_pinv123[0]), color='r', alpha=0.5)
ax.plot_surface(x1, x2, hyperplane(x1, x2, weights_pinv123[1]), color='g', alpha=0.5)
ax.plot_surface(x1, x2, hyperplane(x1, x2, weights_pinv123[2]), color='b', alpha=0.5)
ax.legend(['Decision Boundary for Class 1: Iris Setosa',
           'Decision Boundary for Class 2: Iris Versicolor',
           'Decision Boundary for Class 3: Iris Virginica'])
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
ax.set_xlim3d([3, 8])
ax.set_ylim3d([1.5, 5])
ax.set_zlim3d([1, 8])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Decision Boundaries for each class using features 2,3,4")
ax.scatter(iris_data_f_234[0][:50], iris_data_f_234[1][:50], iris_data_f_234[2][:50], c='r', marker='o', label='Iris Setosa')
ax.scatter(iris_data_f_234[0][50:100], iris_data_f_234[1][50:100], iris_data_f_234[2][50:100], c='g', marker='o', label='Iris Versicolor')
ax.scatter(iris_data_f_234[0][100:], iris_data_f_234[1][100:], iris_data_f_234[2][100:], c='b', marker='o', label='Iris Virginica')
ax.plot_surface(x1, x2, hyperplane(x1, x2, weights_pinv234[0]), color='r', alpha=0.5)
ax.plot_surface(x1, x2, hyperplane(x1, x2, weights_pinv234[1]), color='g', alpha=0.5)
ax.plot_surface(x1, x2, hyperplane(x1, x2, weights_pinv234[2]), color='b', alpha=0.5)
ax.legend(['Decision Boundary for Class 1: Iris Setosa',
           'Decision Boundary for Class 2: Iris Versicolor',
           'Decision Boundary for Class 3: Iris Virginica'])
ax.set_xlabel('Sepal Width (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')
ax.set_xlim3d([1.5, 5])
ax.set_ylim3d([1, 8])
ax.set_zlim3d([0, 4])
plt.show()
