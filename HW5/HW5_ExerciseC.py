import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def mse_pseudoinverse(data, bin_labels):
    """
    :param data:  Matrix of all data, Size=(Num_of_features x Number of samples) -> Here 4x150
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


def ho_kashyap(data, bin_labels, learning_rate, max_epochs, b_min):
    """
    :param data: Matrix of all data, Size=(Num_of_features x Number of samples) -> Here 4x150
    :param bin_labels: Vector of labels of samples, Size=(1 x Number of samples) -> Here 1x150 == (150,)
    :param learning_rate: The step size at each iteration while moving toward a minimum of a loss function
    :param max_epochs: The maximum number of epochs
    :param b_min: Stopping criterion
    :return:
    """
    iteration = 0
    pseudo_input = np.ones([1, data.shape[1]])
    data_aug = np.vstack([pseudo_input, data])
    b = np.ones([1, data_aug.shape[1]])
    Y = (data_aug*bin_labels).transpose()
    # weights = np.random.rand(5).reshape(-1, 1)
    weights = np.ones(5).reshape(-1, 1)
    criteria = []
    errors = []
    while iteration < max_epochs:
        e = np.dot(Y, weights).transpose() - b
        iteration += 1
        activation = np.multiply(np.dot(weights.transpose(), data_aug), bin_labels)
        errors.append(np.sum(activation < 0))
        criterion = np.sum(np.abs(e))
        criteria.append(criterion)
        if criterion > b_min:
            e_plus = 0.5*(e+np.abs(e))
            b += 2 * learning_rate * e_plus
            weights = np.dot(np.linalg.pinv(Y), b.transpose())
        else:
            break
    return weights, errors, iteration, criteria


iris_dataset = load_iris()
iris_data = iris_dataset.data.transpose()
iris_labels = iris_dataset.target
iris_data_c12 = iris_data[:, 50:]
iris_labels_c12 = iris_labels[50:]
binary_labels_c12 = np.where(iris_labels_c12 == 1, 1, -1)

weights_pinv, errors_pinv = mse_pseudoinverse(iris_data_c12, binary_labels_c12)
print("Weights using MSE pseudoinverse method are: \n", weights_pinv)
print(errors_pinv, " out of", iris_data_c12.shape[1], " points misclassified using MSE pseudoinverse method ")
print("Classification error using MSE pseudoinverse method  is: ", errors_pinv/iris_data_c12.shape[1])

weights_hk, errors_hk, iterations_hk, criteria_hk = ho_kashyap(iris_data_c12, binary_labels_c12, 0.01, 1000, 0.0001)
print("Final weights using Ho-Kashyap are: \n", weights_hk)
print(errors_hk[-1], " out of", iris_data_c12.shape[1], " points misclassified ")
print("Ho-Kashyap terminated after ", iterations_hk, " epochs with error ", errors_hk[-1]/iris_data_c12.shape[1])
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(np.linspace(1, iterations_hk, iterations_hk), np.array(errors_hk)/iris_data_c12.shape[1])
axes[0].set_title("Ho-Kashyap Classification Error")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Classification Error")
axes[1].plot(np.linspace(1, iterations_hk, iterations_hk), np.array(criteria_hk)/iris_data_c12.shape[1])
axes[1].set_title("Ho-Kashyap Stopping Criterion")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Criterion")
fig.tight_layout()
plt.show()
