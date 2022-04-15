from sklearn.datasets import load_iris
import numpy as np
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


def lms_widrow_hoff(data, bin_labels, learning_rate, max_epochs, init_weights, theta):
    """
    :param data:  Matrix of all data, Size=(Num_of_features x Number of samples) -> Here 4x150
    :param bin_labels: Vector of labels of samples, Size=(1 x Number of samples) -> Here 1x150 == (150,)
    :param learning_rate: The step size at each iteration while moving toward a minimum of a loss function
    :param max_epochs: The maximum number of epochs
    :param init_weights: Initial weights, Size=(Number of features x 1) -> Here 4x1
    :param theta: Stopping tolerance
    :return: weights(final_weights), iteration(last_iteration_number), errors(number of errors at each iteration)
    """
    iteration = 0
    pseudo_input = np.ones([1, data.shape[1]])
    data_aug = np.vstack([pseudo_input, data])
    # weights = np.vstack([1, init_weights])
    weights = np.sum(data_aug, axis=1).reshape(-1, 1)/data_aug.shape[1]
    # weights = np.random.rand(5).reshape(-1, 1)
    b = bin_labels
    criteria = []
    errors = []
    while iteration < max_epochs:
        batch_update = learning_rate * (b - np.dot(weights.transpose(), data_aug))*data_aug
        criterion = np.sum(np.abs(np.sum(batch_update, axis=1)))/data_aug.shape[1]
        criteria.append(criterion)
        iteration += 1
        activation = np.multiply(np.dot(weights.transpose(), data_aug), bin_labels)
        errors.append(np.sum(activation < 0))
        if criterion > theta:
            weights = weights + (np.sum(batch_update, axis=1)/150).reshape(-1, 1)
        else:
            break
        # for i in range(data_aug.shape[1]):
        #     update = learning_rate * (b[i] - np.dot(weights.transpose(), data_aug_temp[:, i]))*data_aug_temp[:, i]
        #     if np.sum(np.abs(update)) > theta:
        #         weights += update.reshape(-1, 1)
        #     else:
        #         activation = np.multiply(np.dot(weights.transpose(), data_aug), bin_labels)
        #         errors = np.sum(activation < 0)
        #         return weights, errors, iteration
        # iteration += 1
    return weights, errors, iteration, criteria


iris_dataset = load_iris()
iris_data = iris_dataset.data.transpose()
iris_labels = iris_dataset.target
binary_labels = np.where(iris_labels == 0, 1, -1)
initial_weights = np.ones([iris_data.shape[0], 1])

weights_pinv, errors_pinv = mse_pseudoinverse(iris_data, binary_labels)
print("Weights using MSE with pseudoinverse method are: \n", weights_pinv)
print(errors_pinv, " out of", iris_data.shape[1], " points misclassified using MSE with pseudoinverse method")
print("Classification error sing MSE with pseudoinverse method is: ", errors_pinv/iris_data.shape[1])

weights_lms, errors_lms, iterations_lms, criteria_lms = lms_widrow_hoff(iris_data, binary_labels, 0.02, 100, initial_weights, 0.02)
print("Final weights using LMS Widrow-Hoff are: \n", weights_lms)
print(errors_lms[-1], " out of", iris_data.shape[1], " points misclassified ")
print("LMS Widrow-Hoff finished terminated after ", iterations_lms, " epochs with error ", errors_lms[-1]/iris_data.shape[1])
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(np.linspace(1, iterations_lms, iterations_lms), np.array(errors_lms)/iris_data.shape[1])
axes[0].set_title("LMS Widrow Hoff Classification Error")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Classification Error")
axes[1].plot(np.linspace(1, iterations_lms, iterations_lms), np.array(criteria_lms)/iris_data.shape[1])
axes[1].set_title("LMS Widrow Hoff Stopping Criterion")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Criterion")
fig.tight_layout()
plt.show()
