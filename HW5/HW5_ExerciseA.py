from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


def batch_perceptron(data, bin_labels, learning_rate, max_epochs, init_weights, theta=10**(-5)):
    """
    :param data: Matrix of all data, Size=(Num_of_features x Number of samples) -> Here 4x150
    :param bin_labels: Vector of labels of samples, Size=(1 x Number of samples) -> Here 1x150 == (150,)
    :param learning_rate: The step size at each iteration while moving toward a minimum of a loss function
    :param max_epochs: The maximum number of epochs
    :param init_weights: Initial weights, Size=(Number of features x 1) -> Here 4x1
    :param theta: Stopping tolerance
    :return: weights(final_weights), iteration(last_iteration_number), errors(number of errors at each iteration), criteria(value of criterion at each iteration)
    """
    iteration = 0
    errors, criteria = [], []
    pseudo_input = np.ones([1, data.shape[1]])
    data_aug = np.vstack([pseudo_input, data])  # Augmented data matrix
    weights = np.vstack([1, init_weights])  # Augmented weight vector
    while iteration < max_epochs:
        activation = np.multiply(np.dot(weights.transpose(), data_aug), bin_labels)
        wrong_class = np.where(activation < 0, 1, 0)
        errors.append(np.sum(wrong_class))
        error_grad = np.sum(np.where(wrong_class == 1, data_aug * bin_labels, 0), axis=1).reshape(-1, 1)
        iteration += 1
        criterion = np.sum(np.abs(learning_rate*error_grad))/data_aug.shape[1]
        criteria.append(criterion)
        if criterion > theta:
            weights = weights + learning_rate*error_grad
        else:
            break
        # if errors[iteration] > 0:
        #     error_grad = np.sum(np.where(wrong_class == 1, data_aug*bin_labels, 0), axis=1).reshape(-1, 1)
        #     weights = weights + learning_rate*error_grad
        #     print(np.sum(abs(learning_rate*error_grad)))
        #     iteration += 1
        # else:
        #     iteration += 1
        #     break
    return weights, iteration, errors, criteria


def batch_relaxation_with_margin(data, bin_labels, learning_rate, max_epochs, init_weights, b):
    """
    :param data: Matrix of all data, Size=(Num_of_features x Number of samples) -> Here 4x150
    :param bin_labels: Vector of labels of samples, Size=(1 x Number of samples) -> Here 1x150 == (150,)
    :param learning_rate: The step size at each iteration while moving toward a minimum of a loss function
    :param max_epochs: The maximum number of epochs
    :param init_weights: Initial weights, Size=(Number of features x 1) -> Here 4x1
    :param b: Margin
    :return: weights(final_weights), iteration(last_iteration_number), errors(number of errors at each iteration)
    """
    iteration = 0
    errors = []
    pseudo_input = np.ones([1, data.shape[1]])
    data_aug = np.vstack([pseudo_input, data])
    weights = np.vstack([1, init_weights])
    while iteration < max_epochs:
        activation = np.multiply(np.dot(weights.transpose(), data_aug), bin_labels)
        wrong_class = np.where(activation < b, 1, 0)
        error_count = np.sum(wrong_class)
        errors.append(error_count)
        iteration += 1
        if error_count > 0:
            numerator = (b - np.dot(weights.transpose(), data_aug)).reshape(-1, 1)
            denominator = np.sum(np.where(wrong_class == 1, data_aug, 0)**2, axis=0).reshape(-1, 1)
            wrong_class_idx = np.where(wrong_class == 1, True, False)
            grad = (numerator[wrong_class_idx.transpose()]/denominator[wrong_class_idx.transpose()]).reshape(-1, 1)
            error_grad = np.dot(np.ones([data_aug.shape[0], 1]), grad.transpose())
            temp = np.multiply(error_grad, data_aug[:, wrong_class_idx[0]])
            update = np.sum(temp, axis=1).reshape(-1, 1)
            # lr = learning_rate * (1 / (1 + 0.5 * iteration))
            # print(lr)
            weights = weights + learning_rate*update
            # weights = weights + lr*update
        else:
            break
    return weights, iteration, errors


iris_dataset = load_iris()
iris_data = iris_dataset.data.transpose()
iris_labels = iris_dataset.target
binary_labels = np.where(iris_labels == 0, -1, 1)
initial_weights = np.ones([iris_data.shape[0], 1])

best_weights_per, iterations_per, error_list_per, criteria_per = batch_perceptron(iris_data, binary_labels, 0.025, 1000, initial_weights, 10**(-5))
print("Final weights using Batch Perceptron are: \n", best_weights_per)
print("Batch Perceptron finished terminated after ", iterations_per, " epochs with error ", error_list_per[-1]/150)
print("Values of Criterion are: ", criteria_per)
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(np.linspace(1, iterations_per, iterations_per), np.array(error_list_per)/150)
ax[0].set_title("Batch Perceptron Classification Errors")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Classification Error")
ax[1].plot(np.linspace(1, iterations_per, iterations_per), np.array(criteria_per))
ax[1].set_title("Batch Perceptron Criterion Values")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Criterion Value")
fig.tight_layout()
plt.show()


best_weights_per_relax, iterations_per_relax, error_list_per_relax = batch_relaxation_with_margin(iris_data, binary_labels, 0.04, 1000, initial_weights, 1.5)
print("Final weights using Batch (Perceptron) Relaxation with Margin are : \n", best_weights_per_relax)
print("Batch (Perceptron) Relaxation with Margin finished terminated  with Margin after ", iterations_per_relax, " epochs with error ", error_list_per_relax[-1]/iris_data.shape[1])
plt.figure()
plt.plot(np.linspace(1, iterations_per_relax, iterations_per_relax), np.array(error_list_per_relax)/iris_data.shape[1])
plt.title("Batch (Perceptron) Relaxation with Margin")
plt.xlabel("Epochs")
plt.ylabel("Classification Error")
plt.show()
