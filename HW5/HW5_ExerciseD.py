import numpy as np
from sklearn.datasets import load_iris


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


iris_dataset = load_iris()
iris_data = iris_dataset.data.transpose()
iris_labels = iris_dataset.target
unique = np.unique(iris_labels)
weights_pinv = []
errors_pinv = []
binary_labels = []
for i in range(len(unique)):
    binary_labels.append(np.where(iris_labels == i, 1, -1))
    w_pinv, e_pinv = mse_pseudoinverse(iris_data, binary_labels[i])
    weights_pinv.append(w_pinv)
    errors_pinv.append(e_pinv)
    print("Classifying Class ", i, " from  other 2")
    print("Weights using MSE pseudoinverse method are: \n", w_pinv)
    print(e_pinv, " out of", iris_data.shape[1], " points misclassified ")
    print("Classification error is: ", e_pinv / iris_data.shape[1])
    print("--------------------------------------------------")
