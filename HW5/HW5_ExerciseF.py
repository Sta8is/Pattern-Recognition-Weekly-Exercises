import numpy as np
from sklearn.datasets import load_iris


def keslers_construction(data_augmented, class_labels, max_epochs):
    dims, num_of_samples = data_augmented.shape
    unique_classes = np.unique(class_labels)
    num_of_classes = len(unique_classes)
    weights = np.zeros([dims, num_of_classes])
    iteration = 0
    flag = False
    while iteration < max_epochs and not flag:
        iteration += 1
        flag = True
        for i in unique_classes:
            class_i = np.argwhere(class_labels == i)
            class_i = np.squeeze(class_i)
            activ_i = np.dot(weights[:, i].transpose(), data_augmented[:, class_i])
            for j in np.setdiff1d(unique_classes, i*np.ones(1)):
                activ_j = np.dot(weights[:, j], data_augmented[:, class_i])
                diff = activ_i - activ_j
                min_diff_idx = np.argmin(diff, axis=0)
                idx = np.atleast_1d(min_diff_idx)
                if diff[idx] <= 0:
                    idx = class_i[idx[0]]
                    weights[:, i] += data_augmented[:, idx]
                    weights[:, j] -= data_augmented[:, idx]
                    flag = False
                    break
            if not flag:
                break
    return weights, iteration


def hyperplane(x_1, x_2, weights):
    x3 = (-weights[0] - weights[1]*x_1 - weights[2]*x_2)/weights[3]
    return x3


iris_dataset = load_iris()
iris_data = iris_dataset.data.transpose()
iris_labels = iris_dataset.target
iris_data_aug = np.vstack([np.ones([1, iris_data.shape[1]]), iris_data])
weights_k, iteration_k = keslers_construction(iris_data_aug, iris_labels, 1000)
print(weights_k)
print(iteration_k)
activations = np.dot(weights_k.transpose(), iris_data_aug)
predictions = np.argmax(activations, axis=0)
errors = np.where(predictions == iris_labels, 0, 1)
print("Total errors are ", errors.sum(), " out of ", iris_data.shape[1])
print("Classification error is ", errors.sum()/iris_data.shape[1])
print("----------------------------------------------------------")
