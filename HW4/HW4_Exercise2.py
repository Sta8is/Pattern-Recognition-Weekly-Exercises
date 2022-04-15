import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
rng = np.random.default_rng(seed=1)


def create_data(num_of_classes, size, priors_list, list_of_means, list_of_std):
    data, labels = [], []
    available_num_of_samples = size
    for j in range(num_of_classes-1):
        classSize = math.floor(size*priors_list[j])
        data.append(rng.normal(list_of_means[j], math.sqrt(list_of_std[j]), size=classSize))
        labels.append(np.ones(classSize) * j)
        available_num_of_samples -= classSize
    # for last class we need to take the remaining samples because rounding error of size*priors()
    data.append(rng.normal(list_of_means[-1], list_of_std[-1], size=available_num_of_samples))
    labels.append(np.ones(available_num_of_samples)*(num_of_classes-1))
    return data, labels


def knn_classifier(number_of_classes, train, test, train_classes, k_neighbours):
    class_votes = np.zeros((number_of_classes, len(test)), dtype=int)
    distances = np.sqrt(np.power(np.subtract(test.reshape(-1, 1).T, train.reshape(-1, 1)), 2))  # Euclidian Distances
    # distances2 = np.abs(np.subtract(test.reshape(-1, 1).T, train.reshape(-1, 1)))  # Manhattan Distance == Euclidian distances as dimension = 1
    sorted_dist_ind = np.argsort(distances, axis=0)
    Nearest_Neighbors_class = train_classes[sorted_dist_ind[0:k_neighbours]]
    for i in range(Nearest_Neighbors_class.shape[1]):
        for j in range(number_of_classes):
            class_votes[j, i] += (Nearest_Neighbors_class[:, i] == j).sum()
    knn_prediction = np.argmax(class_votes, axis=0)
    return knn_prediction


def calculate_wrong_predictions(prediction, real_classes):
    return (prediction != real_classes).sum()


def discriminant_function_1d(x, m, s, prior):
    c = x - m
    S_inv = s ** (-1)
    det_S = abs(s)
    return -0.5 * c ** 2 * S_inv - (1 / 2) * math.log(2*math.pi) - 0.5 * math.log(det_S) + math.log(prior)


def bayes_classifier(test, list_of_means, list_of_covariances, priors_list):
    g1 = np.array([discriminant_function_1d(test[t], list_of_means[0], list_of_covariances[0], priors_list[0]) for t in range(len(test))]).reshape(-1, 1)
    g2 = np.array([discriminant_function_1d(test[t], list_of_means[1], list_of_covariances[1], priors_list[1]) for t in range(len(test))]).reshape(-1, 1)
    g3 = np.array([discriminant_function_1d(test[t], list_of_means[2], list_of_covariances[2], priors_list[2]) for t in range(len(test))]).reshape(-1, 1)
    g = np.concatenate((g1, g2, g3), axis=1)
    bayes_prediction = np.argmax(g, axis=1)
    return bayes_prediction


def parzen_gaussian_kernel(x, v):
    return np.mean(norm.pdf(x, loc=0, scale=1), axis=0)/v


def parzen_classifier(train_list, test, h, apriori):
    u = []
    pdf = []
    g = []
    for i in range(len(train_list)):
        u.append(np.subtract(test.reshape(-1, 1).T, train_list[i].reshape(-1, 1))/h)
        pdf.append(parzen_gaussian_kernel(u[i], h))
        g.append(pdf[i]*apriori[i])
    parzen_prediction = np.argmax(g, axis=0)
    return parzen_prediction


# Exercise 4.2 A
priors = np.array([0.5, 0.3, 0.2])
means = np.array([2, 1, 3])
covariances = np.array([0.5, 1, 1.2])
stds = np.sqrt(covariances)
train_data_list, train_labels_list = create_data(3, 100, priors, means, stds)
test_data_list, test_labels_list = create_data(3, 1000, priors, means, stds)

train_data = np.concatenate(train_data_list)
test_data = np.concatenate(test_data_list)
train_labels = np.concatenate(train_labels_list)
test_labels = np.concatenate(test_labels_list)

# Visualization
sns.displot(train_data_list, bins=20, kde=True, stat="density", legend=False)
plt.legend(["Class 3", "Class 2", "Class 1"], prop={'size': 12})
sns.displot(train_data_list[0], bins=20, kde=True, legend=False)
plt.legend(["Class 1"], prop={'size': 12})
sns.displot(train_data_list[1], bins=20, kde=True, legend=False)
plt.legend(["Class 2"], prop={'size': 12})
sns.displot(train_data_list[2], bins=20, kde=True, legend=False)
plt.legend(["Class 3"], prop={'size': 12})
plt.show()


# Exercise 4.2 B
# Knn Classifier
K_values = [1, 2, 3]
for K in K_values:
    knn_predictions = knn_classifier(3, train_data, test_data, train_labels, K)
    knn_wrong_predictions = calculate_wrong_predictions(knn_predictions, test_labels)
    print("With k = ", K, " Knn Classifier Misclassified ", knn_wrong_predictions, "points. Total error is ", knn_wrong_predictions/len(test_labels))

# Bayes Classifier
estimated_means = [np.mean(td) for td in test_data_list]
estimated_cov = [np.cov(td) for td in test_data_list]
print("Estimated means from MLE: ", estimated_means)
print("Estimated covariances from MLE: ", estimated_cov)
bayes_predictions = bayes_classifier(test_data, estimated_means, estimated_cov, priors)
bayes_wrong_predictions = calculate_wrong_predictions(bayes_predictions, test_labels)
print("Bayes Classifier misclassified ", bayes_wrong_predictions, "points. Total error is ", bayes_wrong_predictions/len(test_labels))

# Rule for K
K_rule = int(len(train_data) ** (1/2))
K_rule_predictions = knn_classifier(3, train_data, test_data, train_labels, K_rule)
K_rule_mistakes = calculate_wrong_predictions(K_rule_predictions, test_labels)
print("Using rule k =sqrt(N)= ", K_rule, " Knn Classifier Misclassified ", K_rule_mistakes, "points. Total error is ", K_rule_mistakes/len(test_labels))

# Bruteforce Knn
knn_mistakes = []
K_test = [i for i in range(1, 50)]
for k in K_test:
    test_knn_predictions = knn_classifier(3, train_data, test_data, train_labels, k)
    knn_mistakes.append(calculate_wrong_predictions(test_knn_predictions, test_labels))
knn_best_idx = np.argmin(knn_mistakes)
k_best = K_test[knn_best_idx]
best_knn_mistakes = knn_mistakes[knn_best_idx]
print("With best k = ", k_best, " Knn Classifier Misclassified ", best_knn_mistakes, "points. Total error is ", best_knn_mistakes/len(test_labels))

# Visualize Bruteforce K-error
error = np.array(knn_mistakes)/len(test_labels)
plt.figure()
plt.plot(K_test, error)
plt.xticks([i for i in range(1, 50, 2)])
plt.xlabel("K Values")
plt.ylabel("Classification Error")
plt.show()


# Parzen Classifier
h_values = [0.05, 0.1, 0.2, 0.4, 0.8, 1]
for H in h_values:
    parzen_predictions = parzen_classifier(train_data_list, test_data, H, priors)
    parzen_wrong_predictions = calculate_wrong_predictions(parzen_predictions, test_labels)
    print("With h = ", H, " Parzen Classifier Misclassified ", parzen_wrong_predictions, "points. Total error is ", parzen_wrong_predictions/len(test_labels))


# Bruteforce Parzen
parzen_mistakes = []
h_test = np.linspace(0.05, 4, 80)
for ht in h_test:
    test_parzen_predictions = parzen_classifier(train_data_list, test_data, ht, priors)
    parzen_mistakes.append(calculate_wrong_predictions(test_parzen_predictions, test_labels))
h_best_idx = np.argmin(parzen_mistakes)
h_best = h_test[h_best_idx]
best_parzen_mistakes = parzen_mistakes[h_best_idx]
print("With best h = ", h_best, " Parzen Classified Misclassified ", best_parzen_mistakes, "points. Total error is ", best_parzen_mistakes/len(test_labels))

# Visualize Bruteforce h-error
error = np.array(parzen_mistakes)/len(test_labels)
plt.figure()
plt.plot(h_test, error)
plt.xticks(np.linspace(0.1, 4, 40))
plt.xlabel("h Values")
plt.ylabel("Classification Error")
plt.show()
