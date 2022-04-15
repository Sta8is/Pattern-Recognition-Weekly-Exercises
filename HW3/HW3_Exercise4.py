import numpy as np
import math
import HW3_Exercise1 as Ex1

# np.random.seed(35)


def create_data(num_of_classes, size, priors_list, list_of_means, list_of_covariance_m):
    rng = np.random.default_rng(seed=0)
    Data = []
    available_num_of_samples = size
    for j in range(num_of_classes-1):
        classSize = math.floor(size*priors_list[j])
        Data.append(rng.multivariate_normal(list_of_means[j], list_of_covariance_m[j], size=classSize))
        available_num_of_samples -= classSize
    # for last class we need to take the remaining samples cause for sum of p we might not have round number of data
    Data.append(rng.multivariate_normal(list_of_means[-1], list_of_covariance_m[-1], size=available_num_of_samples))
    return Data


training_size = 10000
testing_size = 1000

prior_1 = prior_2 = prior_3 = 1/3
priors = [prior_1, prior_2, prior_3]

mean1 = np.array([0, 0, 0])
mean2 = np.array([1, 2, 2])
mean3 = np.array([3, 3, 4])
means = [mean1, mean2, mean3]

covariance_m1 = covariance_m2 = covariance_m3 = np.array([[0.8, 0.2, 0.1],
                                                          [0.2, 0.8, 0.2],
                                                          [0.1, 0.2, 0.8]])
covariance_ms = [covariance_m1, covariance_m2, covariance_m3]


training_data = create_data(3, training_size, priors, means, covariance_ms)
testing_data = create_data(3, testing_size, priors, means, covariance_ms)

print("\n ----- Exercise 3.4 b ----- ")


def euclidian_classifier(test_data, test_size, list_of_means):
    # ---------- Euclidian Classifier ----------
    wrong_euclidian = 0
    # Class 1 Classification with Euclidian Classifier
    for i in range(test_data[0].shape[0]):
        # test of class_1 - euclidian distance
        distance_to_c1 = Ex1.euclidian_distance(test_data[0][i], list_of_means[0], 3)
        distance_to_c2 = Ex1.euclidian_distance(test_data[0][i], list_of_means[1], 3)
        distance_to_c3 = Ex1.euclidian_distance(test_data[0][i], list_of_means[2], 3)

        # A point is misclassified if distance from class_1 is not the smallest
        if distance_to_c1 != min([distance_to_c1, distance_to_c2, distance_to_c3]):
            wrong_euclidian += 1

    # Class 2 Classification with Euclidian Classifier
    for i in range(test_data[1].shape[0]):
        # test of class_2 - euclidian distance
        distance_to_c1 = Ex1.euclidian_distance(test_data[1][i], list_of_means[0], 3)
        distance_to_c2 = Ex1.euclidian_distance(test_data[1][i], list_of_means[1], 3)
        distance_to_c3 = Ex1.euclidian_distance(test_data[1][i], list_of_means[2], 3)

        # A point is misclassified if distance from class_2 is not the smallest
        if distance_to_c2 != min([distance_to_c1, distance_to_c2, distance_to_c3]):
            wrong_euclidian += 1

    # Class 3 Classification with Euclidian Classifier
    for i in range(test_data[2].shape[0]):
        # test of class_3 - euclidian distance
        distance_to_c1 = Ex1.euclidian_distance(test_data[2][i], list_of_means[0], 3)
        distance_to_c2 = Ex1.euclidian_distance(test_data[2][i], list_of_means[1], 3)
        distance_to_c3 = Ex1.euclidian_distance(test_data[2][i], list_of_means[2], 3)

        # A point is misclassified if distance from class_2 is not the smallest
        if distance_to_c3 != min([distance_to_c1, distance_to_c2, distance_to_c3]):
            wrong_euclidian += 1

    print("Euclidian Minimum Distance Classifier misclassified ", wrong_euclidian)
    print("The error is:", wrong_euclidian/test_size)


def mahalanobis_classifier(test_data, test_size, list_of_means, list_of_covariance_m):
    # ---------- Mahalanobis Classifier ----------
    wrong_Mahalanobis = 0
    # Class 1 Classification with Mahalanobis Classifier
    for i in range(test_data[0].shape[0]):
        # test of class_1 - Mahalanobis distance
        distance_to_c1 = Ex1.mahalanobis_distance(test_data[0][i], list_of_means[0], list_of_covariance_m[0], 3)
        distance_to_c2 = Ex1.mahalanobis_distance(test_data[0][i], list_of_means[1], list_of_covariance_m[1], 3)
        distance_to_c3 = Ex1.mahalanobis_distance(test_data[0][i], list_of_means[2], list_of_covariance_m[2], 3)

        # A point is misclassified if distance from class_1 is not the smallest
        if distance_to_c1 != min([distance_to_c1, distance_to_c2, distance_to_c3]):
            wrong_Mahalanobis += 1

    # Class 2 Classification with Mahalanobis Classifier
    for i in range(test_data[1].shape[0]):
        # test of class_2 - Mahalanobis distance
        distance_to_c1 = Ex1.mahalanobis_distance(test_data[1][i], list_of_means[0], list_of_covariance_m[0], 3)
        distance_to_c2 = Ex1.mahalanobis_distance(test_data[1][i], list_of_means[1], list_of_covariance_m[1], 3)
        distance_to_c3 = Ex1.mahalanobis_distance(test_data[1][i], list_of_means[2], list_of_covariance_m[2], 3)

        # A point is misclassified if distance from class_2 is not the smallest
        if distance_to_c2 != min([distance_to_c1, distance_to_c2, distance_to_c3]):
            wrong_Mahalanobis += 1

    # Class 3 Classification with Mahalanobis Classifier
    for i in range(test_data[2].shape[0]):
        # test of class_3 - Mahalanobis distance
        distance_to_c1 = Ex1.mahalanobis_distance(test_data[2][i], list_of_means[0], list_of_covariance_m[0], 3)
        distance_to_c2 = Ex1.mahalanobis_distance(test_data[2][i], list_of_means[1], list_of_covariance_m[1], 3)
        distance_to_c3 = Ex1.mahalanobis_distance(test_data[2][i], list_of_means[2], list_of_covariance_m[2], 3)

        # A point is misclassified if distance from class_2 is not the smallest
        if distance_to_c3 != min([distance_to_c1, distance_to_c2, distance_to_c3]):
            wrong_Mahalanobis += 1

    print("Mahalanobis Minimum Distance Classifier misclassified ", wrong_Mahalanobis)
    print("The error is:", wrong_Mahalanobis/test_size)


def bayes_classifier(test_data, test_size, list_of_means, list_of_covariance_m, priors_list):
    # ---------- Bayes Classifier ----------
    wrong_Bayes = 0
    # Class 1 Classification with Bayes Classifier
    for i in range(test_data[0].shape[0]):
        # test of class_1 - Bayes distance
        g1 = Ex1.discriminant_function(test_data[0][i], list_of_means[0], list_of_covariance_m[0], priors_list[0], 3)
        g2 = Ex1.discriminant_function(test_data[0][i], list_of_means[1], list_of_covariance_m[1], priors_list[1], 3)
        g3 = Ex1.discriminant_function(test_data[0][i], list_of_means[2], list_of_covariance_m[2], priors_list[2], 3)

        # A point is misclassified if value from class_1 is not the biggest
        if g1 != max([g1, g2, g3]):
            wrong_Bayes += 1

    # Class 2 Classification with Bayes Classifier
    for i in range(test_data[1].shape[0]):
        # test of class_1 - Bayes distance
        g1 = Ex1.discriminant_function(test_data[1][i], list_of_means[0], list_of_covariance_m[0], priors_list[0], 3)
        g2 = Ex1.discriminant_function(test_data[1][i], list_of_means[1], list_of_covariance_m[1], priors_list[1], 3)
        g3 = Ex1.discriminant_function(test_data[1][i], list_of_means[2], list_of_covariance_m[2], priors_list[2], 3)

        # A point is misclassified if value from class_2 is not the smallest
        if g2 != max([g1, g2, g3]):
            wrong_Bayes += 1

    # Class 3 Classification with Euclidian Classifier
    for i in range(test_data[2].shape[0]):
        # test of class_3 - Bayes distance
        g1 = Ex1.discriminant_function(test_data[2][i], list_of_means[0], list_of_covariance_m[0], priors_list[0], 3)
        g2 = Ex1.discriminant_function(test_data[2][i], list_of_means[1], list_of_covariance_m[1], priors_list[1], 3)
        g3 = Ex1.discriminant_function(test_data[2][i], list_of_means[2], list_of_covariance_m[2], priors_list[2], 3)

        # A point is misclassified if value from class_2 is not the biggest
        if g3 != max([g1, g2, g3]):
            wrong_Bayes += 1

    print("Bayes Classifier misclassified ", wrong_Bayes)
    print("The error is:", wrong_Bayes/test_size)


print("\nClassification Errors results using real means and Covariance matrices \n")
euclidian_classifier(testing_data, testing_size, means)
mahalanobis_classifier(testing_data, testing_size, means, covariance_ms)
bayes_classifier(testing_data, testing_size, means, covariance_ms, priors)

# Exercise 3.4 c
print("\n ----- Exercise 3.4 c -----")


def maximum_likelihood_estimation(train_data):
    # Estimate means and covariance matrices from our training data
    class_1_mean = np.mean(train_data[0], axis=0)
    class_2_mean = np.mean(train_data[1], axis=0)
    class_3_mean = np.mean(train_data[2], axis=0)
    class_1_cov = np.cov(train_data[0], rowvar=False)
    class_2_cov = np.cov(train_data[1], rowvar=False)
    class_3_cov = np.cov(train_data[2], rowvar=False)
    return [class_1_mean, class_2_mean, class_3_mean], [class_1_cov, class_2_cov, class_3_cov]


means_estimates, covariance_ms_estimate = maximum_likelihood_estimation(training_data)

print("\nClassification Errors results using estimated means and Covariance matrices  \n")
euclidian_classifier(testing_data, testing_size, means_estimates)
mahalanobis_classifier(testing_data, testing_size, means_estimates, covariance_ms_estimate)
bayes_classifier(testing_data, testing_size, means_estimates, covariance_ms_estimate, priors)


# Exercise 3.4 c
print("\n ----- Exercise 3.4 d -----\n")
prior_1, prior_2, prior_3 = 1/6, 1/6, 2/3
priors = [prior_1, prior_2, prior_3]

mean1 = np.array([0, 0, 0])
mean2 = np.array([1, 2, 2])
mean3 = np.array([3, 3, 4])
means = [mean1, mean2, mean3]

covariance_m1 = np.array([[0.8, 0.2, 0.1],
                          [0.2, 0.8, 0.2],
                          [0.1, 0.2, 0.8]])
covariance_m2 = np.array([[0.6, 0.2, 0.01],
                          [0.2, 0.8, 0.01],
                          [0.01, 0.01, 0.6]])
covariance_m3 = np.array([[0.6, 0.1, 0.1],
                          [0.1, 0.6, 0.1],
                          [0.1, 0.1, 0.6]])

covariance_ms = [covariance_m1, covariance_m2, covariance_m3]

training_data = create_data(3, training_size, priors, means, covariance_ms)
testing_data = create_data(3, testing_size, priors, means, covariance_ms)
print("\nClassification Errors results using real means and Covariance matrices \n")
euclidian_classifier(testing_data, testing_size, means)
mahalanobis_classifier(testing_data, testing_size, means, covariance_ms)
bayes_classifier(testing_data, testing_size, means, covariance_ms, priors)

means_estimates, covariance_ms_estimate = maximum_likelihood_estimation(training_data)

print("\nClassification Errors results using estimated means and Covariance matrices  \n")
euclidian_classifier(testing_data, testing_size, means_estimates)
mahalanobis_classifier(testing_data, testing_size, means_estimates, covariance_ms_estimate)
bayes_classifier(testing_data, testing_size, means_estimates, covariance_ms_estimate, priors)
