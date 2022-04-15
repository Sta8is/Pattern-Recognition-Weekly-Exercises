import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


def my_perceptron(inp, out, tolerance=1e-3, max_iterations=10000):
    """
    :param inp: MxN matrix (M number of features, N number of input samples). Here 2x6
    :param out: {N,} vector
    :param tolerance: The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol).
    :param max_iterations: The maximum number of epochs
    :return: errors, weights (1xM), bias (1x1), w_list, b_list
    """
    iteration = 1
    errors, criteria = [], []
    w = np.ones([inp.shape[0], 1])
    b = 1
    w_list, b_list = [], []
    w_list.append(w.ravel().tolist())
    b_list.append(b)
    while iteration < max_iterations:
        error = 0
        for i in range(inp.shape[1]):
            neuron_sum = np.dot(w.T, inp[:, i].reshape(-1, 1)) + b
            activation = 1 if int(neuron_sum) >= 0 else 0
            err = out[i] - activation
            w += err*inp[:, i].reshape(-1, 1)
            b += err
            error += abs(err)
        w_list.append(w.ravel().tolist())
        b_list.append(b)
        criterion = error/inp.shape[1]
        criteria.append(criterion)
        errors.append(error)
        if criterion < tolerance:
            break
        iteration += 1
    return errors, w, b, iteration, w_list, b_list


def line(x, w, b):
    return (-w[0]*x - b)/w[1]


X = np.array([[-2, 1], [1, 2],
              [0, 0], [-2, -1], [2, 0], [2, 1]])
Y = np.array([0, 0, 1, 1, 1, 1])

# # Plot only Points
# plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "bo", X[:, 0][Y == 0], X[:, 1][Y == 0], "ro", markersize=15)
# plt.title("Dataset", fontsize=18)
# plt.legend(["$\omega_1$", "$\omega_2$"], fontsize=18)
# plt.xlabel("X", fontsize=18)
# plt.ylabel("Y", fontsize=18)
# plt.show()


model = Perceptron(shuffle=False, verbose=1, n_iter_no_change=2, tol=0.2)
model.fit(X, Y, coef_init=[1, 1], intercept_init=1)
final_weights = model.coef_.tolist()[0]
final_bias = int(model.intercept_)
print("-"*25)
print("Sklearn Perceptron Results: ")
print('weights: ', final_weights, "and bias", final_bias)
print("Single Perceptron Accuracy", model.score(X, Y))
print("Number of iterations: ", model.n_iter_)
print("-"*25)

errs, weights, bias, iterations, weights_list, bias_list = my_perceptron(X.T, Y)
print("My Perceptron Results: ")
print('weights: ', weights.ravel().tolist(), "and bias", bias)
print("Number of my iterations ", iterations)
print("Single Perceptron Accuracy: ", 1-errs[-1])
print("Errors at each epoch: ", errs)


# Plot points and decision boundaries
plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "bo", X[:, 0][Y == 0], X[:, 1][Y == 0], "ro", markersize=15,)
x_axis = np.linspace(-2.5, 2.5, 100)
plt.plot(x_axis, line(x_axis, final_weights, final_bias), "y-", linewidth=5)
plt.plot(x_axis, line(x_axis, weights.ravel().tolist(), bias), "k-", linewidth=1)
plt.title("Dataset and Decision Boundaries", fontsize=18)
plt.legend(["$\omega_1$", "$\omega_2$", "Decision Boundary Sklearn", "My Decision Boundary"], fontsize=18)
plt.xlabel("X", fontsize=18)
plt.ylabel("Y", fontsize=18)
plt.show()


plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "bo", X[:, 0][Y == 0], X[:, 1][Y == 0], "ro", markersize=15)
for i in range(len(weights_list)-1):
    plt.plot(x_axis, line(x_axis, weights_list[i], bias_list[i]), "--", linewidth=2, label="Epoch "+str(i+1))
plt.plot(x_axis, line(x_axis, weights_list[-1], bias_list[-1]), "y-", linewidth=1, label="Last Epoch")
plt.title("Dataset and Decision Boundaries at each Epoch", fontsize=18)
plt.legend(fontsize=18)
plt.xlabel("X", fontsize=18)
plt.ylabel("Y", fontsize=18)
plt.show()
