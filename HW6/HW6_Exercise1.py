import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    XX, YY = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return XX, YY


def plot_contours(axs, classifier, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    axs: matplotlib axes object
    classifier: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = axs.contourf(xx, yy, Z, **params)
    return out


def plot_svc(x, y, classifer):
    fig, axs = plt.subplots()
    # Set-up grid for plotting.
    X0, X1 = x[:, 0], x[:, 1]
    XX, YY = make_meshgrid(X0, X1)
    plot_contours(axs, clf, XX, YY, cmap=plt.cm.coolwarm, alpha=0.8)
    axs.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=40, edgecolors="k")
    axs.set_ylabel("x2")
    axs.set_xlabel("x1")
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    axs.set_xlim([xlim[0], xlim[1]])
    axs.set_ylim([ylim[0], ylim[1]])
    params = classifer.get_params()
    title = "XOR Problem with polynomial of degree="+str(params["degree"])+" and coef="\
            + str(params["coef0"])+" with C="+str(params["C"])
    axs.set_title(title)
    plt.show()


def plot_svc_decision_function(x, y, classifier, axs=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if axs is None:
        axs = plt.gca()
    X0, X1 = x[:, 0], x[:, 1]
    plt.scatter(X0, X1, c=y, s=50, cmap=plt.cm.coolwarm)
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    # create grid to evaluate model
    XX, YY = make_meshgrid(X0, X1)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    axs.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    # plot support vectors
    if plot_support:
        axs.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=300,
                    linewidth=1, facecolors="none", edgecolors="k")
    axs.set_xlim([xlim[0]-1, xlim[1]+1])
    axs.set_ylim([ylim[0]-1, ylim[1]+1])
    plt.show()


def kernel_phi(x0, x1):
    return [x0**2, x1**2, np.sqrt(2)*x0*x1]


def plot_3d(x, y, el=30, az=30):
    axs = plt.subplot(projection='3d')
    X0, X1 = x[:, 0], x[:, 1]
    kern = kernel_phi(X0, X1)
    axs.scatter3D(kern[0], kern[1], kern[2], c=y, s=50, cmap=plt.cm.coolwarm)
    axs.view_init(elev=el, azim=az)
    axs.set_xlabel('$x_{1}^2$')
    axs.set_ylabel('$x_{2}^2$')
    axs.set_zlabel('$\sqrt{2}x_{1}x_{2}$')
    axs.set_title("Data in feature space")
    plt.show()


def plot_linear_dec(weights):
    points = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(points, points)
    ax = plt.subplot(111, projection='3d')
    kernel = kernel_phi(x_0, x_1)
    ax.scatter3D(kernel[0], kernel[1], kernel[2], c=Y, s=50, cmap=plt.cm.coolwarm)
    ax.plot_surface(xx**2, yy**2, -(weights[0]*xx**2+weights[1]*yy**2+bias)/(np.sqrt(2)*weights[2]), color='y', alpha=0.5)
    ax.set_xlabel('$x_{0}^2$', fontsize=14)
    ax.set_ylabel('$x_{1}^2$', fontsize=14)
    ax.set_zlabel('$\sqrt{2}x_{0}x_{1}$', fontsize=14)
    ax.set_title("Linearly Separable data in feature space", fontsize=14)
    plt.show()


# Define XOR data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([-1, 1, 1, -1]).reshape(-1, 1)
x_0, x_1 = X[:, 0], X[:, 1]
# # Plot points
# plt.scatter(x_0, x_1, c=Y, cmap=plt.cm.coolwarm, s=200)
# plt.title("XOR function", fontsize=18)
# plt.xlabel("$x_{0}$", fontsize=18)
# plt.ylabel("$x_{1}$", fontsize=18)
# plt.show()

# plot_3d(X, Y)

# SVM classifier
c = 6  # SVM regularization parameter
model = svm.SVC(kernel='poly', degree=2, coef0=0, C=c, gamma=1)
clf = model.fit(X, Y.ravel())
ay_values = clf.dual_coef_
bias = clf.intercept_
sv_order = clf.support_
print("a*y values = ", ay_values[0])
print("bias", bias)
print("support vector order", sv_order)

list_w = []
for i in range(len(clf.support_)):
    list_w.append(np.array(kernel_phi(x_0[sv_order[i]], x_1[sv_order[i]]))*ay_values[0][i])
w = np.sum(np.array(list_w), axis=0)
print("weights", w)


plot_svc(X, Y, clf)
plot_svc_decision_function(X, Y, clf)
plot_linear_dec(w)

