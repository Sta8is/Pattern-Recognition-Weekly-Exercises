from scipy.integrate import quad
import numpy as np
import math
import matplotlib.pyplot as plt


def p_theta_0(theta2):
    return np.sin(math.pi*theta2)


A = 1/quad(p_theta_0, 0, 1)[0]
print("A = ", A)



def numerator(theta1, k):
    return (theta1 ** k)*(1-theta1)**(i-k)*A*p_theta_0(theta1)



d = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1])  # Results of flip
Num_Of_Points = 1000
y = []
for i in [1, 5, 10]:
    K = (d[0:i] == 1).sum()
    denominator = quad(numerator, 0, 1, args=K)[0]
    theta = np.linspace(0, 1, Num_Of_Points)
    y.append(numerator(theta, K)/denominator)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(theta, y[0], 'r')
plt.plot(theta, y[1], 'g')
plt.plot(theta, y[2], 'b')
plt.legend(["P(theta|D1)", "P(theta|D5)", "P(theta|D10)"], prop={'size': 15})
plt.show()

print("The max value of P(θ|D10) is:", max(y[2]), "for x ", (np.argmax(y[2])) / Num_Of_Points)


