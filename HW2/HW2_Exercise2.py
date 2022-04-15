import numpy as np
import matplotlib.pyplot as plt
likelihood = np.array([[0.3, 0.2, 0.1, 0.1, 0.2, 0.1],
                      [0.2, 0.2, 0.4, 0.05, 0.1, 0.05],
                      [0.1, 0.3, 0.15, 0.05, 0.3, 0.1]])
a_priori = np.array([0.3, 0.3, 0.4])
prod = np.zeros((3, 6))
for i in range(likelihood.shape[1]):
    prod[:, i] = np.multiply(likelihood[:, i], a_priori)
print(prod)
print("Total risk = ", 1 - np.sum(np.max(prod, axis= 0)))
for j in range(prod.shape[0]):
    plt.plot(prod[j, :])
plt.xticks(np.arange(prod.shape[1]), np.arange(1, prod.shape[1]+1))
plt.legend(['p(x|$ω_1$)p($ω_1$)', 'p(x|$ω_2$)p($ω_2$)', 'p(x|$ω_3$)p($ω_3$)'], prop={'size': 15})
plt.show()
