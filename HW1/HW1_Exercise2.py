from scipy.stats import kurtosis, skew
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def die_rolls(num_of_throws):
    rng = np.random.default_rng(seed=123)
    total_throws = rng.integers(low=1, high=6, endpoint=True, size=num_of_throws)
    return total_throws


def auto_corr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


# Ερώτημα β
n_rolls_a = [20, 100, 1000]
throws_list_a = [die_rolls(n) for n in n_rolls_a]
print(throws_list_a)
fig, axes = plt.subplots(nrows=1, ncols=len(n_rolls_a))
for ax in range(len(axes)):
    axes[ax].hist(throws_list_a[ax], bins=[i+0.5 for i in range(0, 7)], rwidth=0.5, density=True)
    axes[ax].set_title("N="+str(n_rolls_a[ax]))

fig.tight_layout()
plt.savefig("Fig1.svg", bbox_inches="tight")
plt.show()


# Ερώτημα γ
n_rolls_c = [10, 20, 50, 100, 500, 1000]
throws_list_c = [die_rolls(n) for n in n_rolls_c]
mean_values = [np.mean(thl) for thl in throws_list_c]
variance_values = [np.var(thl) for thl in throws_list_c]
skewness_values = [skew(thl) for thl in throws_list_c]
kurtosis_values = [kurtosis(thl, fisher=False) for thl in throws_list_c]
for i in range(len(n_rolls_c)):
    print(30*"==")
    print(str(n_rolls_c[i])+" die rolls results:")
    print("Mean value: ", format(mean_values[i], '.4f'), "Difference from theoretical value: ", format(abs(3.5-mean_values[i]), '.4f'))
    print("Variance: ", format(variance_values[i], '.4f'), "Difference from theoretical value: ", format(abs(2.9166-variance_values[i]), '.4f'))
    print("Skewness is", format(skewness_values[i], '.4f'), "Difference from theoretical value: ", format(abs(skewness_values[i]), '.4f'))
    print("Kurtosis is", format(kurtosis_values[i], '.4f'), "Difference from theoretical value: ", format(abs(1.731428-kurtosis_values[i]), '.4f'))
    print(30*"==")
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax0 in range(len(axes)):
    for ax1 in range(len(axes[0])):
        axes[ax0][ax1].hist(throws_list_c[3*ax0+ax1], bins=[i+0.5 for i in range(0, 7)], rwidth=0.5, density=True)
        axes[ax0][ax1].set_title("N="+str(n_rolls_c[3*ax0+ax1]))
fig.tight_layout()
plt.show()


# Ερώτημα ε
s = 10000
throws_list_e = die_rolls(s)
mean_values = [np.mean(throws_list_e[0:i]) for i in range(10, s, 10)]
var_values = [np.var(throws_list_e[0:i]) for i in range(10, s, 10)]
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(mean_values)
axes[0].set_title("Mean Values for increasing Number of Throws")
axes[0].set_xticks(range(50, 1050, 50))
axes[1].plot(var_values)
axes[1].set_title("Variance Values for increasing Number of Throws")
axes[1].set_xticks(range(50, 1050, 50))
fig.tight_layout()
plt.show()
plot_acf(throws_list_e, lags=range(0, s, 100), title="Auto-correlation for N="+str(s))
plt.show()
