def gradient_descent(df, x_init, max_iterations, min_step, learning_rate=0.01):
    iterations = 0
    x = x_init
    step = 10
    while iterations < max_iterations and abs(step) > min_step:
        prev_x = x  # Store current x value in prev_x
        x = prev_x - learning_rate * df(prev_x)  # Gradient descent rule
        iterations += 1  # Iteration count
        step = prev_x - x
    print("The local minimum using Gradient Descent is ", x, 'after', iterations, 'iterations')
    return x, iterations


def newton_method(fir_derivative, sec_derivative, x_init, max_iterations, min_step):
    iterations = 0
    x = x_init
    step = 10
    while iterations < max_iterations and abs(step) > min_step:
        prev_x = x  # Store current x value in prev_x
        x = prev_x - fir_derivative(x)/sec_derivative(x)  # Newton-Raphson rule
        iterations += 1  # Iteration count
        step = prev_x - x
    print("The local minimum using Newton Method is ", x, 'after', iterations, 'iterations')
    return x, iterations


def first_der(x):
    return 4 * pow((x - 5), 3) + 3


def second_der(x):
    return 12*pow((x-5), 2)


# Ερώτημα α
analytic_sol = -pow(3/4, 1/3)+5
print("Analytic Solution with Derivative Method: ", analytic_sol)
# Ερώτημα β
[x_g, it_g] = gradient_descent(first_der, 1, 2000, 1e-9, 0.03)
print("Difference from analytic Solution :", abs(x_g-analytic_sol))
# Ερώτημα γ
[x_n, it_n] = newton_method(first_der, second_der, 1, 2000, 1e-9)
print("Difference from analytic Solution :", abs(x_n-analytic_sol))
