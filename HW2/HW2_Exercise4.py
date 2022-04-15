import numpy as np
rng = np.random.default_rng(seed=123)


def gauss_decision(number_of_samples):
    p_omega1, p_omega2 = 1/3, 2/3
    w1_samples_size = int(p_omega1 * number_of_samples)
    w2_samples_size = int(p_omega2 * number_of_samples)
    l_11, l_12, l_21, l_22 = 1, 2, 3, 1
    # Generating samples from normal distribution
    w1 = rng.normal(2, np.sqrt(0.5), w1_samples_size)
    w2 = rng.normal(1.5, np.sqrt(0.2), w2_samples_size)
    w1_as_w2, w1_as_w1, w2_as_w1, w2_as_w2 = 0, 0, 0, 0
    # Using Decision Boundaries to classify data
    for p in w1:
        if 0.402986090944180 < p < 1.930347242389154:
            w1_as_w2 += 1
        else:
            w1_as_w1 += 1
    for p in w2:
        if 0.402986090944180 < p < 1.930347242389154:
            w2_as_w2 += 1
        else:
            w2_as_w1 += 1
    # Calculating Probabilities
    p_w1_as_w1 = w1_as_w1 / w1_samples_size
    p_w1_as_w2 = w1_as_w2 / w1_samples_size
    p_w2_as_w1 = w2_as_w1 / w2_samples_size
    p_w2_as_w2 = w2_as_w2 / w2_samples_size
    # Total Cost
    cost = p_omega1 * ((l_11 * p_w1_as_w1) + (l_21 * p_w1_as_w2)) + \
           p_omega2 * ((l_12 * p_w2_as_w1) + (l_22 * p_w2_as_w2))
    # Print results
    print("Percentage of w1 classified as w1:", p_w1_as_w1)
    print("Percentage of w1 classified as w2:", p_w1_as_w2)
    print("Percentage of w2 classified as w1:", p_w2_as_w1)
    print("Percentage of w2 classified as w2", p_w2_as_w2)
    print("The cost is:", cost)


gauss_decision(300000)
