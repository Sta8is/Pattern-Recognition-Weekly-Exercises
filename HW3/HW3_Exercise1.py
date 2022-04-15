import numpy as np
import math


def discriminant_function(x, m, s, prior, d):
    c = x - m
    if d > 1:
        S_inv = np.linalg.inv(s)
        det_S = round(abs(np.linalg.det(s)), 5)
        return -0.5*np.linalg.multi_dot([c.T, S_inv, c]) - (d/2)*math.log(2*math.pi) - 0.5*math.log(det_S) + math.log(prior)
    else:
        S_inv = s ** (-1)
        det_S = abs(s)
        return -0.5 * c ** 2 * S_inv - (d / 2) * math.log(2*math.pi) - 0.5 * math.log(det_S) + math.log(prior)


def euclidian_distance(x1, x2, d):
    if d > 1:
        return np.sqrt(np.sum(np.dot((x1-x2).T, x1-x2)))
        # return math.sqrt(np.sum(np.square(x1 - x2)))
    else:
        return abs(x1 - x2)


def mahalanobis_distance(x, m, s, d):
    c = x - m
    if d > 1:
        S_inv = np.linalg.inv(s)
        return np.sqrt(np.linalg.multi_dot([c.T, S_inv, c]))
    else:
        S_inv = s ** (-1)
        return math.sqrt(c*S_inv*c)

