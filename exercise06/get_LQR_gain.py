

import numpy as np
from  scipy.linalg import solve_discrete_are


def get_LQR_gain(A, B, Q, R):

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve((B.T @ P @ B + R), B.T @ P @ A)
    return (K, P)