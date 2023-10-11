import numpy as np

def up_cross(epsilon):
    res = np.array([[0, -epsilon[2], epsilon[1]], 
                    [epsilon[2], 0, -epsilon[0]], 
                    [-epsilon[1], epsilon[0], 0]])
    return res

def up_plus_circ(q):
    yita = q[3]
    epsilon = q[:3]
    b1 = up_cross(epsilon) + yita * np.eye(3)
    epsilon = epsilon.reshape(-1, 1)
    res = np.block([[b1, epsilon], 
                    [-epsilon.T, yita]])
    return res

def cross_circ(q1, q2):
    res = up_plus_circ(q2) @ q1
    return res

