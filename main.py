import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def cross(vector1, vector2):
    return np.cross(vector1, vector2)


G = 1e-2
m1 = 1000
m2 = 3

R = 1
R1 = R * m2 / (m1 + m2)
R2 = R * m1 / (m1 + m2)

w = np.array([0, 0, np.sqrt(G * (m1 + m2) / R ** 3)])

r1 = np.array([-R1, 0, 0])
r2 = np.array([R2, 0, 0])

T = 100

r0 = np.array([R / 2 - R1, np.sqrt(3) / 2 * R, 0])
v0 = np.array([0.01, 0.1, 0.05])
state0 = np.concatenate((r0, v0))


def l4(_, state):
    r = state[:3]
    v = state[3:6]

    a_grav1 = -G * m1 / np.linalg.norm(r - r1) ** 3 * (r - r1)
    a_grav2 = -G * m2 / np.linalg.norm(r - r2) ** 3 * (r - r2)
    a_centrifugal = -cross(w, cross(w, r))
    a_coriolis = -2 * cross(w, v)
    acc = a_grav1 + a_grav2 + a_centrifugal + a_coriolis

    return np.concatenate((v, acc))


solution = solve_ivp(l4, [0, T], state0, max_step=0.03)

rx, ry, rz, vx, vy, vz = zip(solution.y)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(r1[0], r1[1], r1[2])
ax.scatter(r2[0], r2[1], r2[2])
ax.plot(rx[0], ry[0], rz[0])

plt.show()
