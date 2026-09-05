import matplotlib.pyplot as plt
import numpy as np

from ode.initial_value import (
    forward_euler_system,
    runge_kutta_4_system,
)


# Coupled three-mass spring system
def spring_mass_system(t, y):
    derivative = np.zeros(6)

    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    k4 = 1.0

    l1 = 1.0
    l2 = 1.0
    l3 = 1.0

    m1 = 1.0
    m2 = 1.0
    m3 = 1.0

    x1 = y[0]
    v1 = y[1]

    x2 = y[2]
    v2 = y[3]

    x3 = y[4]
    v3 = y[5]

    derivative[0] = v1

    derivative[1] = (
        -k1 * (x1 - l1)
        + k2 * (x2 - x1 - l2)
    ) / m1

    derivative[2] = v2

    derivative[3] = (
        -k2 * (x2 - x1 - l2)
        + k3 * (x3 - x2 - l3)
    ) / m2

    derivative[4] = v3

    derivative[5] = (
        -k3 * (x3 - x2 - l3)
        + k4 * (l1 + l2 + l3 - x3)
    ) / m3

    return derivative


# Initial conditions
initial_conditions = np.array(
    [
        0.5,
        0.0,
        1.5,
        0.0,
        2.5,
        0.0,
    ]
)

t0 = 0.0
t_end = 20.0
h = 0.01


# Solve using Forward Euler
t_forward, y_forward = forward_euler_system(
    spring_mass_system,
    initial_conditions,
    t0,
    t_end,
    h,
)


# Solve using RK4
t_rk4, y_rk4 = runge_kutta_4_system(
    spring_mass_system,
    initial_conditions,
    t0,
    t_end,
    h,
)


# Plot Forward Euler results
plt.figure(figsize=(8, 6))

plt.plot(
    t_forward,
    y_forward[0],
    label="Mass 1",
)

plt.plot(
    t_forward,
    y_forward[2],
    label="Mass 2",
)

plt.plot(
    t_forward,
    y_forward[4],
    label="Mass 3",
)

plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Coupled Spring-Mass System - Forward Euler")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Plot RK4 results
plt.figure(figsize=(8, 6))

plt.plot(
    t_rk4,
    y_rk4[0],
    label="Mass 1",
)

plt.plot(
    t_rk4,
    y_rk4[2],
    label="Mass 2",
)

plt.plot(
    t_rk4,
    y_rk4[4],
    label="Mass 3",
)

plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Coupled Spring-Mass System - RK4")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()