import matplotlib.pyplot as plt
import numpy as np

from ode.initial_value import runge_kutta_4_system


# Double pendulum system
def double_pendulum_system(t, y):
    derivative = np.zeros(4)

    g = 9.81

    l1 = 1.0
    l2 = 0.5

    m1 = 2.0
    m2 = 1.0

    theta1 = y[0]
    omega1 = y[1]
    theta2 = y[2]
    omega2 = y[3]

    angle_difference = theta1 - theta2

    sin_difference = np.sin(angle_difference)
    cos_difference = np.cos(angle_difference)

    derivative[0] = omega1

    derivative[1] = (
        m2 * g * np.sin(theta2) * cos_difference
        - m2
        * sin_difference
        * (
            l1 * omega1**2 * cos_difference
            + l2 * omega2**2
        )
        - (m1 + m2) * g * np.sin(theta1)
    ) / (
        l1
        * (
            m1
            + m2 * sin_difference**2
        )
    )

    derivative[2] = omega2

    derivative[3] = (
        (m1 + m2)
        * (
            l1 * omega1**2 * sin_difference
            + g
            * np.sin(theta1)
            * cos_difference
            - g * np.sin(theta2)
        )
        + m2
        * l2
        * omega2**2
        * sin_difference
        * cos_difference
    ) / (
        l2
        * (
            m1
            + m2 * sin_difference**2
        )
    )

    return derivative


# Initial conditions
initial_conditions = np.array(
    [
        np.pi / 4,
        0.0,
        -np.pi / 4,
        0.0,
    ]
)

t0 = 0.0
t_end = 20.0
h = 0.002


# Solve the system
t, solution = runge_kutta_4_system(
    double_pendulum_system,
    initial_conditions,
    t0,
    t_end,
    h,
)

theta1 = solution[0]
theta2 = solution[2]


# Plot angular displacement
plt.figure(figsize=(8, 6))

plt.plot(
    t,
    np.degrees(theta1),
    label="Pendulum 1",
)

plt.plot(
    t,
    np.degrees(theta2),
    label="Pendulum 2",
)

plt.xlabel("Time (s)")
plt.ylabel("Angular Displacement (degrees)")
plt.title("Double Pendulum Motion")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Calculate mass positions
l1 = 1.0
l2 = 0.5

x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)

x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)


# Plot pendulum trajectories
plt.figure(figsize=(7, 6))

plt.plot(
    x1,
    y1,
    label="Mass 1",
)

plt.plot(
    x2,
    y2,
    label="Mass 2",
)

plt.xlabel("Horizontal Position (m)")
plt.ylabel("Vertical Position (m)")
plt.title("Double Pendulum Trajectories")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()