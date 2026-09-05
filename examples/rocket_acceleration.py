import matplotlib.pyplot as plt
import numpy as np

from differentiation.differentiation import forward_derivative


# Load the discrete rocket data
y = np.loadtxt("data/rocket.txt")

# Time interval between measurements
h = 100.0

# Calculate the second derivative
derivative_order = 2
acceleration = forward_derivative(
    y,
    h,
    derivative_order,
)

# Construct the time array for the original data
time = np.arange(y.size) * h

# The forward derivative produces N - k values
acceleration_time = time[:-derivative_order]


# Plot the calculated acceleration
plt.figure(figsize=(8, 6))

plt.plot(
    acceleration_time,
    acceleration,
    marker="o",
)

plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.title("Rocket Acceleration from Discrete Data")
plt.grid(True)

plt.tight_layout()
plt.show()