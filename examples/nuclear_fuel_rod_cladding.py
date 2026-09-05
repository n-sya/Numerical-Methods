import matplotlib.pyplot as plt
import numpy as np

from ode.boundary_value import finite_difference_robin


# Material and boundary-condition properties
heat_transfer_coefficient = 6.0e4
thermal_conductivity = 16.75

fuel_radius = 0.015
cladding_thickness = 0.003

coolant_temperature = 473.0


# Governing equation written as:
# T'' + f(r)T' + g(r)T = p(r)
def coefficients(r):
    f = 1 / r
    g = 0.0

    p = (
        -1.0e8
        * np.exp(-r / fuel_radius)
        / (thermal_conductivity * r)
    )

    return f, g, p


# Inner boundary condition
# dT/dr = prescribed heat flux / k
left_boundary = (
    1.0,
    0.0,
    -6.32e5 / thermal_conductivity,
)


# Outer convective boundary condition
# dT/dr + (h/k)T = (h/k)T_coolant
right_boundary = (
    1.0,
    heat_transfer_coefficient / thermal_conductivity,
    (
        heat_transfer_coefficient
        / thermal_conductivity
        * coolant_temperature
    ),
)


# Radial domain
inner_radius = fuel_radius
outer_radius = fuel_radius + cladding_thickness

number_of_intervals = 100


# Solve the boundary-value problem
radius, temperature = finite_difference_robin(
    coefficients,
    a=inner_radius,
    b=outer_radius,
    left_boundary=left_boundary,
    right_boundary=right_boundary,
    number_of_intervals=number_of_intervals,
)


# Plot temperature distribution
plt.figure(figsize=(8, 6))

plt.plot(
    radius,
    temperature,
    linewidth=2,
)

plt.xlabel("Radius (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Distribution Through Nuclear Fuel Cladding")
plt.grid(True)

plt.tight_layout()
plt.show()