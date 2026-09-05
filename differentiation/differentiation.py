import math

import numpy as np


# First derivative using the forward difference method
def forward_difference(function, x, h):
    if h <= 0:
        raise ValueError("Step size h must be positive.")

    derivative = (function(x + h) - function(x)) / h

    return derivative


# First derivative using the backward difference method
def backward_difference(function, x, h):
    if h <= 0:
        raise ValueError("Step size h must be positive.")

    derivative = (function(x) - function(x - h)) / h

    return derivative


# First derivative using the central difference method
def central_difference(function, x, h):
    if h <= 0:
        raise ValueError("Step size h must be positive.")

    derivative = (
        function(x + h) - function(x - h)
    ) / (2 * h)

    return derivative


# General kth derivative using a forward finite difference scheme
def forward_derivative(y, h, derivative_order):
    y = np.asarray(y, dtype=float)

    if h <= 0:
        raise ValueError("Step size h must be positive.")

    if derivative_order < 1:
        raise ValueError(
            "Derivative order must be at least 1."
        )

    if derivative_order >= y.size:
        raise ValueError(
            "Derivative order must be smaller than "
            "the number of data points."
        )

    number_of_results = y.size - derivative_order
    derivative = np.zeros(number_of_results)

    for n in range(number_of_results):
        for i in range(derivative_order + 1):
            coefficient = (
                (-1) ** i
                * math.comb(derivative_order, i)
            )

            derivative[n] += (
                coefficient
                * y[n + derivative_order - i]
            )

        derivative[n] /= h**derivative_order

    return derivative


# General kth derivative using a backward finite difference scheme
def backward_derivative(y, h, derivative_order):
    y = np.asarray(y, dtype=float)

    if h <= 0:
        raise ValueError("Step size h must be positive.")

    if derivative_order < 1:
        raise ValueError(
            "Derivative order must be at least 1."
        )

    if derivative_order >= y.size:
        raise ValueError(
            "Derivative order must be smaller than "
            "the number of data points."
        )

    number_of_results = y.size - derivative_order
    derivative = np.zeros(number_of_results)

    for n in range(
        derivative_order,
        y.size,
    ):
        result_index = n - derivative_order

        for i in range(derivative_order + 1):
            coefficient = (
                (-1) ** i
                * math.comb(derivative_order, i)
            )

            derivative[result_index] += (
                coefficient
                * y[n - i]
            )

        derivative[result_index] /= h**derivative_order

    return derivative