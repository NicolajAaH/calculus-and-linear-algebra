import datetime

from sympy import *

x, y, z = symbols('x y z')
init_printing(use_unicode=True)

epsilon = 0.0001  # Epsilon value
iterMax = 50  # Maximum number of iterations


def derivative(function):  # Returns the derivative of the provided functions with respect to x
    return diff(function, x)


def gradient_descent(function, x_0):
    x_k = x_0
    count = 0
    f_prime = derivative(function)  # Get f'(x)
    f_prime_prime = derivative(f_prime)  # Get f''(x)
    while abs(round(f_prime.subs(x, x_k), 15)) > epsilon and count < iterMax:  # Check for under epsilon and iterMax
        alpha = round(1 / f_prime_prime.subs(x, x_k), 15)  # Calculate alpha
        x_k = round(x_k - alpha * f_prime.subs(x, x_k), 15)  # Calculate new x_k
        count += 1  # Increment
    print("The function: ")
    pretty_print(function)
    print("Has the derivative: ")
    pretty_print(f_prime)
    print("The refined x-value was found in " + str(count) + " iterations, with starting x_0 as x_0=" + str(
        x_0) + ", and the refined x was found to be x=", end='')
    return evaluate_extrema(f_prime_prime, x_k)


def evaluate_extrema(function, x_0):
    val = function.subs(x, x_0)  # Calculate f''(x_0)
    if abs(val) < 0.05:  # If its between -0.05 and 0.05 it should be 0 due to rounding errors
        return str(round(x_0, 2)) + " which is an inflection point, since f''(x)=0"
    elif val > 0:  # Minimum
        return str(round(x_0, 2)) + " which is a minimum, since f''(x)>0"
    return str(round(x_0, 2)) + " which is a maximum, since f''(x)<0"  # Maximum


if __name__ == '__main__':
    start_time = datetime.datetime.now()  # Start timer
    print(gradient_descent(-20 * exp(-x ** 2 / 8) - exp((1 / 2) * cos(2 * pi * x)) + 20 + exp(1), 0.1))
    # print(gradient_descent(x**2, -1))
    # print(gradient_descent(x ** 3, -1))
    end_time = datetime.datetime.now()  # End timer
    print("Algorithm took " + str((end_time - start_time).total_seconds()) + " seconds to run")
