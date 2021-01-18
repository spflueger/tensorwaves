# Small test example

# import inspect

# import numpy as np
# from sympy import cos, lambdify, sin, symbols

# x = symbols("x")
# expr = sin(x) + cos(x)
# print(expr)

# f = lambdify(x, expr, "numpy")
# a = np.array([1, 2])
# print(inspect.getsource(f))
# print(f(a))

import inspect
import math
import timeit

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from jax import pmap
from jax import scipy as jsp
from jax import vmap
from jax.config import config

config.update("jax_enable_x64", True)
from sympy import cos, exp, lambdify, simplify, sin, sqrt, symbols

# import numpy as np


# 1. create a double gaussian amp with sympy
# when building the model, we should be careful to pass the parameters as arguments as well
# otherwise frameworks like jax cant determine the gradient
x, A1, mu1, sigma1, A2, mu2, sigma2 = symbols(
    "x, A1, mu1, sigma1, A2, mu2, sigma2"
)
gaussian1 = (
    A1 / (sigma1 * sqrt(2.0 * math.pi)) * exp(-((x - mu1) ** 2) / (2 * sigma1))
)
# A1, mu1, sigma1 = 1, 0, 1
# A2, mu2, sigma2 = 2, 1, 0.5
gaussian2 = (
    A2 / (sigma2 * sqrt(2.0 * math.pi)) * exp(-((x - mu2) ** 2) / (2 * sigma2))
)

gauss_sum = gaussian1 + gaussian2
# gauss_sum = simplify(gauss_sum)

# 2. use this as a function in tensorwaves (using lambdify)
tf_gauss_sum = lambdify(
    (x, A1, mu1, sigma1, A2, mu2, sigma2), gauss_sum, "tensorflow"
)
numpy_gauss_sum = lambdify(
    (x, A1, mu1, sigma1, A2, mu2, sigma2), gauss_sum, "numpy"
)
# print(inspect.getsource(numpy_gauss_sum))

jax_gauss_sum = lambdify(
    (x, A1, mu1, sigma1, A2, mu2, sigma2),
    gauss_sum,
    modules=(jnp, jsp.special),
)
print(inspect.getsource(jax_gauss_sum))


# 3. create the same function in tf natively
def gaussian(x, A, mu, sigma):
    return (
        A
        / (sigma * tf.sqrt(tf.constant(2.0, dtype=tf.float64) * math.pi))
        * tf.exp(
            -tf.pow(-tf.constant(0.5, dtype=tf.float64) * (x - mu) / sigma, 2)
        )
    )


def native_tf_gauss_sum(x_, A1_, mu1_, sigma1_, A2_, mu2_, sigma2_):
    return gaussian(x_, A1_, mu1_, sigma1_) + gaussian(x_, A2_, mu2_, sigma2_)


# @pmap
def jax_gaussian(x, A, mu, sigma):
    return (
        A
        / (sigma * jnp.sqrt(2.0 * math.pi))
        * jnp.exp(-((-0.5 * (x - mu) / sigma) ** 2))
    )


def native_jax_gauss_sum(x_, A1_, mu1_, sigma1_, A2_, mu2_, sigma2_):
    return jax_gaussian(x_, A1_, mu1_, sigma1_) + jax_gaussian(
        x_, A2_, mu2_, sigma2_
    )


# 4. compare performance

parameter_values = (1.0, 0.0, 0.1, 2.0, 2.0, 0.2)
np_x = np.random.uniform(-1, 3, 10000)
tf_x = tf.constant(np_x)


def evaluate_with_parameters(function):
    def wrapper():
        return function(np_x, *(parameter_values))

    return wrapper


def call_native_tf():
    func = native_tf_gauss_sum
    params = tuple(tf.Variable(v, dtype=tf.float64) for v in parameter_values)

    def wrapper():
        return func(tf_x, *params)

    return wrapper


print(
    "sympy tf lambdify",
    timeit.timeit(evaluate_with_parameters(tf_gauss_sum), number=100),
)
print(
    "sympy numpy lambdify",
    timeit.timeit(evaluate_with_parameters(numpy_gauss_sum), number=100),
)
print(
    "sympy jax lambdify",
    timeit.timeit(evaluate_with_parameters(jax_gauss_sum), number=100),
)
print("native tf", timeit.timeit(call_native_tf(), number=100))

print(
    "native jax",
    timeit.timeit(evaluate_with_parameters(native_jax_gauss_sum), number=100),
)

# 5. Handling parameters
# 5.1 Changing parameter values
#     Can be done in the model itself...
#     But how can the values be propagated to the `AmplitudeModel`?
#     Well, if an amplitude model only defines parameters with a name and the
#     values are supplied in the function evaluation, then everything is
#     decoupled and there are no problems.

# 5.2 Changing parameter names
#     Names can be changed in the sympy `AmplitudeModel`. Since this sympy model
#     serves as the source of truth for the `Function`, all things generated from
#     this model will reflect the name changes as well.
#     But going even further, since the Parameters are passed into the functions
#     as arguments, the whole naming becomes irrelevant anyways.
#     tf_var_A1 = tf.Variable(1.0, dtype=tf.float64)  <- does not carry a name!!

# 5.3 Coupling parameters
#     This means that one parameter is just assigned to another one?
#
result = evaluate_with_parameters(jax_gauss_sum)()
print(result)
print(np_x)
plt.hist(np_x, bins=100, weights=result)
plt.show()

parameter_values = (1.0, 0.0, 0.1, 2.0, 2.0, 0.1)
result = evaluate_with_parameters(jax_gauss_sum)()
plt.hist(np_x, bins=100, weights=result)
plt.show()

# 6. Exchange a gaussian with some other function (this should be easy)
from sympy.abc import C, a, b, x

expr = sin(a * x) + cos(b * x)
print(expr)
substituted = expr.subs(sin(a * x), C)
print(substituted)


# 7. Matrix operations?
from sympy import Abs, I, Matrix, MatrixSymbol, re
from sympy.physics.quantum.dagger import Dagger

spin_density = MatrixSymbol("rho", 3, 3)
amplitudes = Matrix([[1 + I], [2 + I], [3 + I]])

dummy_intensity = re(
    Dagger(amplitudes) * spin_density * amplitudes,
    evaluate=False
    # evaluate=False is important otherwise it generates some function that cant
    # be lambdified anymore
)

tf_intensity = lambdify(
    spin_density,
    dummy_intensity,
    modules=(tf,),
)
print(inspect.getsource(tf_intensity))
real0 = tf.constant(0, dtype=tf.float64)
real1 = tf.constant(1, dtype=tf.float64)
intensity_result = tf_intensity(
    np.array(
        [
            [
                tf.complex(real1, real0),
                tf.complex(real0, real0),
                -tf.complex(real0, real1),
            ],
            [
                tf.complex(real0, real0),
                tf.complex(real1, real0),
                tf.complex(real0, real0),
            ],
            [
                tf.complex(real0, real1),
                tf.complex(real0, real0),
                tf.complex(real1, real0),
            ],
        ]
    ),
)

print(intensity_result)
