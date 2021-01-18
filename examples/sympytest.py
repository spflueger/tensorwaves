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
from sympy import exp, lambdify, pi, simplify, symbols

# 1. create a double gaussian amp with sympy
# when building the model, we should be careful to pass the parameters as arguments as well
# otherwise frameworks like jax cant determine the gradient
x, A1, mu1, sigma1, A2, mu2, sigma2 = symbols(
    "x, A1, mu1, sigma1, A2, mu2, sigma2"
)
gaussian1 = (
    A1
    / (sigma1 * tf.sqrt(2.0 * math.pi))
    * exp(-((x - mu1) ** 2) / (2 * sigma1))
)
# A1, mu1, sigma1 = 1, 0, 1
# A2, mu2, sigma2 = 2, 1, 0.5
gaussian2 = (
    A2
    / (sigma2 * tf.sqrt(2.0 * math.pi))
    * exp(-((x - mu2) ** 2) / (2 * sigma2))
)

gauss_sum = gaussian1 + gaussian2
# gauss_sum = simplify(gauss_sum)

# 2. use this as a function in tensorwaves (using lambdify)
tf_gauss_sum = lambdify(
    (x, A1, mu1, sigma1, A2, mu2, sigma2), gauss_sum, "tensorflow"
)
print(inspect.getsource(tf_gauss_sum))


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


# 4. compare performance

tf_var_A1 = tf.Variable(1.0, dtype=tf.float64)
tf_var_mu1 = tf.Variable(0.0, dtype=tf.float64)
tf_var_sigma1 = tf.Variable(0.1, dtype=tf.float64)

tf_var_A2 = tf.Variable(2.0, dtype=tf.float64)
tf_var_mu2 = tf.Variable(2.0, dtype=tf.float64)
tf_var_sigma2 = tf.Variable(0.2, dtype=tf.float64)

tf_x = tf.constant(np.random.uniform(-1, 3, 10000))


def call_sympy_lambdify():
    return tf_gauss_sum(
        tf_x,
        tf_var_A1,
        tf_var_mu1,
        tf_var_sigma1,
        tf_var_A2,
        tf_var_mu2,
        tf_var_sigma2,
    )


def call_native_tf():
    return native_tf_gauss_sum(
        tf_x,
        tf_var_A1,
        tf_var_mu1,
        tf_var_sigma1,
        tf_var_A2,
        tf_var_mu2,
        tf_var_sigma2,
    )


print("sympy lambdify", timeit.timeit(call_sympy_lambdify, number=100))
print("native tf", timeit.timeit(call_native_tf, number=100))

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
result = call_sympy_lambdify()
# print(result)
# print(tf_x)
plt.hist(tf_x.numpy(), bins=100, weights=result.numpy())
plt.show()

tf_var_sigma2 = tf_var_sigma1
result = call_sympy_lambdify()
plt.hist(tf_x.numpy(), bins=100, weights=result.numpy())
plt.show()

# 6. Exchange a gaussian with some other function (this should be easy)
