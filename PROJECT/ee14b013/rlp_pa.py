# @Author: aravind ( Aravind S; EE14B013 )
# @Email: arvindsuresh2009@gmail.com
# @Github: (Aravind-Suresh)[https://github.com/Aravind-Suresh]
# @Date:   2017-01-24T17:58:43+05:30
# @Last modified by:   aravind
# @Last modified time: 2017-01-25T09:39:09+05:30

"""
    Python script for the preliminary project task ( PA ).

    Contains methods for computing masked softmax, given a list of values ( float ).
"""

# Necessary modules imported
import tensorflow as tf
import numpy as np

# For time calculations
from datetime import datetime

def f(x):
    """
    Method to compute masked softmax, given a list of values ( float ).

    :param list x: Input list of float values. Float only.

    :return matrix: Matrix containing masked softmax values, if is_valid_float_list(x) is True.
                    Else, returns None.
    """

    if not is_valid_float_list(x):
        return None

    # Tensorflow graph definitions

    # Length of x
    n = len(x)

    # Placeholder for the input - type is float64 ( float ), and shape is (n, 1)
    ph_x = tf.placeholder(tf.float64, shape = [n, 1])

    # Extracting max-element from x
    x_max_ele = tf.reduce_max(ph_x, reduction_indices=[0])

    # Subtracting max-element from all entries in x
    # Prevents overflows when used with tf.exp
    # Works because softmax([a, b, c]) = softmax([a-m, b-m, c-m])s
    x_sub = tf.sub(ph_x, x_max_ele)

    # Lower triangular matrix of size (n, n)
    # Used for cumulative sum ( appears in denominator of softmax )
    #
    # Example: n = 3
    # L = [[ 1.  0.  0.]
    #       [ 1.  1.  0.]
    #       [ 1.  1.  1.]]
    L = tf.constant(np.tril(np.ones((n, n))), shape = [n, n])

    # Reverse-transposed upper triangular matrix of size (n, n)
    # Used for masking upper triangle
    #
    # Example: n = 3
    # U = [[ 1.  1.  1.]
    #       [ 1.  1.  0.]
    #       [ 1.  0.  0.]]
    U = tf.reverse(tf.transpose(tf.constant(np.triu(np.ones((n, n))), shape = [n, n])), [True, False])

    # Holds exp(xx) for xx in x
    x_exp = tf.exp(x_sub)

    # Vertically stacks vector x of length n to create a matrix of size (n, n)
    x_exp_vstack = tf.transpose(tf.tile(x_exp, [1, n]))

    # Masks the upper triangle part alone ( specified by U ) with a tf.mul operation
    x_exp_vstack_mask = tf.mul(U, x_exp_vstack)

    # Stores the cumulative sum of exponentials
    x_exp_cum_sum = tf.matmul(L, x_exp)

    # Reverses the rows of x_exp_cum_sum so as to fit output format
    x_exp_cum_sum_rev = tf.reverse(x_exp_cum_sum, [ True, False ])

    # Vertically stacks vector x_exp_cum_sum_rev of length n to create a matrix of size (n, n)
    x_exp_cum_sum_rev_vstack = tf.tile(x_exp_cum_sum_rev, [1, n])

    # Final output y
    # Element-to-element division of exponentials to cumulative sums
    y = tf.div(x_exp_vstack_mask, x_exp_cum_sum_rev_vstack)

    # Starting tensorflow session
    with tf.Session() as sess:
        # Reshaping x to fit placeholder definitions
        x_res = np.asarray(x).reshape((n, 1))

        # Running the computation for y
        # Feed forwarding x using feed_dict
        ret = sess.run(y, feed_dict = { ph_x: x_res })

        # Returns output
        return ret

def is_valid_float_list(x):
    """
    Method to check whether the given list is a list of floats.

    :param list x: Arbitrary input.
    :return bool: True, if x is a list of float. False, otherwise.
    """
    return (type(x) == list) and all(isinstance(xx, float) for xx in x)

def eval_func(f, x):
    """
    Utility method to evaluate execution time of a function f, taking x as argument

    Computes the execution time ( using datetime.now() ), and prints it.

    :param function f: Arbitrary function.
    :param object x: Any arbitrary input.
    :return void: No return value
    """
    # Measure running time
    f_eval_start_time = datetime.now()
    y = f(x)
    f_eval_end_time = datetime.now()

    # Printing input, output
    print x
    print y
    print 'Time taken:', (f_eval_end_time - f_eval_start_time)

def main():
    # Sample execution 1
    x = [1.0, 2.0, 3.0]
    eval_func(f, x)

    # Sample execution 2
    x = [2.0, 3.0, 4.0, 5.0]
    eval_func(f, x)

    # Sample execution 3
    n = 1000
    x = list(np.random.normal(0, 1, n))
    eval_func(f, x)

if __name__ == '__main__':
    main()
