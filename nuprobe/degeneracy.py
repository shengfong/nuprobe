import numpy as np

from nuprobe.params import rtol, atol


def calc_relative_difference_between_all_elements(input_vec, d):
    """ Calculate relative difference between all elements
    params:
    - input_vec: a vector of real numbers
    - d (int): dimension of the vector
    """ 
    first_diff_elem = np.repeat(input_vec, d).reshape((d, d))
    second_diff_elem = first_diff_elem.T
    return np.abs(first_diff_elem - second_diff_elem) / np.abs(first_diff_elem)


def calc_unique_values(input_vec, tol=rtol):
    """ Determine unique values in a vector
    params:
    - input_vec: a vector of real numbers
    - tol: relative difference for two numbers to be considered equal
    """ 
    unique_input_vec = np.unique(input_vec)
    d = len(unique_input_vec)

    # If the absolute value of the eigenvalue is smaller than atol, set to atol
    for i in range(d):
        if np.abs(unique_input_vec[i]) < atol: unique_input_vec[i] = atol

    relative_diff = calc_relative_difference_between_all_elements(unique_input_vec, d)
    # Use a triangular matrix to store the differences (put zero in elements below diagonal)
    triu_diff = np.triu(relative_diff)
    # Avoid mistaking the smallest element with the zero from the triangular matrix
    triu_diff[triu_diff == 0] = 1 + tol

    min_per_row = np.min(triu_diff, axis=1)

    unique_indices = list(np.where(min_per_row > tol)[0])
    return unique_input_vec[unique_indices]



