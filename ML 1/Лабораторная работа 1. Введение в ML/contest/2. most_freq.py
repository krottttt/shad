import numpy as np

def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    return np.argmax(np.bincount(np.array(lst)))