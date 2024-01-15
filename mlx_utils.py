import mlx.core as mx
import mlx.nn as nn

def softmax(x):
    """
    Compute the softmax function for the given input array.

    Parameters:
    x (ndarray): Input array.

    Returns:
    ndarray: Output array after applying the softmax function.
    """
    e_x = mx.exp(x - mx.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def sample_from_categorical(probs):
    """
    Samples from a categorical distribution.

    Parameters:
    - probs (ndarray): The probability distribution over categories.

    Returns:
    - sample (ndarray): The sampled categories.
    """
    sample = mx.zeros(probs.shape[:-1])
    for i in range(len(probs)):
        sample[i] = mx.random.categorical(probs[i])
    sample = sample.astype(mx.int16)
    return sample

def log_prob(probs, actions):
    """
    Calculates the logarithm of the probabilities of the given actions.

    Parameters:
    probs (ndarray): Array of probabilities.
    actions (ndarray): Array of actions.

    Returns:
    ndarray: Array of logarithm of probabilities for the given actions.
    """
    return mx.log(probs[mx.arange(len(probs)), actions])

def calc_entropy(probs):
    """
    Calculates the entropy of a probability distribution.

    Parameters:
    probs (ndarray): The probability distribution.

    Returns:
    float: The entropy of the probability distribution.
    """
    return -mx.sum(probs * mx.log(probs + 1e-10), axis=-1)

def shuffle_array(arr):
    """
    Shuffles the elements of the input array randomly.

    Parameters:
    arr (list): The input array to be shuffled.

    Returns:
    list: The shuffled array.
    """
    n = len(arr)
    for i in range(n):
        j = mx.random.randint(i, n)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def custom_std(arr, axis=0):
    """
    Compute the standard deviation of a numpy array along a specified axis 
    without using the numpy std function.
    """
    # Calculate mean
    mean = mx.mean(arr, axis=axis, keepdims=True)

    # Subtract mean and square the result
    squared_diff = (arr - mean) ** 2

    # Sum the squared differences and divide by the number of elements
    variance = mx.sum(squared_diff, axis=axis) / arr.shape[axis]

    # Take the square root to get the standard deviation
    std_dev = mx.sqrt(variance)
    return std_dev