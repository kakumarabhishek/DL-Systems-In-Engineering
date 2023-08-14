import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline


# Specifying seed value for reproducibility.
seedval = 8888
np.random.seed(seedval)


def grad_descent(X, Y, lr=1e-4, n_params=None, n_epochs=100, batch_size=-1):

    """
    Gradient descent implementation using NumPy.

    Args:
        X (np.array): Training data input.
        Y (np.array): Training data labels.
        n_params (int): Number of parameters in the trained model.
        lr (float, optional): Learning rate for weight update.
        n_epochs (int, optional): Number of epochs for gradient descent update.
        batch_size (int, optional): Number of training samples used to update
                                    the weight in each iteration.
                                    -1 indicates that the entire data is used at a time.
                                    1 indicates stochastic gradient descent.
                                    Any other value indicates mini-batch gradient descent.

    Returns:
        W (np.array): Trained model parameters.
        epoch_W (np.array): Traced model parameters over iterations.
        epoch_loss (np.array): Traced model loss over iterations.

    """

    # Check if number of training samples is equal to number of training labels.
    assert X.shape[1] == Y.shape[0]

    # Infer m (#training samples) from shape of X.
    # If n_params is not provided, infer it from shape of X.
    if n_params:
        m = X.shape[1]
    else:
        n_params, m = X.shape

    # Initialize W of the appropriate shape with values randomly sampled from
    # a zero-mean unit-variance Gaussian distribution.
    W = np.random.normal(0, 1, (n_params, 1))

    # Initialize empty lists to keep track of model parameters and loss values
    # for each iteration.
    epoch_W, epoch_loss = [], []

    # If batch_size is -1, all training samples are used to update the parameters.
    if batch_size == -1:
        # Iterate for n_epochs.
        for epoch in range(n_epochs):

            # Generate model predictions based on current parameters.
            pred_Y = X.T @ W

            # Calculate the mean squared error loss using the predictions
            # and ground truth values.
            loss = np.sum(np.power((pred_Y - Y), 2))
            # Keep track of loss values by appending to a list.
            epoch_loss.append(loss / m)

            # Parameter update step.
            W = W - 2 * (lr / m) * (X @ (pred_Y - Y))
            # Keep track of model parameters by appending to a list.
            epoch_W.append(W)

    # If batch_size is some other value, we perform mini-batch or stochastic
    # gradient descent as specified.
    else:
        # Iterate for n_epochs.
        for epoch in range(n_epochs):
            # Depending on batch_size, we need to sample training data.
            for idx in range(m // batch_size):

                # Finding starting and ending indices for each batch.
                start_idx = idx * batch_size
                end_idx = start_idx + batch_size

                # Sampling data points and labels based on starting and
                # ending indices calculated above.
                elem_X = X[:, start_idx:end_idx].reshape(-1, batch_size)
                elem_Y = Y[start_idx:end_idx].reshape(batch_size, -1)

                # Generate model predictions based on current parameters.
                pred_Y = elem_X.T @ W

                # Calculate the mean squared error loss using the predictions
                # and ground truth values.
                loss = np.sum(np.power((pred_Y - elem_Y), 2))
                # Keep track of loss values by appending to a list.
                epoch_loss.append(loss / batch_size)

                # Parameter update step.
                W = W - 2 * (lr / batch_size) * (elem_X @ (pred_Y - elem_Y))
                # Keep track of model parameters by appending to a list.
                epoch_W.append(W)

    # Converting epoch_W and epoch_loss to NumPy arrays for added functionality.
    epoch_W = np.array(epoch_W)[..., 0]
    epoch_loss = np.array(epoch_loss)

    return W, epoch_W, epoch_loss


def analytical_linreg(X, Y):

    """
    Analytical solution for linear regression implementation using NumPy.

    Args:
        X (np.array): Training data input.
        Y (np.array): Training data labels.

    Returns:
        W_analytical (np.array): Trained model parameters.
    """

    # Calculating the analytical solution.
    # We use pinv() here to generate the Moore-Penrose pseudoinverse,
    # in case of a singular matrix.
    W_analytical = (X @ np.linalg.pinv(X.T @ X)) @ Y

    return W_analytical


def generate_data(n_samples, poly_coeffs, l_lim, u_lim):

    """
    Generate synthetic data based on coefficients from a polynomial function.

    Args:
        n_samples (int): Number of samples to be generated.
        poly_coeffs (np.array): Coefficients of the polynomial to be used.
        l_lim (float): Lower limit of the values to be generated.
        u_lim (float): Upper limit of the values to be generated.

    Returns:
        X (np.array): Generated samples.
        Y_clean (np.array): Labels for the generated samples based on the 
                            polynomial coefficients.
        Y_noisy (np.array): Labels for the generated samples corrupted with
                            zero-mean unit-variance Gaussian noise.
    """

    # Infer the order of the polynomial from the shape of the coefficients.
    poly_order = poly_coeffs.shape[0] - 1

    # Generate equally spaced input points between the specified limits.
    X = np.linspace(l_lim, u_lim, n_samples).reshape(-1, 1)

    # Generate polynomial features for these generated input points.
    poly_X = X
    # Traverse over all the exponent values of the polynomial.
    for exp in range(1, poly_order + 1):
        # Skip exponent = 1 since we already have those values.
        if exp != 1:
            # Raise the input points to the exponent and concatenate.
            poly_X = np.hstack((poly_X, X ** exp))

    # Finally, add ones to the features by exponentiating with 0.
    poly_X = np.hstack((poly_X, X ** 0))

    # Generate labels for the generated data points using the coefficients.
    Y_clean = (poly_X @ poly_coeffs.T).reshape(-1, 1)

    # Add Gaussian noise to corrupt the labels.
    Y_noisy = Y_clean + np.random.normal(0, 1, (n_samples, 1))

    return X, Y_clean, Y_noisy


def generate_data_eval(n_samples, poly_coeffs, l_lim, u_lim):

    """
    Generate synthetic data based on coefficients from a polynomial function,
    but for evaluating a trained model.

    Args:
        n_samples (int): Number of samples to be generated.
        poly_coeffs (np.array): Coefficients of the polynomial to be used.
        l_lim (float): Lower limit of the values to be generated.
        u_lim (float): Upper limit of the values to be generated.

    Returns:
        X (np.array): Generated samples.
        Y_clean (np.array): Labels for the generated samples based on the 
                            polynomial coefficients.
    """

    # Infer the order of the polynomial from the shape of the coefficients.
    poly_order = poly_coeffs.shape[0] - 1

    # Generate equally spaced input points between the specified limits.
    X = np.linspace(l_lim, u_lim, n_samples).reshape(-1, 1)

    # Generate polynomial features for these generated input points.
    poly_X = X
    # Traverse over all the exponent values of the polynomial.
    for exp in range(1, poly_order + 1):
        # Skip exponent = 1 since we already have those values.
        if exp != 1:
            # Raise the input points to the exponent and concatenate.
            poly_X = np.hstack((poly_X, X ** exp))

    # Finally, add ones to the features by exponentiating with 0.
    poly_X = np.hstack((poly_X, X ** 0))

    # Generate labels for the generated data points using the coefficients.
    Y_clean = (poly_X @ poly_coeffs).reshape(-1, 1)

    return X, Y_clean


def cost_fn(params, X, Y):

    """
    Calculate the cost for a set of input points, given the input labels and
    model parameters.

    Args:
        params (np.array): Parameters of the model whose cost is to be calculated.
        X (np.array): Data points.
        Y (np.array): Corresponding data labels.
    
    Returns:
        cost (np.array): Model cost for all the data points.
    """

    # Calculating the cost for each training sample.
    cost = ((X.T @ params) - Y) ** 2
    # Averaging the cost over all training samples.
    cost = np.average(cost, axis=0)

    return cost
