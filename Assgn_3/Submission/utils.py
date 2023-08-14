import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.model_selection import train_test_split
from scipy.special import softmax

__author__ = "Kumar Abhishek"
__email__ = "kabhishe@sfu.ca"

# Specifying seed value for reproducibility.
seedval = 8888
np.random.seed(seedval)


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom softmax multi-class classification module.
    Inspired from these sources:
    - https://scikit-learn.org/stable/developers/develop.html
    - https://stackoverflow.com/a/47224641

    Args:
        BaseEstimator (class): Base class for all estimators in scikit-learn.
        ClassifierMixin (class): Mixin class for all classifiers in scikit-learn.
    """

    def __init__(self, n_features, n_classes, reg_param, lr, batch_size, n_epochs):
        """
        Initializes a SoftmaxClassifier object with input parameters.

        Args:
            n_features (int): Number of features in the input.
            n_classes (int): Number of classes for multi-class classification.
            reg_param (float): L2 regularization parameter.
            lr (float): Learning rate for the neuron.
            batch_size (int): Number of training samples used to update the model
                              parameters in each iteration of gradient descent.
            n_epochs (int): Number of epochs for the gradient descent update.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.reg_param = reg_param
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Store the model parameters as a dictionary in class attributes.
        self.params_ = {}

        # Initialize the model parameters with values randomly sampled from a zero-mean
        # unit-variance Gaussian distribution.
        self.params_["W"] = np.random.RandomState(seed=seedval).random(
            (self.n_features, self.n_classes)
        )
        self.params_["b"] = np.random.RandomState(seed=seedval).random((self.n_classes))

        # Initialize an empty list to track each epoch's loss value.
        self.loss_values_ = []

    def fit(self, X, y, calculate_loss=False):
        """
        Trains the SoftmaxClassifier model for the given input data-label
        pairs.

        Args:
            X (np.array): Training data input.
            y (np.array): Training data labels.
            calculate_loss (bool): Specify if the prediction loss should be calculated.

        Returns:
            SoftmaxClassifier: The trained SoftmaxClassifier
                               model.
        """

        # Use scikit-learn's inbuilt sanity checks on the inputs and the labels.
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Extract the class labels present in the training data. This will be used in
        # calculating the loss values.
        self.classes_ = np.unique(y)

        # Assign the training data as class attributes.
        self.X_, self.y_ = X, y

        # Infer the number of features from the input shape.
        self.n_features_ = X.shape[1]

        # Infer the number of training samples from the input shape.
        train_size = X.shape[0]
        # Calculate how many mini-batches can be constructed based on the number of
        # training samples and the batch size.
        batch_idx_range = (
            (train_size // self.batch_size)
            if (train_size % self.batch_size == 0)
            else (train_size // self.batch_size + 1)
        )

        # Iterate the gradient descent method for n_epochs epochs.
        for _ in range(self.n_epochs):
            # Shuffle the training data indices to ensure that a random order of
            # training data is fed to the model in each epoch.
            shuffled_idxs = np.random.RandomState(seed=seedval).permutation(train_size)

            # Initialize epoch_loss as 0 to keep track of the loss value.
            epoch_loss = 0.0

            # Iterate the gradient descent update for each mini-batch of training data.
            for idx in range(batch_idx_range):

                # Find the start and end indices for each mini-batch.
                start_idx = idx * self.batch_size
                # Limit end_idx to the last index in case the calculated end_idx is
                # outside the training indices' range.
                end_idx = (
                    (start_idx + self.batch_size)
                    if (start_idx + self.batch_size) < train_size
                    else train_size
                )

                # Construct training data and label mini-batches based on start and end
                # indices from the shuffled indices.
                batch_X, batch_y = (
                    X[shuffled_idxs[start_idx:end_idx]],
                    y[shuffled_idxs[start_idx:end_idx]],
                )

                # Convert the training labels to one-hot encoded format for multi-class
                # classification targets.
                batch_y_onehot = integer_to_onehot(
                    y_integer=batch_y, n_classes=self.n_classes
                )

                # Calculate the compute step of the first layer for the mini-batch.
                z_1 = (batch_X @ self.params_["W"]) + self.params_["b"]

                # Calculate the activation step of the first layer for the mini-batch.
                a_1 = softmax(z_1, axis=1)

                # Calculate the loss for each mini-batch and add it to the epoch loss
                # if the calculate_loss parameter is set to True.
                # Loss function formulation inspired by https://bit.ly/2SRAnXE
                if calculate_loss:
                    epoch_loss += -np.sum(np.log(a_1[np.arange(len(batch_y)), batch_y]))

                # Calculate the gradients of the loss w.r.t. the parameters, 'W' and
                # 'b', denoted by 'dW' and 'db' respectively.
                dW = np.dot(batch_X.T, (a_1 - batch_y_onehot)) + (
                    2 * self.reg_param * self.params_["W"]
                )
                db = np.sum(a_1 - batch_y_onehot)

                # Update the model parameters: 'W' and 'b' based on the calculated
                # gradients.
                self.params_["W"] -= (1 / batch_X.shape[0]) * self.lr * dW
                self.params_["b"] -= (1 / batch_X.shape[0]) * self.lr * db

            # Calculate the loss for the epoch by averaging over all the training
            # samples. Append this loss value to keep track of training loss.
            if calculate_loss:
                self.loss_values_.append(epoch_loss / train_size)

        return self

    def predict(self, X, y=None, calculate_loss=False):
        """
        Predicts the labels for input data using a trained SoftmaxClassifier
        model.

        Args:
            X (np.array): Test data input.
            y (np.array): Training data labels.
            calculate_loss (bool): Specify if the prediction loss should be calculated.
                                   Set to True only when y is provided.

        Returns:
            np.array: Predicted labels for the test data.
        """

        if calculate_loss and y is None:
            print("Labels needed to calculate loss. Exiting.")
            return None

        # Use scikit-learn's inbuilt validation to check if the model has been trained.
        check_is_fitted(self, ["X_", "y_", "classes_"])

        if calculate_loss:
            # Use scikit-learn's inbuilt sanity checks on the inputs and the labels.
            X, y = check_X_y(X, y)
            check_classification_targets(y)
        else:
            # Use scikit-learn's inbuilt sanity checks on the inputs.
            X = check_array(X)

        # Calculate the compute step of the first layer for the input data.
        z_1 = (X @ self.params_["W"]) + self.params_["b"]

        # Calculate the activation step of the first layer for the input data.
        a_1 = softmax(z_1, axis=1)

        # Calculate the loss for the test samples if the calculate_loss parameter is
        # set to True.
        if calculate_loss:
            loss = -np.mean(np.log(a_1[np.arange(len(y)), y]))

        # Generate model predictions by extracting the most likely label.
        pred_y = np.argmax(a_1, axis=1)

        if calculate_loss:
            return pred_y, loss
        else:
            return pred_y


def prepare_data(X, Y, class0, class1, class2, class3, test_split_fraction, seedval):
    """
    Prepares the data for binary classification.
    Assigns binary labels ({0, 1}) to training data.
    Performs stratified train-test split.

    Args:
        X (np.array): Input data.
        Y (np.array): Labels for the input data.
        class0 (int): The digit (0-9) which should be classified as class 0.
        class1 (int): The digit (0-9) which should be classified as class 0.
        test_split_fraction (float): The fraction of the dataset which should be set
                                     aside for evaluating (testing) the model.
        seedval (int): Seed value for shuffling the data and generating random splits
                       of the data for training and testing.

    Returns:
        X_train (np.array): Training split of the input data.
        X_test (np.array): Training split of the data labels.
        Y_train (np.array): Testing split of the input data.
        Y_test (np.array): Testing split of the data labels.
    """

    # Check if the number of training data samples is equal to the number of training
    # labels.
    assert X.shape[0] == Y.shape[0]

    # Define the class labels for multi-class classification.
    class_labels = [0, 1, 2, 3]

    # Initialize dictionaries to store classes and indices for the classification.
    data_cls, data_idxs = {}, {}

    # Assign the four specified classes.
    data_cls[0], data_cls[1], data_cls[2], data_cls[3] = class0, class1, class2, class3

    # Extract indices of the four specified classes for classification.
    for cl in class_labels:
        data_idxs[cl] = np.where(Y == data_cls[cl])[0]

    # Concatenate the indices of the four specified classes.
    all_idxs = np.concatenate((data_idxs[0], data_idxs[1], data_idxs[2], data_idxs[3]))

    # Extract data and labels for the four specified classes.
    all_X, all_Y = X[all_idxs], Y[all_idxs]

    # Change the labels in the data to labels by mapping class0 to 0, class1 to 1,
    # and so on.
    for cl in class_labels:
        all_Y[all_Y == data_cls[cl]] = cl

    # Generate training and testing splits of the data.
    #
    # shuffle=True shuffles the data before splitting into training and testing splits.
    #
    # stratify=all_Y ensures that the class ratios remain the same in both training and
    # testing splits.
    # Source: https://stackoverflow.com/a/38889389
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_X,
        all_Y,
        test_size=test_split_fraction,
        random_state=seedval,
        stratify=all_Y,
        shuffle=True,
    )

    return X_train, X_test, Y_train, Y_test


def integer_to_onehot(y_integer, n_classes):
    """
    Converts the labels from integer labels to one-hot encoded format for multi-class
    classification targets.
    Inspired by https://stackoverflow.com/a/29831596

    Args:
        y_integer (np.array): Original integer labels.
        n_classes (np.array): Number of classes in the dataset.

    Returns:
        y_onehot (np.array): One-hot encoded labels.
    """

    # Initialize a matrix of zeros as a placeholder to fill values in.
    y_onehot = np.zeros((len(y_integer), n_classes))

    # Use indexing to specify the indices where the values should be set to 1.
    y_onehot[np.arange(len(y_integer)), y_integer] = 1

    return y_onehot
