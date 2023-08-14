import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import copy

# Specify the random seed value to ensure reproducibility.
seedval = 8888

# Set the seed for all PyTorch related modules.
torch.manual_seed(seedval)

# Set the amount of padding to be applied in order to preserve the spatial size of 
# feature maps, i.e., "same" padding.
# The values in the dictionary below denote that when a kernel of size 3 x 3 is used,
# a padding of 1 pixel across all borders is to be applied. Similarly, a padding of 
# 2 pixels is to be applued when using a kernel of size 5 x 5.
pad_dict = {3: 1, 5: 2}

# Specify the directory where the datasets' DataFrames are to be saved.
savedir = "Data/"


class LogisticRegression(nn.Module):
    """
    Custom implementation of a logistic regression module.
    Accepts a structured input (image), flattens it, and outputs a single value.
    """

    def __init__(self, input_dim):
        """
        Initializes a LogisticRegression model.

        Args:
            input_dim (int): Number of dimensions in the input to the neuron.
        """
        super(LogisticRegression, self).__init__()

        # Define the neuron by defining its weights using the number of input features
        # (input_dim) and the number of output features (1).
        self.neuron = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        """
        Forward pass of the LogisticRegression model.

        Args:
            x (torch.Tensor): Input to the LogisticRegression model.

        Returns:
            compute (torch.Tensor): Output of the neuron's computation.
        """
        # Flatten the input along axis=1 so that it can be fed to a Linear layer.
        x = torch.flatten(x, 1)
        # Calculate neuron's computation using the flattened input.
        compute = self.neuron(x)
        return compute


class FCN(nn.Module):
    """
    Custom implementation of a fully connected network (FCN) module.
    Accepts a structured input (image), flattens it, and outputs a vector of 
    predictions.
    """

    def __init__(self, input_dim, hidden_dim, n_classes):
        """
        Initializes an FCN model.

        Args:
            input_dim (int): Number of dimensions in the input to the neuron.
            hidden_dim (int): Number of neurons in the single hidden layer of the FCN.
            n_classes (int): Number of classes to predict from.
        """
        super(FCN, self).__init__()
        # Define the fully connected layer using first the number of input features 
        # (input_dim) and the number of output features (hidden_dim) and then the 
        # corresponding number of input features (hidden_dim) and the number of output 
        # features (n_classes).
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)

    def forward(self, x):
        """
        Forward pass of the FCN model.

        Args:
            x (torch.Tensor): Input to the FCN model.

        Returns:
            x (torch.Tensor): Output of the FCN model.
        """
        # Flatten the input along axis=1 so that it can be fed to a Linear layer.
        x = torch.flatten(x, 1)
        # Pass the flattened input to the fully connected layer and apply the ReLU 
        # activation to the layer's output.
        x = F.relu(self.fc1(x))
        # Pass the ReLU activation output from the hidden layer to the generate the 
        # model's output.
        x = self.fc2(x)
        return x

class FCN_Dropout(nn.Module):
    """
    Custom implementation of a fully connected network (FCN) module with dropout.
    Accepts a structured input (image), flattens it, and outputs a vector of 
    predictions.
    """

    def __init__(self, input_dim, hidden_dim, n_classes):
        """
        Initializes an FCN model with a dropout layer.

        Args:
            input_dim (int): Number of dimensions in the input to the neuron.
            hidden_dim (int): Number of neurons in the single hidden layer of the FCN.
            n_classes (int): Number of classes to predict from.
            dropout (bool, optional): Specify if Dropout layer(s) should be used.
                                      Defaults to False.
        """
        super(FCN_Dropout, self).__init__()
        # Define the dropout layer with an element being zeroed out with 
        # probability = 0.1.
        self.drop1 = nn.Dropout(0.1)
        # Define the fully connected layer using first the number of input features 
        # (input_dim) and the number of output features (hidden_dim) and then the 
        # corresponding number of input features (hidden_dim) and the number of output 
        # features (n_classes).
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)

    def forward(self, x):
        """
        Forward pass of the FCN model.

        Args:
            x (torch.Tensor): Input to the FCN model.

        Returns:
            x (torch.Tensor): Output of the FCN model.
        """
        # Flatten the input along axis=1 so that it can be fed to a Linear layer.
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        # Pass the flattened input to the fully connected layer and apply the ReLU 
        # activation to the layer's output.
        x = F.relu(self.fc1(x))
        # Pass the ReLU activation output from the hidden layer to the generate the 
        # model's output.
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """
    Custom implementation of a convolutional neural network (CNN) module.
    Accepts a structured input (image) and passes it through a series of layers to 
    output a vector of predictions.
    """

    def __init__(self, filter_1, filter_2, n_channels, fc_dim, n_classes):
        """
        Initializes a CNN model.

        Args:
            filter_1 (int): Size of the convolutional kernel in the first 
                            convolutional layer.
            filter_2 (int): Size of the convolutional kernel in the second 
                            convolutional layer.
            n_channels (int): Number of channels in the first convolutional layer.
            fc_dim (int): Number of neurons in the fully connected layer.
            n_classes (int): Number of classes to predict from.
        """
        super(CNN, self).__init__()
        # Define the first convolutional layer and choose the padding accordingly so as
        # to maintain the feature size.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=filter_1,
            padding=pad_dict[filter_1],
        )
        # Define the first pooling layer.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the second convolutional layer and choose the padding accordingly so as
        # to maintain the feature size.
        self.conv2 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=(n_channels * 2),
            kernel_size=filter_2,
            padding=pad_dict[filter_2],
        )
        # Define the second pooling layer.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the number of resulting features that will be fed to the fully 
        # connected layer.
        self.fc_in = (n_channels * 2) * 50 * 50

        # Define the fully connected layer using first the number of input features 
        # (fc_in) and the number of output features (fc_dim) and then the corresponding
        # number of input features (fc_dim) and the number of output features 
        # (n_classes).
        self.fc1 = nn.Linear(in_features=self.fc_in, out_features=fc_dim)
        self.fc2 = nn.Linear(in_features=fc_dim, out_features=n_classes)

    def forward(self, x):
        # Pass the input to the first convolutional layer, followed by ReLU activation 
        # and max pooling operations.
        x = self.pool1(F.relu(self.conv1(x)))
        # Pass the input to the second convolutional layer, followed by ReLU activation 
        # and max pooling operations.
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten the output from the convolutional layers along axis=1 so that it can
        # be fed to a Linear layer.
        x = torch.flatten(x, 1)
        # Pass the flattened input to the fully connected layer and apply the ReLU 
        # activation to the layer's output.
        x = F.relu(self.fc1(x))
        # Pass the ReLU activation output from the hidden layer to the generate the 
        # model's output.        
        x = self.fc2(x)
        return x


def prepare_utk_data(utk_imgs_path, write_data=False):
    """
    Prepare the UTK Aligned and Cropped Faces dataset for classification.

    Args:
        utk_imgs_path (str): Path of the UTK dataset images directory.
        write_data (bool, optional): Specify if the DataFrame should be saved as a CSV 
                                     file. Defaults to False.

    Returns:
        utk_df_final (pd.DataFrame): The final DataFrame containing the image path, 
                                     age, and gender labels from the UTK dataset. 
    """
    # List all files in the directory by filtering those only with extension '.jpg'.
    utk_files = [f for f in os.listdir(utk_imgs_path) if f.endswith(".jpg")]
    # Extract the ages of all the images based on their file names.
    ages = [int(f.split("_")[0]) for f in utk_files]

    # Check if the ages have been extracted for all the images.
    assert len(ages) == len(utk_files)

    # List all the files in the directory which are images and have age >= 13.
    files_13plus = [
        f
        for f in os.listdir(utk_imgs_path)
        if (f.endswith(".jpg") and (int(f.split("_")[0]) >= 13))
    ]
    # Filter the list of files to remove any files which have a gender apart from 
    # male (0) or female (1).
    # For example: UTK dataset had a file called 61_3_20170109150557335.jpg.chip.jpg
    # whose label indicates gender=3.
    files_13plus_filtered = [f for f in files_13plus if int(f.split("_")[1]) in [0, 1]]

    # Initialize a DataFrame and store the list of filtered images in it as a column.
    utk_df = pd.DataFrame({"path": files_13plus_filtered})

    # For each image, store its absolute path "img_path" by adding a prefix of the 
    # directory's path to it.
    utk_df["img_path"] = utk_imgs_path + utk_df["path"]
    # For each image, store its gender by checking if its gender label is male (0) or
    # female (1).
    utk_df["female"] = utk_df["path"].map(lambda x: int(x.split("_")[1]))
    # For each image, store its age by extracting it from the image name.
    utk_df["age_simple"] = utk_df["path"].map(lambda x: int(x.split("_")[0]))

    # Create a copy of the DataFrame which only stores the relevant columns.
    # Inspired by https://stackoverflow.com/a/34683105
    utk_df_final = utk_df[["img_path", "age_simple", "female"]].copy()

    # Print the statistics of the UTK dataset.
    print("UTK data statistics:")
    # Group and count the images by gender.
    tmp_df = utk_df_final.groupby("female").agg("count")["img_path"]
    # Calculate the fraction of females in the dataset. This will help understand if 
    # there is class imbalance and if we need to address it.
    female_frac = tmp_df[1] / (tmp_df[1] + tmp_df[0])
    print(f"{utk_df_final.shape[0]} images \t {female_frac:.4%} females")

    # If write_data=True, write the DataFrame to a CSV file.
    if write_data:
        utk_df_final.to_csv(f"{savedir}/utk_data.csv", index=False)

    return utk_df_final


def prepare_adience_data(adience_path, adience_imgs_path, seedval, write_data=False):
    """
    Prepare the Adience dataset for classification.

    Args:
        adience_path (str): Path of the Adience dataset directory.
        adience_imgs_path (str): Path of the Adience dataset images directory.
        seedval (int): Random seed to ensure data split reproducibility.
        write_data (bool, optional): Specify if the dataset should be saved as a CSV 
                                     file. Defaults to False.

    Returns:
        adience_final (pd.DataFrame): The final DataFrame containing the image path, 
                                      age, and gender labels from the Adience dataset.
    """
    # Initialize empty dictionaries to store DataFrames.
    adience_dfs, adience_final = {}, {}

    # The Adience dataset has metadata in the form of 5 .txt files, which are read and 
    # stored in a dictionary of DataFrames.
    for idx in range(5):
        adience_dfs[idx] = pd.read_csv(
            adience_path + f"fold_{idx}_data.txt", delimiter="\t"
        )

    # Next, these 5 DataFrames are combined into a single DataFrame in order to 
    # consolidate the entire dataset.
    adience_dfs["combined"] = pd.DataFrame().append(
        [adience_dfs[idx] for idx in range(5)]
    )

    # In order to restrict the age to >= 13, list all ages/age intervals that need to 
    # be removed.
    restricted_age = ["(0, 2)", "(4, 6)", "(8, 12)", "(8, 23)", "2", "3"]
    # Similarly, filter the genders so as to only permit 'f' and 'm' labels.
    permitted_gender = ["f", "m"]

    # The Adience dataset contains some images with exact age and some with age 
    # intervals. For images with age intervals (identified by checking if the age field
    # is a tuple), calculate the average age.
    avg_age = (
        lambda x: (eval(x)[0] + eval(x)[1]) / 2
        if isinstance(eval(x), tuple)
        else eval(x)
    )

    # Filter the DataFrame to only keep rows which do not have age in restricted_age.
    # Inspired by https://stackoverflow.com/a/12098586
    adience_dfs_filtered = adience_dfs["combined"][
        ~adience_dfs["combined"]["age"].isin(restricted_age)
    ]

    # Calculate and store the avg_age for each image as age_simple.
    # Inspired by https://stackoverflow.com/a/34962518
    adience_dfs_filtered["age_simple"] = adience_dfs_filtered["age"].map(avg_age)
    # For each image, store the absolute path of the image.
    # The Adience dataset contains images named as
    # "coarse_tilt_aligned_face.<face_id>.<face_id>.<image_name>.jpg".
    adience_dfs_filtered["img_path"] = (
        adience_imgs_path
        + adience_dfs_filtered["user_id"]
        + "/coarse_tilt_aligned_face."
        + adience_dfs_filtered["face_id"].astype(str)
        + "."
        + adience_dfs_filtered["original_image"]
    )
    # For each image, store the gender as male (0) or female (1).
    adience_dfs_filtered["female"] = adience_dfs_filtered["gender"].map(
        lambda x: 1 if x == "f" else 0
    )

    # Filter the DataFrame to only include images with gender male (0) or female (1).
    adience_dfs_filtered = adience_dfs_filtered[
        adience_dfs_filtered["gender"].isin(permitted_gender)
    ]

    # Remove any rows from the DataFrame containing null entries.
    adience_dfs_filtered = adience_dfs_filtered.dropna()

    # Create a copy of the DataFrame which only stores the relevant columns.
    # Inspired by https://stackoverflow.com/a/34683105
    adience_dfs_filtered = adience_dfs_filtered[
        ["img_path", "age_simple", "female"]
    ].copy()

    # Since this DataFrame needs to be split into three partitions (training, testin 
    # and validation) and stratification is used to ensure that gender and age labels
    # are equally distributed across all the partitions.
    # Therefore, only those combinations of (age_simple + female) that have at least 3 
    # samples (since 3 partitions are needed) are retained and the rest are filtered 
    # out and later added to the training partition.
    adience_train_permitted = adience_dfs_filtered.groupby(
        ["age_simple", "female"]
    ).filter(lambda x: len(x) < 3)
    adience_dfs_filtered = adience_dfs_filtered.groupby(
        ["age_simple", "female"]
    ).filter(lambda x: len(x) > 3)

    # First, the dataset (DataFrame) is split into training and evaluation partitions.
    # 70% of the data is retained for training and 30% is held out for evaluation.
    # This is done by stratifying the data based on 2 columns: age_simple and female.
    # Inspired by https://stackoverflow.com/a/45526792
    adience_train, adience_eval = train_test_split(
        adience_dfs_filtered,
        test_size=0.3,
        random_state=seedval,
        stratify=adience_dfs_filtered[["age_simple", "female"]],
    )

    # Next, the evaluation partition is split into validation and testing sets.
    # Similar as before, only those combinations of (age_simple + female) that have at 
    # least 2 samples (since 2 partitions are needed) are retained and the rest are 
    # filtered out and later added to the training partition.
    adience_eval_permitted = adience_eval.groupby(["age_simple", "female"]).filter(
        lambda x: len(x) < 2
    )
    adience_eval = adience_eval.groupby(["age_simple", "female"]).filter(
        lambda x: len(x) > 2
    )

    # After that, the evaluation partition is split into validation and testing 
    # partitions. 75% of the evaluation data is retained for testing and the remaining
    # 25% is used for validation. Again, the split is based on stratification across 2
    # columns: age_simple and female.
    # Inspired by https://stackoverflow.com/a/45526792
    adience_final["valid"], adience_final["test"] = train_test_split(
        adience_eval,
        test_size=0.75,
        random_state=seedval,
        stratify=adience_eval[["age_simple", "female"]],
    )

    # Finally, the samples that were left out before the two split steps are merged 
    # into the training split as discussed before.
    adience_final["train"] = pd.DataFrame().append(
        [adience_train, adience_train_permitted, adience_eval_permitted]
    )

    # Print the statistics of the Adience dataset.
    print("\nAdience data split statistics:")
    # Iterate over all the 3 partitions: training, validation, and testing.
    for split in adience_final.keys():
        # Group and count the images by gender.
        tmp_df = adience_final[split].groupby("female").agg("count")["img_path"]
        # Calculate the fraction of females in the dataset. This will help understand if 
        # there is class imbalance and if we need to address it.
        female_frac = tmp_df[1] / (tmp_df[1] + tmp_df[0])
        print(
            f"{split} \t {adience_final[split].shape[0]} images \t {female_frac:.4%} females"
        )

    # If write_data=True, write all the DataFrames to corresponding CSV files.
    if write_data:
        for split in adience_final.keys():
            adience_final[split].to_csv(
                f"{savedir}/adience_data_{split}.csv", index=False
            )

    return adience_final


def prepare_data(
    utk_imgs_path, adience_path, adience_imgs_path, seedval, write_data=False
):
    """
    Prepare the UTK and the Adience datasets for classification.

    Args:
        utk_imgs_path (str): Path of the UTK dataset images directory.
        adience_path (str): Path of the Adience dataset directory.
        adience_imgs_path (str): Path of the Adience dataset images directory.
        seedval (int): Random seed to ensure data split reproducibility.
        write_data (bool, optional): Specify if the datasets should be saved as CSV 
                                     files. Defaults to False.

    Returns:
        train_data (pd.DataFrame): The final training data's DataFrame containing the 
                                   image path, age, and gender labels from both the UTK
                                   and the Adience datasets.
        valid_data (pd.DataFrame): The final validation data's DataFrame containing the 
                                   image path, age, and gender labels from both the UTK
                                   and the Adience datasets.
        test_data (pd.DataFrame): The final testing data's DataFrame containing the 
                                  image path, age, and gender labels from both the UTK
                                  and the Adience datasets.
        all_data (pd.DataFrame): The DataFrame containing the image path, age, and 
                                 gender labels of all the images from both the UTK and 
                                 the Adience datasets.
    """
    # Call the corresponding functions to prepare each of the two datasets.
    utk_data = prepare_utk_data(utk_imgs_path, write_data)
    adience_data = prepare_adience_data(
        adience_path, adience_imgs_path, seedval, write_data
    )

    # Merge all the data from both the datasets to create a single all_data DataFrame.
    all_data = pd.DataFrame().append(
        [utk_data, adience_data["train"], adience_data["valid"], adience_data["test"]]
    )
    # Create the training data by merging the UTK dataset with the training split of  
    # the Adience dataset.
    train_data = pd.DataFrame().append([adience_data["train"], utk_data])
    # The validation and the testing data are as created from the Adience dataset.
    valid_data, test_data = adience_data["valid"], adience_data["test"]

    # # If write_data=True, write all the DataFrames to corresponding CSV files.
    if write_data:
        train_data.to_csv(f"{savedir}/all_data_train.csv", index=False)
        all_data.to_csv(f"{savedir}/all_data.csv", index=False)

    return train_data, valid_data, test_data, all_data


class GenderRecognitionDataset(torch.utils.data.Dataset):
    """
    A custom torchvision dataset object for training gender recognition models.
    """

    def __init__(self, data_file_path, transform=None):
        """
        Initialize a GenderRecognitionDataset object.

        Args:
            data_file_path (str): Path of the CSV file containing the dataset.
            transform (torchvision.transforms, optional): Transforms to be applied to 
                                                          the sampled data. 
                                                          Defaults to None.
        """
        super(GenderRecognitionDataset, self).__init__()
        # Read and store the CSV file as a DataFrame.
        self.data = pd.read_csv(data_file_path)
        # Store the transforms to be applied to the sampled data.
        self.transform = transform

    def __getitem__(self, index):
        """
        Fetch one batch of the dataset.

        Args:
            index (int): Index(s) of the dataset to be fetched.

        Returns:
            image (torch.Tensor): A (batch of) image(s) from the dataset with 
                                  transforms applied.
            label (torch.long): A (batch of) label(s) of the corresponding images from
                                the dataset. 
        """
        # Get the image path and the image label based on the 
        img_path, label = self.data.iloc[index, 0], self.data.iloc[index, 2]
        # Read the image from the image path.
        image = Image.open(img_path)

        # Apply transforms to the image if specified.
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """
        Calculate the size of the dataset.

        Returns:
            self.data (int): The size (number of data points) in the dataset.
        """
        return len(self.data)


def calculate_data_statistics(dataloader):
    """
    Calculate the mean and standard deviation intensities of the images in a dataset.
    Inspired by https://bit.ly/3wJAAdv

    Args:
        dataloader (torch.utils.data.Dataset): The DataLoader object to sample data 
                                               samples from.

    Returns:
        mean (torch.float): The mean intensity of images in the dataset.
        std (torch.float): The standard deviation of the intensity of images in the 
                           dataset.
    """
    # Initialize the variables to keep track of.
    n_imgs, mean, var = 0, 0.0, 0.0

    # Iterate over the dataset using the DataLoader object.
    for _, data in enumerate(dataloader):
        # Fetch the image from a batch of sampled data.
        image = data[0]

        # Reshape the image so that the height and the width are flattened.
        image = image.view(image.size(0), image.size(1), -1)
        # Keep track of the number of images using the batch size.
        n_imgs += image.size(0)

        # Calculate the mean and the variance of the images along axis=2.
        mean += image.mean(2).sum(0)
        var += image.var(2).sum(0)

    # Normalize the mean and the variance by dividing by the number of images.
    mean /= n_imgs
    var /= n_imgs

    # Calculate the standard deviation by taking the square root of the mean.
    std = torch.sqrt(var)

    return mean, std


def train_model(
    model,
    dloaders,
    dset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device="cuda",
    logreg=False,
):
    """
    Train a classification model for gender recognition from images.
    Inspired by https://bit.ly/3kXA2OW

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dloaders (dict): A dictionary of DataLoader objects.
        dset_sizes (dict): A dictionary of the sizes of the splits of the dataset.
        criterion (torch.nn.Module): The loss function to be used to optimize the 
                                     model.
        optimizer (torch.optim): The PyTorch optimizer to be used for parameter update.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler to be used in
                                              the conjunction with the optimizer to 
                                              update the model's parameters.
        num_epochs (int): The number of epochs the model is to be optimized for.
        device (str, optional): The device to be used for model training. 
                                Defaults to "cuda".
        logreg (bool, optional): Specify if the model is a single-neuron logistic 
                                 regression model. 
                                 Defaults to False.

    Returns:
        model (torch.nn.Module): The trained classification model.
        losses (dict): A dictionary containing losses tracked during the training.
        accs (dict): A dictionary containing accuracies tracked during the training.
    """
    # Create a copy of model's initial weights. This will be updated as the model's 
    # performance on the validation split improves.
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize the best validation accuracy and empty dictionaries for losses and 
    # accuracies to keep track of.
    best_acc = 0.0
    losses, accs = {}, {}
    for phase in ["train", "valid", "test"]:
        losses[phase], accs[phase] = torch.zeros(num_epochs), torch.zeros(num_epochs)

    # Iterate over the number of training epochs.
    for epoch in range(num_epochs):
        # For each epoch, perform training, validation, and testing steps.
        for phase in ["train", "valid", "test"]:
            # Check if the model is being trained. If yes, set the model to "train", 
            # mode, else set the model in "eval" mode.
            # This is important for dynamic components such as Dropout, 
            # Batch Normalization, etc.
            if phase == "train":
                model.train()
            else:
                model.eval()

            # For each epoch, keep track of the loss and the number of correct 
            # predictions.
            epoch_loss, epoch_corrects = 0.0, 0

            # For each phase ("train", "valid", "test"), iterate over the 
            # corresponding partition using the respective dataloader.
            for inputs, labels in dloaders[phase]:
                # Move the images and the labels to the specified device.
                # Also convert the labels to long data type as PyTorch needs it to be 
                # so for the loss calculations.
                inputs = inputs.to(device)
                labels = labels.long()
                labels = labels.to(device)

                # Zero out the gradients accumulated in the optimizer.
                optimizer.zero_grad()

                # Perform the forward pass through the model.
                # Check if the model is being trained, and if so, enable gradient 
                # tracking.
                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass the input data through the model.
                    outputs = model(inputs)
                    # If the model is a logistic regression neuron, the labels need to
                    # be cast as float and the outputs need to be reshaped in order to 
                    # compute the loss.
                    if logreg:
                        # Cast the labels as float.
                        labels = labels.float()
                        # Calculate the loss value.
                        loss = criterion(outputs.reshape(-1), labels)
                    else:
                        # Calculate the loss value.
                        loss = criterion(outputs, labels)
                    
                    # Calculate the final predictions based on the argmax operation.
                    _, preds = torch.max(outputs, 1)

                    # If the model is being trained, perform a backward pass using the 
                    # calculated loss to update the model parameters.
                    if phase == "train":
                        # Backward propagation of the loss function and model 
                        # parameters' update.
                        loss.backward()
                        optimizer.step()

                # Keep track of each epoch's loss value and number of correctly 
                # predicted images.
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            # Check if a learning rate scheduler is being used and if the model is 
            # being trained. If yes to both, then update the learning rate using the
            # scheduler.
            if scheduler and phase == "train":
                scheduler.step()

            # Normalize and store each epoch's loss value and number of correctly 
            # predicted images (i.e., model's classification accuracy).
            epoch_loss = epoch_loss / dset_sizes[phase]
            epoch_acc = epoch_corrects.double() / dset_sizes[phase]
            losses[phase][epoch], accs[phase][epoch] = epoch_loss, epoch_acc

            # In the validation phase, keep track of the best accuracy and copy the 
            # weights of the model producing the best validation accuracy.
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # When model training is done, load the weights which produced the best validation 
    # accuracy into the model.
    model.load_state_dict(best_model_wts)

    return model, losses, accs


def test_model(model, dloaders, dset_sizes, device="cuda"):
    """
    Test a trained classification model for gender recognition from images.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dloaders (dict): A dictionary of DataLoader objects.
        dset_sizes (dict): A dictionary of the sizes of the splits of the dataset.
        device (str, optional): The device to be used for model evaluation. 
                                Defaults to "cuda".

    Returns:
        test_acc (torch.Tensor): The accuracy of the model on the testing dataset.
    """
    # Initialize the number of correct test predictions as 0.
    test_corrects = 0

    # Iterate over the images and the labels from the testing data.
    for inputs, labels in dloaders["test"]:
        # Move the images and the labels to the specified device.
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Perform the forward pass through the model.
        # Disable gradient tracking since the model is being used in inference mode.
        with torch.set_grad_enabled(False):
            # Forward pass the input data through the model.
            outputs = model(inputs)
            # Calculate the final predictions based on the argmax operation.
            _, preds = torch.max(outputs, 1)

        # Keep track of the number of correct predictions in each batch.
        test_corrects += torch.sum(preds == labels.data)

    # Calculate the test accuracy by dividing the number of correct predictions by the 
    # size of the test dataset.
    test_acc = test_corrects.double() / dset_sizes["test"]

    return test_acc


def calculate_confmat(model, dloader, device="cuda"):
    """
    Calculate the confusion matrix of the classifier on the test dataset.
    Inspired by https://stackoverflow.com/a/53291323

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dloader (dict): A dictionary of DataLoader objects.
        device (str, optional): The device to be used for model evaluation. 
                                Defaults to "cuda".

    Returns:
        confusion_matrix (torch.Tensor): The confusion matrix of the classifier on the 
                                         test dataset.
    """
    # Initialize an empty confusion matrix.
    confusion_matrix = torch.zeros(2, 2)

    # Disable gradient tracking since this is an inference step.
    with torch.no_grad():
        # Iterate over the images and the labels from the testing data.
        for _, (inputs, labels) in enumerate(dloader["test"]):
            # Move the images and the labels to the specified device.
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Perform the forward pass through the model.
            outputs = model(inputs)

            # Calculate the final predictions based on the argmax operation.
            _, preds = torch.max(outputs, 1)

            # For every pair of (ground truth, prediction), increment the corresponding
            # entry in the confusion matrix.
            for label, pred in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[label.long(), pred.long()] += 1

    return confusion_matrix
