# CAMARADERIE

This repository contains the implementation of **Discrete Classifier Supervised Autoencoders** (DC-SAE) and the **CAMARADERIE** algorithm to perform Content-based Knowledge Transfer in a privacy-preserving setting.

## Problem Statement

There are two clients **A** and **B**. Client **A** (Donor) has a small labelled dataset whereas Client **B** (Recipient) has a large unlabelled dataset. Both these datasets consist examples belonging to only 2 classes. The objective of this algorithm is to label the dataset possessed by Client **B** using the features that can be learned from the dataset possessed by Client **A** albeit without sharing it with Client **B**.

## Functionalities

In order to execute the CAMARADERIE algorithm, DC-SAE offers four functionalities :-

1. `train` : Train the DC-SAE on the given dataset. In order to prevent overfitting, we have implemented early stopping. 

2. `visualize` : Visualize the latent space of the DC-SAE.

3. `reconstruct` : This functionality is relevant for image data. This involves displaying the reconstructed output of DC-SAE.

4. `classify` : Implements the CAMARADERIE algorithm and classifies the test dataset on the basis of features learnt from the training dataset.

## Dataset Directory Structure

For image data, DC-SAE requires the following directory structure :-

```
Dataset
├── Client-A
│   ├── Training
│   │   ├── Negative
│   │   └── Positive
│   └── Validation
│       ├── Negative
│       └── Positive
└── Client-B
    └── Test
        ├── Negative
        └── Positive
```

To aid the user to construct this directory structure, we have provided another functionality `create`. It takes the following arguments :-

1. `train_size` : Size of the training set (Positive + Negative examples)
2. `test_size` : Size of the test dataset (Positive + Negative examples)
3. `positive_set` : Path to the folder containing the positive examples
4. `negative_set` : Path to the folder containing the negative examples

An example command for dataset creation :-

```bash
python3 main.py --task create --train_size 1000 --test_size 100 --positive_set ./foo --negative_set ./bar
```

## Parameters 

- In addition to the datasets, the users will also need to provide values for several hyperparameters that affect the training of DC-SAE.

  1. `n_latent` : The dimensionality of the latent space of DC-SAE
  2. `alpha` : The weightage of reconstruction loss in the total loss function used for training the DC-SAE
  3. `beta` : The weightage of KL Divergence loss in the total loss function used for training the DC-SAE
  4. `gamma` : The weightage of repulsion loss in the total loss function used for training the DC-SAE
  5. `rho` : The desired separation between the clusters in the latent space of DC-SAE

- For image data, we need to specify some more parameters :- 

  1. `n_chan` : The number of channels present in the images. 1 for grayscale and 3 for color images.
  2. `input_d` : The dimensions of the input image. It should be specified as `hxw` where **h** is the height of the image and **w** is the width of the image. Eg. 28x28

- For visualize, reconstruct and classify tasks, we need to provide the following paths :-

  1. `weights` : Path to the weights file obtained after training the DC-SAE on the training dataset
  2. `hyperparameters` : Path to the file containing hyperparameters used while training DC-SAE

  We have hardcoded the paths to the following files outputed by the program :-

  1. `weights.pt` : The weights of the model are stored in this file
  2. `hyperparameters.pt` : The hyperparameters used for training the DC-SAE are stored in this file
  3. `enc_only_weights.pt` : The weights associated with the encoder portion of the trained DC-SAE are stored in this file.

- For tabular data, we need to specify the following paths :-

  1. `train_path` : Path to the csv file containing the training dataset
  2. `val_path` : Path to the csv file containing the validation dataset
  3. `test_path` : Path to the csv file containing the test dataset

- Currently, DC-SAE requries the tabular data to be provided in `csv` format. Moreover, we perform normalization on this dataset before feeding it to DC-SAE.

## Setting up the Environment

Before executing the algorithm, we need to install the necessary Python packages

```bash
pip install -r requirements.txt
```

## Running the Code

- DC-SAE can handle image data as well as tabular data. This is specified using the following command-line argument
  - Use `image` if the input dataset is an image dataset.
  - Use `num` if the input dataset is a tabular dataset.
```bash
python3 main.py --type <data format>
```

  

- Training the DC-SAE is done using the following command :- 

  ```bash
  python3 main.py --task train --type image --n_latent <> --alpha <> --beta <> --gamma <> --rho <> --n_chan <> --input_d <>
  ```

