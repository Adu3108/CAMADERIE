import argparse
import torch
import pandas as pd
from sklearn import preprocessing

from DCSAE.DCSAE_camaraderie import CAMARADERIE
from DCSAE.DCSAE_camaraderie import NumCAMARADERIE
from VAE.VAE_camaraderie import VAE_CAMARADERIE
from VAE.VAE_camaraderie import VAE_NumCAMARADERIE
from preprocess import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--type', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--model', type=str, required=True)

parser.add_argument('--n_latent', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--rho', type=float)

parser.add_argument('--weights', type=str)
parser.add_argument('--hyperparameters', type=str)

# Case 1 : Convolutional Networks
# Hyperparameters
parser.add_argument('--n_chan', type=int)
parser.add_argument('--input_d', type=str)

# Dataset Creation
parser.add_argument('--train_size', type=int)
parser.add_argument('--test_size', type=int)
parser.add_argument('--positive_set', type=str)
parser.add_argument('--negative_set', type=str)

# Case 2 : Numerical Data
parser.add_argument('--train_path', type=str)
parser.add_argument('--val_path', type=str)
parser.add_argument('--test_path', type=str)

args = parser.parse_args()

train_dataset = "./Dataset/Client-A/Training"
validation_dataset = "./Dataset/Client-A/Validation"
test_dataset = "./Dataset/Client-B/Test"

encoder_weights_path = "./enc_only_weights.pt"
weights_path = "./weights.pt"
hyperparameters_path = "./hyperparameters.pt"

if args.input_d != None:
    input_dimensions = tuple([int(i) for i in args.input_d.split('x')])

eps = 1e-5

if not (args.train_path == None):
    train_df = pd.read_csv(args.train_path)
    train_labels = torch.Tensor(train_df["y"])
    train_df.drop(["y"], axis=1, inplace=True)
    train_scaler = preprocessing.StandardScaler().fit(train_df)
    train_data = train_scaler.transform(train_df)
    train_data = torch.Tensor(train_data) + eps

if not (args.val_path == None):
    val_df = pd.read_csv(args.val_path)
    val_labels = torch.Tensor(val_df["y"])
    val_df.drop(["y"], axis=1, inplace=True)
    val_scaler = preprocessing.StandardScaler().fit(val_df)
    val_data = val_scaler.transform(val_df)
    val_data = torch.Tensor(val_data) + eps

if not (args.test_path == None):
    test_df = pd.read_csv(args.test_path)
    test_labels = torch.Tensor(test_df["y"])
    test_df.drop(["y"], axis=1, inplace=True)
    test_scaler = preprocessing.StandardScaler().fit(test_df)
    test_data = test_scaler.transform(test_df)
    test_data = torch.Tensor(test_data) + eps

if (args.task=="create"):
    if not (args.train_size == None or args.test_size == None or args.positive_set == None or args.negative_set == None):
        dataloader = DataLoader(args.train_size, args.test_size, args.positive_set, args.negative_set)
        dataloader.create()
        dataloader.load()
    else:
        print("Please enter train_size, test_size and paths to the folder containing the positive and negative training examples")
        
elif (args.task=="train"):
    model = None
    if (args.model == "dcsae"):
        if (args.type=="image"):
            model = CAMARADERIE(args.n_chan, input_dimensions, args.n_latent, args.alpha, args.beta, args.gamma, args.rho, train_dataset, validation_dataset, test_dataset, encoder_weights_path, weights_path, hyperparameters_path)
        elif (args.type=="num"):
            model = NumCAMARADERIE(args.n_latent, args.alpha, args.beta, args.gamma, args.rho, train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, weights_path, hyperparameters_path)
    elif (args.model == "vae"):
        if (args.type=="image"):
            model = VAE_CAMARADERIE(args.n_chan, input_dimensions, args.n_latent, args.beta, train_dataset, validation_dataset, test_dataset, encoder_weights_path, weights_path, hyperparameters_path)
        elif (args.type=="num"):
            model = VAE_NumCAMARADERIE(args.n_latent, args.beta, train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, weights_path, hyperparameters_path)
    model.train()

elif (args.task=="visualise"):
    hyperparameters = torch.load(args.hyperparameters)
    model = None
    if (args.model == "dcsae"):
        if (args.type=="image"):
            model = CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
        elif (args.type=="num"):
            model = NumCAMARADERIE(hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, args.weights, args.hyperparameters)
    elif (args.model == "vae"):
        if (args.type=="image"):
            model = VAE_CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["beta"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
        elif (args.type=="num"):
            model = VAE_NumCAMARADERIE(hyperparameters["n_latent"], hyperparameters["beta"], train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, args.weights, args.hyperparameters)
 
    model.visualise()

elif (args.task=="reconstruct"):
    hyperparameters = torch.load(args.hyperparameters)
    if (args.model == "dcsae"):
        model = CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
    elif (args.model == "vae"):
        model = VAE_CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["beta"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
    model.reconstruct()

elif (args.task=="classify"):
    hyperparameters = torch.load(args.hyperparameters)
    model = None
    if (args.model == "dcsae"):
        if (args.type=="image"):
            model = CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
        elif (args.type=="num"):
            model = NumCAMARADERIE(hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, args.weights, args.hyperparameters)
    elif (args.model == "vae"):
        if (args.type=="image"):
            model = VAE_CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["beta"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
        elif (args.type=="num"):
            model = VAE_NumCAMARADERIE(hyperparameters["n_latent"], hyperparameters["beta"], train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, args.weights, args.hyperparameters)
    model.convert()
    ClientB_negative_features, ClientB_class = model.extract()
    model.classify(ClientB_negative_features, ClientB_class)
