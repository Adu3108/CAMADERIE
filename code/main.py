import argparse
import torch
import pandas as pd

from camaraderie import CAMARADERIE
from camaraderie import NumCAMARADERIE
from preprocess import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--type', type=str, required=True)
parser.add_argument('--task', type=str, required=True)

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

if not (args.train_path == None or args.val_path == None or args.test_path == None):
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)
    train_labels = torch.Tensor(train_df["y"])
    val_labels = torch.Tensor(val_df["y"])
    test_labels = torch.Tensor(test_df["y"])
    train_data = torch.Tensor(train_df.drop(["y"], axis=1))
    val_data = torch.Tensor(val_df.drop(["y"], axis=1))
    test_data = torch.Tensor(test_df.drop(["y"], axis=1))

if (args.task=="create"):
    if not (args.train_size == None or args.test_size == None or args.positive_set == None or args.negative_set == None):
        dataloader = DataLoader(args.train_size, args.test_size, args.positive_set, args.negative_set)
        dataloader.create()
        dataloader.load()
    else:
        print("Please enter train_size, test_size and paths to the folder containing the positive and negative training examples")
        
elif (args.task=="train"):
    model = None
    if (args.type=="image"):
        model = CAMARADERIE(args.n_chan, input_dimensions, args.n_latent, args.alpha, args.beta, args.gamma, args.rho, train_dataset, validation_dataset, test_dataset, encoder_weights_path, weights_path, hyperparameters_path)
    elif (args.type=="num"):
        model = NumCAMARADERIE(args.n_latent, args.alpha, args.beta, args.gamma, args.rho, train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, weights_path, hyperparameters_path)
    model.train()

elif (args.task=="visualise"):
    hyperparameters = torch.load(args.hyperparameters)
    model = None
    if (args.type=="image"):
        model = CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
    elif (args.type=="num"):
        model = NumCAMARADERIE(hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, args.weights_path, args.hyperparameters)
    model.visualise()

elif (args.task=="reconstruct"):
    hyperparameters = torch.load(args.hyperparameters)
    model = CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
    model.reconstruct()

elif (args.task=="classify"):
    hyperparameters = torch.load(args.hyperparameters)
    model = None
    if (args.type=="image"):
        model = CAMARADERIE(hyperparameters["n_chan"], hyperparameters["input_d"], hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_dataset, validation_dataset, test_dataset, encoder_weights_path, args.weights, args.hyperparameters)
    elif (args.type=="num"):
        model = NumCAMARADERIE(hyperparameters["n_latent"], hyperparameters["alpha"], hyperparameters["beta"], hyperparameters["gamma"], hyperparameters["rho"], train_data, val_data, test_data, train_labels, val_labels, test_labels, encoder_weights_path, args.weights_path, args.hyperparameters)
    model.convert()
    ClientB_features, ClientB_class = model.extract()
    model.classify(ClientB_features, ClientB_class)