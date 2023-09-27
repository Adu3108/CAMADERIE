import argparse

from ConvNetworks import ConvEncoder
from ConvNetworks import ConvDecoder
from camaraderie import CAMARADERIE
from preprocess import DataLoader

parser = argparse.ArgumentParser()

# Case 1 : Convolutional Networks
parser.add_argument('--n_chan', type=int)
parser.add_argument('--input_d', type=str)

parser.add_argument('--n_latent', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--beta', type=float, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--rho', type=float, required=True)
parser.add_argument('--train_size', type=int, required=True)
parser.add_argument('--test_size', type=int, required=True)
parser.add_argument('--positive_set', type=str, required=True)
parser.add_argument('--negative_set', type=str, required=True)

args = parser.parse_args()

# print(args.n_chan)
# print(args.input_d)
# print(args.n_latent)
# print(args.alpha)
# print(args.beta)
# print(args.gamma)
# print(args.rho)
# print(args.train_size)
# print(args.test_size)
# print(args.positive_set)
# print(args.negative_set)

train_dataset = "./Dataset/Client-A/Training"
validation_dataset = "./Dataset/Client-A/Validation"
test_dataset = "./Dataset/Client-B/Test"
input_dimensions = tuple([int(i) for i in args.input_d.split('x')])

dataloader = DataLoader(args.train_size, args.test_size, args.positive_set, args.negative_set)
encoder = ConvEncoder(args.n_latent, args.n_chan, input_dimensions)
decoder = ConvDecoder(args.n_latent, args.n_chan, input_dimensions)
model = CAMARADERIE(encoder, decoder, args.n_latent, args.alpha, args.beta, args.gamma, args.rho, train_dataset, validation_dataset, test_dataset)

# dataloader.create()
# dataloader.load()
model.train()
model.visualise()
model.convert()
model.extract()
model.classify()