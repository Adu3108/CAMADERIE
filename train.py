import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

from DC_SAE import DCSAE
from vae import StandardVAE

class DCSAE_Trainer:
    def __init__(self, n_latent, alpha, beta, gamma, rho, n_chan, input_d, train_path, val_path, weights_path, hyperparameters_path):
        super(DCSAE_Trainer, self).__init__()
        self.n_latent = n_latent
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.input_d = input_d
        self.n_chan = n_chan
        self.num_epochs = 100
        self.lr = 1e-4

        self.dataset = train_path
        self.validation_dataset = val_path
        self.weights_path = weights_path
        self.hyperparameters_path = hyperparameters_path

        # Creating an instance of VAE Network for training at Client A
        self.ClientA_Network = DCSAE(
            self.n_latent,
            self.alpha,
            self.beta,
            self.gamma,
            self.rho,
            self.n_chan,
            self.input_d
        )
        
    def train(self):
        print(f'alpha={self.alpha}')
        print(f'beta={self.beta}')
        print(f'gamma={self.gamma}')
        print(f'rho={self.rho}')
        print(f'n_latent={self.n_latent}')
        print(f'Using data set {self.dataset}')

        # Training the created VAE instance on the labelled dataset at Client A
        self.ClientA_Network.train_self(
            data_path=self.dataset,
            val_path=self.validation_dataset,
            epochs=self.num_epochs,
            learning_rate = self.lr,
            weights_file=self.weights_path,
            hyperparameters_file = self.hyperparameters_path)

    def latent(self):
        self.result = self.ClientA_Network.testing(data_path=self.dataset, weight_file=self.weights_path)

        self.ClientA_class = self.result["final_class"]

        # Extracting latent space representation of each image in the training dataset
        self.ClientA_Z = []
        for i in range(len(self.result["final_negative_mean"])):
            eps = torch.randn_like(self.result["final_negative_var"][i])
            mean_cpu = self.result["final_negative_mean"][i].cpu().detach().numpy()
            variance_cpu = self.result["final_negative_var"][i].cpu().detach().numpy()
            eps = eps.cpu().detach().numpy()
            z = mean_cpu + variance_cpu * eps
            self.ClientA_Z.append(z)

        return self.ClientA_Z, self.ClientA_class

    def visualize(self):
        colors = ['red','green']
        for i in range(len(self.ClientA_Z)):
            z = self.ClientA_Z[i]
            plt.scatter(z[0][0], z[0][1], c=colors[self.ClientA_class[i]])
        # plt.savefig("/content/test.svg", format="svg")
    
    def reconstruct(self):
        # Generate a white image
        white_array = torch.ones((3, 224, 2))
        white_array.to(torch.int32)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        white_array = white_array.to(device)

        for i in range(len(self.result["final_input"]))[:10]:
            # Squeeze the image tensor to 3 dimensions (channels x height x width)
            # In case of GrayScale Images, repeat the same tensor in the 3 channels
            input_image_tensor = torch.squeeze(self.result["final_input"][i])
            if input_image_tensor.shape[0] == 1:
                input_image_tensor = input_image_tensor.repeat(3, 1, 1)
            reconstruction_image_tensor = torch.squeeze(self.result["final_output"][i])
            if reconstruction_image_tensor.shape[0] == 1:
                reconstruction_image_tensor = reconstruction_image_tensor.repeat(3, 1, 1)

            input_image_tensor = input_image_tensor.to(device)
            reconstruction_image_tensor = reconstruction_image_tensor.to(device)

            # Concatenate the input image and the reconstructed image
            concatenate_image = torch.cat([input_image_tensor, white_array, reconstruction_image_tensor], 2)
            image = torchvision.transforms.ToPILImage()(concatenate_image)
            image = image.resize((384,128), Image.LANCZOS)
            image.show()
            print()

class StandardVAE_Trainer:
    def __init__(self, n_latent, beta, n_chan, input_d, train_path, val_path, weights_path, hyperparameters_path):
        super(StandardVAE_Trainer, self).__init__()
        self.n_latent = n_latent
        self.beta = beta
        self.n_chan = n_chan
        self.input_d = input_d
        self.num_epochs = 100
        self.lr = 1e-4

        self.dataset = train_path
        self.validation_dataset = val_path
        self.weights_path = weights_path
        self.hyperparameters_path = hyperparameters_path
        
    def train(self):
        self.network = StandardVAE(
            self.n_latent,
            self.beta,
            self.n_chan,
            self.input_d)

        # Training the created VAE instance on the labelled dataset at Client A
        self.network.train_self(
            data_path=self.dataset,
            val_path=self.validation_dataset,
            epochs=self.num_epochs,
            learning_rate = self.lr,
            weights_file=self.weights_path,
            hyperparameters_file = self.hyperparameters_path)
        
    def latent(self):
        self.result = self.network.testing(data_path=self.dataset, weight_file=self.weights_path)
        self.ClientA_class = self.result["final_class"]

        # Extracting latent space representation of each image in the training dataset
        self.ClientA_Z = []
        for i in range(len(self.result["final_mean"])):
            eps = torch.randn_like(self.result["final_var"][i])
            mean_cpu = self.result["final_mean"][i].cpu().detach().numpy()
            variance_cpu = self.result["final_var"][i].cpu().detach().numpy()
            eps = eps.cpu().detach().numpy()
            z = mean_cpu + variance_cpu * eps
            self.ClientA_Z.append(z)

        return self.ClientA_Z, self.ClientA_class
    
    def visualize(self):
        colors = ['red','green']
        for i in range(len(self.ClientA_Z)):
            z = self.ClientA_Z[i]
            plt.scatter(z[0][0], z[0][1], c=colors[self.ClientA_class[i]])

    def reconstruct(self):
        # Generate a white image
        white_array = torch.ones((3, 224, 2))
        white_array.to(torch.int32)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        white_array = white_array.to(device)

        for i in range(len(self.result["final_input"]))[:10]:
            # Squeeze the image tensor to 3 dimensions (channels x height x width)
            # In case of GrayScale Images, repeat the same tensor in the 3 channels
            standard_vae_input_image_tensor = torch.squeeze(self.result["final_input"][i])
            if standard_vae_input_image_tensor.shape[0] == 1:
                standard_vae_input_image_tensor = standard_vae_input_image_tensor.repeat(3, 1, 1)
            standard_vae_reconstruction_image_tensor = torch.squeeze(self.result["final_output"][i])
            if standard_vae_reconstruction_image_tensor.shape[0] == 1:
                standard_vae_reconstruction_image_tensor = standard_vae_reconstruction_image_tensor.repeat(3, 1, 1)

            standard_vae_input_image_tensor = standard_vae_input_image_tensor.to(device)
            standard_vae_reconstruction_image_tensor = standard_vae_reconstruction_image_tensor.to(device)

            # Concatenate the input image and the reconstructed image
            concatenate_image = torch.cat([standard_vae_input_image_tensor, white_array, standard_vae_reconstruction_image_tensor], 2)
            image = torchvision.transforms.ToPILImage()(concatenate_image)
            image = image.resize((384,128), Image.LANCZOS)
            image.show()
            print()