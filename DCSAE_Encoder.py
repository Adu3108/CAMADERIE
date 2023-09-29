from typing import Tuple
import torch
import torchvision
import math

class DCSAE_Encoder(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int]) -> None:
        super(DCSAE_Encoder, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d

        # Calculating the size of intermediate output of convolutional layers
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.hidden_units = self.y_5 * self.x_5 * 16 # Number of neurons after convolution layers (height x width x n_chan)

        # Convolutional Encoder network
        self.enc_conv1 = torch.nn.Conv2d(
            in_channels=self.n_chan,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv1_bn = torch.nn.BatchNorm2d(128)
        self.enc_conv1_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv1_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv2_bn = torch.nn.BatchNorm2d(64)
        self.enc_conv2_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv2_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv3_bn = torch.nn.BatchNorm2d(32)
        self.enc_conv3_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv3_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True, ceil_mode=True)

        self.enc_conv4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv4_bn = torch.nn.BatchNorm2d(16)
        self.enc_conv4_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv4_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)
        
        # Dense Encoder Bottleneck
        self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        self.enc_dense1_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense2 = torch.nn.Linear(2048, 1000)
        self.enc_dense2_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3_af = torch.nn.LeakyReLU(0.1)

        # Latent Variables Calculation
        self.enc_dense4_mu_positive = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4_mu_positive_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4_mu_negative = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4_mu_negative_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4_var_positive = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4_var_positive_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4_var_negative = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4_var_negative_af = torch.nn.LeakyReLU(0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional Encoder Network
        z = x
        z = self.enc_conv1(z)
        z = self.enc_conv1_bn(z)
        z = self.enc_conv1_af(z)
        z, self.indices1 = self.enc_conv1_pool(z)

        z = self.enc_conv2(z)
        z = self.enc_conv2_bn(z)
        z = self.enc_conv2_af(z)
        z, self.indices2 = self.enc_conv2_pool(z)

        z = self.enc_conv3(z)
        z = self.enc_conv3_bn(z)
        z = self.enc_conv3_af(z)
        z, self.indices3 = self.enc_conv3_pool(z)

        z = self.enc_conv4(z)
        z = self.enc_conv4_bn(z)
        z = self.enc_conv4_af(z)
        z, self.indices4 = self.enc_conv4_pool(z)

        # Dense Encoder Bottleneck
        z = z.view(z.size(0), -1) # Converting a 4D tensor (output of convolutional layer) to 1D tensor
        z = self.enc_dense1(z)
        z = self.enc_dense1_af(z)

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        return z
    
    def get_layer_size(self, layer: int) -> Tuple[int]:
        y_l, x_l = self.input_d
        for i in range(layer - 1):
            y_l = math.ceil((y_l - 2) / 2 + 1)
            x_l = math.ceil((x_l - 2) / 2 + 1)
        return y_l, x_l

    def positive_latent_calc(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        mu = self.enc_dense4_mu_positive(z)
        mu = self.enc_dense4_mu_positive_af(mu)

        var = self.enc_dense4_var_positive(z)
        var = self.enc_dense4_var_positive_af(var)

        return mu, var
    
    def negative_latent_calc(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        mu = self.enc_dense4_mu_negative(z)
        mu = self.enc_dense4_mu_negative_af(mu)

        var = self.enc_dense4_var_negative(z)
        var = self.enc_dense4_var_negative_af(var)

        return mu, var
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        positive_mean, positive_logvar = self.positive_latent_calc(self.encode(x))
        negative_mean, negative_logvar = self.negative_latent_calc(self.encode(x))
        return positive_mean, positive_logvar, negative_mean, negative_logvar

    def testing(self,
                data_path: str,
                weight_file: str):
        # Using cuda (GPU) if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
        network.load_state_dict(torch.load(weight_file))# Load weights from the .pt file
        network.eval() # Set the network in evalution mode

        # Defining Image Transformations
        # 1. Convert the input image to a PyTorch Tensor
        # 2. Resizing the image to (batch x channels x height x width)
        # (Optional) 3. Incase of only one input channel, we convert the input image to a GrayScale Image
        if self.n_chan == 1:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d),
                torchvision.transforms.Grayscale()])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d)])

        # Applying image transformations to the input test dataset
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transforms)

        # Set up a Python iterable over the input test dataset
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True)

        final_input = []
        final_positive_mean = []
        final_negative_mean = []
        final_positive_var = []
        final_negative_var = []
        final_class = []
        for data in test_loader:
            input, class_name = data
            final_input.append(input)
            final_class.append(int(class_name))
            input = input.to(device)
            with torch.no_grad():
                positive_mean, positive_logvariance, negative_mean, negative_logvariance = network.forward(input)
                final_positive_mean.append(positive_mean)
                final_positive_var.append(torch.exp(positive_logvariance/2))
                final_negative_mean.append(negative_mean)
                final_negative_var.append(torch.exp(negative_logvariance/2))
        
        result = dict()
        result['final_input'] = final_input
        result['final_class'] = final_class
        result['final_positive_mean'] = final_positive_mean
        result['final_positive_var'] = final_positive_var
        result['final_negative_mean'] = final_negative_mean
        result['final_negative_var'] = final_negative_var

        return result