from typing import Tuple
import math
import torch
import torchvision

class ConvEncoder(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int],
                 batch: int = 1) -> None:
        super(ConvEncoder, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.batch = batch

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
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

        return z, self.indices1, self.indices2, self.indices3, self.indices4
    
    def get_layer_size(self, layer: int) -> Tuple[int]:
        y_l, x_l = self.input_d
        for i in range(layer - 1):
            y_l = math.ceil((y_l - 2) / 2 + 1)
            x_l = math.ceil((x_l - 2) / 2 + 1)
        return y_l, x_l
    
    def preprocess(self, data_path):
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

        # Applying image transformations to the input training dataset
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transforms)
        
        return dataset
    
    def output_size(self):
        return self.hidden_units
    
class ConvDecoder(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int],
                 batch : int = 1) -> None:
        super(ConvDecoder, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.batch = batch

        # Calculating the size of intermediate output of convolutional layers
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.hidden_units = self.y_5 * self.x_5 * 16 # Number of neurons after convolution layers (height x width x n_chan)

        # Convolutional Encoder network
        self.dec_conv4_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv4 = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv4_bn = torch.nn.BatchNorm2d(32)
        self.dec_conv4_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv3_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv3 = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv3_bn = torch.nn.BatchNorm2d(64)
        self.dec_conv3_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv2_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv2 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv2_bn = torch.nn.BatchNorm2d(128)
        self.dec_conv2_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv1_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv1 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=self.n_chan,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv1_bn = torch.nn.BatchNorm2d(self.n_chan)
        self.dec_conv1_af = torch.nn.Sigmoid()

    def forward(self, y: torch.Tensor, indices1, indices2, indices3, indices4) -> torch.Tensor:
        y = torch.reshape(y, [self.batch, 16, self.y_5, self.x_5]) # Converting the 1D tensor (output of dense layers) to a 4D tensor
        y = self.dec_conv4_pool(
            y,
            indices4,
            output_size=torch.Size([self.batch, 16, self.y_4, self.x_4]))
        y = self.dec_conv4(y)
        y = self.dec_conv4_bn(y)
        y = self.dec_conv4_af(y)

        y = self.dec_conv3_pool(
            y,
            indices3,
            output_size=torch.Size([self.batch, 32, self.y_3, self.x_3]))
        y = self.dec_conv3(y)
        y = self.dec_conv3_bn(y)
        y = self.dec_conv3_af(y)

        y = self.dec_conv2_pool(
            y,
            indices2,
            output_size=torch.Size([self.batch, 64, self.y_2, self.x_2]))
        y = self.dec_conv2(y)
        y = self.dec_conv2_bn(y)
        y = self.dec_conv2_af(y)

        y = self.dec_conv1_pool(
            y,
            indices1,
            output_size=torch.Size([self.batch, 128, self.input_d[0], self.input_d[1]]))
        y = self.dec_conv1(y)
        y = self.dec_conv1_bn(y)
        y = self.dec_conv1_af(y)

        return y
    
    def get_layer_size(self, layer: int) -> Tuple[int]:
        y_l, x_l = self.input_d
        for i in range(layer - 1):
            y_l = math.ceil((y_l - 2) / 2 + 1)
            x_l = math.ceil((x_l - 2) / 2 + 1)
        return y_l, x_l