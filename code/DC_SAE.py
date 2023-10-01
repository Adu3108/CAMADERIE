from typing import Tuple
import torch
import torchvision
import math

class DCSAE(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 rho: float,
                 n_chan: int,
                 input_d: Tuple[int],
                 batch: int = 1, 
                 ) -> None:
        super(DCSAE, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.batch = batch
        self.n_chan = n_chan
        self.input_d = input_d

        # Early Stopping
        self.patience = 10
        self.delta = 1e-4
        self.best_score = None
        self.num_bad_epochs = 0

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
            return_indices=True,
            ceil_mode=True)

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

        # Dense Decoder Bottleneck
        self.dec_dense4 = torch.nn.Linear(self.n_latent, 250)
        self.dec_dense4_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense3 = torch.nn.Linear(250, 1000)
        self.dec_dense3_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense2 = torch.nn.Linear(1000, 2048)
        self.dec_dense2_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense1 = torch.nn.Linear(2048, self.hidden_units)
        self.dec_dense1_af = torch.nn.LeakyReLU(0.1)

        # Convolutional Decoder Network
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

    def get_layer_size(self, layer: int) -> Tuple[int]:
        y_l, x_l = self.input_d
        for i in range(layer - 1):
            y_l = math.ceil((y_l - 2) / 2 + 1)
            x_l = math.ceil((x_l - 2) / 2 + 1)
        return y_l, x_l

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
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Dense Decoder Bottleneck
        y = self.dec_dense4(z)
        y = self.dec_dense4_af(y)

        y = self.dec_dense3(y)
        y = self.dec_dense3_af(y)

        y = self.dec_dense2(y)
        y = self.dec_dense2_af(y)

        y = self.dec_dense1(y)
        y = self.dec_dense1_af(y)

        # Convolutional Decoder Network
        y = torch.reshape(y, [self.batch, 16, self.y_5, self.x_5]) # Converting the 1D tensor (output of dense layers) to a 4D tensor
        y = self.dec_conv4_pool(
            y,
            self.indices4,
            output_size=torch.Size([self.batch, 16, self.y_4, self.x_4]))
        y = self.dec_conv4(y)
        y = self.dec_conv4_bn(y)
        y = self.dec_conv4_af(y)

        y = self.dec_conv3_pool(
            y,
            self.indices3,
            output_size=torch.Size([self.batch, 32, self.y_3, self.x_3]))
        y = self.dec_conv3(y)
        y = self.dec_conv3_bn(y)
        y = self.dec_conv3_af(y)

        y = self.dec_conv2_pool(
            y,
            self.indices2,
            output_size=torch.Size([self.batch, 64, self.y_2, self.x_2]))
        y = self.dec_conv2(y)
        y = self.dec_conv2_bn(y)
        y = self.dec_conv2_af(y)

        y = self.dec_conv1_pool(
            y,
            self.indices1,
            output_size=torch.Size([self.batch, 128, self.input_d[0], self.input_d[1]]))
        y = self.dec_conv1(y)
        y = self.dec_conv1_bn(y)
        y = self.dec_conv1_af(y)

        return y
    
    def forward(self, x: torch.Tensor, class_name: int) -> Tuple[torch.Tensor]:
        z = self.encode(x)

        if class_name==0:
            mu, logvar = self.negative_latent_calc(z)
        else:
            mu, logvar = self.positive_latent_calc(z)

        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps

        out = self.decode(z)

        return out, mu, logvar
    
    def testing(self,
                data_path: str,
                weight_file: str):
        # Using cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
        network.load_state_dict(torch.load(weight_file)) # Load weights from the .pt file
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
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)

        final_input = []
        final_class = []
        final_output = []
        final_positive_mean = []
        final_negative_mean = []
        final_positive_var = []
        final_negative_var = []
        for data in test_loader:
            input, class_name = data
            final_input.append(input)
            final_class.append(int(class_name))
            input = input.to(device)
            with torch.no_grad():
                output, current_mean, current_logvariance = network.forward(input, int(class_name))
                other_output, other_mean, other_logvariance = network.forward(input, abs(int(class_name)-1))
                final_positive_mean.append(current_mean)
                final_positive_var.append(torch.exp(current_logvariance/2))
                final_negative_mean.append(other_mean)
                final_negative_var.append(torch.exp(other_logvariance/2))
                final_output.append(output)

        result = dict()
        result['final_input'] = final_input
        result['final_class'] = final_class
        result['final_output'] = final_output
        result['final_positive_mean'] = final_positive_mean
        result['final_positive_var'] = final_positive_var
        result['final_negative_mean'] = final_negative_mean
        result['final_negative_var'] = final_negative_var
        return result
    
    def check(self, curr_score, model, weights_file) :
        if self.best_score==None:
            self.best_score = curr_score
            state_dict = model.state_dict()
            torch.save(state_dict, weights_file)
        else:
            if self.best_score-curr_score>self.delta:
                self.best_score = curr_score
                self.num_bad_epochs = 0
                state_dict = model.state_dict()
                torch.save(state_dict, weights_file)
            else:
                self.num_bad_epochs += 1
        if self.num_bad_epochs==self.patience:
            return True
        else:
            return False
        
    def train_self(self,
                data_path: str,
                val_path: str,
                epochs: int,
                learning_rate: float,
                weights_file: str,
                hyperparameters_file: str) -> None:
        # Using cuda (GPU) if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')

        # Set the network in training mode
        network = self.to(device)
        network.train()

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
        
        # Applying image transformations to the input validation dataset
        val_dataset = torchvision.datasets.ImageFolder(
            root=val_path,
            transform=transforms)
    
        # Sets up a Python iterable over the input training dataset
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)

        # Sets up a Python iterable over the input validation dataset
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)

        # Save the hyperparameters used for training the VAE Network
        hyperparameters = {}
        hyperparameters["n_latent"] = self.n_latent
        hyperparameters["alpha"] = self.alpha
        hyperparameters["beta"] = self.beta
        hyperparameters["gamma"] = self.gamma
        hyperparameters["rho"] = self.rho
        hyperparameters["n_chan"] = self.n_chan
        hyperparameters["input_d"] = self.input_d
        torch.save(hyperparameters, hyperparameters_file)

        # Using Adam Optimizer for training
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            for data in train_loader:
                input, class_name = data
                input = input.to(device)

                # Passing the input image through VAE Network
                out, current_mu, current_logvar = network.forward(input, int(class_name))

                if int(class_name)==1:
                    other_mu, other_logvar = network.negative_latent_calc(network.encode(input))
                else:
                    other_mu, other_logvar = network.positive_latent_calc(network.encode(input))

                # KL-Divergence Loss (Regularization loss in VAE Loss function)
                kl_loss = torch.mul(input=torch.sum(current_mu.pow(2) + current_logvar.exp() - current_logvar - 1), other=0.5)

                # Mean Square Error Loss (Reconstruction loss in VAE Loss function)
                mse_loss = torch.nn.functional.mse_loss(out, input)

                # Repulsion Loss (For increasing the distance between the clusters belonging to positive and negative classes)
                repulsion_loss = max(0, self.rho-torch.sqrt(torch.sum(torch.square(current_mu-other_mu))))**2/self.rho

                # Total Loss =  (Alpha * Regularization Loss + Beta * Reconstruction Loss + Gamma * Repulsion Loss)/(Alpha + Beta + Gamma)
                loss = (torch.mul(kl_loss, self.alpha) + torch.mul(mse_loss, self.beta) + torch.mul(repulsion_loss, self.gamma)) / (self.alpha + self.beta + self.gamma)

                optimizer.zero_grad() # Sets gradients of all model parameters to zero
                loss.backward() # Perform Back-Propogation
                optimizer.step() # Performs a single optimization step (parameter update)
                epoch_loss += loss

            val_loss = 0
            for data in val_loader:
                input, class_name = data
                input = input.to(device)
                output, mu, logvar = network.forward(input, int(class_name))
                mse_val_loss = torch.nn.functional.mse_loss(output, input)
                val_loss += mse_val_loss
            if self.check(val_loss, network, weights_file):
                print(f'Early Stopping at epoch {epoch}')
                break
            print(f'Epoch: {epoch}; Training Loss: {epoch_loss}; Validation Loss: {val_loss}')
        print('Training finished, saving weights...')