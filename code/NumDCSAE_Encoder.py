from typing import Tuple
import torch

class NumDCSAE_Encoder(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 input_size: int) -> None:
        super(NumDCSAE_Encoder, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.input_size = input_size

        # Dense Encoder Bottleneck
        self.enc_dense1 = torch.nn.Linear(self.input_size, 2048)
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
        # Dense Encoder Bottleneck
        z = self.enc_dense1(x)
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        positive_mean, positive_logvar = self.positive_latent_calc(self.encode(x))
        negative_mean, negative_logvar = self.negative_latent_calc(self.encode(x))
        return positive_mean, positive_logvar, negative_mean, negative_logvar

    def testing(self,
                test_data: torch.Tensor,
                labels: torch.Tensor,
                weight_file: str):
        # Using cuda (GPU) if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
        network.load_state_dict(torch.load(weight_file))# Load weights from the .pt file
        network.eval() # Set the network in evalution mode

        final_input = []
        final_positive_mean = []
        final_negative_mean = []
        final_positive_var = []
        final_negative_var = []
        final_class = []
        for i in range(len(test_data)):
            input = test_data[i]
            class_name = labels[i]
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