from typing import Tuple
import torch

class DCSAE_Encoder(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 encoder) -> None:
        super(DCSAE_Encoder, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.encoder = encoder
        self.hidden_units = encoder.output_size()

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
        z = self.encoder.forward(x)

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

        dataset = self.encoder.preprocess(data_path)

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