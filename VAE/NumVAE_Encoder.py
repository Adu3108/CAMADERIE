import torch

class NumVAE_Encoder(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 input_size: int) -> None:
        super(NumVAE_Encoder, self).__init__()
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
        self.enc_dense4_mu = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4_mu_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4_var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4_var_af = torch.nn.LeakyReLU(0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Dense Encoder Bottleneck
        z = self.enc_dense1(x)
        z = self.enc_dense1_af(z)

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        mu = self.enc_dense4_mu(z)
        mu = self.enc_dense4_mu_af(mu)

        var = self.enc_dense4_var(z)
        var = self.enc_dense4_var_af(var)

        return mu, var

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
        final_mean = []
        final_var = []
        final_class = []
        for i in range(len(test_data)):
            input = test_data[i]
            class_name = labels[i]
            final_input.append(input)
            final_class.append(int(class_name))
            input = input.to(device)
            with torch.no_grad():
                mu, logvar = network.encode(input)
                final_mean.append(mu)
                final_var.append(torch.exp(logvar/2))
        
        result = dict()
        result['final_input'] = final_input
        result['final_class'] = final_class
        result['final_mean'] = final_mean
        result['final_var'] = final_var

        return result