from typing import Tuple
import torch

class NumDCSAE(torch.nn.Module):
    def __init__(self,
                 n_latent: int,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 rho: float,
                 input_size: int,
                 batch: int = 1, 
                 ) -> None:
        super(NumDCSAE, self).__init__()
        # Initializing the class variables
        self.n_latent = n_latent
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.batch = batch
        self.input_size = input_size

        # Early Stopping
        self.patience = 10
        self.delta = 1e-4
        self.best_score = None
        self.num_bad_epochs = 0

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

        # Dense Decoder Bottleneck
        self.dec_dense4 = torch.nn.Linear(self.n_latent, 250)
        self.dec_dense4_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense3 = torch.nn.Linear(250, 1000)
        self.dec_dense3_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense2 = torch.nn.Linear(1000, 2048)
        self.dec_dense2_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense1 = torch.nn.Linear(2048, self.input_size)
        self.dec_dense1_af = torch.nn.LeakyReLU(0.1)

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
                test_data: torch.Tensor,
                labels: torch.Tensor,
                weight_file: str):
        # Using cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
        network.load_state_dict(torch.load(weight_file)) # Load weights from the .pt file
        network.eval() # Set the network in evalution mode

        final_input = []
        final_class = []
        final_output = []
        final_positive_mean = []
        final_negative_mean = []
        final_positive_var = []
        final_negative_var = []
        for i in range(len(test_data)):
            input = test_data[i]
            class_name = int(labels[i])
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
                train_data: torch.Tensor,
                val_data: torch.Tensor,
                train_labels: torch.Tensor,
                val_labels: torch.Tensor,
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

        # Save the hyperparameters used for training the VAE Network
        hyperparameters = {}
        hyperparameters["n_latent"] = self.n_latent
        hyperparameters["alpha"] = self.alpha
        hyperparameters["beta"] = self.beta
        hyperparameters["gamma"] = self.gamma
        hyperparameters["rho"] = self.rho
        torch.save(hyperparameters, hyperparameters_file)

        # Using Adam Optimizer for training
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(train_data)):
                input = train_data[i]
                input = input.to(device)
                class_name = int(train_labels[i])

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
            for i in range(len(val_data)):
                input = val_data[i]
                input = input.to(device)
                class_name = int(val_labels[i])
                output, mu, logvar = network.forward(input, int(class_name))
                mse_val_loss = torch.nn.functional.mse_loss(output, input)
                val_loss += mse_val_loss
            if self.check(val_loss, network, weights_file):
                print(f'Early Stopping at epoch {epoch}')
                break
            print(f'Epoch: {epoch}; Training Loss: {epoch_loss}; Validation Loss: {val_loss}')
        print('Training finished, saving weights...')